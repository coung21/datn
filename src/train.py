"""Train 1 phương pháp MR -> CT theo config.

    python -X utf8 -m src.train --exp configs/exp/regression.yaml
    python -X utf8 -m src.train --resume outputs/runs/regression_20261016-231500
    python -X utf8 -m src.train --exp ... --set train.total_images=4000 data.num_workers=0

Đầu ra trong <output_dir>/runs/<run_name>_<YYYYmmdd-HHMMSS>/:
    config.yaml, train_log.csv, val_log.csv,
    checkpoints/step_XXXXXX.pt (chỉ trọng số), last.pt (để train tiếp), best.pt,
    val_full.csv (val toàn volume cho top-k checkpoint theo val nhanh),
    wandb_id.txt (để --resume ghi tiếp vào cùng run wandb)
"""

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import build_dataset, load_split
from .eval.evaluate import evaluate_volumes, load_checkpoint, val_fast
from .methods import build_method
from .models import UNetModel
from .utils.config import load_config
from .utils.logger import WandbLogger

AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp32": None}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_with_warmup(total_steps: int, warmup_steps: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def init_from_checkpoint(model: torch.nn.Module, path) -> None:
    """Khởi tạo từ checkpoint (vd teacher của reflow). Chỉ cho phép thiếu label_emb (khởi tạo 0,
    nên model có điều kiện lớp lúc đầu cho kết quả y hệt checkpoint)."""
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:  # lệch kích thước tensor
        raise ValueError(f"init_from {path} không khớp model: {e}") from e
    if unexpected or any(not k.startswith("label_emb.") for k in missing):
        raise ValueError(f"init_from {path} không khớp model: thiếu {missing}, thừa {unexpected}")
    print(f"Khởi tạo từ {path}" + (f" (mới: {missing})" if missing else ""), flush=True)


def infinite(loader):
    while True:
        yield from loader


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fields = None

    def log(self, row: dict) -> None:
        new = not self.path.exists()
        if self.fields is None:
            self.fields = list(row) if new else next(csv.reader(self.path.open()))
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--exp", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè config")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="train tiếp từ last.pt của run này (config đọc từ RUN_DIR/config.yaml, --set vẫn áp dụng)")
    args = p.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        if not (run_dir / "config.yaml").exists():
            raise SystemExit(f"{run_dir} không phải thư mục run (thiếu config.yaml)")
        cfg = load_config(run_dir / "config.yaml", None, args.set)
    else:
        cfg = load_config(args.config, args.exp, args.set)
        cfg.setdefault("run_name", Path(args.exp or "base").stem)
        run_dir = Path(cfg["output_dir"]) / "runs" / f"{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
        if run_dir.exists():
            raise SystemExit(f"{run_dir} đã tồn tại")
    tc, dc = cfg["train"], cfg["data"]
    print(f"Run: {run_dir}", flush=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = tc["cudnn_benchmark"]
    if tc["precision"] not in AUTOCAST_DTYPES:
        raise ValueError(f"precision {tc['precision']} chưa hỗ trợ (bf16 | fp32)")
    amp_dtype = AUTOCAST_DTYPES[tc["precision"]]

    method = build_method(**cfg["method"])
    expected_in = method.extra_in_channels + 2 * dc["context_slices"] + 1
    if cfg["model"]["in_channels"] != expected_in:
        raise ValueError(f"model.in_channels phải là {expected_in} cho method {cfg['method']['name']}")
    if method.needs_pairs:
        if not dc.get("reflow"):
            raise ValueError(f"method {cfg['method']['name']} cần data.reflow (cặp sinh bởi teacher)")
        if dc["patch_size"] != cfg["inference"]["roi_size"]:
            raise ValueError("reflow cần data.patch_size == inference.roi_size để noise sinh lại khớp với teacher")
    model = UNetModel(**cfg["model"])
    if tc.get("init_from") and not args.resume:
        init_from_checkpoint(model, tc["init_from"])
    model = model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham số, device {device}", flush=True)

    split = load_split(dc["split"]["file"])
    train_ds = build_dataset(cfg, split["train"], "train")
    val_ds = build_dataset(cfg, split["val"], "val")
    workers = dc["num_workers"]
    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True, num_workers=workers,
        pin_memory=dc["pin_memory"] and device.type == "cuda", persistent_workers=dc["persistent_workers"] and workers > 0,
    )

    images_per_step = tc["batch_size"] * tc["grad_accum"]
    total_steps = tc["total_images"] // images_per_step
    oc, sc = tc["optimizer"], tc["lr_schedule"]
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], betas=tuple(oc["betas"]), weight_decay=oc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(total_steps, sc["warmup_steps"], sc["min_lr"] / oc["lr"])
    )

    step = 0
    if args.resume:
        if not (run_dir / "last.pt").exists():
            raise SystemExit(f"{run_dir} chưa có last.pt để train tiếp")
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"Train tiếp từ step {step}", flush=True)

    print(f"{len(train_ds)} lát train, {len(val_ds)} lát val | {total_steps} optimizer step x {images_per_step} ảnh", flush=True)
    train_log, val_log = CsvLogger(run_dir / "train_log.csv"), CsvLogger(run_dir / "val_log.csv")
    wb = WandbLogger(cfg, run_dir, resume=bool(args.resume))
    progress = cfg.get("logging", {}).get("tqdm", True)
    # stdout bị chuyển ra file (chạy nền qua ssh) -> cập nhật thưa để log không bị ngập dòng
    bar = tqdm(total=total_steps, initial=step, desc=run_dir.name, dynamic_ncols=True, disable=not progress,
               mininterval=0.5 if sys.stderr.isatty() else 60)
    data = infinite(loader)
    start, window_start, window_loss, window_steps = time.perf_counter(), time.perf_counter(), 0.0, 0

    while step < total_steps:
        model.train()
        step_loss = 0.0
        for _ in range(tc["grad_accum"]):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in next(data).items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _ = method.loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở step {step}: {loss.item()}")
            (loss / tc["grad_accum"]).backward()
            step_loss += loss.item() / tc["grad_accum"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        window_loss += step_loss
        window_steps += 1
        bar.update(1)

        if step % tc["log_every"] == 0 or step == total_steps:
            now = time.perf_counter()
            row = {
                "step": step, "images": step * images_per_step, "lr": sched.get_last_lr()[0],
                "loss": window_loss / window_steps, "grad_norm": float(grad_norm),
                "img_per_s": window_steps * images_per_step / (now - window_start),
                "elapsed_h": (now - start) / 3600,
            }
            train_log.log(row)
            wb.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step)
            bar.set_postfix(loss=f"{row['loss']:.4f}", lr=f"{row['lr']:.1e}", img_s=f"{row['img_per_s']:.1f}")
            if not progress or not sys.stderr.isatty():
                tqdm.write(f"step {step}/{total_steps} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                           f"gn {row['grad_norm']:.2f} {row['img_per_s']:.1f} img/s {row['elapsed_h']:.2f}h")
            window_start, window_loss, window_steps = now, 0.0, 0

        if step % tc["save_every"] == 0 or step == total_steps:
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                       run_dir / "checkpoints" / f"step_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "step": step, "cfg": cfg}, run_dir / "last.pt")

        if step % tc["val_every"] == 0 or step == total_steps:
            t0 = time.perf_counter()
            res, examples = val_fast(method, model, val_ds, tc["val_fast_nfe"], cfg["inference"], device, progress)
            seconds = time.perf_counter() - t0
            val_log.log({"step": step, **res, "seconds": seconds})
            wb.log({**{f"val/{k}": v for k, v in res.items()}, "val/seconds": seconds}, step=step)
            wb.log_examples("val/examples", examples, step=step)
            tqdm.write(f"[val] step {step}: " + ", ".join(f"{k} {v:.2f}" for k, v in res.items())
                       + f" ({seconds:.0f}s)")

    bar.close()
    run_val_full(run_dir, cfg, method, split, device, wb, progress)
    wb.finish()


def run_val_full(run_dir: Path, cfg: dict, method, split: dict, device, wb=None, progress: bool = True) -> None:
    """Val toàn volume cho top-k checkpoint theo val nhanh; checkpoint MAE thấp nhất -> best.pt."""
    import pandas as pd

    vf = cfg["train"]["val_full"]
    val_log = pd.read_csv(run_dir / "val_log.csv").drop_duplicates("step", keep="last")
    key = f"mae_nfe{max(method.nfe_values([vf['nfe']]))}"
    if key not in val_log:
        key = sorted(c for c in val_log.columns if c.startswith("mae_nfe"))[-1]
    top = val_log.nsmallest(vf["top_k"], key)["step"].astype(int).tolist()

    results = []
    for s in top:
        path = run_dir / "checkpoints" / f"step_{s:06d}.pt"
        if not path.exists():
            continue
        print(f"[val full] step {s}", flush=True)
        model, _, _, _ = load_checkpoint(path, device)
        df = evaluate_volumes(method, model, cfg, split["val"], [vf["nfe"]], device, split.get("special"),
                              progress=progress)
        df.insert(0, "step", s)
        results.append(df)
    df = pd.concat(results)
    df.to_csv(run_dir / "val_full.csv", index=False)
    means = df.groupby("step")[["mae", "psnr", "ssim"]].mean()
    best = int(means["mae"].idxmin())
    shutil.copy(run_dir / "checkpoints" / f"step_{best:06d}.pt", run_dir / "best.pt")
    print(means.round(3).to_string(), flush=True)
    print(f"best.pt <- step {best}", flush=True)
    if wb is not None:
        wb.log_table("val_full/per_patient", df)
        wb.summary({"best_step": best, **{f"val_full/{k}": float(v) for k, v in means.loc[best].items()}})


if __name__ == "__main__":
    main()
