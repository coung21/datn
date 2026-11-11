"""So sánh chi phí tính toán giữa các mô hình: tham số, FLOPs, thời gian và VRAM khi suy luận, tốc độ train.

    python -m scripts.measure_cost \
        --run "UNet 2.5D" outputs/runs/unet2d_baseline_<...> \
        --run "GN + AG" outputs/runs/unet2d_gn_ag_<...> \
        --run "Multi-view" outputs/runs/unet2d_gn_ag_mv_<...> \
        [--n 10] [--out outputs/cost]

Mỗi mô hình:
  - số tham số;
  - FLOPs của 1 lần forward trên 1 patch roi x roi (torch.utils.flop_counter: 1 phép nhân-cộng = 2 FLOPs,
    chỉ tính conv / linear / matmul); in thêm GMACs = FLOPs / 2 (quy ước của fvcore, thop);
  - thời gian suy luận sliding window đúng cấu hình inference của run, trên n bệnh nhân test đầu mỗi vùng:
    mỗi bệnh nhân chạy 2 lần, lấy lần 2 (lần 1 làm nóng cudnn và cache đĩa); gồm cả đọc dữ liệu;
  - VRAM đỉnh allocated / reserved trong các lần đo đó (chỉ tiến trình này);
  - tốc độ train: trung vị img_per_s trong train_log.csv của run.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.data.dataset import build_dataset, load_split
from src.data.preprocess import parse_id
from src.eval.evaluate import load_checkpoint
from src.inference.predict import predict_volume

GB = 1024**3
REGIONS = ("brain", "pelvis")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, in_channels: int, size: int, device="cpu") -> int:
    """FLOPs của 1 lần forward với đầu vào [1, in_channels, size, size]."""
    x = torch.zeros(1, in_channels, size, size, device=device)
    kw = {"region": torch.zeros(1, dtype=torch.long, device=device)} if getattr(model, "needs_region", False) else {}
    was_training = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, **kw)
    model.train(was_training)
    return int(fc.get_total_flops())


def train_speed(run: Path) -> float:
    """Trung vị ảnh/giây khi train (nan nếu run không có train_log.csv)."""
    f = run / "train_log.csv"
    return float(pd.read_csv(f)["img_per_s"].median()) if f.exists() else float("nan")


def pick_patients(ids: list[str], n: int) -> list[str]:
    """n bệnh nhân đầu của mỗi vùng, giữ thứ tự trong split."""
    return [pid for r in REGIONS for pid in [i for i in ids if parse_id(i)["region"] == r][:n]]


def measure_inference(method, model, cfg: dict, pids: list[str], device) -> pd.DataFrame:
    ds = build_dataset(cfg, pids, mode="val")
    rows = []
    for pid in pids:
        predict_volume(method, model, ds, pid, 1, cfg["inference"], device)  # làm nóng
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        predict_volume(method, model, ds, pid, 1, cfg["inference"], device)
        torch.cuda.synchronize()
        rows.append({"id": pid, "region": parse_id(pid)["region"], "slices": ds.metas[pid]["shape"][0],
                     "seconds": time.perf_counter() - t,
                     "alloc_gb": torch.cuda.max_memory_allocated() / GB,
                     "reserved_gb": torch.cuda.max_memory_reserved() / GB})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "RUN"))
    p.add_argument("--ckpt", default="best")
    p.add_argument("--n", type=int, default=10, help="số bệnh nhân test mỗi vùng")
    p.add_argument("--out", default=None, help="thư mục ghi cost.csv và cost_per_patient.csv")
    args = p.parse_args()

    device = torch.device("cuda")
    rows, details = [], []
    for label, run in args.run:
        run = Path(run)
        model, method, cfg, _ = load_checkpoint(run / f"{args.ckpt}.pt", device)
        roi = cfg["inference"]["roi_size"]
        pids = pick_patients(load_split(cfg["data"]["split"]["file"])["test"], args.n)
        d = measure_inference(method, model, cfg, pids, device)
        d.insert(0, "mô hình", label)
        details.append(d)
        row = {"mô hình": label, "tham số (M)": count_params(model) / 1e6,
               "GFLOPs/patch": count_flops(model, cfg["model"]["in_channels"], roi, device) / 1e9}
        row["GMACs/patch"] = row["GFLOPs/patch"] / 2
        for r in REGIONS:
            g = d[d["region"] == r]
            row[f"s/BN {r}"] = g["seconds"].mean()
        row["ms/lát"] = 1000 * d["seconds"].sum() / d["slices"].sum()
        row["VRAM alloc (GB)"] = d["alloc_gb"].max()
        row["VRAM reserved (GB)"] = d["reserved_gb"].max()
        row["train img/s"] = train_speed(run)
        rows.append(row)
        print(f"{label}: xong ({len(pids)} bệnh nhân)", flush=True)
        del model
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    inf = cfg["inference"]
    print(f"\n== Chi phí (roi {inf['roi_size']}, sw_batch {inf['sw_batch_size']}, overlap {inf['overlap']}, "
          f"{inf['precision']}; {torch.cuda.get_device_name(0)})")
    print(table.round(3).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "cost.csv", index=False)
        pd.concat(details).to_csv(out / "cost_per_patient.csv", index=False)
        print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
