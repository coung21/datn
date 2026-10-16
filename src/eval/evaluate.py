"""Đánh giá: val nhanh (lát cố định) trong lúc train, và val/test toàn volume.

    python -X utf8 -m src.eval.evaluate --run outputs/runs/regression_20261016-231500 --ckpt best --split test
    python -X utf8 -m src.eval.evaluate --run outputs/runs/regression_20261016-231500 --ckpt best --split test --save-nii
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data.dataset import SliceDataset, build_dataset, load_split
from ..data.preprocess import parse_id
from ..inference.predict import predict_volume, save_nifti
from ..methods import Method, build_method
from ..models import UNetModel
from .metrics import all_metrics, mae

METRICS = ["mae", "psnr", "ssim"]


def load_checkpoint(path, device) -> tuple[torch.nn.Module, Method, dict, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]
    model = UNetModel(**cfg["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), build_method(cfg["method"]["name"]), cfg, ckpt


def val_fast(
    method: Method, model, val_ds: SliceDataset, nfe_list, inf_cfg, device, progress: bool = True
) -> tuple[dict[str, float], list[dict]]:
    """MAE (HU, trong mask) trên các lát val cố định, trung bình theo bệnh nhân.

    Trả thêm ảnh mẫu (NFE lớn nhất): lát giữa của bệnh nhân val đầu tiên mỗi region.
    """
    by_pid = defaultdict(list)
    for pid, z in val_ds.index:
        by_pid[pid].append(z)
    example_pids = {}
    for pid in by_pid:
        example_pids.setdefault(parse_id(pid)["region"], pid)

    model.eval()
    res, examples = {}, []
    nfes = method.nfe_values(list(nfe_list))
    for nfe in nfes:
        maes = []
        for pid, zs in tqdm(by_pid.items(), desc=f"val nfe={nfe}", leave=False, disable=not progress):
            sct = predict_volume(method, model, val_ds, pid, nfe, inf_cfg, device, slices=zs)
            ct, mask = val_ds.load_volume(pid)
            maes.append(mae(sct, ct[zs], mask[zs]))
            if nfe == nfes[-1] and pid in example_pids.values():
                i = len(zs) // 2
                mr = val_ds.load_slice(pid, zs[i])[0]
                examples.append({"id": pid, "z": zs[i], "mr": mr[len(mr) // 2], "sct": sct[i],
                                 "ct": ct[zs[i]], "mask": mask[zs[i]]})
        res[f"mae_nfe{nfe}"] = float(np.mean(maes))
    return res, examples


def evaluate_volumes(
    method: Method, model, cfg: dict, ids, nfe_list, device, special=None, nii_dir: Path | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Metric toàn volume cho từng bệnh nhân x NFE."""
    ds = build_dataset(cfg, list(ids), "val")
    special = special or {}
    rows = []
    for nfe in method.nfe_values(list(nfe_list)):
        bar = tqdm(ids, desc=f"eval nfe={nfe}", disable=not progress)
        for pid in bar:
            start = time.perf_counter()
            sct = predict_volume(method, model, ds, pid, nfe, cfg["inference"], device)
            seconds = time.perf_counter() - start
            ct, mask = ds.load_volume(pid)
            rows.append({
                "id": pid, **parse_id(pid), "special": pid in special, "nfe": nfe,
                **all_metrics(sct, ct, mask, cfg["eval"]["hu_range"]), "seconds": seconds,
            })
            if nii_dir is not None:
                save_nifti(sct, ds.metas[pid], nii_dir / f"nfe{nfe}" / f"{pid}.nii.gz")
            bar.set_postfix(id=pid, mae=f"{rows[-1]['mae']:.1f}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn theo NFE x region, có thêm 'all' và bản bỏ ca đặc biệt."""
    parts = []
    for subset, d in (("all", df), ("no_special", df[~df["special"]])):
        for (nfe, region), g in list(d.groupby(["nfe", "region"])) + [((n, "all"), g) for n, g in d.groupby("nfe")]:
            row = {"subset": subset, "nfe": nfe, "region": region, "n": len(g)}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
            parts.append(row)
    return pd.DataFrame(parts).sort_values(["subset", "nfe", "region"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="thư mục run, vd outputs/runs/regression_20261016-231500")
    p.add_argument("--ckpt", default="best", help="best | last | đường dẫn .pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--nfe", type=int, nargs="+", default=None, help="mặc định: inference.nfe trong config")
    p.add_argument("--save-nii", action="store_true")
    args = p.parse_args()

    run = Path(args.run)
    ckpt_path = run / f"{args.ckpt}.pt" if args.ckpt in ("best", "last") else Path(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method, cfg, _ = load_checkpoint(ckpt_path, device)
    split = load_split(cfg["data"]["split"]["file"])
    nfe_list = args.nfe or cfg["inference"]["nfe"]

    out = run / "eval" / f"{args.split}_{ckpt_path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate_volumes(
        method, model, cfg, split[args.split], nfe_list, device, split.get("special"),
        nii_dir=out / "nii" if args.save_nii else None,
    )
    df.to_csv(out / "per_patient.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
