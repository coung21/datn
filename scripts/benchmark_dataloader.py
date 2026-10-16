"""Đo tốc độ dataloader (img/s) trên dữ liệu đã preprocess, so với ~16 img/s của GPU.

    python -X utf8 scripts/benchmark_dataloader.py --workers 0 2 4 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SliceDataset, load_split  # noqa: E402

GPU_IMG_PER_SEC = 16.1  # bf16, batch 8, outputs/benchmark/bf16_nockpt.csv


def build_train_dataset(cfg: dict) -> SliceDataset:
    d = cfg["data"]
    split = load_split(d["split"]["file"])
    return SliceDataset(
        d["processed_dir"], split["train"], mode="train",
        patch_size=d["patch_size"], context_slices=d["context_slices"], context_mm=d["context_mm"],
        min_foreground=d["min_foreground"], hflip=d["hflip"], ct_clip=d["ct_clip"],
        mr_percentiles=d["mr_percentiles"], pad_value=d["pad_value"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = build_train_dataset(cfg)
    bs = cfg["train"]["batch_size"]
    print(f"{len(ds)} lát train, batch {bs}, patch {cfg['data']['patch_size']}", flush=True)

    for w in args.workers:
        loader = DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=w, drop_last=True,
            pin_memory=cfg["data"]["pin_memory"], persistent_workers=w > 0,
        )
        it = iter(loader)
        for _ in range(args.warmup):  # gồm cả thời gian khởi động worker
            next(it)
        start = time.perf_counter()
        for _ in range(args.batches):
            batch = next(it)
        elapsed = time.perf_counter() - start
        img_s = args.batches * bs / elapsed
        status = "OK" if img_s >= GPU_IMG_PER_SEC else "NGHẼN"
        print(f"workers={w:2}: {img_s:7.1f} img/s ({img_s / GPU_IMG_PER_SEC:4.1f}x GPU) {status}", flush=True)
        assert batch["cond"].shape[1:] == (2 * cfg["data"]["context_slices"] + 1, *[cfg["data"]["patch_size"]] * 2)
        del it, loader


if __name__ == "__main__":
    main()
