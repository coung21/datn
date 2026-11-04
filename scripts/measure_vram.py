"""Đo VRAM đỉnh khi inference (sliding window, cấu hình inference của run) trên vài bệnh nhân test.

    python -m scripts.measure_vram --run outputs/runs/<run> [--n 2] [--sw-batch 16]

In theo từng bệnh nhân: kích thước volume, thời gian, VRAM đỉnh allocated / reserved của tiến trình này
(không tính tiến trình khác đang dùng GPU)."""

import argparse
import time
from pathlib import Path

import torch

from src.data.dataset import build_dataset, load_split
from src.data.preprocess import parse_id
from src.eval.evaluate import load_checkpoint
from src.inference.predict import predict_volume

GB = 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--n", type=int, default=2, help="số bệnh nhân mỗi vùng")
    p.add_argument("--sw-batch", type=int, default=None, help="ghi đè inference.sw_batch_size")
    args = p.parse_args()

    device = torch.device("cuda")
    model, method, cfg, _ = load_checkpoint(Path(args.run) / f"{args.ckpt}.pt", device)
    inf = dict(cfg["inference"])
    if args.sw_batch:
        inf["sw_batch_size"] = args.sw_batch
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / GB
    print(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham số, trọng số trên GPU {weights:.3f} GB")
    print(f"inference: roi {inf['roi_size']}, sw_batch {inf['sw_batch_size']}, overlap {inf['overlap']}, {inf['precision']}")

    ids = load_split(cfg["data"]["split"]["file"])["test"]
    picked = [pid for r in ("brain", "pelvis") for pid in [i for i in ids if parse_id(i)["region"] == r][: args.n]]
    ds = build_dataset(cfg, picked, mode="val")
    for pid in picked:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        predict_volume(method, model, ds, pid, 1, inf, device)
        torch.cuda.synchronize()
        print(f"{pid} shape {tuple(ds.metas[pid]['shape'])}: {time.perf_counter() - t:.1f}s, "
              f"đỉnh allocated {torch.cuda.max_memory_allocated() / GB:.2f} GB, "
              f"reserved {torch.cuda.max_memory_reserved() / GB:.2f} GB")


if __name__ == "__main__":
    main()
