"""Benchmark model trong configs/base.yaml trên GPU: VRAM đỉnh, tốc độ train step, ước tính thời gian.

Chỉ đo phần tính toán của model (dữ liệu sinh ngẫu nhiên trên GPU), chưa tính dataloader.

Ví dụ:
    python -X utf8 scripts/benchmark.py
    python -X utf8 scripts/benchmark.py --batch-sizes 8 16 32 --precisions bf16 --set model.norm=group
"""

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# Kích thước trung vị (x, y, số lát) theo outputs/dataset_meta/dataset_meta.csv
MEDIAN_VOLUMES = {"brain": (219, 248, 190), "pelvis": (448, 294, 119)}


def gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def usable_gb(args, device) -> float:
    """VRAM dùng được: min(budget, VRAM thật), chừa 5% cho desktop/driver."""
    total = gb(torch.cuda.get_device_properties(device).total_memory)
    return min(args.budget_gb, total) * 0.95


def make_batch(model, batch_size: int, image_size: int, device):
    cond = torch.randn(batch_size, model.in_channels, image_size, image_size, device=device)
    target = torch.randn(batch_size, 1, image_size, image_size, device=device)
    return cond, target


def bench_train(args, batch_size: int, precision: str, device) -> dict:
    row = dict(batch_size=batch_size, precision=precision)
    model = build_model(args.cfg["model"]).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    amp = precision != "fp32"
    cond, target = make_batch(model, batch_size, args.image_size, device)

    def step():
        with torch.autocast("cuda", dtype=DTYPES[precision], enabled=amp):
            loss = (model(cond).float() - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize(device)
        sec_per_iter = (time.perf_counter() - start) / args.steps
    except torch.cuda.OutOfMemoryError:
        row.update(status="OOM")
        return row
    finally:
        del model, opt
        torch.cuda.empty_cache()

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    status = "OVER_BUDGET" if gb(peak_reserved) > usable_gb(args, device) else "OK"
    row.update(
        status=status,
        peak_alloc_gb=round(gb(peak_alloc), 2),
        peak_reserved_gb=round(gb(peak_reserved), 2),
        sec_per_iter=round(sec_per_iter, 4),
        samples_per_sec=round(batch_size / sec_per_iter, 1),
    )
    return row


def windows_per_slice(h: int, w: int, roi: int, overlap: float) -> int:
    """Số cửa sổ sliding window theo cách tính của MONAI."""
    interval = max(int(roi * (1 - overlap)), 1)

    def n(dim):
        return 1 if dim <= roi else math.ceil((dim - roi) / interval) + 1

    return n(h) * n(w)


@torch.no_grad()
def bench_inference(args, precision: str, device) -> float:
    """Trả về số giây cho mỗi patch (1 lần forward) ở batch args.infer_batch."""
    model = build_model(args.cfg["model"]).to(device).eval()
    cond, _ = make_batch(model, args.infer_batch, args.image_size, device)
    amp = precision != "fp32"
    with torch.autocast("cuda", dtype=DTYPES[precision], enabled=amp):
        for _ in range(args.warmup):
            model(cond)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            model(cond)
        torch.cuda.synchronize(device)
    del model
    torch.cuda.empty_cache()
    return (time.perf_counter() - start) / (args.steps * args.infer_batch)


def fmt_hours(sec: float) -> str:
    h = sec / 3600
    return f"{h:.1f} h" if h >= 1 else f"{sec / 60:.0f} min"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè configs/base.yaml")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--precisions", nargs="+", default=["bf16"], choices=list(DTYPES))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--budget-gb", type=float, default=16.0)
    p.add_argument("--total-images", type=int, nargs="+", default=[320_000, 400_000])
    p.add_argument("--num-slices", type=int, default=39_000,
                   help="số lát train cho 1 epoch (~70%% của 55.6k lát axial)")
    p.add_argument("--infer-batch", type=int, default=16)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--out", default="outputs/benchmark/benchmark.csv")
    args = p.parse_args()
    args.cfg = load_config("configs/base.yaml", None, args.set)

    if not torch.cuda.is_available():
        sys.exit("Cần CUDA để benchmark.")
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    props = torch.cuda.get_device_properties(device)
    # Trên Windows (WDDM) driver tràn sang RAM hệ thống thay vì báo OOM, chạy cực chậm.
    # Giới hạn allocator theo VRAM thật để cấu hình quá lớn báo OOM ngay.
    # (không chặn được hoàn toàn, nên còn kiểm tra thêm SPILL theo throughput ở dưới)
    torch.cuda.set_per_process_memory_fraction(usable_gb(args, device) * 1024**3 / props.total_memory, device)
    print(f"GPU: {props.name}, {gb(props.total_memory):.1f} GB, torch {torch.__version__}")
    print(f"Patch {args.image_size}x{args.image_size}, model {args.cfg['model']}, "
          f"warmup={args.warmup}, steps={args.steps}\n")

    rows = []
    header = f"{'bs':>4} {'prec':>5} {'status':>11} {'alloc GB':>9} {'resv GB':>8} {'s/iter':>8} {'img/s':>7}"
    print(header)
    print("-" * len(header))
    oom = set()
    prev_throughput = {}
    for prec, bs in itertools.product(args.precisions, args.batch_sizes):
        key = prec
        if key in oom:  # batch nhỏ hơn đã OOM/tràn thì batch lớn hơn cũng vậy
            r = dict(batch_size=bs, precision=prec, status="SKIP")
        else:
            r = bench_train(args, bs, prec, device)
            # tăng batch mà throughput giảm mạnh -> driver đang tràn VRAM sang RAM hệ thống
            if r["status"] == "OK" and r["samples_per_sec"] < 0.6 * prev_throughput.get(key, 0):
                r["status"] = "SPILL"
            if r["status"] != "OK":
                oom.add(key)
            else:
                prev_throughput[key] = r["samples_per_sec"]
        rows.append(r)
        print(f"{bs:>4} {prec:>5} {r['status']:>11} {r.get('peak_alloc_gb', '-'):>9} "
              f"{r.get('peak_reserved_gb', '-'):>8} {r.get('sec_per_iter', '-'):>8} {r.get('samples_per_sec', '-'):>7}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu {out}")

    ok = [r for r in rows if r["status"] == "OK"]
    if not ok:
        print("Không có cấu hình nào nằm trong budget.")
        return

    print("\n===== Ước tính thời gian train (chỉ tính compute, chưa gồm dataloader/validation) =====")
    for r in sorted(ok, key=lambda r: -r["samples_per_sec"])[:5]:
        tag = f"bs={r['batch_size']} {r['precision']}"
        iters = "  ".join(f"{n // 1000}k ảnh: {fmt_hours(n / r['samples_per_sec'])}" for n in args.total_images)
        epoch = args.num_slices / r["samples_per_sec"]
        print(f"{tag:<28} {r['samples_per_sec']:>6} img/s | 1 epoch ({args.num_slices} lát): "
              f"{fmt_hours(epoch)} | {iters}")

    prec = "bf16" if "bf16" in args.precisions else args.precisions[-1]
    sec_patch = bench_inference(args, prec, device)
    print(f"\n===== Ước tính inference ({prec}, batch {args.infer_batch}, overlap {args.overlap}) =====")
    print(f"{sec_patch * 1000:.2f} ms / patch")
    for site, (h, w, n_slices) in MEDIAN_VOLUMES.items():
        n_win = windows_per_slice(h, w, args.image_size, args.overlap)
        print(f"{site:<7} {h}x{w}x{n_slices}, {n_win} cửa sổ/lát | 1 volume: {n_slices * n_win * sec_patch:.1f}s")


if __name__ == "__main__":
    main()
