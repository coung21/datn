"""Đếm tham số và đo VRAM đỉnh của UNet bằng torch.cuda.max_memory_allocated.

Ví dụ:
    python scripts/check_vram.py --batch-size 8 --image-size 256 --amp
    python scripts/check_vram.py --batch-size 16 --checkpoint --amp
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import UNetModel  # noqa: E402

BUDGET_GB = 16.0


def gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def make_inputs(model: UNetModel, batch_size: int, image_size: int, cond_channels: int, device):
    x = torch.randn(batch_size, model.in_channels - cond_channels, image_size, image_size, device=device)
    cond = torch.randn(batch_size, cond_channels, image_size, image_size, device=device)
    t = torch.rand(batch_size, device=device)
    return x, t, cond


def measure_inference(model, args, device) -> int:
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x, t, cond = make_inputs(model, args.batch_size, args.image_size, args.cond_channels, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
        out = model(x, t, cond)
    torch.cuda.synchronize(device)
    assert out.shape == (args.batch_size, model.out_channels, args.image_size, args.image_size)
    assert torch.isfinite(out).all(), "Output có NaN/Inf"
    return torch.cuda.max_memory_allocated(device)


def measure_train_step(model, args, device) -> int:
    """Một bước train đầy đủ: forward + backward + AdamW step (tính cả optimizer state)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # chạy 2 bước để optimizer state (exp_avg, exp_avg_sq) đã được cấp phát
    for _ in range(2):
        x, t, cond = make_inputs(model, args.batch_size, args.image_size, args.cond_channels, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            loss = (model(x, t, cond) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--cond-channels", type=int, default=3)
    p.add_argument("--amp", action="store_true", help="autocast float16")
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing khi train")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    model = UNetModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        use_checkpoint=args.checkpoint,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params / 1e6:.2f}M")

    if not torch.cuda.is_available():
        print("Không có CUDA: bỏ qua đo VRAM (cần chạy trên GPU, vd Colab/Kaggle).")
        return

    device = torch.device("cuda")
    model.to(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Batch {args.batch_size}, patch {args.image_size}x{args.image_size}, "
          f"AMP={args.amp}, checkpoint={args.checkpoint}")

    peak_inf = measure_inference(model, args, device)
    print(f"Inference peak VRAM : {gb(peak_inf):.2f} GB")

    if not args.skip_train:
        peak_train = measure_train_step(model, args, device)
        print(f"Train step peak VRAM: {gb(peak_train):.2f} GB")
        peak = max(peak_inf, peak_train)
    else:
        peak = peak_inf

    status = "OK" if gb(peak) <= BUDGET_GB else "VƯỢT"
    print(f"Budget {BUDGET_GB:.0f} GB: {status}")


if __name__ == "__main__":
    main()
