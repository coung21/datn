"""Chi phí tính toán của U-Net 3D (trọng số ngẫu nhiên, chỉ đo chi phí, không train) để so với UNet 2.5D.

    python -m scripts.measure_cost_3d --config configs/exp/unet2d_gn_ag.yaml [--roi 128] [--sw-batch 2] [--n 10] \
        [--out outputs/cost_3d]

U-Net 3D giữ đúng cấu trúc UNet 2.5D + GN + AG (src/models/unet2d.py), chỉ đổi mọi phép 2D sang 3D:
2 x [Conv3x3x3 -> GN -> ReLU] mỗi tầng, MaxPool 2x2x2, ConvTranspose 2x2x2, attention gate trên skip, đầu ra Conv1x1x1.
Đầu vào 1 kênh MR (khối 3D tự có ngữ cảnh xuyên lát). Hai biến thể:
  - kênh như bản 2D: 64 -> 128 -> 256 -> 512 -> 1024;
  - bản nhẹ kiểu nnU-Net: 32 -> 64 -> 128 -> 256 -> 320.
Đo giống scripts/measure_cost.py: số tham số; FLOPs 1 lần forward trên 1 patch roi^3 (1 nhân-cộng = 2 FLOPs);
thời gian sliding window 3D (overlap, gaussian, bf16 theo config) trên n bệnh nhân test đầu mỗi vùng,
mỗi bệnh nhân chạy 2 lần lấy lần 2, gồm cả đọc dữ liệu; VRAM đỉnh.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.utils.flop_counter import FlopCounterMode

from scripts.measure_cost import GB, REGIONS, count_params, pick_patients
from src.data.dataset import build_dataset, load_split
from src.data.preprocess import parse_id
from src.inference.predict import AUTOCAST_DTYPES
from src.utils.config import load_config

VARIANTS = {
    "UNet 3D (64→1024)": [64, 128, 256, 512, 1024],
    "UNet 3D nhẹ (32→320)": [32, 64, 128, 256, 320],
}


def gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch), ch)


class ConvBlock3D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False), gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False), gn(out_ch), nn.ReLU(inplace=True),
        )


class AttentionGate3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        inter = max(ch // 2, 1)
        self.w_x = nn.Sequential(nn.Conv3d(ch, inter, 1, bias=False), gn(inter))
        self.w_g = nn.Sequential(nn.Conv3d(ch, inter, 1, bias=False), gn(inter))
        self.psi = nn.Conv3d(inter, 1, 1)

    def forward(self, x, g):
        return x * torch.sigmoid(self.psi(torch.relu(self.w_x(x) + self.w_g(g))))


class UNet3D(nn.Module):
    def __init__(self, chs: list[int], in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.enc = nn.ModuleList([ConvBlock3D(in_ch if i == 0 else chs[i - 1], c) for i, c in enumerate(chs)])
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList([nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2) for i in range(len(chs) - 1)])
        self.gate = nn.ModuleList([AttentionGate3D(chs[i]) for i in range(len(chs) - 1)])
        self.dec = nn.ModuleList([ConvBlock3D(2 * chs[i], chs[i]) for i in range(len(chs) - 1)])
        self.head = nn.Conv3d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(len(self.dec))):
            g = self.up[i](x)
            x = self.dec[i](torch.cat([self.gate[i](skips[i], g), g], 1))
        return self.head(x)


def count_flops_3d(model: nn.Module, roi: int, device) -> int:
    x = torch.zeros(1, 1, roi, roi, roi, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return int(fc.get_total_flops())


def predict_3d(model, ds, pid: str, inf: dict, roi: int, sw_batch: int, device) -> None:
    mr = np.asarray(ds._arrays(pid)["mr"], dtype=np.float32)  # [Z, H, W], gồm cả đọc từ đĩa
    x = torch.from_numpy(mr)[None, None].to(device)
    dtype = AUTOCAST_DTYPES[inf["precision"]]
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        out = sliding_window_inference(x, roi_size=(roi,) * 3, sw_batch_size=sw_batch, predictor=model,
                                       overlap=inf["overlap"], mode=inf["blend_mode"], cval=inf["pad_value"])
    out.float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp/unet2d_gn_ag.yaml")
    p.add_argument("--roi", type=int, default=128)
    p.add_argument("--sw-batch", type=int, default=2)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(exp=args.config)
    inf = cfg["inference"]
    device = torch.device("cuda")
    pids = pick_patients(load_split(cfg["data"]["split"]["file"])["test"], args.n)
    ds = build_dataset(cfg, pids, mode="val")

    rows, details = [], []
    for label, chs in VARIANTS.items():
        model = UNet3D(chs).to(device).eval()
        row = {"mô hình": label, "tham số (M)": count_params(model) / 1e6,
               "GFLOPs/patch": count_flops_3d(model, args.roi, device) / 1e9, "patch": f"{args.roi}^3"}
        row["MFLOPs/voxel"] = row["GFLOPs/patch"] * 1e3 / args.roi**3
        d = []
        try:
            for pid in pids:
                predict_3d(model, ds, pid, inf, args.roi, args.sw_batch, device)  # làm nóng
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                predict_3d(model, ds, pid, inf, args.roi, args.sw_batch, device)
                torch.cuda.synchronize()
                d.append({"mô hình": label, "id": pid, "region": parse_id(pid)["region"],
                          "slices": ds.metas[pid]["shape"][0], "seconds": time.perf_counter() - t,
                          "alloc_gb": torch.cuda.max_memory_allocated() / GB,
                          "reserved_gb": torch.cuda.max_memory_reserved() / GB})
        except torch.cuda.OutOfMemoryError:
            row["ghi chú"] = f"hết VRAM ở sw_batch {args.sw_batch}"
        d = pd.DataFrame(d)
        if len(d):
            for r in REGIONS:
                row[f"s/BN {r}"] = d[d["region"] == r]["seconds"].mean()
            row["ms/lát"] = 1000 * d["seconds"].sum() / d["slices"].sum()
            row["VRAM alloc (GB)"] = d["alloc_gb"].max()
            row["VRAM reserved (GB)"] = d["reserved_gb"].max()
        rows.append(row)
        details.append(d)
        print(f"{label}: xong", flush=True)
        del model
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n== U-Net 3D (roi {args.roi}^3, sw_batch {args.sw_batch}, overlap {inf['overlap']}, {inf['precision']}; "
          f"{torch.cuda.get_device_name(0)})")
    print(table.round(3).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "cost.csv", index=False)
        pd.concat(details).to_csv(out / "cost_per_patient.csv", index=False)
        print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
