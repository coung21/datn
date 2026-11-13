"""Phân tích sai số sCT theo vùng giải phẫu trên tập test: sai số nằm ở đâu, lệch về phía nào.

    python -m scripts.error_regions \
        --run "GN + AG" outputs/runs/unet2d_gn_ag_<...> \
        --run "Multi-view" outputs/runs/unet2d_gn_ag_mv_<...> \
        [--split test] [--out outputs/error_regions]

Mỗi khối (trong mask) được chia thành 5 vùng rời nhau theo CT thật, giãn / co trong mặt phẳng lát:
  bone_edge  dải EDGE_PX điểm ảnh hai bên mép vùng xương (CT > BONE_HU)
  bone_inner xương, trừ dải biên
  body_edge  dải BODY_PX điểm ảnh sát mép mask, trừ 2 vùng trên
  air        CT < AIR_HU, trừ các vùng trên (xoang, khí ruột)
  soft       phần còn lại
Với mỗi mô hình x bệnh nhân x vùng: số điểm ảnh, tỉ lệ thể tích, tỉ lệ trong tổng sai số tuyệt đối,
MAE và độ lệch có dấu (trung bình sCT - CT). Bảng tổng hợp lấy trung bình theo bệnh nhân, tách não / chậu.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ZONES = ["bone_edge", "bone_inner", "body_edge", "air", "soft"]
BONE_HU, AIR_HU = 200.0, -400.0
EDGE_PX, BODY_PX = 2, 4


def dilate(x: torch.Tensor, r: int) -> torch.Tensor:
    """Giãn mask bool [Z, H, W] r điểm ảnh trong mặt phẳng (cửa sổ vuông)."""
    return F.max_pool2d(x[:, None].float(), 2 * r + 1, stride=1, padding=r)[:, 0] > 0


def erode(x: torch.Tensor, r: int) -> torch.Tensor:
    """Co mask bool; ngoài khung ảnh coi như thuộc mask (mép khung không bị co)."""
    return ~dilate(~x, r)


def zone_masks(ct: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """ct (HU), mask (bool) [Z, H, W] -> {vùng: mask bool}; các vùng rời nhau, hợp lại bằng mask."""
    bone = (ct > BONE_HU) & mask
    bone_edge = dilate(bone, EDGE_PX) & ~erode(bone, EDGE_PX) & mask
    bone_inner = bone & ~bone_edge
    used = bone_edge | bone_inner
    body_edge = mask & ~erode(mask, BODY_PX) & ~used
    used = used | body_edge
    air = mask & (ct < AIR_HU) & ~used
    return {"bone_edge": bone_edge, "bone_inner": bone_inner, "body_edge": body_edge, "air": air,
            "soft": mask & ~used & ~air}


def zone_stats(sct: torch.Tensor, ct: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    """Thống kê từng vùng của 1 bệnh nhân."""
    err = sct - ct
    total_n, total_abs = int(mask.sum()), float(err[mask].abs().sum())
    rows = []
    for zone, m in zone_masks(ct, mask).items():
        n = int(m.sum())
        e = err[m]
        rows.append({"zone": zone, "n": n, "vol_frac": n / max(total_n, 1),
                     "err_frac": float(e.abs().sum()) / max(total_abs, 1e-9),
                     "mae": float(e.abs().mean()) if n else float("nan"),
                     "bias": float(e.mean()) if n else float("nan")})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình theo bệnh nhân cho từng mô hình x vùng cơ thể x vùng sai số."""
    g = df.groupby(["model", "region", "zone"], sort=False)
    out = g[["vol_frac", "err_frac", "mae", "bias"]].mean()
    out[["vol_frac", "err_frac"]] *= 100
    return out.rename(columns={"vol_frac": "vol_%", "err_frac": "err_%"}).reset_index()


def main():
    from src.data.dataset import build_dataset, load_split
    from src.data.preprocess import parse_id
    from src.eval.evaluate import load_checkpoint
    from src.inference.predict import predict_volume

    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "RUN"))
    p.add_argument("--ckpt", default="best")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default="outputs/error_regions")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for label, run in args.run:
        model, method, cfg, _ = load_checkpoint(Path(run) / f"{args.ckpt}.pt", device)
        ids = load_split(cfg["data"]["split"]["file"])[args.split]
        ds = build_dataset(cfg, ids, mode="val")
        for pid in ids:
            sct = torch.from_numpy(predict_volume(method, model, ds, pid, 1, cfg["inference"], device)).to(device)
            ct, mask = ds.load_volume(pid)
            ct, mask = torch.from_numpy(ct).to(device), torch.from_numpy(mask).to(device) > 0
            mae_all = float((sct - ct)[mask].abs().mean())
            rows += [{"model": label, "id": pid, **parse_id(pid), "mae_all": mae_all, **r}
                     for r in zone_stats(sct, ct, mask)]
        del model
        print(f"{label}: xong ({len(ids)} bệnh nhân)", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_patient.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
