"""Tiền xử lý SynthRAD2023 Task1: .nii.gz -> .npy từng bệnh nhân (đọc bằng mmap khi train).

Mỗi bệnh nhân lưu ở <out>/<region>/<id>/:
    ct.npy    int16    HU, clip [-1024, 3000], ngoài mask = -1024
    mr.npy    float16  cường độ gốc, ngoài mask = 0
    mask.npy  uint8
    meta.json hình học gốc, bbox, thống kê MR trong mask, diện tích mask từng lát
Mảng có trục (z, y, x) và đã cắt theo bbox của mask. Chuẩn hóa về [-1, 1] làm trong dataset.

Chạy (trên server):
    python -X utf8 -m src.data.preprocess --raw-root data/raw/Task1_train --out data/processed
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CT_CLIP = (-1024, 3000)
MR_PERCENTILES = (0.5, 1.0, 50.0, 99.0, 99.5)
REGIONS = {"B": "brain", "P": "pelvis"}


def parse_id(pid: str) -> dict:
    """1BA001 -> task 1, region brain, center A."""
    return {"region": REGIONS[pid[1]], "center": pid[2]}


def mask_bbox(mask: np.ndarray, margin: tuple[int, int, int]) -> list[list[int]]:
    """Bbox [start, stop) theo từng trục của vùng mask > 0, nới thêm margin, kẹp trong ảnh."""
    bbox = []
    for axis, m in enumerate(margin):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(mask.any(axis=other))
        if idx.size == 0:
            raise ValueError("mask rỗng")
        bbox.append([max(int(idx[0]) - m, 0), min(int(idx[-1]) + 1 + m, mask.shape[axis])])
    return bbox


def preprocess_arrays(
    ct: np.ndarray, mr: np.ndarray, mask: np.ndarray, margin=(2, 8, 8)
) -> tuple[dict[str, np.ndarray], dict]:
    """Làm sạch và cắt bbox. Input (z, y, x); trả về (mảng, thông tin cho meta)."""
    if not (ct.shape == mr.shape == mask.shape):
        raise ValueError(f"shape lệch: ct {ct.shape}, mr {mr.shape}, mask {mask.shape}")
    m = mask > 0
    bbox = mask_bbox(m, margin)
    sl = tuple(slice(a, b) for a, b in bbox)

    ct = np.clip(ct, *CT_CLIP)
    ct = np.where(m, ct, CT_CLIP[0])[sl].round().astype(np.int16)
    mr = np.where(m, mr, 0)[sl].astype(np.float32)
    m = m[sl]

    inside = mr[m]
    info = {
        "bbox": bbox,
        "shape": list(m.shape),
        "mr_percentiles": {str(p): float(v) for p, v in zip(MR_PERCENTILES, np.percentile(inside, MR_PERCENTILES))},
        "mr_mean": float(inside.mean()),
        "mr_std": float(inside.std()),
        "slice_mask_area": m.sum(axis=(1, 2)).astype(int).tolist(),
    }
    arrays = {"ct": ct, "mr": mr.astype(np.float16), "mask": m.astype(np.uint8)}
    return arrays, info


def save_patient(out_dir: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def restore_full_volume(arr: np.ndarray, meta: dict, fill: float) -> np.ndarray:
    """Đặt mảng đã cắt bbox trở lại khung kích thước gốc (dùng khi ghép sCT)."""
    full = np.full(meta["orig_shape"], fill, dtype=arr.dtype)
    full[tuple(slice(a, b) for a, b in meta["bbox"])] = arr
    return full


def process_patient(pdir: Path, out_root: Path, margin) -> str:
    import SimpleITK as sitk

    pid = pdir.name
    imgs = {k: sitk.ReadImage(str(pdir / f"{k}.nii.gz")) for k in ("ct", "mr", "mask")}
    ref = imgs["ct"]
    for k in ("mr", "mask"):
        im = imgs[k]
        same = (
            im.GetSize() == ref.GetSize()
            and np.allclose(im.GetSpacing(), ref.GetSpacing())
            and np.allclose(im.GetOrigin(), ref.GetOrigin(), atol=1e-3)
            and np.allclose(im.GetDirection(), ref.GetDirection())
        )
        if not same:
            raise ValueError(f"{pid}: hình học {k} khác ct")

    arrays, info = preprocess_arrays(
        *(sitk.GetArrayFromImage(imgs[k]) for k in ("ct", "mr", "mask")), margin=margin
    )
    meta = {
        "id": pid,
        **parse_id(pid),
        "spacing": list(ref.GetSpacing()),       # (x, y, z) theo SimpleITK
        "origin": list(ref.GetOrigin()),
        "direction": list(ref.GetDirection()),
        "orig_shape": list(sitk.GetArrayFromImage(imgs["mask"]).shape),  # (z, y, x)
        **info,
    }
    save_patient(out_root / meta["region"] / pid, arrays, meta)
    return pid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="data/raw/Task1_train")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--margin", type=int, nargs=3, default=[2, 8, 8], help="lề bbox (z, y, x)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    raw, out = Path(args.raw_root), Path(args.out)
    pdirs = sorted(d for r in REGIONS.values() for d in (raw / r).iterdir() if d.is_dir() and d.name != "overview")
    print(f"{len(pdirs)} bệnh nhân -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(process_patient, d, out, tuple(args.margin)) for d in pdirs]
        for i, f in enumerate(futures, 1):
            f.result()
            if i % 20 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
