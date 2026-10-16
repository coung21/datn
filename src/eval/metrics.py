"""Metric theo cách đánh giá SynthRAD2023: tính trong mask, trên thang HU đã clip."""

import numpy as np
from skimage.metrics import structural_similarity

HU_RANGE = (-1024.0, 3000.0)


def _prepare(sct: np.ndarray, ct: np.ndarray, mask: np.ndarray, hu_range=HU_RANGE):
    lo, hi = hu_range
    m = mask > 0
    sct = np.where(m, np.clip(sct, lo, hi), lo).astype(np.float32)
    ct = np.where(m, np.clip(ct, lo, hi), lo).astype(np.float32)
    return sct, ct, m


def mae(sct, ct, mask, hu_range=HU_RANGE) -> float:
    sct, ct, m = _prepare(sct, ct, mask, hu_range)
    return float(np.abs(sct - ct)[m].mean())


def psnr(sct, ct, mask, hu_range=HU_RANGE) -> float:
    sct, ct, m = _prepare(sct, ct, mask, hu_range)
    mse = float(((sct - ct) ** 2)[m].mean())
    data_range = hu_range[1] - hu_range[0]
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(sct, ct, mask, hu_range=HU_RANGE) -> float:
    """SSIM map trên cả ảnh (ngoài mask = nền), lấy trung bình trong mask."""
    sct, ct, m = _prepare(sct, ct, mask, hu_range)
    _, smap = structural_similarity(
        ct, sct, data_range=hu_range[1] - hu_range[0], full=True
    )
    return float(smap[m].mean())


def all_metrics(sct, ct, mask, hu_range=HU_RANGE) -> dict[str, float]:
    return {
        "mae": mae(sct, ct, mask, hu_range),
        "psnr": psnr(sct, ct, mask, hu_range),
        "ssim": ssim(sct, ct, mask, hu_range),
    }
