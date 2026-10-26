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
    """SSIM map trên cả ảnh (ngoài mask = nền), lấy trung bình trong mask.

    Như code chính thức SynthRAD2023: dịch về [0, hi - lo] trước khi tính (số hạng độ sáng
    của SSIM phụ thuộc giá trị tuyệt đối, nên không dịch sẽ cho kết quả khác).
    """
    sct, ct, m = _prepare(sct, ct, mask, hu_range)
    lo, hi = hu_range
    _, smap = structural_similarity(ct - lo, sct - lo, data_range=hi - lo, full=True)
    return float(smap[m].mean())


# 5 lớp mô theo HU của CT thật (báo cáo SynthRAD2023): [cận dưới, cận trên)
TISSUES = {
    "air": (-np.inf, -210),
    "fat": (-210, -20),
    "soft": (-20, 120),
    "marrow": (120, 555),
    "cortical": (555, np.inf),
}


def tissue_mae(sct, ct, mask, hu_range=HU_RANGE) -> dict[str, float]:
    """MAE trong mask, tách theo lớp mô của CT thật; lớp không có điểm nào -> NaN."""
    sct, ct, m = _prepare(sct, ct, mask, hu_range)
    err = np.abs(sct - ct)
    out = {}
    for name, (a, b) in TISSUES.items():
        sel = m & (ct >= a) & (ct < b)
        out[f"mae_{name}"] = float(err[sel].mean()) if sel.any() else float("nan")
    return out


def all_metrics(sct, ct, mask, hu_range=HU_RANGE) -> dict[str, float]:
    return {
        "mae": mae(sct, ct, mask, hu_range),
        "psnr": psnr(sct, ct, mask, hu_range),
        "ssim": ssim(sct, ct, mask, hu_range),
        **tissue_mae(sct, ct, mask, hu_range),
    }
