"""Sinh sCT bằng sliding window theo từng lát, ghép volume và ghi .nii.gz."""

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from ..data.dataset import SliceDataset, denormalize_ct
from ..data.preprocess import CT_CLIP, restore_full_volume
from ..methods import Method

AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp32": None}


def predict_slices(
    method: Method, model, cond: torch.Tensor, nfe: int, inf_cfg: dict, device: torch.device
) -> torch.Tensor:
    """cond [N, C, H, W] (H, W tùy ý) -> CT chuẩn hóa [N, 1, H, W] trên CPU."""
    dtype = AUTOCAST_DTYPES[inf_cfg["precision"]]
    roi = inf_cfg["roi_size"]
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        out = sliding_window_inference(
            cond.to(device),
            roi_size=(roi, roi),
            sw_batch_size=inf_cfg["sw_batch_size"],
            predictor=lambda x: method.predict(model, x, nfe),
            overlap=inf_cfg["overlap"],
            mode=inf_cfg["blend_mode"],
            cval=inf_cfg["pad_value"],
        )
    return out.float().cpu()


def predict_volume(
    method: Method,
    model,
    ds: SliceDataset,
    pid: str,
    nfe: int,
    inf_cfg: dict,
    device: torch.device,
    slices: list[int] | None = None,
    chunk: int = 16,
) -> np.ndarray:
    """sCT (HU, đã cắt bbox) cho các lát `slices` (mặc định: cả volume) của bệnh nhân pid.

    Ngoài mask gán -1024 như ground truth.
    """
    meta = ds.metas[pid]
    zs = list(range(meta["shape"][0])) if slices is None else list(slices)
    out = np.empty((len(zs), *meta["shape"][1:]), np.float32)
    for s in range(0, len(zs), chunk):
        part = zs[s:s + chunk]
        cond = torch.from_numpy(np.stack([ds.load_slice(pid, z)[0] for z in part]))
        out[s:s + len(part)] = predict_slices(method, model, cond, nfe, inf_cfg, device)[:, 0].numpy()

    hu = denormalize_ct(out, ds.ct_clip)
    _, mask = ds.load_volume(pid)
    return np.where(mask[zs] > 0, hu, CT_CLIP[0]).astype(np.float32)


def save_nifti(hu: np.ndarray, meta: dict, path: Path) -> None:
    """Ghi sCT về đúng kích thước và hình học của ảnh gốc."""
    import SimpleITK as sitk

    full = restore_full_volume(np.round(hu).astype(np.int16), meta, fill=CT_CLIP[0])
    img = sitk.GetImageFromArray(full)
    img.SetSpacing(meta["spacing"])
    img.SetOrigin(meta["origin"])
    img.SetDirection(meta["direction"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)
