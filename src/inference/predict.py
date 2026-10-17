"""Sinh sCT theo từng lát, ghép volume và ghi .nii.gz.

Mỗi lát được pad lên >= roi (giá trị nền, giống lúc train) rồi đưa cả lát cho method.sample;
model được bọc sliding window nên mỗi lần gọi net là 1 trường kết quả trên cả lát
(flow matching: 1 quỹ đạo duy nhất cho cả lát, không ghép mẫu độc lập của từng cửa sổ).
"""

import zlib
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from ..data.dataset import SliceDataset, denormalize_ct, pad_to
from ..data.preprocess import CT_CLIP, restore_full_volume
from ..methods import Method

AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp32": None}


def slice_noise(pid: str, z: int, shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    """Noise [1, H, W] cố định theo (bệnh nhân, lát): kết quả tái lập được trên mọi máy."""
    g = torch.Generator().manual_seed(zlib.crc32(f"{pid}:{z}".encode()) ^ seed)
    return torch.randn((1, *shape), generator=g)


def windowed(model, inf_cfg: dict):
    """net(inp, t): model chạy bằng sliding window trên cả lát (H, W >= roi_size)."""
    roi = inf_cfg["roi_size"]

    def net(inp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return sliding_window_inference(
            inp, roi_size=(roi, roi), sw_batch_size=inf_cfg["sw_batch_size"],
            predictor=lambda w: model(w, t), overlap=inf_cfg["overlap"],
            mode=inf_cfg["blend_mode"], cval=inf_cfg["pad_value"],
        ).float()

    return net


def padded_shape(h: int, w: int, roi: int) -> tuple[int, int]:
    return max(h, roi), max(w, roi)


def predict_slices(
    method: Method, model, cond: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor, nfe: int,
    inf_cfg: dict, device: torch.device,
) -> torch.Tensor:
    """cond [N, C, H, W], mask [N, 1, H, W] (H, W tùy ý), noise [N, 1, *padded_shape(H, W)]
    -> CT chuẩn hóa [N, 1, H, W] trên CPU."""
    h, w = cond.shape[-2:]
    roi = inf_cfg["roi_size"]
    cond_p = torch.from_numpy(pad_to(cond.numpy(), roi, inf_cfg["pad_value"]))
    mask_p = torch.from_numpy(pad_to(mask.numpy(), roi, 0.0))
    if tuple(noise.shape[-2:]) != tuple(cond_p.shape[-2:]):
        raise ValueError(f"noise {tuple(noise.shape[-2:])} phải có kích thước sau pad {tuple(cond_p.shape[-2:])}")
    top, left = (cond_p.shape[-2] - h) // 2, (cond_p.shape[-1] - w) // 2  # pad_to pad đều 2 phía

    dtype = AUTOCAST_DTYPES[inf_cfg["precision"]]
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        out = method.sample(windowed(model, inf_cfg), cond_p.to(device), mask_p.to(device), noise.to(device), nfe)
    return out.float().cpu()[..., top:top + h, left:left + w]


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
        loaded = [ds.load_slice(pid, z) for z in part]
        cond = torch.from_numpy(np.stack([mr for mr, _, _ in loaded]))
        mask = torch.from_numpy(np.stack([m for _, _, m in loaded]))
        # noise trên cả lát đã pad: vùng pad cũng có noise như lúc train (pad CT trước, thêm noise sau)
        shape = padded_shape(*meta["shape"][1:], inf_cfg["roi_size"])
        noise = torch.stack([slice_noise(pid, z, shape) for z in part])
        out[s:s + len(part)] = predict_slices(method, model, cond, mask, noise, nfe, inf_cfg, device)[:, 0].numpy()

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
