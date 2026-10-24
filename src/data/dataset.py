"""Dataset 2.5D MR -> CT đọc từ data/processed (xem src/data/preprocess.py).

Mỗi mẫu là 1 lát axial z:
    cond : [2*context_slices+1, H, W]  MR các lát z + k*stride, stride = round(context_mm / spacing_z)
    ct   : [1, H, W]                   CT lát z
    mask : [1, H, W]
Giá trị MR/CT đã chuẩn hóa về [-1, 1].

mode="train": pad lên patch_size (giá trị nền), crop ngẫu nhiên có tâm trong mask, lật ngang.
mode="val"  : val_slices_per_patient lát cố định mỗi bệnh nhân, trả nguyên lát (chạy sliding window).
"""

import json
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import CT_CLIP, parse_id


def normalize_ct(hu: np.ndarray, clip=CT_CLIP) -> np.ndarray:
    lo, hi = clip
    return (np.clip(hu.astype(np.float32), lo, hi) - lo) / (hi - lo) * 2 - 1


def denormalize_ct(x, clip=CT_CLIP):
    lo, hi = clip
    return (x + 1) / 2 * (hi - lo) + lo


def normalize_mr(mr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    mr = np.clip(mr.astype(np.float32), p_low, p_high)
    return (mr - p_low) / max(p_high - p_low, 1e-6) * 2 - 1


def slice_stride(spacing_z: float, context_mm: float) -> int:
    return max(1, round(context_mm / spacing_z))


def neighbor_indices(z: int, n_slices: int, context_slices: int, stride: int) -> list[int]:
    """Chỉ số các lát z + k*stride, k in [-context_slices, context_slices], kẹp vào [0, n-1]."""
    return [min(max(z + k * stride, 0), n_slices - 1) for k in range(-context_slices, context_slices + 1)]


def val_slice_indices(slice_mask_area, n: int, min_area_ratio: float = 0.25) -> list[int]:
    """n lát cách đều trong các lát có diện tích mask >= min_area_ratio * lớn nhất."""
    area = np.asarray(slice_mask_area)
    cand = np.flatnonzero(area >= min_area_ratio * area.max())
    picks = np.linspace(0, len(cand) - 1, min(n, len(cand))).round().astype(int)
    return sorted({int(cand[i]) for i in picks})


def pad_to(arr: np.ndarray, size: int, value: float) -> np.ndarray:
    """Pad đều 2 phía để H, W >= size."""
    h, w = arr.shape[-2:]
    ph, pw = max(size - h, 0), max(size - w, 0)
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)]
    return np.pad(arr, pad, constant_values=value)


REGION_INDEX = {"brain": 0, "pelvis": 1}  # nhãn lớp cho model có num_classes


def padded_shape(h: int, w: int, size: int) -> tuple[int, int]:
    """Kích thước lát sau pad_to(size)."""
    return max(h, size), max(w, size)


def slice_noise(pid: str, z: int, shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    """Noise [1, H, W] cố định theo (bệnh nhân, lát, seed) trên lát đã pad: tái lập được trên mọi máy.

    Dùng chung cho suy luận (seed 0) và để sinh lại noise của cặp reflow lúc train.
    """
    g = torch.Generator().manual_seed(zlib.crc32(f"{pid}:{z}".encode()) ^ seed)
    return torch.randn((1, *shape), generator=g)


def load_split(path) -> dict:
    return json.loads(Path(path).read_text())


def build_dataset(cfg: dict, ids: list[str], mode: str) -> "SliceDataset":
    """Tạo SliceDataset từ phần data của config."""
    d = cfg["data"]
    return SliceDataset(
        d["processed_dir"], ids, mode=mode,
        patch_size=d["patch_size"], context_slices=d["context_slices"], context_mm=d["context_mm"],
        min_foreground=d["min_foreground"], hflip=d["hflip"], ct_clip=d["ct_clip"],
        mr_percentiles=d["mr_percentiles"], pad_value=d["pad_value"],
        val_slices_per_patient=d["val_slices_per_patient"], reflow=d.get("reflow"),
    )


class SliceDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        ids: list[str],
        mode: str = "train",
        patch_size: int = 256,
        context_slices: int = 1,
        context_mm: float = 2.0,
        min_foreground: float = 0.1,
        hflip: float = 0.5,
        ct_clip=CT_CLIP,
        mr_percentiles=(0.5, 99.5),
        pad_value: float = -1.0,
        val_slices_per_patient: int = 5,
        max_crop_tries: int = 10,
        reflow: dict | None = None,
    ):
        assert mode in ("train", "val"), mode
        self.root = Path(processed_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.context_slices = context_slices
        self.context_mm = context_mm
        self.min_foreground = min_foreground
        self.hflip = hflip
        self.ct_clip = tuple(ct_clip)
        self.mr_keys = tuple(str(float(p)) for p in mr_percentiles)
        self.pad_value = pad_value
        self.max_crop_tries = max_crop_tries
        # reflow: {"file": tên .npy sCT của teacher (HU), "noise_seed": seed noise đã dùng khi sinh}
        # -> mẫu train có thêm x1 (sCT teacher) và noise (sinh lại, cùng crop/lật)
        self.reflow = reflow

        self.metas = {pid: json.loads((self._dir(pid) / "meta.json").read_text()) for pid in ids}
        self.index: list[tuple[str, int]] = []
        for pid in ids:
            area = self.metas[pid]["slice_mask_area"]
            if mode == "train":
                zs = np.flatnonzero(np.asarray(area) > 0).tolist()
            else:
                zs = val_slice_indices(area, val_slices_per_patient)
            self.index += [(pid, z) for z in zs]
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _dir(self, pid: str) -> Path:
        return self.root / parse_id(pid)["region"] / pid

    def __getstate__(self):
        # mmap không pickle sang worker (Windows spawn sẽ copy cả mảng) -> mỗi worker tự mở lại
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def _arrays(self, pid: str) -> dict[str, np.ndarray]:
        if pid not in self._cache:
            d = self._dir(pid)
            # cặp reflow chỉ có cho tập train và chỉ dùng khi train
            with_pairs = self.reflow and self.mode == "train"
            keys = ("ct", "mr", "mask") + ((self.reflow["file"],) if with_pairs else ())
            self._cache[pid] = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in keys}
        return self._cache[pid]

    def load_volume(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """CT (HU, float32) và mask của cả volume (sau bbox) - làm ground truth khi đánh giá."""
        arr = self._arrays(pid)
        return np.asarray(arr["ct"], dtype=np.float32), np.asarray(arr["mask"])

    def load_slice(self, pid: str, z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lát z nguyên kích thước (sau bbox), đã chuẩn hóa: mr [C,H,W], ct [1,H,W], mask [1,H,W]."""
        meta, arr = self.metas[pid], self._arrays(pid)
        stride = slice_stride(meta["spacing"][2], self.context_mm)
        zs = neighbor_indices(z, meta["shape"][0], self.context_slices, stride)
        p = meta["mr_percentiles"]
        mr = normalize_mr(np.asarray(arr["mr"][zs]), p[self.mr_keys[0]], p[self.mr_keys[1]])
        ct = normalize_ct(np.asarray(arr["ct"][z])[None], self.ct_clip)
        mask = np.asarray(arr["mask"][z])[None].astype(np.float32)
        return mr, ct, mask

    def _crop_origin(self, mask: np.ndarray) -> tuple[int, int]:
        """Góc trên-trái của patch có tâm là 1 điểm ngẫu nhiên trong mask."""
        size = self.patch_size
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        best = (-1.0, (h - size) // 2, (w - size) // 2)
        if len(ys) == 0:
            return best[1], best[2]
        for _ in range(self.max_crop_tries):
            k = int(torch.randint(len(ys), (1,)))
            y0 = min(max(int(ys[k]) - size // 2, 0), h - size)
            x0 = min(max(int(xs[k]) - size // 2, 0), w - size)
            frac = float(mask[y0:y0 + size, x0:x0 + size].mean())
            if frac >= self.min_foreground:
                return y0, x0
            if frac > best[0]:
                best = (frac, y0, x0)
        return best[1], best[2]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        pid, z = self.index[i]
        mr, ct, mask = self.load_slice(pid, z)
        sample = {"id": pid, "z": z, "y": REGION_INDEX[parse_id(pid)["region"]]}

        if self.mode == "train":
            size = self.patch_size
            mr = pad_to(mr, size, self.pad_value)
            ct = pad_to(ct, size, self.pad_value)
            mask = pad_to(mask, size, 0.0)
            extra = {}
            if self.reflow:  # (mode train)
                x1 = normalize_ct(np.asarray(self._arrays(pid)[self.reflow["file"]][z])[None], self.ct_clip)
                extra["x1"] = pad_to(x1, size, self.pad_value)
                # noise trên cả lát đã pad, đúng như lúc teacher sinh x1 (predict_volume, roi = patch)
                extra["noise"] = slice_noise(pid, z, mask.shape[-2:], self.reflow["noise_seed"]).numpy()
            y0, x0 = self._crop_origin(mask[0])
            win = (slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
            mr, ct, mask = mr[win], ct[win], mask[win]
            extra = {k: v[win] for k, v in extra.items()}
            flip = bool(torch.rand(()) < self.hflip)
            if flip:
                mr, ct, mask = mr[..., ::-1], ct[..., ::-1], mask[..., ::-1]
                extra = {k: v[..., ::-1] for k, v in extra.items()}
            sample.update(origin=(y0, x0), flip=flip)
            sample.update({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in extra.items()})

        sample.update(
            cond=torch.from_numpy(np.ascontiguousarray(mr)),
            ct=torch.from_numpy(np.ascontiguousarray(ct)),
            mask=torch.from_numpy(np.ascontiguousarray(mask)),
        )
        return sample
