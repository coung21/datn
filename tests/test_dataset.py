import pickle

import numpy as np
import pytest
import torch

from src.data.dataset import (
    SliceDataset,
    neighbor_indices,
    normalize_ct,
    normalize_mr,
    slice_stride,
    val_slice_indices,
)
from src.data.preprocess import parse_id, preprocess_arrays, restore_full_volume, save_patient

SHAPE = (20, 48, 56)  # (z, y, x)
BOX = (slice(4, 16), slice(10, 38), slice(12, 44))


def make_volume(seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.uniform(-1500, 3500, SHAPE).astype(np.float32)
    mr = rng.uniform(1, 2000, SHAPE).astype(np.float32)
    mask = np.zeros(SHAPE, np.uint8)
    mask[BOX] = 1
    return ct, mr, mask


def make_patient(root, pid, spacing_z, seed=0):
    ct, mr, mask = make_volume(seed)
    arrays, info = preprocess_arrays(ct, mr, mask, margin=(2, 8, 8))
    meta = {"id": pid, **parse_id(pid), "spacing": [1.0, 1.0, spacing_z],
            "orig_shape": list(SHAPE), **info}
    save_patient(root / meta["region"] / pid, arrays, meta)
    return arrays, meta


@pytest.fixture
def processed(tmp_path):
    make_patient(tmp_path, "1BA001", spacing_z=1.0, seed=0)
    make_patient(tmp_path, "1PA001", spacing_z=2.5, seed=1)
    return tmp_path


def test_preprocess_arrays():
    ct, mr, mask = make_volume()
    arrays, info = preprocess_arrays(ct, mr, mask, margin=(2, 8, 8))
    assert info["bbox"] == [[2, 18], [2, 46], [4, 52]]
    assert arrays["ct"].dtype == np.int16 and arrays["mr"].dtype == np.float16
    m = arrays["mask"].astype(bool)
    assert arrays["ct"].min() >= -1024 and arrays["ct"].max() <= 3000
    assert (arrays["ct"][~m] == -1024).all() and (arrays["mr"][~m] == 0).all()
    assert len(info["slice_mask_area"]) == arrays["mask"].shape[0]
    full = restore_full_volume(arrays["mask"], {"orig_shape": SHAPE, **info}, fill=0)
    np.testing.assert_array_equal(full, mask)


def test_stride_and_neighbors():
    assert slice_stride(1.0, 2.0) == 2  # brain
    assert slice_stride(2.5, 2.0) == 1  # pelvis
    assert neighbor_indices(5, 10, 1, 2) == [3, 5, 7]
    assert neighbor_indices(0, 10, 1, 2) == [0, 0, 2]
    assert neighbor_indices(9, 10, 1, 2) == [7, 9, 9]


def test_normalization_range():
    assert normalize_ct(np.array([-2000, -1024, 3000, 5000])).tolist() == [-1, -1, 1, 1]
    assert normalize_mr(np.array([0.0, 50.0, 100.0, 200.0]), 0, 100).tolist() == [-1, 0, 1, 1]


def test_load_slice_uses_stride(processed):
    ds = SliceDataset(processed, ["1BA001", "1PA001"], patch_size=32)
    for pid, stride in (("1BA001", 2), ("1PA001", 1)):
        z = 6
        mr, ct, mask = ds.load_slice(pid, z)
        raw = np.load(processed / parse_id(pid)["region"] / pid / "mr.npy")
        p = ds.metas[pid]["mr_percentiles"]
        for c, zz in enumerate((z - stride, z, z + stride)):
            np.testing.assert_allclose(mr[c], normalize_mr(raw[zz], p["0.5"], p["99.5"]))
        assert ct.shape == mask.shape == (1, *raw.shape[1:])


@pytest.mark.parametrize("patch", [16, 64])  # nhỏ hơn lát -> crop; lớn hơn -> pad
def test_train_sample(processed, patch):
    torch.manual_seed(0)
    ds = SliceDataset(processed, ["1BA001", "1PA001"], patch_size=patch, hflip=0.0)
    for i in range(0, len(ds), 3):
        s = ds[i]
        assert s["cond"].shape == (3, patch, patch)
        assert s["ct"].shape == s["mask"].shape == (1, patch, patch)
        for k in ("cond", "ct"):
            assert s[k].min() >= -1 and s[k].max() <= 1
        assert s["mask"].mean() >= 0.1


def test_crop_alignment_and_flip(processed):
    torch.manual_seed(0)
    patch = 16
    for hflip in (0.0, 1.0):
        ds = SliceDataset(processed, ["1PA001"], patch_size=patch, hflip=hflip)
        s = ds[5]
        mr, ct, mask = ds.load_slice(s["id"], s["z"])
        y0, x0 = s["origin"]
        win = (slice(None), slice(y0, y0 + patch), slice(x0, x0 + patch))
        exp = [a[win] for a in (mr, ct, mask)]
        if s["flip"]:
            exp = [a[..., ::-1] for a in exp]
        for got, want in zip((s["cond"], s["ct"], s["mask"]), exp):
            np.testing.assert_array_equal(got.numpy(), want)
        assert s["flip"] == bool(hflip)


def test_pad_value(processed):
    ds = SliceDataset(processed, ["1BA001"], patch_size=64, hflip=0.0)
    s = ds[0]
    # lát sau bbox là 44 x 48 -> pad lên 64: viền ngoài là nền
    assert (s["cond"][:, 0] == -1).all() and (s["ct"][:, 0] == -1).all() and (s["mask"][:, 0] == 0).all()


def test_val_mode(processed):
    ds = SliceDataset(processed, ["1BA001", "1PA001"], mode="val", val_slices_per_patient=5)
    assert len(ds) == 10
    assert [z for _, z in ds.index] == [z for _, z in SliceDataset(processed, ["1BA001", "1PA001"], mode="val").index]
    s = ds[0]
    assert s["cond"].shape[1:] == tuple(ds.metas[s["id"]]["shape"][1:])
    assert "origin" not in s
    assert val_slice_indices([0, 1, 10, 10, 10, 10, 2], 3) == [2, 4, 5]


def test_pickle_drops_mmap(processed):
    ds = SliceDataset(processed, ["1BA001"], patch_size=32)
    ds[0]
    assert ds._cache
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._cache == {}
    assert clone[0]["cond"].shape == (3, 32, 32)


def test_dataloader_batch(processed):
    ds = SliceDataset(processed, ["1BA001", "1PA001"], patch_size=32)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch["cond"].shape == (4, 3, 32, 32) and batch["ct"].dtype == torch.float32
