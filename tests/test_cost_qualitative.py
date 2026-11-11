import json

import numpy as np
import pandas as pd

from scripts.make_qualitative import draw, middle_index, select_patients, take_view
from scripts.measure_cost import count_flops, count_params, pick_patients
from src.models import UNet2D


def test_count_flops_single_conv():
    import torch

    conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
    # mỗi điểm ra: 3*3*3 phép nhân-cộng (2 FLOPs mỗi phép), 4 kênh ra, 8x8 điểm
    assert count_flops(conv, 3, 8) == 2 * 27 * 4 * 64


def test_multiview_costs_more_than_plain():
    kw = dict(base_channels=8, depth=3, norm="group", attention_gates=True, in_channels=9)
    plain, mv = UNet2D(**kw), UNet2D(**kw, multiview_input=True)
    assert count_params(mv) > count_params(plain) * 0.99
    assert count_flops(mv, 9, 32) > count_flops(plain, 9, 32)


def test_pick_patients():
    ids = ["1BA001", "1PA001", "1BB002", "1PC003", "1BC004"]
    assert pick_patients(ids, 2) == ["1BA001", "1BB002", "1PA001", "1PC003"]


def test_select_patients():
    ids = [f"1BA{i:03d}" for i in range(5)] + [f"1PA{i:03d}" for i in range(5)]
    ref = pd.DataFrame({"region": ["brain"] * 5 + ["pelvis"] * 5,
                        "mae": [70, 80, 60, 90, 75, 50, 65, 55, 70, 60.0]}, index=ids)
    new = ref.copy()
    new["mae"] += [0, 1, 0, 8, 0, 0, -9, 0, 0, 0]
    picked = select_patients(ref, new)
    assert picked == {"1BA004": "median brain", "1PA004": "median pelvis",
                      "1PA001": "largest gain (pelvis)", "1BA003": "largest loss"}


def test_views():
    vol = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert take_view(vol, "axial", 1).shape == (3, 4)
    cor = take_view(vol, "coronal", 2)
    assert cor.shape == (2, 4) and (cor[0] == vol[1, 2]).all()
    mask = np.zeros((6, 5, 5), np.uint8)
    mask[2:5, 1:3] = 1
    assert middle_index(mask, 0) == 3 and middle_index(mask, 1) == 2


def test_draw_writes_figures(tmp_path):
    rng = np.random.default_rng(0)
    labels = ["A", "B"]
    info = {"labels": labels, "patients": {"1BA001": "median brain", "1PA001": "median pelvis"},
            "mae": {p: {l: 60.0 for l in labels} for p in ("1BA001", "1PA001")}}
    data = {}
    for pid, spacing in (("1BA001", (1, 1, 1)), ("1PA001", (1, 1, 2.5))):
        mask = np.zeros((12, 20, 24), np.uint8)
        mask[2:10, 4:16, 5:20] = 1
        ct = rng.integers(-1000, 1500, mask.shape).astype(np.int16)
        data[pid] = {"mask": mask, "ct": ct, "mr": rng.uniform(-1, 1, mask.shape).astype(np.float16),
                     "sct_0": ct + 50, "sct_1": ct - 50, "spacing": np.asarray(spacing, np.float32)}
    for view in ("axial", "coronal"):
        draw(view, json.loads(json.dumps(info)), data, tmp_path / f"q_{view}")
        assert (tmp_path / f"q_{view}.png").stat().st_size > 0 and (tmp_path / f"q_{view}.pdf").exists()
