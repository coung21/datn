import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import TINY_MODEL, make_tiny_run
from test_dataset import make_patient

from src.data.dataset import REGION_INDEX, SliceDataset, normalize_ct, padded_shape, slice_noise
from src.eval.evaluate import build_dataset, load_checkpoint
from src.inference.predict import predict_volume
from src.methods import Reflow, build_method
from src.models import UNetModel
from src.train import init_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_reflow", ROOT / "scripts" / "generate_reflow.py")
generate_reflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_reflow)


def test_unet_class_conditioning_starts_as_unconditional():
    torch.manual_seed(0)
    base = UNetModel(**TINY_MODEL, in_channels=4).eval()
    with torch.no_grad():  # các conv khởi tạo 0 (kiểu ADM) làm embedding chưa có tác dụng -> nhiễu nhẹ
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))
    cond = UNetModel(**TINY_MODEL, in_channels=4, num_classes=2).eval()
    missing, unexpected = cond.load_state_dict(base.state_dict(), strict=False)
    assert missing == ["label_emb.weight"] and not unexpected
    x, t = torch.randn(2, 4, 32, 32), torch.rand(2)
    with torch.no_grad():
        torch.testing.assert_close(cond(x, t, y=torch.tensor([0, 1])), base(x, t))
        cond.label_emb.weight.data.normal_()
        assert not torch.allclose(cond(x, t, y=torch.tensor([0, 0])), cond(x, t, y=torch.tensor([1, 1])))
    with pytest.raises(AssertionError):
        cond(x, t)  # thiếu nhãn
    with pytest.raises(AssertionError):
        base(x, t, y=torch.tensor([0, 1]))  # model không có lớp


def test_init_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    teacher = UNetModel(**TINY_MODEL, in_channels=4)
    torch.save({"model": teacher.state_dict()}, tmp_path / "t.pt")
    student = UNetModel(**TINY_MODEL, in_channels=4, num_classes=2)
    init_from_checkpoint(student, tmp_path / "t.pt")
    for k, v in teacher.state_dict().items():
        assert torch.equal(student.state_dict()[k], v)
    assert (student.label_emb.weight == 0).all()
    with pytest.raises(ValueError):
        init_from_checkpoint(UNetModel(**TINY_MODEL, in_channels=5), tmp_path / "t.pt")


def write_fake_reflow(d: Path) -> np.ndarray:
    ct, mask = np.load(d / "ct.npy"), np.load(d / "mask.npy") > 0
    x1 = np.where(mask, np.clip(ct + 37, -1024, 3000), -1024).astype(np.int16)
    np.save(d / "reflow_nfe2.npy", x1)
    return x1


@pytest.mark.parametrize("hflip", [0.0, 1.0])
def test_dataset_reflow_pairs_aligned(tmp_path, hflip):
    make_patient(tmp_path, "1PA001", spacing_z=2.5, seed=0)
    x1_vol = write_fake_reflow(tmp_path / "pelvis" / "1PA001")
    ds = SliceDataset(tmp_path, ["1PA001"], patch_size=16, hflip=hflip,
                      reflow={"file": "reflow_nfe2", "noise_seed": 7})
    torch.manual_seed(0)
    s = ds[4]
    z, (y0, x0) = s["z"], s["origin"]
    shape = padded_shape(*x1_vol.shape[1:], 16)
    want_noise = slice_noise("1PA001", z, shape, 7)[:, y0:y0 + 16, x0:x0 + 16]
    want_x1 = normalize_ct(x1_vol[z][None])[:, y0:y0 + 16, x0:x0 + 16]
    if s["flip"]:
        want_noise, want_x1 = want_noise.flip(-1), want_x1[..., ::-1]
    torch.testing.assert_close(s["noise"], want_noise)
    np.testing.assert_allclose(s["x1"].numpy(), want_x1)
    assert s["y"] == REGION_INDEX["pelvis"] and s["x1"].shape == s["noise"].shape == s["mask"].shape


def test_reflow_loss_uses_pairs_and_region():
    seen = {}

    class Spy(torch.nn.Module):
        num_classes = 2

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t, y=None):
            seen.update(x=x, t=t, y=y)
            return x[:, :1] * self.w

    torch.manual_seed(1)
    mask = torch.zeros(2, 1, 16, 16)
    mask[..., 4:12, 4:12] = 1
    batch = {"x1": torch.rand(2, 1, 16, 16), "noise": torch.randn(2, 1, 16, 16), "cond": torch.randn(2, 3, 16, 16),
             "mask": mask, "y": torch.tensor([0, 1]), "ct": torch.full((2, 1, 16, 16), 9.0)}
    method = build_method("reflow")
    assert isinstance(method, Reflow) and method.needs_pairs
    loss, _ = method.loss(Spy(), batch)
    t = seen["t"].view(-1, 1, 1, 1)
    want = (1 - t) * batch["noise"] + t * batch["x1"]
    inside = mask > 0
    torch.testing.assert_close(seen["x"][:, :1][inside], want[inside])  # dùng cặp, không dùng CT thật
    assert (seen["x"][:, :1][~inside] == -1).all() and torch.equal(seen["y"], batch["y"])
    target = (batch["x1"] - batch["noise"])[inside]
    assert loss.item() == pytest.approx((target ** 2).mean().item(), rel=1e-5)


def test_generate_pairs_and_train_reflow_end_to_end(tmp_path, monkeypatch):
    # teacher: flow matching nhỏ
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/flow_matching.yaml", in_channels=4)
    teacher = run()

    monkeypatch.setattr(sys, "argv", ["g", "--teacher", str(teacher), "--nfe", "2", "--seed", "1"])
    generate_reflow.main()
    root = tmp_path / "processed"
    manifest = json.loads((root / "reflow_manifest.json").read_text())["reflow_nfe2"]
    assert manifest["info"]["nfe"] == 2 and len(manifest["patients"]) == 4

    # x̂1 đã lưu = teacher chạy lại với cùng seed -> noise sinh lại trong dataset khớp noise lúc sinh
    model, method, cfg, _ = load_checkpoint(teacher / "best.pt", torch.device("cpu"))
    ds = build_dataset(cfg, ["1BA001"], "val")
    again = predict_volume(method, model, ds, "1BA001", 2, cfg["inference"], torch.device("cpu"), noise_seed=1)
    saved = np.load(root / "brain" / "1BA001" / "reflow_nfe2.npy")
    assert np.abs(np.round(again) - saved).max() <= 1

    # chạy lại: bỏ qua; đổi cấu hình: báo lỗi
    mtime = (root / "brain" / "1BA001" / "reflow_nfe2.npy").stat().st_mtime_ns
    generate_reflow.main()
    assert (root / "brain" / "1BA001" / "reflow_nfe2.npy").stat().st_mtime_ns == mtime
    monkeypatch.setattr(sys, "argv", ["g", "--teacher", str(teacher), "--nfe", "2", "--seed", "5"])
    with pytest.raises(SystemExit, match="cấu hình khác"):
        generate_reflow.main()

    # student: reflow có điều kiện region, khởi tạo từ teacher
    _, run_student = make_tiny_run(tmp_path / "student", monkeypatch, "configs/exp/reflow.yaml", in_channels=4)
    student = run_student(f"data.processed_dir={root.as_posix()}", f"train.init_from={(teacher / 'best.pt').as_posix()}")
    val = pd.read_csv(student / "val_log.csv")
    assert {"mae_nfe1", "mae_nfe2"} <= set(val.columns) and (val["mae_nfe1"] > 0).all()
    m, _, scfg, _ = load_checkpoint(student / "best.pt", torch.device("cpu"))
    assert m.num_classes == 2 and scfg["train"]["init_from"].endswith("best.pt")


def test_reflow_requires_pairs(tmp_path, monkeypatch):
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/reflow.yaml", in_channels=4)
    with pytest.raises(ValueError, match="data.reflow"):
        run("data.reflow=null", "train.init_from=null")
