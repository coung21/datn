import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_tiny_run, write_fake_reg
from test_dataset import make_patient

from src.data.dataset import SliceDataset, normalize_ct
from src.methods import ResidualFlow, build_method, masked_mse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("predict_reg", ROOT / "scripts" / "predict_reg.py")
predict_reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_reg)


def make_batch(n=2, size=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, 1, size, size)
    mask[..., 8:24, 6:26] = 1
    ct = torch.where(mask > 0, torch.rand(n, 1, size, size, generator=g) * 2 - 1, -1.0)
    y_hat = torch.where(mask > 0, (ct + 0.05 * torch.randn(n, 1, size, size, generator=g)).clamp(-1, 1), -1.0)
    cond = torch.cat([torch.randn(n, 3, size, size, generator=g), y_hat], dim=1)
    return {"cond": cond, "ct": ct, "mask": mask}


def test_scale_required():
    with pytest.raises(ValueError):
        build_method("residual_flow", sigma=0.5)
    m = build_method("residual_flow", sigma=0.5, scale=0.04)
    assert isinstance(m, ResidualFlow) and m.needs_reg and m.extra_in_channels == 1


def test_dataset_with_reg(tmp_path):
    make_patient(tmp_path, "1BA001", spacing_z=1.0, seed=0)
    write_fake_reg(tmp_path / "brain" / "1BA001", seed=1)
    reg = np.load(tmp_path / "brain" / "1BA001" / "reg.npy")
    val = SliceDataset(tmp_path, ["1BA001"], mode="val", with_reg=True)
    cond, _, _ = val.load_slice("1BA001", 6)
    assert cond.shape[0] == 4
    np.testing.assert_allclose(cond[-1], normalize_ct(reg[6]))
    plain = SliceDataset(tmp_path, ["1BA001"], mode="val")
    np.testing.assert_array_equal(plain.load_slice("1BA001", 6)[0], cond[:3])  # 3 kênh MR không đổi

    torch.manual_seed(0)
    ds = SliceDataset(tmp_path, ["1BA001"], patch_size=16, hflip=1.0, with_reg=True)
    s = ds[3]
    full_cond, full_ct, _ = ds.load_slice(s["id"], s["z"])
    y0, x0 = s["origin"]
    want = full_cond[:, y0:y0 + 16, x0:x0 + 16][..., ::-1]
    np.testing.assert_array_equal(s["cond"].numpy(), want)  # ŷ crop + lật cùng MR/CT
    assert s["cond"].shape == (4, 16, 16)


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.seen = {}

    def forward(self, x, t):
        self.seen["x"], self.seen["t"] = x, t
        return x[:, :1] * self.w  # = 0


def test_loss_target_and_background():
    batch = make_batch()
    method = ResidualFlow(sigma=0.0, scale=0.05)
    spy = Spy()
    loss, logs = method.loss(spy, batch)
    loss.backward()
    mask, x1, y_hat = batch["mask"], batch["ct"], batch["cond"][:, -1:]
    z1 = torch.where(mask > 0, (x1 - y_hat) / 0.05, 0.0)
    # sigma = 0: z0 = 0, model trả 0 -> loss = mean(z1^2) trong mask
    assert loss.item() == pytest.approx(masked_mse(torch.zeros_like(z1), z1, mask).item(), rel=1e-5)
    x = spy.seen["x"]
    assert x.shape[1] == 5 and (x[:, :1][mask == 0] == 0).all()  # nền phần dư = 0
    torch.testing.assert_close(x[:, 1:], batch["cond"])
    t = spy.seen["t"].view(-1, 1, 1, 1)
    torch.testing.assert_close(x[:, :1], t * z1)  # z_t = t * z1 khi sigma = 0


@pytest.mark.parametrize("sigma", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("nfe", [1, 4])
def test_oracle_velocity_reaches_ct(sigma, nfe):
    batch = make_batch()
    mask, x1, cond = batch["mask"], batch["ct"], batch["cond"]
    scale = 0.05
    noise = torch.randn_like(x1)
    z0 = torch.where(mask > 0, sigma * noise, 0.0)
    z1 = torch.where(mask > 0, (x1 - cond[:, -1:]) / scale, 0.0)
    calls = []

    def net(inp, t):
        calls.append(float(t))
        assert inp.shape[1] == 5
        return z1 - z0

    out = ResidualFlow(sigma=sigma, scale=scale).sample(net, cond, mask, noise, nfe)
    assert len(calls) == nfe
    torch.testing.assert_close(out[mask > 0], x1[mask > 0], atol=1e-5, rtol=0)
    assert (out[mask == 0] == -1).all()


def test_sigma_zero_ignores_noise_and_nfe1_is_yhat_plus_residual():
    batch = make_batch()
    method = ResidualFlow(sigma=0.0, scale=0.05)
    net = lambda inp, t: torch.full_like(inp[:, :1], 2.0)  # noqa: E731  ẑ1 = 2 -> sCT = ŷ + 0.1
    a = method.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 1)
    b = method.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 1)
    assert torch.equal(a, b)
    inside = batch["mask"] > 0
    torch.testing.assert_close(a[inside], (batch["cond"][:, -1:] + 0.1).clamp(-1, 1)[inside])


def test_train_requires_with_reg(tmp_path, monkeypatch):
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/residual_flow.yaml", in_channels=5)
    with pytest.raises(ValueError, match="with_reg"):
        run("method.scale=0.05", "data.with_reg=false")


def test_residual_flow_end_to_end(tmp_path, monkeypatch):
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/residual_flow.yaml", in_channels=5, with_reg=True)
    run_dir = run("method.scale=0.05")
    assert run_dir.name.startswith("residual_flow_")
    val = pd.read_csv(run_dir / "val_log.csv")
    assert {"mae_nfe1", "mae_nfe2"} <= set(val.columns) and (val["mae_nfe1"] > 0).all()
    assert set(pd.read_csv(run_dir / "val_full.csv")["nfe"]) == {2}


def test_predict_reg_and_stats(tmp_path, monkeypatch):
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/regression.yaml", in_channels=3)
    run_dir = run()
    processed = tmp_path / "processed"

    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["predict_reg", *args])
        predict_reg.main()

    cli("--run", str(run_dir), "--subset", "train", "val", "test")
    manifest = json.loads((processed / "reg_manifest.json").read_text())
    assert len(manifest) == 7 and all(v["run"] == run_dir.name for v in manifest.values())
    for pid in manifest:
        d = processed / ("brain" if pid[1] == "B" else "pelvis") / pid
        reg, ct, mask = (np.load(d / f"{k}.npy") for k in ("reg", "ct", "mask"))
        assert reg.dtype == np.int16 and reg.shape == ct.shape and (reg[mask == 0] == -1024).all()

    before = (processed / "brain" / "1BA001" / "reg.npy").stat().st_mtime_ns
    cli("--run", str(run_dir), "--subset", "train")  # đã có, cùng run -> bỏ qua
    assert (processed / "brain" / "1BA001" / "reg.npy").stat().st_mtime_ns == before
    other = run_dir.with_name(run_dir.name + "_other")
    run_dir.rename(other)
    with pytest.raises(SystemExit, match="run khác"):
        cli("--run", str(other), "--subset", "train")

    cli("--stats", "--set", f"data.processed_dir={processed.as_posix()}",
        f"data.split.file={(tmp_path / 'split.json').as_posix()}", f"output_dir={tmp_path.as_posix()}")
    stats = json.loads((tmp_path / "residual_stats.json").read_text())
    assert set(stats) == {"train", "val", "test"} and stats["train"]["std"] > 0
