import pandas as pd
import pytest
import torch
from conftest import make_tiny_run

from src.methods import MRBridge, build_method


def make_batch(n=2, size=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, 1, size, size)
    mask[..., 8:24, 6:26] = 1
    cond = torch.where(mask > 0, torch.rand(n, 3, size, size, generator=g) * 2 - 1, -1.0)
    ct = torch.where(mask > 0, torch.rand(n, 1, size, size, generator=g) * 2 - 1, -1.0)
    return {"cond": cond, "ct": ct, "mask": mask}


def test_loss_starts_from_mr():
    seen = {}

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t):
            seen.update(x=x, t=t)
            return x[:, :1] * self.w

    batch = make_batch()
    method = build_method("mr_bridge", sigma=0.0)
    assert isinstance(method, MRBridge) and method.extra_in_channels == 1
    loss, _ = method.loss(Spy(), batch)
    loss.backward()
    mask, mr, ct = batch["mask"], batch["cond"][:, 1:2], batch["ct"]
    t = seen["t"].view(-1, 1, 1, 1)
    inside = mask > 0
    torch.testing.assert_close(seen["x"][:, :1][inside], ((1 - t) * mr + t * ct)[inside])  # đường MR -> CT
    assert (seen["x"][:, :1][~inside] == -1).all()
    assert loss.item() == pytest.approx(((ct - mr)[inside] ** 2).mean().item(), rel=1e-5)  # v* = CT - MR


@pytest.mark.parametrize("sigma", [0.0, 0.3])
@pytest.mark.parametrize("nfe", [1, 4])
def test_oracle_velocity_reaches_ct(sigma, nfe):
    batch = make_batch()
    cond, mask, ct = batch["cond"], batch["mask"], batch["ct"]
    noise = torch.randn_like(ct)
    x0 = torch.where(mask > 0, cond[:, 1:2] + sigma * noise, -1.0)
    calls = []

    def net(inp, t):
        calls.append(float(t))
        return ct - x0

    out = MRBridge(sigma=sigma).sample(net, cond, mask, noise, nfe)
    assert len(calls) == nfe
    torch.testing.assert_close(out[mask > 0], ct.clamp(-1, 1)[mask > 0], atol=1e-5, rtol=0)
    assert (out[mask == 0] == -1).all()


def test_sigma_zero_is_deterministic():
    batch = make_batch()
    net = lambda inp, t: torch.full_like(inp[:, :1], 0.1)  # noqa: E731
    method = MRBridge(sigma=0.0)
    a = method.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 2)
    b = method.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 2)
    assert torch.equal(a, b)
    noisy = MRBridge(sigma=0.5)
    c = noisy.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 2)
    d = noisy.sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 2)
    assert not torch.equal(c, d)


def test_mr_bridge_end_to_end(tmp_path, monkeypatch):
    _, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/mr_bridge.yaml", in_channels=4)
    run_dir = run()
    assert run_dir.name.startswith("mr_bridge_")
    val = pd.read_csv(run_dir / "val_log.csv")
    assert {"mae_nfe1", "mae_nfe2", "mae_nfe4"} <= set(val.columns) and (val["mae_nfe1"] > 0).all()
    assert set(pd.read_csv(run_dir / "val_full.csv")["nfe"]) == {2}
