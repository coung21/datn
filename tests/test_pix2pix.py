import pandas as pd
import pytest
import torch
from test_regression import TINY_MODEL, tiny_run  # noqa: F401  (fixture dùng chung)

from src.methods import Pix2Pix, build_method, masked_l1
from src.methods.pix2pix import gan_loss
from src.models import PatchGAN, UNet2D
from src.utils.config import load_config

TINY_DISC = ["method.disc.base_channels=8", "method.disc.n_layers=2"]


def test_patchgan_matches_pix2pix():
    d = PatchGAN()
    assert 2.7e6 < sum(p.numel() for p in d.parameters()) < 2.8e6
    assert d(torch.randn(1, 4, 256, 256)).shape == (1, 1, 30, 30)  # 70x70 receptive field


def test_gan_loss_modes():
    x = torch.zeros(2, 1, 3, 3)
    assert gan_loss(x, True, "vanilla").item() == pytest.approx(0.6931, abs=1e-4)  # log 2
    assert gan_loss(x, True, "lsgan").item() == 1.0
    with pytest.raises(ValueError):
        Pix2Pix(gan_mode="wgan")


def _setup(lambda_l1=100.0):
    torch.manual_seed(0)
    cfg = load_config("configs/base.yaml", "configs/exp/pix2pix.yaml")
    method = build_method(**{**cfg["method"], "lambda_l1": lambda_l1, "disc": {"base_channels": 8, "n_layers": 2}})
    method.setup(cfg, torch.device("cpu"), lambda step: 1.0)
    G = UNet2D(**TINY_MODEL)
    batch = {"cond": torch.randn(2, 3, 32, 32), "ct": torch.rand(2, 1, 32, 32) * 2 - 1,
             "mask": (torch.rand(2, 1, 32, 32) > 0.3).float()}
    return method, G, batch


def test_config_is_standard_pix2pix():
    cfg = load_config("configs/base.yaml", "configs/exp/pix2pix.yaml")
    assert cfg["method"]["gan_mode"] == "vanilla" and cfg["method"]["lambda_l1"] == 100
    assert cfg["train"]["optimizer"]["betas"] == [0.5, 0.999] and cfg["train"]["grad_clip"] is None
    assert cfg["model"]["norm"] == "batch"


def test_gradients_do_not_mix():
    """Gradient của D chỉ đến từ L_D, của G chỉ đến từ L_G."""
    method, G, batch = _setup()
    total, parts = method.loss(G, batch)
    total.backward()
    g_grad = [p.grad.clone() for p in G.parameters()]
    d_grad = [p.grad.clone() for p in method.D.parameters()]
    assert parts["g"] + parts["d"] == pytest.approx(total.item(), rel=1e-5)
    assert parts["g"] == pytest.approx(parts["g_adv"] + 100 * parts["l1"], rel=1e-5)

    G.zero_grad(), method.D.zero_grad()
    fake = G(batch["cond"])
    D = method.D
    loss_d = 0.5 * (gan_loss(D(torch.cat([batch["cond"], batch["ct"]], 1)), True, "vanilla")
                    + gan_loss(D(torch.cat([batch["cond"], fake.detach()], 1)), False, "vanilla"))
    loss_d.backward()
    for p, g in zip(D.parameters(), d_grad):
        assert torch.allclose(p.grad, g, atol=1e-6)
    D.zero_grad()
    for p in D.parameters():
        p.requires_grad_(False)
    loss_g = gan_loss(D(torch.cat([batch["cond"], fake], 1)), True, "vanilla") \
        + 100 * masked_l1(fake, batch["ct"], batch["mask"])
    loss_g.backward()
    for p, g in zip(G.parameters(), g_grad):
        assert torch.allclose(p.grad, g, atol=1e-5)


def test_optimizer_step_updates_d_and_state_roundtrip():
    method, G, batch = _setup()
    before = [p.clone() for p in method.D.parameters()]
    method.loss(G, batch)[0].backward()
    method.optimizer_step()
    assert any(not torch.equal(a, b) for a, b in zip(before, method.D.parameters()))
    assert all(p.grad is None for p in method.D.parameters())
    other, _, _ = _setup()
    other.load_state_dict(method.state_dict())
    for a, b in zip(method.D.parameters(), other.D.parameters()):
        assert torch.equal(a, b)


def test_train_pix2pix_end_to_end(tiny_run):
    _, run = tiny_run
    run_dir = run("method.name=pix2pix", "run_name=pix2pix", "train.optimizer.betas=[0.5, 0.999]",
                  "train.grad_clip=null", *TINY_DISC)
    log = pd.read_csv(run_dir / "train_log.csv")
    for k in ("loss_l1", "loss_g_adv", "loss_d", "loss_d_real", "loss_d_fake"):
        assert log[k].notna().all(), k
    last = torch.load(run_dir / "last.pt", weights_only=True)
    assert set(last["method"]) == {"D", "opt_d", "sched_d"}
    assert (run_dir / "best.pt").exists() and len(pd.read_csv(run_dir / "val_full.csv")) > 0
    run("train.total_images=32", resume=run_dir)  # train tiếp, khôi phục D
    assert pd.read_csv(run_dir / "train_log.csv")["step"].tolist() == [2, 4, 6, 8]
