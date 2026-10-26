import pytest
import torch

from src.models import UNet2D, build_model
from src.utils.config import load_config


def test_param_count_from_base_config():
    model = build_model(load_config("configs/base.yaml")["model"])
    n = sum(p.numel() for p in model.parameters())
    assert 30e6 <= n <= 32e6, f"{n / 1e6:.2f}M"


@pytest.mark.parametrize("norm", ["batch", "instance", "group"])
@pytest.mark.parametrize("size", [32, 64])
def test_forward_backward(norm, size):
    torch.manual_seed(0)
    model = UNet2D(base_channels=8, depth=3, norm=norm)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 1, size, size) and torch.isfinite(out).all()
    out.mean().backward()
    assert model.down[0][0].weight.grad.abs().sum() > 0


def test_bad_input_raises():
    model = UNet2D(base_channels=8, depth=3).eval()
    with pytest.raises(AssertionError):
        model(torch.randn(1, 3, 36, 36))  # 36 không chia hết cho 2^3
    with pytest.raises(AssertionError):
        model(torch.randn(1, 4, 32, 32))  # sai số kênh
    with pytest.raises(ValueError):
        UNet2D(norm="layer")


def test_attention_gate_formula():
    """alpha = sigmoid(psi(ReLU(GN(W_x x) + GN(W_g g)))), x_hat = alpha * x."""
    from src.models.unet2d import AttentionGate
    torch.manual_seed(0)
    gate = AttentionGate(64, "group")
    x, g = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)
    out = gate(x, g)
    alpha = torch.sigmoid(gate.psi(torch.relu(gate.w_x(x) + gate.w_g(g))))
    assert gate.w_x[0].out_channels == 32 and gate.psi.bias is not None
    assert alpha.shape == (2, 1, 16, 16) and torch.allclose(out, alpha * x)
    assert torch.allclose(gate.last_alpha, alpha) and 0 <= alpha.min() and alpha.max() <= 1


def test_attention_gates_option():
    torch.manual_seed(0)
    plain = UNet2D(base_channels=8, depth=3, norm="group")
    gated = UNet2D(base_channels=8, depth=3, norm="group", attention_gates=True)
    # tắt gate thì state_dict giữ nguyên như trước -> checkpoint cũ vẫn nạp được
    assert all(not k.startswith("gates") for k in plain.state_dict())
    assert sum(p.numel() for p in gated.parameters()) > sum(p.numel() for p in plain.parameters())
    out = gated(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)
    out.mean().backward()
    assert all(g.psi.weight.grad is not None for g in gated.gates)
    assert [tuple(g.last_alpha.shape[-2:]) for g in gated.gates] == [(8, 8), (16, 16), (32, 32)]


def test_gn_ag_config():
    cfg = load_config("configs/base.yaml", "configs/exp/unet2d_gn_ag.yaml")
    assert cfg["model"]["norm"] == "group" and cfg["model"]["attention_gates"] is True
    assert load_config("configs/base.yaml")["model"]["attention_gates"] is False
    n = sum(p.numel() for p in build_model(cfg["model"]).parameters())
    assert 31e6 < n < 32e6
