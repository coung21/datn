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
