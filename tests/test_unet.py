import pytest
import torch

from src.models import UNetModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return UNetModel(image_size=64).eval()


def test_param_count():
    n = sum(p.numel() for p in UNetModel().parameters())
    assert 25e6 <= n <= 40e6, f"{n / 1e6:.2f}M"


@pytest.mark.parametrize("size", [64, 128])
def test_forward_shape(model, size):
    x = torch.randn(2, 1, size, size)
    cond = torch.randn(2, 3, size, size)
    t = torch.rand(2)
    with torch.no_grad():
        out = model(x, t, cond)
    assert out.shape == (2, 1, size, size)
    assert torch.isfinite(out).all()


def test_scalar_timestep(model):
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out = model(x, torch.tensor(0.5))
    assert out.shape == (2, 1, 64, 64)


def test_backward_with_checkpoint():
    m = UNetModel(image_size=32, use_checkpoint=True).train()
    # conv ra bị zero-init -> phá đối xứng để gradient lan tới mọi tầng
    torch.nn.init.normal_(m.out[-1].weight, std=0.02)
    x = torch.randn(1, 4, 32, 32)
    m(x, torch.rand(1)).mean().backward()
    assert m.input_blocks[0][0].weight.grad.abs().sum() > 0


def test_bad_spatial_size_raises(model):
    with pytest.raises(AssertionError):
        model(torch.randn(1, 4, 40, 40), torch.rand(1))
