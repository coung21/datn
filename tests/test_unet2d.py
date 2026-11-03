import numpy as np
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


def test_bottleneck_attention_matches_manual_mhsa():
    """h = x + Proj(softmax(QK^T / sqrt(d)) V), Q/K/V từ GN(x), mỗi head d = 64 kênh."""
    from src.models.unet2d import SelfAttention2d
    torch.manual_seed(0)
    attn = SelfAttention2d(128, 64, "group").eval()
    x = torch.randn(2, 128, 4, 4)
    assert torch.equal(attn(x), x)  # Proj khởi tạo 0 -> hàm đồng nhất
    torch.nn.init.normal_(attn.proj.weight, std=0.05)
    q, k, v = attn.qkv(attn.norm(x)).reshape(2, 3, 128, 16).unbind(1)  # [B, C, N]
    heads = []
    for hd in range(2):
        sl = slice(64 * hd, 64 * (hd + 1))
        w = torch.softmax(q[:, sl].transpose(1, 2) @ k[:, sl] / 8.0, dim=-1)  # [B, N, N]
        heads.append((w @ v[:, sl].transpose(1, 2)).transpose(1, 2))           # [B, 64, N]
    ref = x + attn.proj(torch.cat(heads, 1).reshape(2, 128, 4, 4))
    assert torch.allclose(attn(x), ref, atol=1e-5)


def test_bottleneck_attention_option():
    plain = UNet2D(base_channels=8, depth=2, norm="group")
    model = UNet2D(base_channels=8, depth=2, norm="group", bottleneck_attention=True, attention_head_channels=8)
    assert all(not k.startswith("bottom_attn") for k in plain.state_dict())
    assert model.bottom_attn.heads == 4  # 32 kênh bottleneck / 8
    out = model(torch.randn(2, 3, 32, 32))
    out.mean().backward()
    assert out.shape == (2, 1, 32, 32) and model.bottom_attn.proj.weight.grad.abs().sum() > 0


def test_gn_ag_mha_config():
    cfg = load_config("configs/base.yaml", "configs/exp/unet2d_gn_ag_mha.yaml")
    m = cfg["model"]
    assert m["norm"] == "group" and m["attention_gates"] and m["bottleneck_attention"]
    model = build_model(m)
    assert model.bottom_attn.heads == 16
    assert 35e6 < sum(p.numel() for p in model.parameters()) < 36e6


def test_resblock_formula():
    """y = ReLU(GN(Conv(ReLU(GN(Conv(x))))) + shortcut(x)); shortcut = x khi cùng số kênh."""
    from src.models.unet2d import ResBlock
    torch.manual_seed(0)
    same, diff = ResBlock(16, 16, "group"), ResBlock(8, 16, "group")
    assert isinstance(same.shortcut, torch.nn.Identity) and isinstance(diff.shortcut[0], torch.nn.Conv2d)
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(same(x), torch.relu(same.body(x) + x))
    x8 = torch.randn(2, 8, 8, 8)
    assert torch.allclose(diff(x8), torch.relu(diff.body(x8) + diff.shortcut(x8)))


def test_res_block_option():
    conv = UNet2D(base_channels=8, depth=2, norm="group")
    res = UNet2D(base_channels=8, depth=2, norm="group", block="res", attention_gates=True)
    # cùng số conv 3x3, chỉ thêm shortcut
    n3 = lambda m: sum(1 for x in m.modules() if isinstance(x, torch.nn.Conv2d) and x.kernel_size == (3, 3))
    assert n3(conv) == n3(res)
    out = res(torch.randn(2, 3, 32, 32))
    out.mean().backward()
    assert out.shape == (2, 1, 32, 32) and res.down[0].body[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        UNet2D(block="dense")


def test_gn_ag_res_config():
    cfg = load_config("configs/base.yaml", "configs/exp/unet2d_gn_ag_res.yaml")
    m = cfg["model"]
    assert m["block"] == "res" and m["norm"] == "group" and m["attention_gates"] and not m["bottleneck_attention"]
    assert load_config("configs/base.yaml")["model"]["block"] == "conv"


def test_structure_guided_gate():
    torch.manual_seed(0)
    plain = UNet2D(base_channels=8, depth=3, norm="group", attention_gates=True)
    sg = UNet2D(base_channels=8, depth=3, norm="group", attention_gates=True, structure_guided=True)
    extra = sum(p.numel() for p in sg.parameters()) - sum(p.numel() for p in plain.parameters())
    # mỗi gate: W_p (C*3 + 3) + W_s (3*C/2) + GN (2*C/2); đầu P0: 8*3 + 3
    assert extra == sum(c * 3 + 3 + 3 * c // 2 + c for c in (8, 16, 32)) + 8 * 3 + 3
    x = torch.randn(2, 3, 32, 32)
    out = sg(x)
    assert out.shape == (2, 1, 32, 32)
    out2, structs = sg(x, return_struct=True)
    assert torch.allclose(out, out2)
    assert [(lv, tuple(lg.shape)) for lv, lg in structs] == [
        (2, (2, 3, 8, 8)), (1, (2, 3, 16, 16)), (0, (2, 3, 32, 32)), (0, (2, 3, 32, 32))]
    with pytest.raises(ValueError):
        UNet2D(attention_gates=False, structure_guided=True)
    with pytest.raises(AssertionError):
        plain(x, return_struct=True)


def test_structure_guided_gate_formula():
    from src.models.unet2d import StructureGuidedGate
    torch.manual_seed(0)
    gate = StructureGuidedGate(64, "group")
    x, g = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)
    out = gate(x, g)
    p = torch.softmax(gate.w_p(g), dim=1)
    alpha = torch.sigmoid(gate.psi(torch.relu(gate.w_x(x) + gate.w_g(g) + gate.w_s(p))))
    assert gate.w_s[0].in_channels == 3 and gate.w_s[0].out_channels == 32
    assert torch.allclose(out, alpha * x, atol=1e-6) and torch.allclose(gate.last_logits, gate.w_p(g))


def test_mixture_head():
    from src.data.dataset import normalize_ct
    torch.manual_seed(0)
    sg = UNet2D(base_channels=8, depth=3, norm="group", attention_gates=True, structure_guided=True)
    mix = UNet2D(base_channels=8, depth=3, norm="group", attention_gates=True, structure_guided=True, mixture_head=True)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(mix) - n(sg) == 2 * (8 + 1)  # thêm 2 kênh đầu ra của Conv1x1 cuối
    expect = torch.from_numpy(normalize_ct(np.array([-1000.0, 40.0, 700.0])))
    assert torch.allclose(mix.out.bias.detach(), expect, atol=1e-6)
    x = torch.randn(2, 3, 32, 32)
    out, structs = mix(x, return_struct=True)
    p0 = torch.softmax(structs[-1][1], dim=1)
    assert out.shape == (2, 1, 32, 32) and torch.allclose(out, (p0 * mix.out(_last_feature(mix, x))).sum(1, keepdim=True), atol=1e-5)
    assert torch.allclose(mix(x), out)
    out.mean().backward()
    assert mix.struct_head.weight.grad.abs().sum() > 0  # L1 chảy vào P0
    with pytest.raises(ValueError):
        UNet2D(attention_gates=True, mixture_head=True)


def _last_feature(model, x):
    """Đặc trưng decoder cuối (đầu vào của model.out)."""
    feats = {}
    h = model.out.register_forward_hook(lambda m, i, o: feats.update(x=i[0]))
    model(x)
    h.remove()
    return feats["x"]
