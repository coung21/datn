"""Discriminator PatchGAN của pix2pix (Isola et al. 2017), bản 'basic' 70x70.

Nhận MR (điều kiện) ghép với CT thật/giả theo kênh, trả bản đồ logit thật/giả cho từng vùng.
"""

import torch.nn as nn

from .unet2d import make_norm


class PatchGAN(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 64, n_layers: int = 3, norm: str = "batch"):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = base_channels
        for i in range(1, n_layers + 1):
            out = base_channels * min(2**i, 8)
            stride = 2 if i < n_layers else 1  # tầng cuối giữ kích thước, như pix2pix
            layers += [nn.Conv2d(ch, out, 4, stride=stride, padding=1, bias=False), make_norm(norm, out),
                       nn.LeakyReLU(0.2, True)]
            ch = out
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


def _init_weights(m: nn.Module) -> None:
    """Khởi tạo N(0, 0.02) như pix2pix."""
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)) and m.weight is not None:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
