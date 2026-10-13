"""Khối giảm/tăng độ phân giải không gian."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """Giảm 2 lần độ phân giải, bằng conv stride 2 hoặc average pooling."""

    def __init__(self, channels: int, use_conv: bool = True, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """Tăng 2 lần độ phân giải bằng nearest, tùy chọn thêm conv 3x3."""

    def __init__(self, channels: int, use_conv: bool = True, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
