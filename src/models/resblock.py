"""ResBlock có điều kiện theo timestep (ADM-style)."""

from abc import abstractmethod

import torch
import torch.nn as nn

from .nn import normalization, zero_module
from .sampling import Downsample, Upsample


class TimestepBlock(nn.Module):
    """Module nhận thêm embedding timestep trong forward."""

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor: ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential tự truyền emb cho các layer con là TimestepBlock."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class ResBlock(TimestepBlock):
    """Residual block điều kiện theo timestep.

    - use_scale_shift_norm: dùng AdaGN, h = norm(h) * (1 + scale) + shift (như ADM).
    - up/down: resample ngay trong block (BigGAN-style resblock_updown).
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: int | None = None,
        use_scale_shift_norm: bool = True,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_norm = normalization(channels)
        self.in_act = nn.SiLU()
        self.in_conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, use_conv=False)
            self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False)
            self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_act(self.in_norm(x))
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = self.in_conv(h)

        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = self.out_norm(h + emb_out)
        h = self.out_layers(h)
        return self.skip_connection(x) + h
