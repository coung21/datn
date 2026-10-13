"""Các hàm/lớp tiện ích dùng chung cho UNet ADM-style."""

import math

import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """GroupNorm tính trong float32 để ổn định khi train mixed precision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int, num_groups: int = 32) -> nn.Module:
    return GroupNorm32(num_groups, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Khởi tạo toàn bộ tham số của module bằng 0 (ADM dùng cho conv cuối mỗi nhánh residual)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding cho timestep.

    Args:
        t: tensor 1-D [N], có thể là số thực (flow matching) hoặc số nguyên (DDPM).
        dim: số chiều output.
    Returns:
        tensor [N, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
