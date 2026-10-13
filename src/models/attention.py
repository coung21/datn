"""Self-attention không gian (ADM-style) dùng F.scaled_dot_product_attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn import normalization, zero_module


class AttentionBlock(nn.Module):
    """Multi-head self-attention trên các vị trí không gian, có residual."""

    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels {channels} không chia hết cho num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        assert channels % self.num_heads == 0

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww = x.shape
        x_flat = x.reshape(b, c, hh * ww)
        qkv = self.qkv(self.norm(x_flat))
        # [b, 3*c, l] -> 3 x [b, heads, l, d]
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, hh * ww).permute(
            1, 0, 2, 4, 3
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, hh * ww)
        out = self.proj_out(out)
        return (x_flat + out).reshape(b, c, hh, ww)
