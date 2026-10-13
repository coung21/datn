"""UNet ADM-style (Dhariwal & Nichol, 2021) cho bài toán MRI -> sCT.

Đầu vào là ảnh nhiễu/trạng thái x_t (CT) ghép kênh với ảnh điều kiện (MRI 2.5D),
đầu ra là dự đoán cùng kích thước (noise / velocity / x0 tùy phương pháp).
"""

import torch
import torch.nn as nn

from .attention import AttentionBlock
from .nn import normalization, timestep_embedding, zero_module
from .resblock import ResBlock, TimestepEmbedSequential
from .sampling import Downsample, Upsample


class UNetModel(nn.Module):
    """
    Args:
        image_size: kích thước patch vào (dùng để xác định độ phân giải đặt attention).
        in_channels: số kênh x_t + số kênh điều kiện (concat ở trục kênh).
        model_channels: số kênh cơ sở.
        out_channels: số kênh output.
        num_res_blocks: số ResBlock mỗi mức độ phân giải.
        attention_resolutions: các độ phân giải (vd 32, 16) có AttentionBlock.
        channel_mult: hệ số nhân kênh cho từng mức.
        time_scale: nhân t trước khi embed; với t liên tục trong [0, 1] (flow matching)
            đặt 1000 để giữ dải tần giống ADM, với t nguyên (DDPM) đặt 1.
        resblock_updown: dùng ResBlock để up/down thay cho Upsample/Downsample.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 4,
        model_channels: int = 64,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32, 16),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 2, 4, 4),
        num_heads: int = 4,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        time_scale: float = 1000.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_scale = time_scale
        self.use_checkpoint = use_checkpoint
        self.num_levels = len(channel_mult)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        def res_block(ch_in, ch_out, up=False, down=False):
            return ResBlock(
                ch_in,
                time_embed_dim,
                dropout,
                out_channels=ch_out,
                use_scale_shift_norm=use_scale_shift_norm,
                up=up,
                down=down,
            )

        def attn_block(ch):
            return AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)

        # ---- Encoder ----
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = int(mult * model_channels)
                layers: list[nn.Module] = [res_block(ch, out_ch)]
                ch = out_ch
                if image_size // ds in attention_resolutions:
                    layers.append(attn_block(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                down = res_block(ch, ch, down=True) if resblock_updown else Downsample(ch)
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_chans.append(ch)
                ds *= 2

        # ---- Bottleneck ----
        self.middle_block = TimestepEmbedSequential(
            res_block(ch, ch),
            attn_block(ch),
            res_block(ch, ch),
        )

        # ---- Decoder ----
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                out_ch = int(model_channels * mult)
                layers = [res_block(ch + skip_ch, out_ch)]
                ch = out_ch
                if image_size // ds in attention_resolutions:
                    layers.append(attn_block(ch))
                if level and i == num_res_blocks:
                    up = res_block(ch, ch, up=True) if resblock_updown else Upsample(ch)
                    layers.append(up)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def _run(self, block: nn.Module, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(block, h, emb, use_reentrant=False)
        return block(h, emb)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [N, C_x, H, W] trạng thái x_t.
            t: [N] (hoặc scalar) timestep.
            cond: [N, C_cond, H, W] ảnh điều kiện, ghép với x theo kênh.
                Tổng C_x + C_cond phải bằng in_channels.
        Returns:
            [N, out_channels, H, W].
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        assert x.shape[1] == self.in_channels, (
            f"Kỳ vọng {self.in_channels} kênh vào, nhận {x.shape[1]}"
        )
        factor = 2 ** (self.num_levels - 1)
        assert x.shape[-2] % factor == 0 and x.shape[-1] % factor == 0, (
            f"H, W phải chia hết cho {factor}"
        )
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        emb = self.time_embed(timestep_embedding(t * self.time_scale, self.model_channels))

        hs = []
        h = x
        for block in self.input_blocks:
            h = self._run(block, h, emb)
            hs.append(h)
        h = self._run(self.middle_block, h, emb)
        for block in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(block, h, emb)
        return self.out(h.type(x.dtype))
