"""UNet 2D cổ điển (Ronneberger 2015) cho MR -> CT 2.5D: vào các lát MR liền kề, ra 1 lát CT.

Mỗi tầng: 2 x [Conv3x3 -> Norm -> ReLU]. Encoder giảm kích thước bằng MaxPool 2x2,
decoder tăng bằng ConvTranspose 2x2 rồi nối skip. Đầu ra Conv1x1, không có hàm kích hoạt.

attention_gates=True: mỗi skip đi qua 1 attention gate (Attention U-Net, Oktay et al. 2018)
trước khi nối với decoder:
    q = psi^T ReLU(Norm(W_x x) + Norm(W_g g)) + b_psi,   alpha = sigmoid(q),   x_hat = alpha * x
    h = ConvBlock([g ; x_hat])
x: đặc trưng skip, g: đặc trưng decoder sau ConvTranspose (cùng độ phân giải với x),
W_x, W_g, psi là Conv1x1, số kênh trung gian C_int = C / 2.
"""

import torch
import torch.nn as nn


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(32, channels), channels)
    raise ValueError(f"norm '{kind}' không có, chọn batch | instance | group")


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, norm: str):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch), nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    def __init__(self, channels: int, norm: str):
        super().__init__()
        inter = channels // 2
        self.w_x = nn.Sequential(nn.Conv2d(channels, inter, 1, bias=False), make_norm(norm, inter))
        self.w_g = nn.Sequential(nn.Conv2d(channels, inter, 1, bias=False), make_norm(norm, inter))
        self.psi = nn.Conv2d(inter, 1, 1)  # có bias b_psi, không norm
        self.last_alpha: torch.Tensor | None = None  # alpha của lần forward gần nhất, để phân tích

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.psi(torch.relu(self.w_x(x) + self.w_g(g))))
        self.last_alpha = alpha.detach()
        return x * alpha


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64, depth: int = 4,
                 norm: str = "batch", attention_gates: bool = False):
        super().__init__()
        self.in_channels, self.depth = in_channels, depth
        chs = [base_channels * 2**i for i in range(depth + 1)]  # 64, 128, 256, 512, 1024

        self.down = nn.ModuleList()
        prev = in_channels
        for ch in chs[:-1]:
            self.down.append(ConvBlock(prev, ch, norm))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottom = ConvBlock(chs[-2], chs[-1], norm)

        self.up, self.dec, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.dec.append(ConvBlock(ch * 2, ch, norm))
            self.gates.append(AttentionGate(ch, norm) if attention_gates else nn.Identity())
        self.out = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels, f"Kỳ vọng {self.in_channels} kênh vào, nhận {x.shape[1]}"
        k = 2**self.depth
        assert x.shape[-2] % k == 0 and x.shape[-1] % k == 0, f"H, W phải chia hết cho {k}, nhận {tuple(x.shape[-2:])}"
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, gate, skip in zip(self.up, self.dec, self.gates, reversed(skips)):
            g = up(x)
            if isinstance(gate, AttentionGate):
                skip = gate(skip, g)
            x = dec(torch.cat([g, skip], dim=1))
        return self.out(x)
