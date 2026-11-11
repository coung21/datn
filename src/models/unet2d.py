"""UNet 2D cổ điển (Ronneberger 2015) cho MR -> CT 2.5D: vào các lát MR liền kề, ra 1 lát CT.

Mỗi tầng: 2 x [Conv3x3 -> Norm -> ReLU]. Encoder giảm kích thước bằng MaxPool 2x2,
decoder tăng bằng ConvTranspose 2x2 rồi nối skip. Đầu ra Conv1x1, không có hàm kích hoạt.

attention_gates=True: mỗi skip đi qua 1 attention gate (Attention U-Net, Oktay et al. 2018)
trước khi nối với decoder:
    q = psi^T ReLU(Norm(W_x x) + Norm(W_g g)) + b_psi,   alpha = sigmoid(q),   x_hat = alpha * x
    h = ConvBlock([g ; x_hat])
x: đặc trưng skip, g: đặc trưng decoder sau ConvTranspose (cùng độ phân giải với x),
W_x, W_g, psi là Conv1x1, số kênh trung gian C_int = C / 2.

structure_guided=True (cần attention_gates=True): đổi mỗi gate thành SG-gate. Gate dự đoán bản đồ cấu trúc
3 lớp (khí / mô mềm / xương) từ g rồi đưa vào cổng như một nguồn thứ ba:
    P = softmax(W_p g + b_p),   q = psi^T ReLU(Norm(W_x x) + Norm(W_g g) + Norm(W_s P)) + b_psi
Thêm 1 đầu P0 (Conv1x1 -> 3 lớp) trên đặc trưng decoder cuối. Logit cấu trúc của mọi tầng được giám sát
bằng nhãn ngưỡng HU (xem methods/regression.py); khi suy luận chỉ dùng sCT.

mixture_head=True (cần structure_guided=True): đầu ra Conv1x1 cho 3 giá trị CT, mỗi giá trị cho 1 lớp mô,
rồi trộn theo bản đồ cấu trúc P0 của đầu struct_head:
    y = sum_k softmax(P0)_k * y_k,   k in {khí, mô mềm, xương}
Bias của y_k khởi tạo ở HU điển hình của lớp đó (-1000 / 40 / 700 HU, đổi sang thang [-1, 1]).

tissue_film=True (cần norm="group", không dùng cùng structure_guided): mỗi tầng decoder có 1 nhánh dự đoán
bản đồ mô P^l = softmax(Conv1x1(g^l)) (khí / mô mềm / xương, giám sát như SG-gate). P^l điều khiển 2 lớp GN
trong khối conv của tầng đó (FiLM theo mô, kiểu SPADE):
    GN_P(x)_i = gamma_i * x_hat_i + beta_i,   gamma_i = sum_k P_ik gamma_k,   beta_i = sum_k P_ik beta_k
gamma_k = 1, beta_k = 0 lúc khởi tạo. Skip connection không dùng P (attention gate thường nếu bật).

multiscale_input=True: thêm khối trích đặc trưng đa quy mô trên MRI (kiểu ASPP, giữ độ phân giải đầy đủ):
    F = ReLU(GN(Conv1x1([ReLU(GN(Conv3x3_d(MR))) for d in dilations])))
F được AvgPool về kích thước tầng s rồi chiếu Conv1x1 (khởi tạo 0) và cộng vào đầu vào khối conv của
tầng encoder s = 1..depth-1 và bottleneck:
    e_s = Block_s(MaxPool(e_{s-1}) + Proj_s(AvgPool_{2^s}(F)))
Tầng 0 vẫn nhận MRI bình thường. Lúc khởi tạo model giống hệt khi không bật.

multiview_input=True (không dùng cùng multiscale_input): đầu vào là khối K lát MR (in_channels = K). UNet chính chỉ
nhận main_in_channels lát giữa; khối K lát đi qua nhánh multi-view 3D nhẹ, mỗi nhánh 1 mặt cắt:
    axial: Conv3d (1x3x3),  coronal: Conv3d (3x1x3),  sagittal: Conv3d (3x3x1)   (2 lớp, mỗi lớp -> GN -> ReLU)
    F = ReLU(GN(Conv3d_(K,1,1)(ReLU(GN(Conv1x1x1([axial; coronal; sagittal]))))))     (gộp trục z về lát giữa)
F (multiview_channels kênh, độ phân giải đầy đủ) được đưa vào encoder giống multiscale_input (AvgPool, Conv1x1 khởi
tạo 0, cộng vào đầu vào khối conv tầng 1..depth-1 và bottleneck). Lúc khởi tạo model giống UNet chỉ nhận lát giữa.

region_film=True (cần norm="group"): điều kiện hóa theo vùng giải phẫu (0 não, 1 chậu). Mọi GN trong các khối
conv của encoder, bottleneck và decoder có gamma/beta riêng cho từng vùng:
    GN_r(x) = gamma_r * x_hat + beta_r
gamma_r = 1, beta_r = 0 lúc khởi tạo. Attention gate và phần còn lại dùng chung. forward cần region.

region_adagn=True (cần norm="group", block="conv"): điều kiện hóa theo vùng kiểu AdaGN của ADM. Nhãn vùng qua
embedding dùng chung e = Linear(SiLU(Linear(Embedding(r)))); GN thứ 2 trong mỗi khối conv của encoder,
bottleneck và decoder thành
    AdaGN(h) = GN(h) * (1 + scale) + shift,   (scale, shift) = Linear(SiLU(e))   (Linear riêng mỗi khối, khởi tạo 0)
nên lúc khởi tạo model giống hệt khi không bật. Attention gate dùng chung. forward cần region.

block="res": thay mỗi ConvBlock bằng 1 ResBlock (basic block của ResNet, He et al. 2016), cùng số conv:
    y = ReLU(Norm(Conv3x3(ReLU(Norm(Conv3x3(x))))) + shortcut(x)),
shortcut là x, hoặc Conv1x1 + Norm khi số kênh vào khác ra.

bottleneck_attention=True: thêm 1 khối self-attention đa đầu sau ConvBlock ở bottleneck (kiểu ADM):
    h = x + Proj(MHSA(Norm(x)))
mỗi vị trí 16x16 (patch 256) là 1 token; qkv và Proj là Conv1x1, Proj khởi tạo 0 nên lúc đầu khối là
hàm đồng nhất. Không positional encoding, không FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.preprocess import CT_CLIP

MIXTURE_INIT_HU = (-1000.0, 40.0, 700.0)  # khí, mô mềm, xương


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(32, channels), channels)
    if kind == "region_group":
        return RegionFiLMGroupNorm(channels)
    raise ValueError(f"norm '{kind}' không có, chọn batch | instance | group")


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, norm: str):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch), nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), make_norm(norm, out_ch),
        )
        self.shortcut = (nn.Identity() if in_ch == out_ch else
                         nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), make_norm(norm, out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


BLOCKS = {"conv": ConvBlock, "res": ResBlock}


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


class StructureGuidedGate(AttentionGate):
    N_CLASSES = 3  # khí, mô mềm, xương

    def __init__(self, channels: int, norm: str):
        super().__init__(channels, norm)
        inter = channels // 2
        self.w_p = nn.Conv2d(channels, self.N_CLASSES, 1)
        self.w_s = nn.Sequential(nn.Conv2d(self.N_CLASSES, inter, 1, bias=False), make_norm(norm, inter))
        self.last_logits: torch.Tensor | None = None  # logit cấu trúc của lần forward gần nhất, để tính loss

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        logits = self.w_p(g)
        self.last_logits = logits
        p = torch.softmax(logits.float(), dim=1).to(g.dtype)
        alpha = torch.sigmoid(self.psi(torch.relu(self.w_x(x) + self.w_g(g) + self.w_s(p))))
        self.last_alpha = alpha.detach()
        return x * alpha


class TissueFiLMGroupNorm(nn.Module):
    """GroupNorm không affine, rồi gamma/beta theo từng voxel = trung bình gamma_k/beta_k theo xác suất mô P."""

    def __init__(self, channels: int, n_classes: int = 3):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, affine=False)
        self.gamma = nn.Parameter(torch.ones(n_classes, channels))
        self.beta = nn.Parameter(torch.zeros(n_classes, channels))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p = p.to(x.dtype)
        gamma = torch.einsum("bkhw,kc->bchw", p, self.gamma.to(x.dtype))
        beta = torch.einsum("bkhw,kc->bchw", p, self.beta.to(x.dtype))
        return gamma * self.norm(x) + beta


class RegionFiLMGroupNorm(nn.Module):
    """GroupNorm không affine, rồi gamma/beta tra theo nhãn vùng của từng mẫu.

    Nhãn vùng do UNet2D gán vào thuộc tính `region` ([B] long) ngay trước mỗi lần forward."""

    def __init__(self, channels: int, n_regions: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, affine=False)
        self.gamma = nn.Parameter(torch.ones(n_regions, channels))
        self.beta = nn.Parameter(torch.zeros(n_regions, channels))
        self.region: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.region is not None, "RegionFiLMGroupNorm cần nhãn vùng (gọi qua UNet2D với region=...)"
        gamma = self.gamma[self.region].to(x.dtype)[:, :, None, None]
        beta = self.beta[self.region].to(x.dtype)[:, :, None, None]
        return gamma * self.norm(x) + beta


class AdaGroupNorm(nn.Module):
    """GN(h) * (1 + scale) + shift, (scale, shift) chiếu từ embedding vùng (ADM AdaGN, use_scale_shift_norm).

    Embedding [B, E] do UNet2D gán vào thuộc tính `emb` ngay trước mỗi lần forward."""

    def __init__(self, channels: int, emb_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
        self.emb: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.emb is not None, "AdaGroupNorm cần embedding vùng (gọi qua UNet2D với region=...)"
        scale, shift = self.proj(self.emb).to(x.dtype)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


class AdaGNConvBlock(nn.Sequential):
    """Conv3x3 -> GN -> ReLU -> Conv3x3 -> AdaGN(vùng) -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, emb_channels: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), make_norm("group", out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), AdaGroupNorm(out_ch, emb_channels),
            nn.ReLU(inplace=True),
        )


class TissueFiLMConvBlock(nn.Module):
    """2 x [Conv3x3 -> GN điều khiển bởi P -> ReLU]."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = TissueFiLMGroupNorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = TissueFiLMGroupNorm(out_ch)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.conv1(x), p))
        return torch.relu(self.norm2(self.conv2(x), p))


class MultiScaleExtractor(nn.Module):
    """Các nhánh Conv3x3 với dilation khác nhau chạy song song trên MRI, ghép kênh rồi trộn bằng Conv1x1."""

    def __init__(self, in_ch: int, branch_ch: int, out_ch: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=d, dilation=d, bias=False),
                          nn.GroupNorm(min(8, branch_ch), branch_ch), nn.ReLU(inplace=True))
            for d in dilations)
        cat = branch_ch * len(dilations)
        self.fuse = nn.Sequential(nn.Conv2d(cat, out_ch, 1, bias=False), make_norm("group", out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class MultiViewExtractor(nn.Module):
    """Nhánh conv 3D dẹt theo 3 mặt cắt trên khối K lát MR, trộn rồi gộp trục z -> đặc trưng 2D của lát giữa.

    planes: các mặt cắt được dựng nhánh (mặc định cả 3), dùng cho ablation theo mặt cắt."""

    KERNELS = {"axial": (1, 3, 3), "coronal": (3, 1, 3), "sagittal": (3, 3, 1)}

    def __init__(self, n_slices: int, branch_ch: int, out_ch: int, planes=None):
        super().__init__()
        planes = list(self.KERNELS) if planes is None else list(planes)
        if not planes or len(set(planes)) != len(planes) or any(p not in self.KERNELS for p in planes):
            raise ValueError(f"multiview_planes phải là tập con khác rỗng, không lặp của {list(self.KERNELS)}: {planes}")

        def conv(i, o, k):
            return [nn.Conv3d(i, o, k, padding=tuple(s // 2 for s in k), bias=False),
                    nn.GroupNorm(min(8, o), o), nn.ReLU(inplace=True)]

        self.branches = nn.ModuleDict(
            {name: nn.Sequential(*conv(1, branch_ch, k), *conv(branch_ch, branch_ch, k))
             for name, k in self.KERNELS.items() if name in planes})
        self.fuse = nn.Sequential(
            nn.Conv3d(len(planes) * branch_ch, out_ch, 1, bias=False), make_norm("group", out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, (n_slices, 1, 1), bias=False), make_norm("group", out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.unsqueeze(1)  # [B, 1, K, H, W]
        return self.fuse(torch.cat([b(v) for b in self.branches.values()], dim=1)).squeeze(2)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, head_channels: int, norm: str):
        super().__init__()
        assert channels % head_channels == 0, f"{channels} kênh không chia hết cho {head_channels} kênh/head"
        self.heads = channels // head_channels
        self.norm = make_norm(norm, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B, 3C, H, W] -> 3 x [B, heads, HW, C/heads]
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).permute(1, 0, 2, 4, 3)
        a = F.scaled_dot_product_attention(q, k, v)  # softmax(QK^T / sqrt(d)) V
        a = a.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(a)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64, depth: int = 4,
                 norm: str = "batch", attention_gates: bool = False, bottleneck_attention: bool = False,
                 attention_head_channels: int = 64, block: str = "conv",
                 structure_guided: bool = False, mixture_head: bool = False, tissue_film: bool = False,
                 multiscale_input: bool = False, multiscale_branch_channels: int = 16,
                 multiscale_channels: int = 64, multiscale_dilations=(1, 2, 4, 8), region_film: bool = False,
                 region_adagn: bool = False, region_emb_channels: int = 256,
                 multiview_input: bool = False, main_in_channels: int = 3, multiview_branch_channels: int = 16,
                 multiview_channels: int = 64, multiview_planes=None):
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"block '{block}' không có, chọn {' | '.join(BLOCKS)}")
        if structure_guided and not attention_gates:
            raise ValueError("structure_guided cần attention_gates=True")
        if tissue_film and (norm != "group" or block != "conv" or structure_guided):
            raise ValueError("tissue_film cần norm=group, block=conv và structure_guided=False")
        if region_film and (norm != "group" or tissue_film):
            raise ValueError("region_film cần norm=group và tissue_film=False")
        if region_adagn and (norm != "group" or block != "conv" or tissue_film or region_film):
            raise ValueError("region_adagn cần norm=group, block=conv, tissue_film=False và region_film=False")
        if multiview_input and (multiscale_input or main_in_channels % 2 == 0 or not 0 < main_in_channels <= in_channels
                                or in_channels % 2 == 0):
            raise ValueError("multiview_input cần multiscale_input=False, in_channels và main_in_channels lẻ, "
                             "main_in_channels <= in_channels")
        if mixture_head and not structure_guided:
            raise ValueError("mixture_head cần structure_guided=True")
        if mixture_head and out_channels != 1:
            raise ValueError("mixture_head chỉ hỗ trợ out_channels=1")
        Block = BLOCKS[block]
        if region_adagn:
            Block = lambda i, o, _norm: AdaGNConvBlock(i, o, region_emb_channels)  # noqa: E731
            e = region_emb_channels
            self.region_emb = nn.Sequential(nn.Embedding(2, e), nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        Gate = StructureGuidedGate if structure_guided else AttentionGate
        self.structure_guided, self.mixture_head, self.tissue_film = structure_guided, mixture_head, tissue_film
        self.in_channels, self.depth = in_channels, depth
        self.region_film, self.region_adagn = region_film, region_adagn
        self.needs_region = region_film or region_adagn
        block_norm = "region_group" if region_film else norm  # chỉ GN trong các khối conv đổi theo vùng
        chs = [base_channels * 2**i for i in range(depth + 1)]  # 64, 128, 256, 512, 1024

        self.down = nn.ModuleList()
        self.multiview_input = multiview_input
        # multiview_input: UNet chính chỉ nhận main_in_channels lát giữa của khối in_channels lát
        self.main_slice = (slice((in_channels - main_in_channels) // 2, (in_channels + main_in_channels) // 2)
                           if multiview_input else slice(None))
        prev = main_in_channels if multiview_input else in_channels
        for ch in chs[:-1]:
            self.down.append(Block(prev, ch, block_norm))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.multiscale_input = multiscale_input
        if multiscale_input:
            self.ms_extract = MultiScaleExtractor(in_channels, multiscale_branch_channels, multiscale_channels,
                                                  tuple(multiscale_dilations))
            # đầu vào khối conv tầng 1..depth-1 và bottleneck có số kênh chs[0..depth-1]
            self.ms_proj = nn.ModuleList(nn.Conv2d(multiscale_channels, ch, 1) for ch in chs[:-1])
            for proj in self.ms_proj:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        if multiview_input:
            self.mv_extract = MultiViewExtractor(in_channels, multiview_branch_channels, multiview_channels,
                                                 multiview_planes)
            self.mv_proj = nn.ModuleList(nn.Conv2d(multiview_channels, ch, 1) for ch in chs[:-1])
            for proj in self.mv_proj:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        self.bottom = Block(chs[-2], chs[-1], block_norm)
        self.bottom_attn = (SelfAttention2d(chs[-1], attention_head_channels, norm) if bottleneck_attention
                            else nn.Identity())

        self.up, self.dec, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.dec.append(TissueFiLMConvBlock(ch * 2, ch) if tissue_film else Block(ch * 2, ch, block_norm))
            self.gates.append(Gate(ch, norm) if attention_gates else nn.Identity())
        n_cls = StructureGuidedGate.N_CLASSES
        if tissue_film:
            self.struct_pred = nn.ModuleList(nn.Conv2d(ch, n_cls, 1) for ch in reversed(chs[:-1]))
        self.out = nn.Conv2d(chs[0], n_cls if mixture_head else out_channels, 1)
        if mixture_head:
            lo, hi = CT_CLIP
            with torch.no_grad():
                self.out.bias.copy_(torch.tensor([(h - lo) / (hi - lo) * 2 - 1 for h in MIXTURE_INIT_HU]))
        if structure_guided:
            self.struct_head = nn.Conv2d(chs[0], n_cls, 1)

    def forward(self, x: torch.Tensor, return_struct: bool = False, region=None):
        """return_struct=True (structure_guided hoặc tissue_film): trả thêm list (tầng l, logit cấu trúc).
        Tầng l có độ phân giải H/2^l; đầu P0 của structure_guided cũng là tầng 0.
        region (region_film / region_adagn): int chung cho cả batch hoặc tensor [B], 0 não, 1 chậu."""
        assert x.shape[1] == self.in_channels, f"Kỳ vọng {self.in_channels} kênh vào, nhận {x.shape[1]}"
        if self.needs_region:
            assert region is not None, "region_film / region_adagn cần region"
            r = torch.as_tensor(region, device=x.device, dtype=torch.long).reshape(-1).expand(x.shape[0])
            emb = self.region_emb(r) if self.region_adagn else None
            for m in self.modules():
                if isinstance(m, RegionFiLMGroupNorm):
                    m.region = r
                elif isinstance(m, AdaGroupNorm):
                    m.emb = emb
        k = 2**self.depth
        assert x.shape[-2] % k == 0 and x.shape[-1] % k == 0, f"H, W phải chia hết cho {k}, nhận {tuple(x.shape[-2:])}"
        feat, proj = None, None
        if self.multiscale_input:
            feat, proj = self.ms_extract(x), self.ms_proj
        elif self.multiview_input:
            feat, proj = self.mv_extract(x), self.mv_proj
            x = x[:, self.main_slice]
        skips = []
        for s, block in enumerate(self.down):
            if feat is not None and s > 0:
                x = x + proj[s - 1](F.avg_pool2d(feat, 2**s))
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        if feat is not None:
            x = x + proj[-1](F.avg_pool2d(feat, 2**self.depth))
        x = self.bottom_attn(self.bottom(x))
        film_logits = []
        for i, (up, dec, gate, skip) in enumerate(zip(self.up, self.dec, self.gates, reversed(skips))):
            g = up(x)
            if isinstance(gate, AttentionGate):
                skip = gate(skip, g)
            h = torch.cat([g, skip], dim=1)
            if self.tissue_film:
                logits = self.struct_pred[i](g)
                film_logits.append((len(self.up) - 1 - i, logits))
                x = dec(h, torch.softmax(logits.float(), dim=1))
            else:
                x = dec(h)
        out = self.out(x)
        head = self.struct_head(x) if self.structure_guided else None
        if self.mixture_head:
            out = (torch.softmax(head.float(), dim=1) * out.float()).sum(dim=1, keepdim=True)
        if not return_struct:
            return out
        if self.tissue_film:
            return out, film_logits
        assert self.structure_guided, "return_struct cần structure_guided hoặc tissue_film"
        n = len(self.gates)
        structs = [(n - 1 - i, gate.last_logits) for i, gate in enumerate(self.gates)]
        return out, structs + [(0, head)]
