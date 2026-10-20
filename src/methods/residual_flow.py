"""Residual flow matching: flow sinh phần dư của regression thay vì cả CT.

    ŷ = sCT regression (kênh cuối của cond, thang [-1, 1])
    z1 = (x1 - ŷ) / scale     phần dư đã chuẩn hóa (0 ngoài mask)
    z0 = sigma * eps          điểm xuất phát (sigma = 0: tất định, gần PMRF khi nhỏ)
    z_t = (1 - t) z0 + t z1,  v* = z1 - z0,  model(cat[z_t, MR, ŷ], t) ~ v*
    sCT = ŷ + scale * ẑ1

Vùng nền: ngoài mask phần dư = 0 (CT và ŷ đều = -1), nên z_t ngoài mask luôn = 0 và vận tốc ở đó = 0.
"""

import torch
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

from .base import Method, masked_mse
from .flow_matching import _Velocity


class ResidualFlow(Method):
    extra_in_channels = 1  # kênh z_t
    needs_reg = True       # cond phải có ŷ ở kênh cuối (data.with_reg)

    def __init__(self, sigma: float = 0.5, scale: float | None = None):
        if scale is None or scale <= 0:
            raise ValueError("method.scale phải > 0: lấy độ lệch chuẩn phần dư từ `scripts/predict_reg.py --stats`")
        self.sigma, self.scale = sigma, scale
        self.path = CondOTProbPath()

    def loss(self, model, batch):
        x1, cond, mask = batch["ct"], batch["cond"], batch["mask"]
        inside = mask > 0
        z1 = torch.where(inside, (x1 - cond[:, -1:]) / self.scale, 0.0)
        z0 = torch.where(inside, self.sigma * torch.randn_like(x1), 0.0)
        t = torch.rand(x1.shape[0], device=x1.device)
        ps = self.path.sample(x_0=z0, x_1=z1, t=t)
        v = model(torch.cat([ps.x_t, cond], dim=1), t)
        loss = masked_mse(v.float(), ps.dx_t, mask)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, net, cond, mask, noise, nfe):
        inside = mask > 0
        z0 = torch.where(inside, self.sigma * noise, 0.0).float()
        solver = ODESolver(velocity_model=_Velocity(net, cond, mask, fix_background=True))
        z1 = solver.sample(
            x_init=z0, step_size=None, method="euler",
            time_grid=torch.linspace(0.0, 1.0, nfe + 1, device=cond.device),
        )
        y_hat = cond[:, -1:].float()
        return torch.where(inside, y_hat + self.scale * z1, -1.0).clamp(-1, 1)
