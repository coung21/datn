"""Flow matching MR -> CT: xuất phát từ ảnh MR của chính bệnh nhân thay vì noise.

    x0 = MR_z (+ sigma * eps trong mask),  x1 = CT_z,  x_t = (1 - t) x0 + t x1,  v* = x1 - x0
    model(cat[x_t, MR_{z-s}, MR_z, MR_{z+s}], t) ~ v*, loss MSE trong mask.

Cặp (MR, CT) đã đăng ký là coupling: điểm xuất phát đã đúng giải phẫu, flow chỉ phải đổi tương phản.
sigma = 0: cầu nối tất định (NFE 1 ~ E[CT | MR]); sigma > 0: có ngẫu nhiên để đo độ bất định.
Ngoài mask MR và CT đều = -1 nên nền giữ nguyên; vẫn cố định nền như flow matching cho chắc.
"""

import torch
from flow_matching.solver import ODESolver

from .base import masked_mse
from .flow_matching import BACKGROUND, FlowMatching, _Velocity


class MRBridge(FlowMatching):
    def __init__(self, sigma: float = 0.0, fix_background: bool = True):
        super().__init__(fix_background=fix_background)
        self.sigma = sigma

    def _start(self, cond: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x0 = lát MR giữa (+ sigma * noise), nền = -1."""
        mr = cond[:, cond.shape[1] // 2:cond.shape[1] // 2 + 1].float()
        x0 = mr + self.sigma * noise if self.sigma > 0 else mr
        return torch.where(mask > 0, x0, BACKGROUND) if self.fix_background else x0

    def loss(self, model, batch):
        x1, cond, mask = batch["ct"], batch["cond"], batch["mask"]
        x0 = self._start(cond, mask, torch.randn_like(x1))
        t = torch.rand(x1.shape[0], device=x1.device)
        ps = self.path.sample(x_0=x0, x_1=x1, t=t)
        x_t = torch.where(mask > 0, ps.x_t, BACKGROUND) if self.fix_background else ps.x_t
        v = model(torch.cat([x_t, cond], dim=1), t)
        loss = masked_mse(v.float(), ps.dx_t, mask)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, net, cond, mask, noise, nfe):
        x0 = self._start(cond, mask, noise)
        solver = ODESolver(velocity_model=_Velocity(net, cond, mask, self.fix_background))
        x1 = solver.sample(
            x_init=x0, step_size=None, method="euler",
            time_grid=torch.linspace(0.0, 1.0, nfe + 1, device=cond.device),
        )
        if self.fix_background:
            x1 = torch.where(mask > 0, x1, BACKGROUND)
        return x1.clamp(-1, 1)
