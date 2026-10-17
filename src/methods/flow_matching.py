"""Rectified flow (conditional flow matching, đường OT thẳng) MR -> CT.

    x_t = (1 - t) x0 + t x1,  x0 ~ N(0, I) (t=0),  x1 = CT (t=1),  v* = x1 - x0
    model(cat[x_t, MR], t) ~ v*, loss MSE chỉ trong mask.

Cố định vùng nền (fix_background): ngoài mask x_t luôn = -1 (nền CT) khi train; khi sinh,
x0 ngoài mask = -1 và vận tốc ngoài mask = 0 nên nền giữ nguyên -1 qua mọi bước Euler.
"""

import torch
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from .base import Method, Net, masked_mse

BACKGROUND = -1.0


class _Velocity(ModelWrapper):
    """Vận tốc trên cả lát cho ODESolver: v(x_t, t | MR), 0 ngoài mask khi cố định nền."""

    def __init__(self, net: Net, cond: torch.Tensor, mask: torch.Tensor, fix_background: bool):
        super().__init__(model=None)
        self.net, self.cond, self.mask, self.fix_background = net, cond, mask, fix_background

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        v = self.net(torch.cat([x, self.cond], dim=1), t).float()
        return torch.where(self.mask > 0, v, 0.0) if self.fix_background else v


class FlowMatching(Method):
    extra_in_channels = 1  # kênh x_t

    def __init__(self, fix_background: bool = True):
        self.fix_background = fix_background
        self.path = CondOTProbPath()

    def loss(self, model, batch):
        x1, cond, mask = batch["ct"], batch["cond"], batch["mask"]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        ps = self.path.sample(x_0=x0, x_1=x1, t=t)
        x_t = torch.where(mask > 0, ps.x_t, BACKGROUND) if self.fix_background else ps.x_t
        v = model(torch.cat([x_t, cond], dim=1), t)
        loss = masked_mse(v.float(), ps.dx_t, mask)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, net, cond, mask, noise, nfe):
        x0 = torch.where(mask > 0, noise, BACKGROUND) if self.fix_background else noise
        solver = ODESolver(velocity_model=_Velocity(net, cond, mask, self.fix_background))
        x1 = solver.sample(
            x_init=x0.float(), step_size=None, method="euler",
            time_grid=torch.linspace(0.0, 1.0, nfe + 1, device=cond.device),
        )
        if self.fix_background:
            x1 = torch.where(mask > 0, x1, BACKGROUND)
        return x1.clamp(-1, 1)
