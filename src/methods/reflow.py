"""Conditional reflow (2-rectified flow) theo region.

Teacher (1-RF, method flow_matching) sinh cặp (noise ε, x̂1 = ODE_teacher(ε | MR)) trên tập train
(scripts/generate_reflow.py). Student học lại đường thẳng giữa đúng các cặp đó:

    x_t = (1 - t) ε + t x̂1,  v* = x̂1 - ε,  model(cat[x_t, MR], t, y=region) ~ v*

Cặp đã được ghép cố định nên quỹ đạo của student thẳng hơn -> ít bước (lý tưởng 1) vẫn đủ.
Sinh ảnh giống hệt flow matching (Euler, cố định nền); nhãn region do phía suy luận truyền qua net.
"""

import torch

from .base import masked_mse
from .flow_matching import BACKGROUND, FlowMatching


class Reflow(FlowMatching):
    needs_pairs = True  # batch phải có "x1", "noise" (data.reflow)

    def loss(self, model, batch):
        x1, x0, cond, mask = batch["x1"], batch["noise"], batch["cond"], batch["mask"]
        t = torch.rand(x1.shape[0], device=x1.device)
        ps = self.path.sample(x_0=x0, x_1=x1, t=t)
        x_t = torch.where(mask > 0, ps.x_t, BACKGROUND) if self.fix_background else ps.x_t
        y = batch["y"] if getattr(model, "num_classes", None) else None
        v = model(torch.cat([x_t, cond], dim=1), t, y=y)
        loss = masked_mse(v.float(), ps.dx_t, mask)
        return loss, {"mse": loss.item()}
