"""Baseline: UNet hồi quy trực tiếp MR -> CT (1 lần forward).

loss l1 (mặc định) học trung vị có điều kiện; l2 học trung bình có điều kiện E[CT | MR].
"""

import torch
import torch.nn as nn

from .base import Method, masked_l1, masked_mse

LOSSES = {"l1": masked_l1, "l2": masked_mse}


class Regression(Method):
    extra_in_channels = 0

    def __init__(self, loss: str = "l1"):
        if loss not in LOSSES:
            raise ValueError(f"loss '{loss}' không có, chọn trong {sorted(LOSSES)}")
        self.loss_name = loss

    def loss(self, model: nn.Module, batch):
        pred = model(batch["cond"])
        loss = LOSSES[self.loss_name](pred.float(), batch["ct"], batch["mask"])
        return loss, {self.loss_name: loss.item()}

    @torch.no_grad()
    def predict(self, model, cond, nfe=1):
        return model(cond).float().clamp(-1, 1)

    def nfe_values(self, requested):
        return [1]  # không lặp, NFE luôn bằng 1
