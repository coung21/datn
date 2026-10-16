"""Baseline: UNet hồi quy trực tiếp MR -> CT (1 lần forward, t = 0 cố định)."""

import torch
import torch.nn as nn

from .base import Method, masked_l1


class Regression(Method):
    extra_in_channels = 0

    @staticmethod
    def _forward(model: nn.Module, cond: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(cond.shape[0], device=cond.device)
        return model(cond, t)

    def loss(self, model, batch):
        pred = self._forward(model, batch["cond"])
        loss = masked_l1(pred.float(), batch["ct"], batch["mask"])
        return loss, {"l1": loss.item()}

    @torch.no_grad()
    def predict(self, model, cond, nfe=1):
        return self._forward(model, cond).float().clamp(-1, 1)

    def nfe_values(self, requested):
        return [1]  # không lặp, NFE luôn bằng 1
