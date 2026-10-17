"""Baseline: UNet hồi quy trực tiếp MR -> CT (1 lần forward, t = 0 cố định)."""

import torch

from .base import Method, masked_l1


class Regression(Method):
    extra_in_channels = 0

    def loss(self, model, batch):
        cond = batch["cond"]
        pred = model(cond, torch.zeros(cond.shape[0], device=cond.device))
        loss = masked_l1(pred.float(), batch["ct"], batch["mask"])
        return loss, {"l1": loss.item()}

    @torch.no_grad()
    def sample(self, net, cond, mask, noise, nfe=1):
        return net(cond, torch.zeros((), device=cond.device)).float().clamp(-1, 1)

    def nfe_values(self, requested):
        return [1]  # không lặp, NFE luôn bằng 1
