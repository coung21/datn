"""Baseline: UNet hồi quy trực tiếp MR -> CT (1 lần forward).

loss l1 (mặc định) học trung vị có điều kiện; l2 học trung bình có điều kiện E[CT | MR].

struct_weight > 0 (dùng với model.structure_guided): cộng thêm loss cấu trúc cho bản đồ khí / mô mềm / xương
mà SG-gate dự đoán ở mỗi tầng:
    L = L1 + lambda * sum_l w_l L_l / sum_l w_l,   w_l = 2^-l
    L_l = CE(P_l, s_l; M_l) + 1/2 (Dice_khí + Dice_xương)
Nhãn s lấy từ CT bằng ngưỡng HU (khí < -400, xương > 200), hạ phân giải nearest về kích thước tầng l.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.preprocess import CT_CLIP
from .base import Method, masked_l1, masked_mse

LOSSES = {"l1": masked_l1, "l2": masked_mse}
AIR, SOFT, BONE = 0, 1, 2


def structure_labels(ct: torch.Tensor, thresholds=(-400.0, 200.0)) -> torch.Tensor:
    """CT [N,1,H,W] ở thang [-1, 1] -> nhãn [N,1,H,W] (long): 0 khí, 1 mô mềm, 2 xương."""
    lo, hi = CT_CLIP
    hu = (ct + 1) / 2 * (hi - lo) + lo
    lab = torch.full_like(hu, SOFT, dtype=torch.long)
    lab[hu < thresholds[0]] = AIR
    lab[hu > thresholds[1]] = BONE
    return lab


def masked_dice_loss(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice mềm trên cả batch, chỉ tính trong mask. Các tensor [N,1,H,W]."""
    inter = (prob * target * mask).sum()
    return 1 - (2 * inter + eps) / ((prob * mask).sum() + (target * mask).sum() + eps)


def structure_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE + 1/2 (Dice khí + Dice xương) trong mask; nhãn/mask đưa về kích thước logits bằng nearest."""
    size = logits.shape[-2:]
    lab = F.interpolate(labels.float(), size=size, mode="nearest").long()
    m = F.interpolate(mask.float(), size=size, mode="nearest")
    logits = logits.float()
    ce = (F.cross_entropy(logits, lab[:, 0], reduction="none") * m[:, 0]).sum() / m.sum().clamp_min(1.0)
    prob = logits.softmax(dim=1)
    dice = sum(masked_dice_loss(prob[:, k:k + 1], (lab == k).float(), m) for k in (AIR, BONE))
    return ce + 0.5 * dice


class Regression(Method):
    extra_in_channels = 0

    def __init__(self, loss: str = "l1", struct_weight: float = 0.0, struct_thresholds=(-400.0, 200.0)):
        if loss not in LOSSES:
            raise ValueError(f"loss '{loss}' không có, chọn trong {sorted(LOSSES)}")
        self.loss_name = loss
        self.struct_weight = float(struct_weight)
        self.struct_thresholds = tuple(struct_thresholds)

    def loss(self, model: nn.Module, batch):
        if self.struct_weight <= 0:
            pred = model(batch["cond"])
            loss = LOSSES[self.loss_name](pred.float(), batch["ct"], batch["mask"])
            return loss, {self.loss_name: loss.item()}
        pred, structs = model(batch["cond"], return_struct=True)
        rec = LOSSES[self.loss_name](pred.float(), batch["ct"], batch["mask"])
        labels = structure_labels(batch["ct"], self.struct_thresholds)
        weights = [2.0 ** -level for level, _ in structs]
        struct = sum(w * structure_loss(lg, labels, batch["mask"]) for w, (_, lg) in zip(weights, structs)) / sum(weights)
        loss = rec + self.struct_weight * struct
        return loss, {self.loss_name: rec.item(), "struct": struct.item()}

    @torch.no_grad()
    def predict(self, model, cond, nfe=1):
        return model(cond).float().clamp(-1, 1)

    def nfe_values(self, requested):
        return [1]  # không lặp, NFE luôn bằng 1
