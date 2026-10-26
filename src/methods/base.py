"""Giao diện chung cho các phương pháp MR -> CT (regression, flow matching, ...)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 trung bình trên các điểm trong mask."""
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên các điểm trong mask."""
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


class Method(ABC):
    """Mỗi phương pháp định nghĩa loss khi train và cách sinh CT từ MR khi suy luận.

    Ảnh CT và MR đều ở thang [-1, 1]; cond là [N, C_mr, H, W].
    """

    # Số kênh model nhận ngoài các kênh MR (vd 1 kênh x_t cho flow matching)
    extra_in_channels: int = 0

    @abstractmethod
    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Trả về (loss để backward, các giá trị để log)."""

    @abstractmethod
    def predict(self, model: nn.Module, cond: torch.Tensor, nfe: int = 1) -> torch.Tensor:
        """Sinh CT [N, 1, H, W] từ cond, dùng nfe lần gọi model."""

    def nfe_values(self, requested: list[int]) -> list[int]:
        """Các giá trị NFE có nghĩa với phương pháp này."""
        return list(requested)

    # Hook cho phương pháp có mạng/optimizer riêng ngoài model chính (vd discriminator của GAN).
    # Mặc định không làm gì, nên regression chạy như cũ.
    def setup(self, cfg: dict, device: torch.device, lr_lambda) -> None:
        """Gọi 1 lần trước khi train; lr_lambda là lịch LR dùng chung với model chính."""

    def optimizer_step(self) -> None:
        """Gọi sau mỗi optimizer step của model chính."""

    def state_dict(self) -> dict:
        """Trạng thái cần lưu vào last.pt để train tiếp."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass
