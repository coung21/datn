"""Giao diện chung cho các phương pháp MR -> CT (regression, flow matching, ...)."""

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

# net(inp, t) -> output: model đã bọc sliding window, chạy trên cả lát (H, W >= roi).
# inp = [x_t (nếu có), MR] theo kênh; t là tensor vô hướng.
Net = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 trung bình trên các điểm trong mask."""
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên các điểm trong mask."""
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


class Method(ABC):
    """Mỗi phương pháp định nghĩa loss khi train và cách sinh CT từ MR khi suy luận.

    Ảnh CT và MR đều ở thang [-1, 1]; cond là [N, C_mr, H, W], mask là [N, 1, H, W].
    """

    # Số kênh model nhận ngoài cond (vd 1 kênh x_t cho flow matching)
    extra_in_channels: int = 0
    # cond có kèm ŷ (sCT regression) ở kênh cuối không (data.with_reg)
    needs_reg: bool = False

    @abstractmethod
    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Trả về (loss để backward, các giá trị để log)."""

    @abstractmethod
    def sample(
        self, net: Net, cond: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor, nfe: int
    ) -> torch.Tensor:
        """Sinh CT [N, 1, H, W] (thang [-1, 1]) với nfe lần gọi net.

        noise [N, 1, H, W] ~ N(0, I) do phía suy luận sinh có seed; phương pháp tất định bỏ qua.
        """

    def nfe_values(self, requested: list[int]) -> list[int]:
        """Các giá trị NFE có nghĩa với phương pháp này."""
        return list(requested)
