"""pix2pix (Isola et al. 2017): G là UNet hồi quy, thêm discriminator PatchGAN có điều kiện.

    L_D = 1/2 [BCE(D(MR, CT), 1) + BCE(D(MR, G(MR)), 0)]
    L_G = BCE(D(MR, G(MR)), 1) + lambda_l1 * L1(G(MR), CT)      (non-saturating, lambda = 100)

Khác bản gốc: L1 chỉ tính trong mask (giống baseline), G không có dropout.

loss() trả L_G + L_D để train loop backward 1 lần; hai phần không lẫn gradient vì L_D dùng
G(MR).detach() còn L_G tính khi D đã đóng băng. G và D step cùng lúc sau mỗi optimizer step,
nên L_G dùng D của step trước (bản gốc cập nhật D xong mới tính L_G); chênh lệch 1 step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import PatchGAN
from .regression import LOSSES, Regression


def gan_loss(logits: torch.Tensor, real: bool, mode: str) -> torch.Tensor:
    logits = logits.float()
    target = torch.full_like(logits, 1.0 if real else 0.0)
    if mode == "vanilla":
        return F.binary_cross_entropy_with_logits(logits, target)
    if mode == "lsgan":
        return F.mse_loss(logits, target)
    raise ValueError(f"gan_mode '{mode}' không có, chọn vanilla | lsgan")


class Pix2Pix(Regression):
    def __init__(self, gan_mode: str = "vanilla", lambda_l1: float = 100.0, disc: dict | None = None,
                 loss: str = "l1"):
        super().__init__(loss=loss)  # loss tái tạo, pix2pix chuẩn là l1
        gan_loss(torch.zeros(1), True, gan_mode)  # kiểm tra gan_mode sớm
        self.gan_mode, self.lambda_l1 = gan_mode, lambda_l1
        self.disc_cfg = disc or {}
        self.D: nn.Module | None = None

    def setup(self, cfg, device, lr_lambda):
        in_ch = cfg["model"]["in_channels"] + cfg["model"]["out_channels"]
        self.D = PatchGAN(in_channels=in_ch, **self.disc_cfg).to(device).train()
        oc = cfg["train"]["optimizer"]
        self.opt_d = torch.optim.AdamW(self.D.parameters(), lr=oc["lr"], betas=tuple(oc["betas"]),
                                       weight_decay=oc["weight_decay"])
        self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda)

    def loss(self, model, batch):
        cond, ct = batch["cond"], batch["ct"]
        fake = model(cond)

        # D: thật -> 1, giả -> 0 (G(MR) detach nên không có gradient về G)
        d_real = self.D(torch.cat([cond, ct], 1))
        d_fake = self.D(torch.cat([cond, fake.detach()], 1))
        loss_d = 0.5 * (gan_loss(d_real, True, self.gan_mode) + gan_loss(d_fake, False, self.gan_mode))

        # G: đóng băng D để loss G không cộng gradient vào D
        self.D.requires_grad_(False)
        loss_adv = gan_loss(self.D(torch.cat([cond, fake], 1)), True, self.gan_mode)
        self.D.requires_grad_(True)
        loss_l1 = LOSSES[self.loss_name](fake.float(), ct, batch["mask"])
        loss_g = loss_adv + self.lambda_l1 * loss_l1
        return loss_g + loss_d, {
            "l1": loss_l1.item(), "g_adv": loss_adv.item(), "g": loss_g.item(), "d": loss_d.item(),
            "d_real": torch.sigmoid(d_real.float()).mean().item(),  # D tin ảnh thật là thật (~1 là D mạnh)
            "d_fake": torch.sigmoid(d_fake.float()).mean().item(),  # D tin ảnh giả là thật
        }

    def optimizer_step(self):
        self.opt_d.step()
        self.sched_d.step()
        self.opt_d.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"D": self.D.state_dict(), "opt_d": self.opt_d.state_dict(), "sched_d": self.sched_d.state_dict()}

    def load_state_dict(self, state):
        self.D.load_state_dict(state["D"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sched_d.load_state_dict(state["sched_d"])
