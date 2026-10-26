from .unet2d import UNet2D


def build_model(cfg_model: dict) -> UNet2D:
    return UNet2D(**cfg_model)


__all__ = ["UNet2D", "build_model"]
