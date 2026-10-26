from .base import Method, masked_l1, masked_mse
from .pix2pix import Pix2Pix
from .regression import Regression

METHODS = {"regression": Regression, "pix2pix": Pix2Pix}


def build_method(name: str, **kwargs) -> Method:
    if name not in METHODS:
        raise ValueError(f"method '{name}' không có, chọn trong {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = ["METHODS", "Method", "Pix2Pix", "Regression", "build_method", "masked_l1", "masked_mse"]
