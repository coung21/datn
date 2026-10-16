from .base import Method, masked_l1
from .regression import Regression

METHODS = {"regression": Regression}


def build_method(name: str, **kwargs) -> Method:
    if name not in METHODS:
        raise ValueError(f"method '{name}' không có, chọn trong {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = ["METHODS", "Method", "Regression", "build_method", "masked_l1"]
