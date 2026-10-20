from .base import Method, masked_l1, masked_mse
from .flow_matching import FlowMatching
from .regression import Regression
from .residual_flow import ResidualFlow

METHODS = {"regression": Regression, "flow_matching": FlowMatching, "residual_flow": ResidualFlow}


def build_method(name: str, **kwargs) -> Method:
    if name not in METHODS:
        raise ValueError(f"method '{name}' không có, chọn trong {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = ["METHODS", "FlowMatching", "Method", "Regression", "ResidualFlow", "build_method", "masked_l1", "masked_mse"]
