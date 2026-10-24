from .base import Method, masked_l1, masked_mse
from .flow_matching import FlowMatching
from .mr_bridge import MRBridge
from .regression import Regression

METHODS = {"regression": Regression, "flow_matching": FlowMatching, "mr_bridge": MRBridge}


def build_method(name: str, **kwargs) -> Method:
    if name not in METHODS:
        raise ValueError(f"method '{name}' không có, chọn trong {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = ["METHODS", "FlowMatching", "MRBridge", "Method", "Regression", "build_method", "masked_l1", "masked_mse"]
