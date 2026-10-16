"""Đọc config: base.yaml <- exp/*.yaml <- ghi đè dòng lệnh dạng key.sub=value."""

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"ghi đè phải có dạng key=value: {item}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(base="configs/base.yaml", exp=None, overrides=()) -> dict:
    cfg = yaml.safe_load(Path(base).read_text())
    if exp:
        cfg = deep_merge(cfg, yaml.safe_load(Path(exp).read_text()) or {})
    return apply_overrides(cfg, list(overrides))
