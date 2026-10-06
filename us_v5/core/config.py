"""Load us_v5/config.yaml into a plain nested dict (with dotted-key overrides)."""
from __future__ import annotations

import copy
import logging
import os
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load(overrides: dict[str, Any] | None = None, path: str = CONFIG_PATH) -> dict:
    """Read the yaml config. `overrides` uses dotted keys, e.g.
    {"label.horizon": 20, "portfolio.top_n": 15}."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def path(cfg: dict, key: str, *parts: str) -> str:
    """Absolute path under one of cfg['paths'] (created if missing)."""
    p = os.path.join(ROOT, cfg["paths"][key], *parts)
    os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p), exist_ok=True)
    return p


def copy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
