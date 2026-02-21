from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base():
    p = Path(__file__).with_name("momentum.py")
    spec = importlib.util.spec_from_file_location(f"base_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed module spec: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mod = _load_base()
Base = getattr(_mod, "LateDominanceStrategy")


def create_strategy(params: dict | None = None):
    p = {'vol_cap_60s': 9.9, 'flip_rate_cap_60s': 1.0, 'jump_cap': 9.9, 'spread_max': 0.10}
    if params:
        p.update(params)
    s = Base(params=p)
    s.name = f"{s.name} NoSafe"
    s.color = "#76b7ff"
    return s
