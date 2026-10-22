from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base():
    p = Path(__file__).with_name("directional_smart_profit.py")
    spec = importlib.util.spec_from_file_location(f"base_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed module spec: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mod = _load_base()
Base = getattr(_mod, "DirectionalSmartProfitStrategy")


def create_strategy(params: dict | None = None):
    p = {
        "bet_usd": 10.0,
        "entry_window_low_s": 5,
        "entry_window_high_s": 90,
        "max_entries_per_market": 3,
        "max_price": 0.98,
        "max_sum_asks": 1.04,
        "max_spread": 0.06,
        "short_n": 6,
        "long_n": 24,
        "w_edge": 1.5,
        "w_strength": 2.2,
        "w_momentum": 2.4,
        "w_spread": 1.0,
        "w_overround": 1.2,
        "prob_scale": 3.6,
        "prob_bias": 0.0,
        "min_score": 0.008,
        "ev_threshold": -0.005,
        "take_profit_price": 0.90,
        "take_profit_price_cap": 0.99,
        "take_profit_entry_slope": 0.8,
        "take_profit_min_gain_cents": 0.007,
        "stop_loss_cents": 0.0,
        "trailing_drop_cents": 0.0,
        "min_hold_s": 1,
        "max_hold_s": 90,
        "force_exit_s": 2,
        "flip_exit_score_gap": 0.025,
    }
    if params:
        p.update(params)
    s = Base(params=p)
    s.name = "Directional Smart Profit HF"
    s.color = "#ff5400"
    return s
