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
        "entry_window_high_s": 60,
        "max_entries_per_market": 1,
        "max_price": 0.95,
        "max_sum_asks": 1.01,
        "max_spread": 0.05,
        "short_n": 6,
        "long_n": 24,
        "w_edge": 1.6,
        "w_strength": 2.0,
        "w_momentum": 2.2,
        "w_spread": 1.2,
        "w_overround": 1.5,
        "prob_scale": 4.0,
        "prob_bias": 0.0,
        "min_score": 0.01,
        "ev_threshold": 0.0,
        "take_profit_price": 0.90,
        "take_profit_price_cap": 0.99,
        "take_profit_entry_slope": 0.9,
        "take_profit_min_gain_cents": 0.01,
        "stop_loss_cents": 0.0,
        "trailing_drop_cents": 0.0,
        "min_hold_s": 1,
        "max_hold_s": 90,
        "force_exit_s": 2,
        "flip_exit_score_gap": 0.03,
    }
    if params:
        p.update(params)
    s = Base(params=p)
    s.name = "Directional Smart Profit LF"
    s.color = "#ff9f1c"
    return s
