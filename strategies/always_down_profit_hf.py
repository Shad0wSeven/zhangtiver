from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base():
    p = Path(__file__).with_name("always_down_tp_sl.py")
    spec = importlib.util.spec_from_file_location(f"base_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed module spec: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mod = _load_base()
Base = getattr(_mod, "AlwaysDownTPSLStrategy")


def create_strategy(params: dict | None = None):
    # Higher-frequency profile: allows multiple re-entries per market.
    p = {
        "bet_usd": 10.0,
        "entry_window_low_s": 5,
        "entry_window_high_s": 90,
        "entry_price_max": 0.98,
        "sum_asks_max": 1.04,
        "take_profit_price": 0.90,
        "take_profit_price_cap": 0.99,
        "take_profit_entry_slope": 0.6,
        "take_profit_min_gain_cents": 0.007,
        "take_profit_cents": 0.0,
        "stop_loss_cents": 0.0,
        "trailing_drop_cents": 0.0,
        "min_hold_s": 1,
        "max_hold_s": 90,
        "force_exit_s": 2,
        "max_entries_per_market": 4,
    }
    if params:
        p.update(params)
    s = Base(params=p)
    s.name = "Always Down Profit HF"
    s.color = "#ff006e"
    return s
