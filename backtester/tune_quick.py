#!/usr/bin/env python3
from __future__ import annotations

from backtester.data import expand_paths, load_markets
from backtester.engine import BacktestEngine
from backtester.gridsearch import tune_strategy


def main() -> None:
    engine = BacktestEngine()
    paths = expand_paths(["btc_5m_prices-archive.jsonl", "btc_5m_prices.jsonl"])
    markets = load_markets(paths)

    pair_grid = {
        "short_n": [8, 10, 12],
        "long_n": [28, 36, 44],
        "trend_gap_min": [0.008, 0.012, 0.016],
        "pullback_min": [0.004, 0.006, 0.008],
        "bid_strength_min": [0.02, 0.03, 0.05],
        "spread_cap": [0.015, 0.02, 0.025],
        "entry_price_cap": [0.94, 0.96, 0.98],
        "vol_cap_60s": [0.03, 0.04],
        "flip_rate_cap_60s": [0.22, 0.30],
        "early_pct": [0.2, 0.25],
        "late_pct": [0.3, 0.4],
        "late_seconds": [8, 10, 12],
    }
    dom_grid = {
        "bid_min": [0.94, 0.95, 0.96],
        "strength_min": [0.04, 0.06, 0.08],
        "spread_max": [0.01, 0.015, 0.02],
        "early_pct": [0.2, 0.3],
        "late_pct": [0.3, 0.4],
        "late_seconds": [8, 10],
        "vol_cap_60s": [0.03, 0.04, 0.05],
        "flip_rate_cap_60s": [0.18, 0.24, 0.30],
        "momentum_gap_min": [0.01, 0.015, 0.02],
        "accel_min": [0.001, 0.002, 0.003],
    }

    pair = tune_strategy(
        engine,
        "strategies/mean_reversion.py",
        markets,
        pair_grid,
        objective="total_pnl",
        top_k=5,
    )
    dom = tune_strategy(
        engine,
        "strategies/momentum.py",
        markets,
        dom_grid,
        objective="total_pnl",
        top_k=5,
    )

    print("PAIR BEST:", pair["best"]["params"])
    print("PAIR STATS:", pair["best"]["stats"])
    print("DOM BEST:", dom["best"]["params"])
    print("DOM STATS:", dom["best"]["stats"])


if __name__ == "__main__":
    main()
