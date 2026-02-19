from __future__ import annotations

import math
from typing import Any

from backtester.types import MarketResult, Trade


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    market_results: list[MarketResult] = result["market_results"]
    trades: list[Trade] = result["trades"]
    pnls = [m.pnl for m in market_results]
    mdds = [m.max_drawdown for m in market_results]

    if not pnls:
        return {
            "markets": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "avg_trades_per_market": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
        }

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / len(pnls)
    std = math.sqrt(var)
    sharpe_like = (mean / std) if std > 1e-9 else 0.0

    return {
        "markets": len(pnls),
        "total_pnl": sum(pnls),
        "avg_pnl": mean,
        "median_pnl": sorted(pnls)[len(pnls) // 2],
        "win_rate": (100.0 * len(wins) / len(pnls)),
        "trade_count": len(trades),
        "avg_trades_per_market": len(trades) / len(pnls),
        "max_drawdown": max(mdds) if mdds else 0.0,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"markets: {summary['markets']}")
    print(f"total_pnl: {summary['total_pnl']:.4f}")
    print(f"avg_pnl: {summary['avg_pnl']:.4f}")
    print(f"median_pnl: {summary['median_pnl']:.4f}")
    print(f"win_rate: {summary['win_rate']:.2f}%")
    print(f"trade_count: {summary['trade_count']}")
    print(f"avg_trades_per_market: {summary['avg_trades_per_market']:.2f}")
    print(f"max_drawdown: {summary['max_drawdown']:.4f}")
    pf = summary["profit_factor"]
    if pf == float("inf"):
        print("profit_factor: inf")
    else:
        print(f"profit_factor: {pf:.4f}")
    print(f"sharpe_like: {summary['sharpe_like']:.4f}")
