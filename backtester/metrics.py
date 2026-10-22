from __future__ import annotations

import math
from typing import Any

from backtester.types import MarketResult, Trade


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    market_results: list[MarketResult] = result["market_results"]
    trades: list[Trade] = result["trades"]
    pnls = [m.pnl for m in market_results]
    mdds = [m.max_drawdown for m in market_results]
    fees = [m.fees for m in market_results]
    durations_s = [max(1, int(getattr(m, "duration_s", 300))) for m in market_results]
    total_notional = sum(t.notional for t in trades)
    total_fees_from_trades = sum(t.fee for t in trades)
    total_5m_windows = sum(d / 300.0 for d in durations_s)

    if not pnls:
        return {
            "markets": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "pnl_per_session": 0.0,
            "pnl_per_5m_window": 0.0,
            "median_pnl": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "avg_trades_per_market": 0.0,
            "total_notional": 0.0,
            "total_fees": 0.0,
            "fee_rate_on_notional": 0.0,
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
        # A "session" maps to one market instance in this backtester.
        "pnl_per_session": mean,
        "pnl_per_5m_window": (
            (sum(pnls) / total_5m_windows) if total_5m_windows > 1e-9 else 0.0
        ),
        "median_pnl": sorted(pnls)[len(pnls) // 2],
        "win_rate": (100.0 * len(wins) / len(pnls)),
        "trade_count": len(trades),
        "avg_trades_per_market": len(trades) / len(pnls),
        "total_notional": total_notional,
        "total_fees": total_fees_from_trades
        if total_fees_from_trades > 0
        else sum(fees),
        "fee_rate_on_notional": (
            (total_fees_from_trades / total_notional) if total_notional > 1e-9 else 0.0
        ),
        "max_drawdown": max(mdds) if mdds else 0.0,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"markets: {summary['markets']}")
    print(f"total_pnl: {summary['total_pnl']:.4f}")
    print(f"avg_pnl: {summary['avg_pnl']:.4f}")
    print(f"pnl_per_session: {summary['pnl_per_session']:.4f}")
    print(f"pnl_per_5m_window: {summary['pnl_per_5m_window']:.4f}")
    print(f"median_pnl: {summary['median_pnl']:.4f}")
    print(f"win_rate: {summary['win_rate']:.2f}%")
    print(f"trade_count: {summary['trade_count']}")
    print(f"avg_trades_per_market: {summary['avg_trades_per_market']:.2f}")
    print(f"total_notional: {summary['total_notional']:.4f}")
    print(f"total_fees: {summary['total_fees']:.4f}")
    print(f"fee_rate_on_notional: {100.0 * summary['fee_rate_on_notional']:.2f}%")
    print(f"max_drawdown: {summary['max_drawdown']:.4f}")
    pf = summary["profit_factor"]
    if pf == float("inf"):
        print("profit_factor: inf")
    else:
        print(f"profit_factor: {pf:.4f}")
    print(f"sharpe_like: {summary['sharpe_like']:.4f}")
