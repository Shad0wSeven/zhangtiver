#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from typing import Callable

WINDOW_START_CASH = 1000.0
DEFAULT_BET_USD = 10.0
SETTLE_WIN = 0.99


@dataclass
class Tick:
    timestamp_ms: int
    time_remaining: int
    up_bid: float
    up_ask: float
    up_mid: float
    down_bid: float
    down_ask: float
    down_mid: float


@dataclass
class Action:
    side: str
    token: str
    size: float = 1.0


class SimState:
    def __init__(self) -> None:
        self.cash = WINDOW_START_CASH
        self.pos = {"up": 0.0, "down": 0.0}
        self.cost = {"up": 0.0, "down": 0.0}
        self.trades = 0


def read_markets(paths: list[str]) -> dict[str, list[Tick]]:
    by_market: dict[str, list[Tick]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    market = d.get("market")
                    if not market:
                        continue
                    if (
                        not d.get("up_bids")
                        or not d.get("up_asks")
                        or not d.get("down_bids")
                        or not d.get("down_asks")
                    ):
                        continue
                    up_bid = float(d["up_bids"][0][0])
                    up_ask = float(d["up_asks"][0][0])
                    down_bid = float(d["down_bids"][0][0])
                    down_ask = float(d["down_asks"][0][0])
                    tick = Tick(
                        timestamp_ms=int(d["timestamp_ms"]),
                        time_remaining=int(d["time_remaining"]),
                        up_bid=up_bid,
                        up_ask=up_ask,
                        up_mid=(up_bid + up_ask) / 2.0,
                        down_bid=down_bid,
                        down_ask=down_ask,
                        down_mid=(down_bid + down_ask) / 2.0,
                    )
                    by_market.setdefault(market, []).append(tick)
                except Exception:
                    continue

    for market in by_market:
        by_market[market].sort(key=lambda x: x.timestamp_ms)
    return by_market


def market_winner(rows: list[Tick]) -> str:
    last = rows[-1]
    return "up" if last.up_mid >= last.down_mid else "down"


def is_complete_market(rows: list[Tick]) -> bool:
    if not rows:
        return False
    min_remaining = min(r.time_remaining for r in rows)
    return min_remaining <= 3


def apply_action(state: SimState, tick: Tick, action: Action) -> None:
    side = action.side
    token = action.token
    price = tick.up_mid if token == "up" else tick.down_mid

    if side == "buy":
        if action.size == 1.0:
            notional = DEFAULT_BET_USD
        elif 0 < action.size < 1:
            notional = state.cash * action.size
        else:
            notional = action.size
        notional = min(notional, state.cash)
        if notional <= 0 or price <= 0:
            return
        qty = notional / price
        state.cash -= notional
        state.pos[token] += qty
        state.cost[token] += notional
        state.trades += 1
        return

    if side == "sell":
        if 0 < action.size <= 1:
            qty = state.pos[token] * action.size
        else:
            qty = action.size
        qty = min(qty, state.pos[token])
        if qty <= 0:
            return
        avg_cost = state.cost[token] / state.pos[token] if state.pos[token] > 0 else 0.0
        state.cash += qty * price
        state.pos[token] -= qty
        state.cost[token] -= qty * avg_cost
        if state.pos[token] <= 1e-9:
            state.pos[token] = 0.0
            state.cost[token] = 0.0
        state.trades += 1


def settle(state: SimState, winner: str) -> float:
    for token in ("up", "down"):
        qty = state.pos[token]
        if qty <= 0:
            continue
        px = SETTLE_WIN if token == winner else 0.0
        state.cash += qty * px
        state.pos[token] = 0.0
        state.cost[token] = 0.0
    return state.cash - WINDOW_START_CASH


class LatePairArbModel:
    def __init__(
        self,
        threshold_20_50: float,
        notional_20_50: float,
        threshold_final: float,
        final_seconds: int,
        notional_final: float,
    ) -> None:
        self.threshold_20_50 = threshold_20_50
        self.notional_20_50 = notional_20_50
        self.threshold_final = threshold_final
        self.final_seconds = final_seconds
        self.notional_final = notional_final
        self.did_20_50 = False
        self.did_final = False

    def reset(self) -> None:
        self.did_20_50 = False
        self.did_final = False

    def on_tick(self, tick: Tick) -> list[Action]:
        ask_sum = tick.up_ask + tick.down_ask
        if (
            not self.did_20_50
            and 20 <= tick.time_remaining <= 50
            and ask_sum <= self.threshold_20_50
        ):
            self.did_20_50 = True
            return [
                Action("buy", "up", self.notional_20_50),
                Action("buy", "down", self.notional_20_50),
            ]
        if (
            not self.did_final
            and tick.time_remaining <= self.final_seconds
            and ask_sum <= self.threshold_final
        ):
            self.did_final = True
            return [
                Action("buy", "up", self.notional_final),
                Action("buy", "down", self.notional_final),
            ]
        return []


class LateDominanceModel:
    def __init__(
        self,
        bid_min: float,
        strength_min: float,
        spread_max: float,
        early_pct: float,
        late_pct: float,
        late_seconds: int,
    ) -> None:
        self.bid_min = bid_min
        self.strength_min = strength_min
        self.spread_max = spread_max
        self.early_pct = early_pct
        self.late_pct = late_pct
        self.late_seconds = late_seconds
        self.entered = False

    def reset(self) -> None:
        self.entered = False

    def on_tick(self, tick: Tick) -> list[Action]:
        if self.entered:
            return []
        in_window = tick.time_remaining <= self.late_seconds or (
            20 <= tick.time_remaining <= 50
        )
        if not in_window:
            return []

        up_strength = tick.up_bid - tick.down_bid
        down_strength = tick.down_bid - tick.up_bid
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        size = self.early_pct if tick.time_remaining > self.late_seconds else self.late_pct

        if (
            tick.up_bid >= self.bid_min
            and up_strength >= self.strength_min
            and up_spread <= self.spread_max
        ):
            self.entered = True
            return [Action("buy", "up", size)]

        if (
            tick.down_bid >= self.bid_min
            and down_strength >= self.strength_min
            and down_spread <= self.spread_max
        ):
            self.entered = True
            return [Action("buy", "down", size)]

        return []


def evaluate(
    markets: dict[str, list[Tick]], model_factory: Callable[[], object]
) -> dict[str, float]:
    market_ids = sorted(markets.keys())
    pnls: list[float] = []
    trades = 0
    for market in market_ids:
        rows = markets[market]
        if not is_complete_market(rows):
            continue
        model = model_factory()
        model.reset()
        state = SimState()
        for tick in rows:
            actions = model.on_tick(tick)
            for action in actions:
                apply_action(state, tick, action)
        pnl = settle(state, market_winner(rows))
        pnls.append(pnl)
        trades += state.trades

    if not pnls:
        return {"markets": 0, "avg_pnl": 0.0, "total_pnl": 0.0, "trade_count": 0, "win_rate": 0.0}

    win_rate = 100.0 * sum(1 for x in pnls if x > 0) / len(pnls)
    return {
        "markets": float(len(pnls)),
        "avg_pnl": sum(pnls) / len(pnls),
        "total_pnl": sum(pnls),
        "trade_count": float(trades),
        "win_rate": win_rate,
    }


def tune_pair_arb(markets: dict[str, list[Tick]]) -> tuple[dict, dict]:
    best_params = None
    best_stats = None
    for t1 in [0.975, 0.98, 0.985, 0.99]:
        for n1 in [20.0, 30.0, 40.0, 50.0]:
            for t2 in [0.97, 0.975, 0.98, 0.985]:
                for sec in [4, 6, 8]:
                    for n2 in [10.0, 20.0, 30.0]:
                        def mk(t1=t1, n1=n1, t2=t2, sec=sec, n2=n2):
                            return LatePairArbModel(t1, n1, t2, sec, n2)
                        stats = evaluate(markets, mk)
                        score = stats["total_pnl"]
                        if best_stats is None or score > best_stats["total_pnl"]:
                            best_params = {
                                "threshold_20_50": t1,
                                "notional_20_50": n1,
                                "threshold_final": t2,
                                "final_seconds": sec,
                                "notional_final": n2,
                            }
                            best_stats = stats
    return best_params, best_stats


def tune_dominance(markets: dict[str, list[Tick]]) -> tuple[dict, dict]:
    best_params = None
    best_stats = None
    for bid in [0.92, 0.94, 0.95, 0.96]:
        for strength in [0.06, 0.08, 0.10, 0.12]:
            for spread in [0.015, 0.02, 0.03]:
                for early in [0.15, 0.2, 0.25, 0.3]:
                    for late in [0.25, 0.3, 0.35, 0.4]:
                        for sec in [6, 8, 10]:
                            def mk(
                                bid=bid,
                                strength=strength,
                                spread=spread,
                                early=early,
                                late=late,
                                sec=sec,
                            ):
                                return LateDominanceModel(
                                    bid_min=bid,
                                    strength_min=strength,
                                    spread_max=spread,
                                    early_pct=early,
                                    late_pct=late,
                                    late_seconds=sec,
                                )

                            stats = evaluate(markets, mk)
                            score = stats["total_pnl"]
                            if best_stats is None or score > best_stats["total_pnl"]:
                                best_params = {
                                    "bid_min": bid,
                                    "strength_min": strength,
                                    "spread_max": spread,
                                    "early_pct": early,
                                    "late_pct": late,
                                    "late_seconds": sec,
                                }
                                best_stats = stats
    return best_params, best_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest/tune late-window strategies on BTC 5m JSONL logs")
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=["btc_5m_prices-archive.jsonl", "btc_5m_prices.jsonl"],
        help="JSONL files or globs",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "tune"],
        default="tune",
        help="Run plain evaluation or parameter tuning",
    )
    args = parser.parse_args()

    paths: list[str] = []
    for pat in args.jsonl:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pat)
    paths = [p for p in paths if p]
    if not paths:
        raise SystemExit("no jsonl files found")

    markets = read_markets(paths)
    print("loaded files:", ", ".join(paths))
    print("markets total:", len(markets))
    complete = sum(1 for rows in markets.values() if is_complete_market(rows))
    print("markets complete:", complete)

    if args.mode == "eval":
        pair_stats = evaluate(markets, lambda: LatePairArbModel(0.975, 50.0, 0.97, 6, 10.0))
        dom_stats = evaluate(markets, lambda: LateDominanceModel(0.95, 0.06, 0.015, 0.30, 0.40, 10))
        print("pair_arb_stats", json.dumps(pair_stats, indent=2))
        print("dominance_stats", json.dumps(dom_stats, indent=2))
        return

    pair_params, pair_stats = tune_pair_arb(markets)
    dom_params, dom_stats = tune_dominance(markets)
    print("best_pair_params", json.dumps(pair_params, indent=2))
    print("best_pair_stats", json.dumps(pair_stats, indent=2))
    print("best_dom_params", json.dumps(dom_params, indent=2))
    print("best_dom_stats", json.dumps(dom_stats, indent=2))


if __name__ == "__main__":
    main()
