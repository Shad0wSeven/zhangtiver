#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict


def read_rows(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if (
                    d.get("up_bids")
                    and d.get("up_asks")
                    and d.get("down_bids")
                    and d.get("down_asks")
                ):
                    yield d
            except Exception:
                continue


def market_rows(path: str):
    by_market: dict[str, list[dict]] = defaultdict(list)
    for d in read_rows(path):
        by_market[d["market"]].append(d)
    for market, rows in by_market.items():
        rows.sort(key=lambda x: x["timestamp_ms"])
        yield market, rows


def best_in_window(rows: list[dict], lo: int, hi: int):
    # time_remaining in [lo, hi]
    w = [r for r in rows if lo <= r["time_remaining"] <= hi]
    if not w:
        return None
    # pick latest in that window
    return w[-1]


def winner(rows: list[dict]) -> str:
    last = rows[-1]
    up_mid = (last["up_bids"][0][0] + last["up_asks"][0][0]) / 2.0
    down_mid = (last["down_bids"][0][0] + last["down_asks"][0][0]) / 2.0
    return "up" if up_mid >= down_mid else "down"


def evaluate(path: str):
    markets = list(market_rows(path))
    print("markets", len(markets))

    # Strategy A: buy both if up_ask + down_ask <= threshold at 20-50s
    for threshold in [0.97, 0.975, 0.98, 0.985, 0.99]:
        trades = 0
        pnl = 0.0
        for _, rows in markets:
            snap = best_in_window(rows, 20, 50)
            if not snap:
                continue
            up_ask = snap["up_asks"][0][0]
            down_ask = snap["down_asks"][0][0]
            cost = up_ask + down_ask
            if cost <= threshold:
                trades += 1
                pnl += 0.99 - cost
        if trades:
            print("pair_arb_20_50", threshold, "trades", trades, "avg", pnl / trades, "sum", pnl)

    # Strategy B: buy strongest side in last N seconds if its ask >= t
    for sec in [3, 5, 8, 12]:
        for t in [0.9, 0.92, 0.94, 0.95, 0.96, 0.97]:
            trades = 0
            pnl = 0.0
            for _, rows in markets:
                snap = best_in_window(rows, 0, sec)
                if not snap:
                    continue
                up_ask = snap["up_asks"][0][0]
                down_ask = snap["down_asks"][0][0]
                side = "up" if up_ask >= down_ask else "down"
                ask = up_ask if side == "up" else down_ask
                if ask < t:
                    continue
                w = winner(rows)
                settle = 0.99 if side == w else 0.0
                pnl += settle - ask
                trades += 1
            if trades >= 5:
                print("late_chase", sec, t, "trades", trades, "avg", pnl / trades, "sum", pnl)

    # Strategy C: buy weakest side cheaply for bounce in 20-50s then settle
    for max_ask in [0.02, 0.03, 0.04, 0.05, 0.06]:
        trades = 0
        pnl = 0.0
        for _, rows in markets:
            snap = best_in_window(rows, 20, 50)
            if not snap:
                continue
            up_ask = snap["up_asks"][0][0]
            down_ask = snap["down_asks"][0][0]
            side = "up" if up_ask < down_ask else "down"
            ask = up_ask if side == "up" else down_ask
            if ask > max_ask:
                continue
            w = winner(rows)
            settle = 0.99 if side == w else 0.0
            pnl += settle - ask
            trades += 1
        if trades:
            print("cheap_lottery_20_50", max_ask, "trades", trades, "avg", pnl / trades, "sum", pnl)


if __name__ == "__main__":
    evaluate("btc_5m_prices.jsonl")
