#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SETTLE_PRICE = 0.99


@dataclass
class Row:
    ts: int
    up: float
    down: float
    btc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit late-window buy/no-trade logistic regression from 1s logs."
    )
    parser.add_argument("--jsonl", default="btc_1s_prices.jsonl")
    parser.add_argument("--fit-min-time-remaining", type=int, default=5)
    parser.add_argument("--fit-max-time-remaining", type=int, default=120)
    parser.add_argument("--fit-price-min", type=float, default=0.75)
    parser.add_argument("--fit-price-max", type=float, default=0.995)
    parser.add_argument("--buy-min-time-remaining", type=int, default=5)
    parser.add_argument("--buy-max-time-remaining", type=int, default=80)
    parser.add_argument("--buy-price-min", type=float, default=0.95)
    parser.add_argument("--buy-price-max", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--output-model", default="backtester/late_logistic_model.json")
    return parser.parse_args()


def parse_ts(s: str) -> int:
    # Input format from logger: YYYY-MM-DD HH:MM:SS (local time).
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


def load_rows(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts = parse_ts(str(d["timestamp"]))
                btc = d.get("btc")
                if btc is None:
                    continue
                rows.append(
                    Row(
                        ts=ts,
                        up=float(d["up"]),
                        down=float(d["down"]),
                        btc=float(btc),
                    )
                )
            except Exception:
                continue
    rows.sort(key=lambda r: r.ts)
    return rows


def group_markets(rows: list[Row]) -> dict[int, list[Row]]:
    by_market: dict[int, list[Row]] = {}
    for r in rows:
        m = (r.ts // 300) * 300
        by_market.setdefault(m, []).append(r)
    for m in by_market:
        by_market[m].sort(key=lambda r: r.ts)
    return by_market


def std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(var)


def flip_rate(values: list[float]) -> float:
    if len(values) <= 2:
        return 0.0
    prev = 0
    flips = 0
    count = 0
    for v in values:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign == 0:
            continue
        count += 1
        if prev != 0 and sign != prev:
            flips += 1
        prev = sign
    if count <= 1:
        return 0.0
    return flips / (count - 1)


def build_dataset(
    market_rows: dict[int, list[Row]],
    min_time_remaining: int,
    max_time_remaining: int,
    price_min: float,
    price_max: float,
) -> tuple[list[list[float]], list[int], list[int], list[float]]:
    x: list[list[float]] = []
    y: list[int] = []
    market_ids: list[int] = []
    leader_prices: list[float] = []

    for market_ts in sorted(market_rows):
        rows = market_rows[market_ts]
        if len(rows) < 60:
            continue

        open_btc = rows[0].btc
        close_btc = rows[-1].btc
        winner = "up" if close_btc >= open_btc else "down"

        for i, r in enumerate(rows):
            time_remaining = market_ts + 300 - r.ts
            if time_remaining < min_time_remaining or time_remaining > max_time_remaining:
                continue

            leader = "up" if r.btc >= open_btc else "down"
            leader_price = r.up if leader == "up" else r.down
            if leader_price < price_min or leader_price > price_max:
                continue

            w_start = max(0, i - 30)
            prev = rows[w_start : i + 1]
            rets: list[float] = []
            deltas: list[float] = []
            for j in range(1, len(prev)):
                p0 = prev[j - 1].btc
                p1 = prev[j].btc
                if p0 > 0:
                    rets.append((p1 - p0) / p0)
                deltas.append(prev[j].btc - open_btc)

            vol30 = std(rets)
            flips30 = flip_rate(deltas)
            delta_abs_pct = abs(r.btc - open_btc) / open_btc if open_btc > 0 else 0.0
            drift10 = 0.0
            if i >= 10 and open_btc > 0:
                drift10 = (r.btc - rows[i - 10].btc) / open_btc

            x.append(
                [
                    leader_price,
                    SETTLE_PRICE - leader_price,
                    delta_abs_pct,
                    vol30,
                    flips30,
                    time_remaining / 300.0,
                    drift10,
                    1.0 if leader == "up" else -1.0,
                ]
            )
            y.append(1 if leader == winner else 0)
            market_ids.append(market_ts)
            leader_prices.append(leader_price)

    return x, y, market_ids, leader_prices


def split_by_market(
    x: list[list[float]], y: list[int], market_ids: list[int], train_frac: float = 0.7
) -> tuple[list[int], list[int]]:
    uniq = sorted(set(market_ids))
    if len(uniq) <= 1:
        idx = list(range(len(x)))
        return idx, idx
    cut = max(1, int(len(uniq) * train_frac))
    train_markets = set(uniq[:cut])
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, m in enumerate(market_ids):
        if m in train_markets:
            train_idx.append(i)
        else:
            test_idx.append(i)
    if not test_idx:
        test_idx = train_idx[:]
    return train_idx, test_idx


def standardize_fit(x: list[list[float]], idx: list[int]) -> tuple[list[float], list[float]]:
    d = len(x[0])
    means = [0.0] * d
    stds = [1.0] * d
    n = len(idx)
    for j in range(d):
        vals = [x[i][j] for i in idx]
        m = sum(vals) / n
        s = std(vals)
        means[j] = m
        stds[j] = s if s > 1e-12 else 1.0
    return means, stds


def apply_standardize(x: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in x:
        out.append([(row[j] - means[j]) / stds[j] for j in range(len(row))])
    return out


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic(
    x: list[list[float]],
    y: list[int],
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[list[float], float]:
    d = len(x[0])
    w = [0.0] * d
    b = 0.0
    n = len(train_idx)
    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for i in train_idx:
            z = b
            xi = x[i]
            for j in range(d):
                z += w[j] * xi[j]
            p = sigmoid(z)
            err = p - y[i]
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(d):
            grad_w[j] = grad_w[j] / n + l2 * w[j]
            w[j] -= lr * grad_w[j]
        b -= lr * (grad_b / n)
    return w, b


def metrics(
    x: list[list[float]],
    y: list[int],
    idx: list[int],
    w: list[float],
    b: float,
    prices: list[float],
) -> dict[str, float]:
    n = len(idx)
    if n == 0:
        return {
            "samples": 0.0,
            "accuracy": 0.0,
            "avg_logloss": 0.0,
            "buy_rate": 0.0,
            "buy_win_rate": 0.0,
            "avg_ev_when_buy": 0.0,
        }

    correct = 0
    ll = 0.0
    buys = 0
    buy_wins = 0
    ev_sum = 0.0
    for i in idx:
        z = b + sum(w[j] * x[i][j] for j in range(len(w)))
        p = sigmoid(z)
        pred = 1 if p >= 0.5 else 0
        if pred == y[i]:
            correct += 1
        p_clip = min(1.0 - 1e-9, max(1e-9, p))
        ll += -(y[i] * math.log(p_clip) + (1 - y[i]) * math.log(1 - p_clip))

        price = prices[i]
        win_ret = (SETTLE_PRICE / price) - 1.0
        ev = p * win_ret + (1.0 - p) * (-1.0)
        p_break_even = price / SETTLE_PRICE
        if p > p_break_even:
            buys += 1
            ev_sum += ev
            if y[i] == 1:
                buy_wins += 1

    return {
        "samples": float(n),
        "accuracy": correct / n,
        "avg_logloss": ll / n,
        "buy_rate": buys / n,
        "buy_win_rate": (buy_wins / buys) if buys > 0 else 0.0,
        "avg_ev_when_buy": (ev_sum / buys) if buys > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.jsonl)
    if not rows:
        raise SystemExit("no rows found")

    markets = group_markets(rows)
    x_raw, y, market_ids, leader_prices = build_dataset(
        markets,
        min_time_remaining=args.fit_min_time_remaining,
        max_time_remaining=args.fit_max_time_remaining,
        price_min=args.fit_price_min,
        price_max=args.fit_price_max,
    )
    if not x_raw:
        raise SystemExit("no candidate rows after filters")

    train_idx, test_idx = split_by_market(x_raw, y, market_ids)
    means, stds = standardize_fit(x_raw, train_idx)
    x = apply_standardize(x_raw, means, stds)

    w, b = fit_logistic(
        x,
        y,
        train_idx=train_idx,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )
    train_m = metrics(x, y, train_idx, w, b, leader_prices)
    test_m = metrics(x, y, test_idx, w, b, leader_prices)

    feature_names = [
        "leader_price",
        "edge_to_settle",
        "abs_open_gap_pct",
        "vol_30s",
        "flip_rate_30s",
        "time_remaining_frac",
        "drift_10s_pct_of_open",
        "leader_is_up",
    ]
    model = {
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "weights": w,
        "bias": b,
        "constraints": {
            "buy_price_min": args.buy_price_min,
            "buy_price_max": args.buy_price_max,
            "buy_min_time_remaining": args.buy_min_time_remaining,
            "buy_max_time_remaining": args.buy_max_time_remaining,
            "settle_price": SETTLE_PRICE,
        },
        "fit_window": {
            "fit_price_min": args.fit_price_min,
            "fit_price_max": args.fit_price_max,
            "fit_min_time_remaining": args.fit_min_time_remaining,
            "fit_max_time_remaining": args.fit_max_time_remaining,
        },
        "train_metrics": train_m,
        "test_metrics": test_m,
    }

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"rows_total: {len(rows)}")
    print(f"markets_total: {len(markets)}")
    print(f"candidates: {len(x_raw)}")
    print(f"train_samples: {int(train_m['samples'])}")
    print(f"test_samples: {int(test_m['samples'])}")
    print("train_metrics:", json.dumps(train_m, indent=2))
    print("test_metrics:", json.dumps(test_m, indent=2))
    print("model_file:", str(out_path))


if __name__ == "__main__":
    main()
