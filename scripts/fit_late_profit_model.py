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
class Tick:
    market: str
    market_ts: int
    ts: int
    time_remaining: int
    up_bid: float
    up_ask: float
    up_mid: float
    down_bid: float
    down_ask: float
    down_mid: float
    btc: float | None


@dataclass
class Sample:
    market: str
    ts: int
    time_remaining: int
    token: str
    x: list[float]
    price: float
    y_win: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit late-window profit model for buying directional contracts and compare "
            "training data sources."
        )
    )
    parser.add_argument(
        "--entry-min-time-remaining",
        type=int,
        default=8,
        help="Only consider entries with >= this many seconds remaining.",
    )
    parser.add_argument(
        "--entry-max-time-remaining",
        type=int,
        default=70,
        help="Only consider entries with <= this many seconds remaining.",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.55,
        help="Ignore entries cheaper than this.",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=0.985,
        help="Ignore entries more expensive than this.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Chronological market split fraction for train.",
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--output-prefix",
        default="backtester/late_profit_model",
    )
    return parser.parse_args()


def parse_ts(v: object) -> int | None:
    if isinstance(v, (int, float)):
        return int(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    try:
        return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())
    except Exception:
        pass
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def market_ts_from_slug(market: str) -> int:
    try:
        return int(market.rsplit("-", 1)[-1])
    except Exception:
        return 0


def load_ticks(paths: list[str]) -> dict[str, list[Tick]]:
    by_market: dict[str, list[Tick]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue

                # Orderbook schema.
                if d.get("market") and d.get("up_bids") and d.get("up_asks"):
                    try:
                        market = str(d["market"])
                        up_bid = float(d["up_bids"][0][0])
                        up_ask = float(d["up_asks"][0][0])
                        down_bid = float(d["down_bids"][0][0])
                        down_ask = float(d["down_asks"][0][0])
                        ts = parse_ts(d.get("timestamp"))
                        if ts is None and d.get("timestamp_ms") is not None:
                            ts = int(float(d["timestamp_ms"]) / 1000.0)
                        if ts is None:
                            continue
                        tick = Tick(
                            market=market,
                            market_ts=market_ts_from_slug(market),
                            ts=ts,
                            time_remaining=int(d["time_remaining"]),
                            up_bid=up_bid,
                            up_ask=up_ask,
                            up_mid=(up_bid + up_ask) / 2.0,
                            down_bid=down_bid,
                            down_ask=down_ask,
                            down_mid=(down_bid + down_ask) / 2.0,
                            btc=float(d["btc"]) if d.get("btc") is not None else None,
                        )
                        by_market.setdefault(market, []).append(tick)
                    except Exception:
                        continue
                    continue

                # 1s schema.
                if d.get("up") is not None:
                    try:
                        ts = parse_ts(d.get("timestamp"))
                        if ts is None and d.get("timestamp_ms") is not None:
                            ts = int(float(d["timestamp_ms"]) / 1000.0)
                        if ts is None:
                            continue
                        up_mid = float(d["up"])
                        down_mid = (
                            float(d["down"])
                            if d.get("down") is not None
                            else (1.0 - up_mid)
                        )
                        market_ts = (ts // 300) * 300
                        market = f"btc-updown-5m-{market_ts}"
                        tick = Tick(
                            market=market,
                            market_ts=market_ts,
                            ts=ts,
                            time_remaining=max(0, market_ts + 300 - ts),
                            up_bid=up_mid,
                            up_ask=up_mid,
                            up_mid=up_mid,
                            down_bid=down_mid,
                            down_ask=down_mid,
                            down_mid=down_mid,
                            btc=float(d["btc"]) if d.get("btc") is not None else None,
                        )
                        by_market.setdefault(market, []).append(tick)
                    except Exception:
                        continue

    # Deduplicate to one point per second per market to reduce overweighting
    # from high-frequency snapshots.
    deduped: dict[str, list[Tick]] = {}
    for market, rows in by_market.items():
        rows.sort(key=lambda t: t.ts)
        uniq: dict[int, Tick] = {}
        for t in rows:
            uniq[t.ts] = t
        out = [uniq[k] for k in sorted(uniq.keys())]
        deduped[market] = out
    return deduped


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    sx = n * (n - 1) / 2.0
    sy = sum(values)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * values[i] for i in range(n))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / den


def flip_rate(values: list[float]) -> float:
    prev = 0
    flips = 0
    count = 0
    for v in values:
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s == 0:
            continue
        count += 1
        if prev != 0 and s != prev:
            flips += 1
        prev = s
    if count <= 1:
        return 0.0
    return flips / (count - 1)


def build_samples(
    markets: dict[str, list[Tick]],
    entry_min_tr: int,
    entry_max_tr: int,
    min_price: float,
    max_price: float,
) -> tuple[list[Sample], list[str]]:
    feature_names = [
        "price",
        "edge_to_settle",
        "price_gap_vs_other",
        "token_spread",
        "other_spread",
        "oriented_gap",
        "abs_gap",
        "sum_asks",
        "time_remaining_frac",
        "token_mom_short_long",
        "oriented_gap_slope",
        "token_ret_vol",
        "gap_flip_rate",
        "btc_available",
        "btc_drift_10s",
        "btc_drift_30s",
        "btc_vol_30s",
        "btc_mom_short_long",
        "btc_token_corr_30s",
    ]

    samples: list[Sample] = []
    for market in sorted(markets):
        rows = markets[market]
        if len(rows) < 70:
            continue
        if min(t.time_remaining for t in rows) > 3:
            continue
        winner = "up" if rows[-1].up_mid >= rows[-1].down_mid else "down"

        up_hist: list[float] = []
        down_hist: list[float] = []
        gap_hist: list[float] = []
        btc_hist: list[float] = []
        for t in rows:
            up_hist.append(t.up_mid)
            down_hist.append(t.down_mid)
            gap_hist.append(t.up_mid - t.down_mid)
            btc_hist.append(t.btc if t.btc is not None else float("nan"))
            i = len(up_hist) - 1
            if i < 35:
                continue
            tr = t.time_remaining
            if tr < entry_min_tr or tr > entry_max_tr:
                continue

            for token in ("up", "down"):
                sign_token = 1.0 if token == "up" else -1.0
                price = t.up_ask if token == "up" else t.down_ask
                other_price = t.down_ask if token == "up" else t.up_ask
                if price < min_price or price > max_price:
                    continue

                token_spread = (
                    (t.up_ask - t.up_bid)
                    if token == "up"
                    else (t.down_ask - t.down_bid)
                )
                other_spread = (
                    (t.down_ask - t.down_bid)
                    if token == "up"
                    else (t.up_ask - t.up_bid)
                )
                oriented_gap = sign_token * gap_hist[-1]
                abs_gap = abs(gap_hist[-1])
                sum_asks = t.up_ask + t.down_ask
                time_frac = tr / 300.0

                token_hist = up_hist if token == "up" else down_hist
                short = token_hist[-8:]
                long = token_hist[-30:]
                token_mom = (sum(short) / len(short)) - (sum(long) / len(long))

                gap_window = [sign_token * g for g in gap_hist[-30:]]
                gap_sl = slope(gap_window)

                ret_w: list[float] = []
                tok_w = token_hist[-30:]
                for j in range(1, len(tok_w)):
                    if tok_w[j - 1] > 1e-12:
                        ret_w.append((tok_w[j] - tok_w[j - 1]) / tok_w[j - 1])
                tok_vol = std(ret_w)
                gap_chop = flip_rate(gap_window)

                # BTC-shift features; default to neutral when unavailable.
                btc_available = 0.0
                btc_drift_10 = 0.0
                btc_drift_30 = 0.0
                btc_vol_30 = 0.0
                btc_mom = 0.0
                btc_tok_corr = 0.0
                btc_w = btc_hist[-30:]
                if all(not math.isnan(v) for v in btc_w):
                    btc_available = 1.0
                    btc_ret_w: list[float] = []
                    for j in range(1, len(btc_w)):
                        if abs(btc_w[j - 1]) > 1e-12:
                            btc_ret_w.append((btc_w[j] - btc_w[j - 1]) / btc_w[j - 1])
                    if len(btc_w) >= 11 and abs(btc_w[-11]) > 1e-12:
                        btc_drift_10 = (btc_w[-1] - btc_w[-11]) / btc_w[-11]
                    if len(btc_w) >= 30 and abs(btc_w[0]) > 1e-12:
                        btc_drift_30 = (btc_w[-1] - btc_w[0]) / btc_w[0]
                    btc_vol_30 = std(btc_ret_w)
                    if len(btc_w) >= 30:
                        btc_short = btc_w[-8:]
                        btc_long = btc_w[-30:]
                        btc_mom = (sum(btc_short) / len(btc_short)) - (
                            sum(btc_long) / len(btc_long)
                        )
                    # Correlate token returns with BTC returns over the same window.
                    if len(btc_ret_w) >= 5 and len(ret_w) >= 5:
                        n = min(len(btc_ret_w), len(ret_w))
                        a = ret_w[-n:]
                        b = btc_ret_w[-n:]
                        ma = sum(a) / n
                        mb = sum(b) / n
                        va = sum((x - ma) ** 2 for x in a)
                        vb = sum((x - mb) ** 2 for x in b)
                        if va > 1e-12 and vb > 1e-12:
                            cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
                            btc_tok_corr = cov / math.sqrt(va * vb)

                x = [
                    price,
                    SETTLE_PRICE - price,
                    price - other_price,
                    token_spread,
                    other_spread,
                    oriented_gap,
                    abs_gap,
                    sum_asks,
                    time_frac,
                    token_mom,
                    gap_sl,
                    tok_vol,
                    gap_chop,
                    btc_available,
                    btc_drift_10,
                    btc_drift_30,
                    btc_vol_30,
                    btc_mom,
                    btc_tok_corr,
                ]
                y = 1 if token == winner else 0
                samples.append(
                    Sample(
                        market=market,
                        ts=t.ts,
                        time_remaining=tr,
                        token=token,
                        x=x,
                        price=price,
                        y_win=y,
                    )
                )
    return samples, feature_names


def split_by_market(
    samples: list[Sample], train_frac: float
) -> tuple[list[int], list[int]]:
    uniq = sorted(set(s.market for s in samples))
    if len(uniq) <= 1:
        idx = list(range(len(samples)))
        return idx, idx
    cut = max(1, int(len(uniq) * train_frac))
    train_markets = set(uniq[:cut])
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, s in enumerate(samples):
        if s.market in train_markets:
            train_idx.append(i)
        else:
            test_idx.append(i)
    if not test_idx:
        test_idx = train_idx[:]
    return train_idx, test_idx


def fit_standardize(
    x: list[list[float]], idx: list[int]
) -> tuple[list[float], list[float]]:
    d = len(x[0])
    means = [0.0] * d
    stds = [1.0] * d
    for j in range(d):
        vals = [x[i][j] for i in idx]
        m = sum(vals) / len(vals)
        s = std(vals)
        means[j] = m
        stds[j] = s if s > 1e-12 else 1.0
    return means, stds


def apply_standardize(
    x: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in x:
        out.append([(row[j] - means[j]) / stds[j] for j in range(len(row))])
    return out


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
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
        gw = [0.0] * d
        gb = 0.0
        for i in train_idx:
            xi = x[i]
            yi = y[i]
            z = b + sum(w[j] * xi[j] for j in range(d))
            p = sigmoid(z)
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            gw[j] = gw[j] / n + l2 * w[j]
            w[j] -= lr * gw[j]
        b -= lr * (gb / n)
    return w, b


def predict(x: list[list[float]], w: list[float], b: float) -> list[float]:
    out: list[float] = []
    for row in x:
        z = b + sum(w[j] * row[j] for j in range(len(w)))
        out.append(sigmoid(z))
    return out


def policy_eval(
    samples: list[Sample],
    probs: list[float],
    idx: list[int],
    ev_threshold: float,
) -> dict[str, float]:
    by_market: dict[str, list[int]] = {}
    for i in idx:
        by_market.setdefault(samples[i].market, []).append(i)

    returns: list[float] = []
    trades = 0
    for market, mids in by_market.items():
        _ = market
        best_i = None
        best_ev = -1e9
        for i in mids:
            p = probs[i]
            px = samples[i].price
            ev = p * ((SETTLE_PRICE / px) - 1.0) + (1.0 - p) * (-1.0)
            if ev > best_ev:
                best_ev = ev
                best_i = i
        if best_i is None or best_ev < ev_threshold:
            continue
        trades += 1
        s = samples[best_i]
        realized = ((SETTLE_PRICE / s.price) - 1.0) if s.y_win == 1 else -1.0
        returns.append(realized)

    markets = len(by_market)
    if not returns:
        return {
            "markets": float(markets),
            "trades": 0.0,
            "trade_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_pnl_per_market": 0.0,
            "win_rate": 0.0,
            "sharpe_like": 0.0,
        }
    avg = sum(returns) / len(returns)
    var = sum((x - avg) ** 2 for x in returns) / len(returns)
    sd = math.sqrt(var)
    return {
        "markets": float(markets),
        "trades": float(trades),
        "trade_rate": trades / max(1, markets),
        "total_pnl": sum(returns),
        "avg_pnl_per_trade": avg,
        "avg_pnl_per_market": sum(returns) / max(1, markets),
        "win_rate": sum(1 for x in returns if x > 0) / len(returns),
        "sharpe_like": (avg / sd) if sd > 1e-12 else 0.0,
    }


def train_and_score(
    name: str,
    paths: list[str],
    args: argparse.Namespace,
) -> dict:
    markets = load_ticks(paths)
    samples, feature_names = build_samples(
        markets=markets,
        entry_min_tr=args.entry_min_time_remaining,
        entry_max_tr=args.entry_max_time_remaining,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    if len(samples) < 100:
        return {
            "dataset": name,
            "paths": paths,
            "error": f"not enough samples ({len(samples)})",
        }

    x_raw = [s.x for s in samples]
    y = [s.y_win for s in samples]
    train_idx, test_idx = split_by_market(samples, args.train_frac)
    means, stds = fit_standardize(x_raw, train_idx)
    x = apply_standardize(x_raw, means, stds)
    w, b = fit_logistic(
        x=x,
        y=y,
        train_idx=train_idx,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )
    probs = predict(x, w, b)

    thresholds = [i / 1000.0 for i in range(-20, 121, 5)]  # -0.020 ... 0.120
    train_rows: list[tuple[float, dict]] = []
    for thr in thresholds:
        st = policy_eval(samples, probs, train_idx, thr)
        train_rows.append((thr, st))
    best_thr, best_train = max(train_rows, key=lambda x: x[1]["total_pnl"])
    test_stats = policy_eval(samples, probs, test_idx, best_thr)

    return {
        "dataset": name,
        "paths": paths,
        "samples": len(samples),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "weights": w,
        "bias": b,
        "selected_ev_threshold": best_thr,
        "train_policy": best_train,
        "test_policy": test_stats,
    }


def main() -> None:
    args = parse_args()
    scenarios = {
        "one_s": ["data/btc_1s_pricesv2.jsonl", "data/btc_1s_pricesv3.jsonl"],
        "five_m": ["btc_5m_prices.jsonl"],
        "combo": [
            "data/btc_1s_pricesv2.jsonl",
            "data/btc_1s_pricesv3.jsonl",
            "btc_5m_prices.jsonl",
        ],
    }

    results = []
    for name, paths in scenarios.items():
        result = train_and_score(name, paths, args)
        results.append(result)

    valid = [r for r in results if r.get("error") is None]
    if not valid:
        raise SystemExit("all scenarios failed")
    best = max(valid, key=lambda r: r["test_policy"]["avg_pnl_per_market"])

    out = {
        "config": {
            "entry_min_time_remaining": args.entry_min_time_remaining,
            "entry_max_time_remaining": args.entry_max_time_remaining,
            "min_price": args.min_price,
            "max_price": args.max_price,
            "train_frac": args.train_frac,
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "settle_price": SETTLE_PRICE,
        },
        "scenarios": results,
        "best_dataset_by_test_avg_pnl_per_market": best["dataset"],
    }

    out_path = Path(f"{args.output_prefix}_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Also write a standalone best-model file for deployment.
    best_model = {
        "dataset": best["dataset"],
        "paths": best["paths"],
        "feature_names": best["feature_names"],
        "means": best["means"],
        "stds": best["stds"],
        "weights": best["weights"],
        "bias": best["bias"],
        "selected_ev_threshold": best["selected_ev_threshold"],
        "config": out["config"],
        "test_policy": best["test_policy"],
    }
    best_model_path = Path(f"{args.output_prefix}_best.json")
    with open(best_model_path, "w", encoding="utf-8") as f:
        json.dump(best_model, f, indent=2)

    print(f"compare_file: {out_path}")
    for r in results:
        if r.get("error"):
            print(f"{r['dataset']}: error={r['error']}")
            continue
        tp = r["test_policy"]
        print(
            f"{r['dataset']}: test_total={tp['total_pnl']:.4f} "
            f"test_avg_market={tp['avg_pnl_per_market']:.4f} trades={int(tp['trades'])} "
            f"win_rate={tp['win_rate']:.2%} thr={r['selected_ev_threshold']:.3f}"
        )
    print(f"best_dataset: {best['dataset']}")
    print(f"best_model_file: {best_model_path}")


if __name__ == "__main__":
    main()
