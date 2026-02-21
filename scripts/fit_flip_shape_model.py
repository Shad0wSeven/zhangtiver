#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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


@dataclass
class Sample:
    market_ts: int
    checkpoint_s: int
    y_flip: int
    x: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an ensemble flip-risk model using time-series shape + variance "
            "features from 1s BTC market logs."
        )
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=["data/btc_1s_pricesv2.jsonl", "data/btc_1s_pricesv3.jsonl"],
    )
    parser.add_argument("--checkpoints", default="40,30,20")
    parser.add_argument("--lookback-s", type=int, default=45)
    parser.add_argument("--min-window-rows", type=int, default=15)
    parser.add_argument("--min-market-rows", type=int, default=80)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", type=int, default=9)
    parser.add_argument("--feature-frac", type=float, default=0.7)
    parser.add_argument("--bag-frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.09)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--output-model",
        default="backtester/flip_shape_ensemble_model.json",
    )
    return parser.parse_args()


def parse_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


def load_rows(paths: list[str]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("up") is None or d.get("btc") is None:
                        continue
                    up = float(d["up"])
                    down = float(d["down"]) if d.get("down") is not None else (1.0 - up)
                    rows.append(
                        Row(
                            ts=parse_ts(str(d["timestamp"])),
                            up=up,
                            down=down,
                            btc=float(d["btc"]),
                        )
                    )
                except Exception:
                    continue
    rows.sort(key=lambda r: r.ts)
    return rows


def group_markets(rows: list[Row]) -> dict[int, list[Row]]:
    by_market: dict[int, list[Row]] = {}
    for r in rows:
        market_ts = (r.ts // 300) * 300
        by_market.setdefault(market_ts, []).append(r)
    for market_ts in by_market:
        by_market[market_ts].sort(key=lambda r: r.ts)
    return by_market


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
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


def realized_var(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return sum((values[i] - values[i - 1]) ** 2 for i in range(1, len(values)))


def flip_rate(values: list[float]) -> float:
    prev = 0
    flips = 0
    count = 0
    for v in values:
        s = sign(v)
        if s == 0:
            continue
        count += 1
        if prev != 0 and s != prev:
            flips += 1
        prev = s
    if count <= 1:
        return 0.0
    return flips / (count - 1)


def corr(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n <= 1:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    ma = mean(aa)
    mb = mean(bb)
    va = sum((x - ma) ** 2 for x in aa)
    vb = sum((x - mb) ** 2 for x in bb)
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    cov = sum((aa[i] - ma) * (bb[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def nearest_checkpoint_row(
    rows: list[Row], market_ts: int, checkpoint_s: int
) -> Row | None:
    best = None
    best_dist = 1_000_000_000
    for r in rows:
        tr = market_ts + 300 - r.ts
        dist = abs(tr - checkpoint_s)
        if dist < best_dist:
            best_dist = dist
            best = r
    return best


def extract_feature_vector(
    market_ts: int,
    checkpoint_s: int,
    checkpoint_row: Row,
    window_rows: list[Row],
) -> list[float]:
    leader_up = checkpoint_row.up >= checkpoint_row.down
    leader_series = [r.up if leader_up else r.down for r in window_rows]
    lagger_series = [r.down if leader_up else r.up for r in window_rows]
    imbalance = [r.up - r.down for r in window_rows]
    btc = [r.btc for r in window_rows]

    leader_last = leader_series[-1]
    lagger_last = lagger_series[-1]
    gap_last = abs(leader_last - lagger_last)

    mid = len(leader_series) // 2
    leader_head = leader_series[:mid] if mid > 0 else leader_series
    leader_tail = leader_series[mid:] if mid > 0 else leader_series
    imb_head = imbalance[:mid] if mid > 0 else imbalance
    imb_tail = imbalance[mid:] if mid > 0 else imbalance

    leader_ret = [
        (leader_series[i] - leader_series[i - 1]) / leader_series[i - 1]
        for i in range(1, len(leader_series))
        if abs(leader_series[i - 1]) > 1e-12
    ]
    btc_ret = [
        (btc[i] - btc[i - 1]) / btc[i - 1]
        for i in range(1, len(btc))
        if abs(btc[i - 1]) > 1e-12
    ]

    leader_short = leader_series[-10:] if len(leader_series) >= 10 else leader_series
    leader_long = leader_series[-25:] if len(leader_series) >= 25 else leader_series
    imb_short = imbalance[-10:] if len(imbalance) >= 10 else imbalance
    imb_long = imbalance[-25:] if len(imbalance) >= 25 else imbalance
    btc_short = btc_ret[-10:] if len(btc_ret) >= 10 else btc_ret
    btc_long = btc_ret[-25:] if len(btc_ret) >= 25 else btc_ret

    leader_max = max(leader_series)
    leader_min = min(leader_series)
    leader_drawdown = (leader_max - leader_last) / max(1e-9, leader_max)
    leader_rebound = (leader_last - leader_min) / max(1e-9, leader_max - leader_min)

    imb_abs = [abs(x) for x in imbalance]
    choppiness = flip_rate(imbalance)
    time_frac = checkpoint_s / 300.0

    # Curvature proxy from local slopes.
    slope_head = slope(imb_head)
    slope_tail = slope(imb_tail)
    curvature = slope_tail - slope_head

    # Compare short vs long momentum to capture inflection.
    leader_mom_short = slope(leader_short)
    leader_mom_long = slope(leader_long)
    imb_mom_short = slope(imb_short)
    imb_mom_long = slope(imb_long)

    # If leader is UP, positive imbalance supports the move. If leader is DOWN,
    # invert imbalance so larger value still means leader support.
    oriented_imb = imbalance if leader_up else [-x for x in imbalance]
    oriented_tail = oriented_imb[-10:] if len(oriented_imb) >= 10 else oriented_imb

    return [
        leader_last,
        lagger_last,
        gap_last,
        SETTLE_PRICE - leader_last,
        leader_last - 0.5,
        time_frac,
        1.0 if leader_up else -1.0,
        mean(leader_series),
        std(leader_series),
        leader_mom_short,
        leader_mom_long,
        leader_mom_short - leader_mom_long,
        leader_drawdown,
        leader_rebound,
        realized_var(leader_series),
        mean(imbalance),
        std(imbalance),
        mean(imb_abs),
        slope(imbalance),
        curvature,
        choppiness,
        mean(oriented_tail),
        std(oriented_tail),
        imb_mom_short,
        imb_mom_long,
        imb_mom_short - imb_mom_long,
        realized_var(imbalance),
        mean(btc_ret),
        std(btc_ret),
        mean(btc_short),
        std(btc_short),
        std(btc_long),
        realized_var(btc_ret),
        corr(imbalance[1:], btc_ret),
        corr(leader_ret, btc_ret),
        (window_rows[-1].btc - window_rows[0].btc) / max(1e-9, window_rows[0].btc),
        abs(window_rows[-1].btc - window_rows[0].btc) / max(1e-9, window_rows[0].btc),
    ]


def build_dataset(
    market_rows: dict[int, list[Row]],
    checkpoints: list[int],
    lookback_s: int,
    min_window_rows: int,
    min_market_rows: int,
) -> tuple[list[Sample], list[str]]:
    feature_names = [
        "leader_price",
        "lagger_price",
        "price_gap",
        "edge_to_settle",
        "leader_premium_to_0_5",
        "time_remaining_frac",
        "leader_is_up",
        "leader_mean",
        "leader_std",
        "leader_slope_short",
        "leader_slope_long",
        "leader_slope_short_minus_long",
        "leader_drawdown_from_max",
        "leader_rebound_from_min",
        "leader_realized_var",
        "imbalance_mean",
        "imbalance_std",
        "imbalance_abs_mean",
        "imbalance_slope",
        "imbalance_curvature",
        "imbalance_flip_rate",
        "oriented_imbalance_tail_mean",
        "oriented_imbalance_tail_std",
        "imbalance_slope_short",
        "imbalance_slope_long",
        "imbalance_slope_short_minus_long",
        "imbalance_realized_var",
        "btc_ret_mean",
        "btc_ret_std",
        "btc_ret_mean_short",
        "btc_ret_std_short",
        "btc_ret_std_long",
        "btc_ret_realized_var",
        "corr_imbalance_btc_ret",
        "corr_leader_ret_btc_ret",
        "btc_drift_window",
        "btc_abs_move_window",
    ]

    samples: list[Sample] = []
    for market_ts in sorted(market_rows):
        rows = market_rows[market_ts]
        if len(rows) < min_market_rows:
            continue
        final = sign(rows[-1].up - rows[-1].down)
        if final == 0:
            continue

        for checkpoint_s in checkpoints:
            cp = nearest_checkpoint_row(rows, market_ts, checkpoint_s)
            if cp is None:
                continue
            cp_side = sign(cp.up - cp.down)
            if cp_side == 0:
                continue
            y_flip = 1 if cp_side != final else 0

            window_rows = []
            lo = checkpoint_s
            hi = checkpoint_s + lookback_s
            for r in rows:
                tr = market_ts + 300 - r.ts
                if lo <= tr <= hi:
                    window_rows.append(r)
            if len(window_rows) < min_window_rows:
                continue
            window_rows.sort(key=lambda r: r.ts)

            x = extract_feature_vector(
                market_ts=market_ts,
                checkpoint_s=checkpoint_s,
                checkpoint_row=cp,
                window_rows=window_rows,
            )
            samples.append(
                Sample(
                    market_ts=market_ts,
                    checkpoint_s=checkpoint_s,
                    y_flip=y_flip,
                    x=x,
                )
            )

    return samples, feature_names


def split_by_market(
    samples: list[Sample], train_frac: float
) -> tuple[list[int], list[int]]:
    markets = sorted(set(s.market_ts for s in samples))
    if len(markets) <= 1:
        idx = list(range(len(samples)))
        return idx, idx
    cut = max(1, int(len(markets) * train_frac))
    train_markets = set(markets[:cut])
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, s in enumerate(samples):
        if s.market_ts in train_markets:
            train_idx.append(i)
        else:
            test_idx.append(i)
    if not test_idx:
        test_idx = train_idx[:]
    return train_idx, test_idx


def fit_standardization(
    x: list[list[float]], train_idx: list[int]
) -> tuple[list[float], list[float]]:
    d = len(x[0])
    means = [0.0] * d
    stds = [1.0] * d
    for j in range(d):
        vals = [x[i][j] for i in train_idx]
        m = mean(vals)
        s = std(vals)
        means[j] = m
        stds[j] = s if s > 1e-12 else 1.0
    return means, stds


def apply_standardization(
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


def fit_logistic_weighted(
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
    pos = sum(y[i] for i in train_idx)
    neg = len(train_idx) - pos
    if pos <= 0 or neg <= 0:
        return w, b
    pos_w = neg / max(1, pos)
    neg_w = 1.0

    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        norm = 0.0
        for i in train_idx:
            xi = x[i]
            yi = y[i]
            z = b + sum(w[j] * xi[j] for j in range(d))
            p = sigmoid(z)
            wt = pos_w if yi == 1 else neg_w
            err = (p - yi) * wt
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
            norm += wt
        if norm <= 0:
            break
        for j in range(d):
            grad_w[j] = grad_w[j] / norm + l2 * w[j]
            w[j] -= lr * grad_w[j]
        b -= lr * (grad_b / norm)
    return w, b


def train_ensemble(
    x: list[list[float]],
    y: list[int],
    train_idx: list[int],
    models: int,
    feature_frac: float,
    bag_frac: float,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    d = len(x[0])
    ensemble: list[dict] = []
    k_feat = max(4, int(d * feature_frac))
    k_bag = max(30, int(len(train_idx) * bag_frac))

    for _ in range(models):
        feat_idx = sorted(rng.sample(range(d), k=min(k_feat, d)))
        bag_idx = [train_idx[rng.randrange(len(train_idx))] for _ in range(k_bag)]
        x_sub = [[x[i][j] for j in feat_idx] for i in range(len(x))]
        w, b = fit_logistic_weighted(
            x=x_sub,
            y=y,
            train_idx=bag_idx,
            epochs=epochs,
            lr=lr,
            l2=l2,
        )
        ensemble.append(
            {
                "feature_idx": feat_idx,
                "weights": w,
                "bias": b,
            }
        )
    return ensemble


def predict_proba(ensemble: list[dict], x: list[list[float]]) -> list[float]:
    if not ensemble:
        return [0.0] * len(x)
    out = [0.0] * len(x)
    for model in ensemble:
        feat_idx = model["feature_idx"]
        w = model["weights"]
        b = model["bias"]
        for i, row in enumerate(x):
            z = b + sum(w[j] * row[feat_idx[j]] for j in range(len(feat_idx)))
            out[i] += sigmoid(z)
    m = float(len(ensemble))
    return [v / m for v in out]


def auc(y: list[int], p: list[float], idx: list[int]) -> float:
    pos = [p[i] for i in idx if y[i] == 1]
    neg = [p[i] for i in idx if y[i] == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def brier(y: list[int], p: list[float], idx: list[int]) -> float:
    if not idx:
        return 0.0
    return sum((p[i] - y[i]) ** 2 for i in idx) / len(idx)


def logloss(y: list[int], p: list[float], idx: list[int]) -> float:
    if not idx:
        return 0.0
    s = 0.0
    for i in idx:
        pi = min(1.0 - 1e-9, max(1e-9, p[i]))
        yi = y[i]
        s += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
    return s / len(idx)


def threshold_metrics(
    y: list[int],
    p: list[float],
    idx: list[int],
    thresholds: list[float],
) -> list[dict[str, float]]:
    pos_total = sum(y[i] for i in idx)
    out: list[dict[str, float]] = []
    for thr in thresholds:
        flagged = [i for i in idx if p[i] >= thr]
        if not flagged:
            continue
        tp = sum(y[i] for i in flagged)
        fp = len(flagged) - tp
        fn = pos_total - tp
        precision = tp / len(flagged)
        recall = tp / max(1, pos_total)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out.append(
            {
                "threshold": thr,
                "flags": float(len(flagged)),
                "flag_rate": len(flagged) / max(1, len(idx)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]

    rows = load_rows(args.jsonl)
    if not rows:
        raise SystemExit("no rows found")

    markets = group_markets(rows)
    samples, feature_names = build_dataset(
        market_rows=markets,
        checkpoints=checkpoints,
        lookback_s=args.lookback_s,
        min_window_rows=args.min_window_rows,
        min_market_rows=args.min_market_rows,
    )
    if not samples:
        raise SystemExit(
            "no samples built; try lowering min-window or min-market filters"
        )

    x_raw = [s.x for s in samples]
    y = [s.y_flip for s in samples]
    train_idx, test_idx = split_by_market(samples, train_frac=args.train_frac)
    means, stds = fit_standardization(x_raw, train_idx)
    x = apply_standardization(x_raw, means, stds)

    ensemble = train_ensemble(
        x=x,
        y=y,
        train_idx=train_idx,
        models=args.models,
        feature_frac=args.feature_frac,
        bag_frac=args.bag_frac,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )
    p = predict_proba(ensemble, x)

    thresholds = [i / 100.0 for i in range(5, 56, 5)]
    train_tm = threshold_metrics(y, p, train_idx, thresholds)
    test_tm = threshold_metrics(y, p, test_idx, thresholds)
    best_test = max(test_tm, key=lambda r: r["f1"]) if test_tm else None

    def by_checkpoint(indexes: list[int]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for cp in checkpoints:
            cp_idx = [i for i in indexes if samples[i].checkpoint_s == cp]
            if not cp_idx:
                continue
            out[str(cp)] = {
                "samples": float(len(cp_idx)),
                "flip_rate": sum(y[i] for i in cp_idx) / len(cp_idx),
                "auc": auc(y, p, cp_idx),
                "brier": brier(y, p, cp_idx),
            }
        return out

    summary = {
        "rows_total": len(rows),
        "markets_total": len(markets),
        "samples_total": len(samples),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_flip_rate": sum(y[i] for i in train_idx) / max(1, len(train_idx)),
        "test_flip_rate": sum(y[i] for i in test_idx) / max(1, len(test_idx)),
        "train_auc": auc(y, p, train_idx),
        "test_auc": auc(y, p, test_idx),
        "train_brier": brier(y, p, train_idx),
        "test_brier": brier(y, p, test_idx),
        "train_logloss": logloss(y, p, train_idx),
        "test_logloss": logloss(y, p, test_idx),
        "best_test_threshold_by_f1": best_test,
        "train_threshold_metrics": train_tm,
        "test_threshold_metrics": test_tm,
        "train_by_checkpoint": by_checkpoint(train_idx),
        "test_by_checkpoint": by_checkpoint(test_idx),
    }

    model = {
        "model_type": "bagged_logistic_ensemble",
        "seed": args.seed,
        "checkpoints": checkpoints,
        "lookback_s": args.lookback_s,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "ensemble": ensemble,
        "meta": {
            "train_frac": args.train_frac,
            "models": args.models,
            "feature_frac": args.feature_frac,
            "bag_frac": args.bag_frac,
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "settle_price": SETTLE_PRICE,
        },
        "summary": summary,
    }

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"rows_total: {len(rows)}")
    print(f"markets_total: {len(markets)}")
    print(f"samples_total: {len(samples)}")
    print(f"train_samples: {len(train_idx)}")
    print(f"test_samples: {len(test_idx)}")
    print(f"train_flip_rate: {summary['train_flip_rate']:.4f}")
    print(f"test_flip_rate: {summary['test_flip_rate']:.4f}")
    print(f"train_auc: {summary['train_auc']:.4f}")
    print(f"test_auc: {summary['test_auc']:.4f}")
    print(f"train_brier: {summary['train_brier']:.4f}")
    print(f"test_brier: {summary['test_brier']:.4f}")
    if best_test is not None:
        print(
            "best_test_threshold_by_f1: "
            f"thr={best_test['threshold']:.2f} precision={best_test['precision']:.3f} "
            f"recall={best_test['recall']:.3f} flag_rate={best_test['flag_rate']:.3f}"
        )
    print(f"model_file: {out_path}")


if __name__ == "__main__":
    main()
