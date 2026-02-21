#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import runpy
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train high-precision late-buy model to maximize number of >90% win "
            "signals using walk-forward validation."
        )
    )
    parser.add_argument(
        "--train-jsonl",
        nargs="+",
        default=["data/btc_1s_pricesv2.jsonl", "data/btc_1s_pricesv3.jsonl"],
    )
    parser.add_argument("--holdout-jsonl", default="data/btc_1s_prices.jsonl")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--entry-min-time-remaining", type=int, default=8)
    parser.add_argument("--entry-max-time-remaining", type=int, default=70)
    parser.add_argument("--min-price", type=float, default=0.55)
    parser.add_argument("--max-price", type=float, default=0.985)
    parser.add_argument("--walkforward-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-model",
        default="backtester/late_profit_model_high_precision.json",
    )
    parser.add_argument(
        "--output-report",
        default="backtester/late_profit_model_high_precision_report.json",
    )
    return parser.parse_args()


def idx_for_markets(samples: list[object], market_set: set[str]) -> list[int]:
    return [i for i, s in enumerate(samples) if str(s.market) in market_set]


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def build_extended_x(x_raw: list[list[float]]) -> tuple[list[list[float]], list[str]]:
    # Extended interaction set to improve ranking calibration for high-confidence picks.
    names = [
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
    out: list[list[float]] = []
    for r in x_raw:
        # raw
        price = r[0]
        edge = r[1]
        gap = r[5]
        abs_gap = r[6]
        time_frac = r[8]
        mom = r[9]
        gap_sl = r[10]
        tok_vol = r[11]
        chop = r[12]
        btc_av = r[13] if len(r) > 13 else 0.0
        btc_d10 = r[14] if len(r) > 14 else 0.0
        btc_d30 = r[15] if len(r) > 15 else 0.0
        btc_vol = r[16] if len(r) > 16 else 0.0
        btc_mom = r[17] if len(r) > 17 else 0.0
        btc_corr = r[18] if len(r) > 18 else 0.0
        ext = list(r)
        # interactions
        ext.extend(
            [
                price * time_frac,
                edge * time_frac,
                gap * time_frac,
                abs_gap * time_frac,
                mom * time_frac,
                gap_sl * time_frac,
                tok_vol * time_frac,
                chop * time_frac,
                price * gap,
                edge * gap,
                mom * gap,
                mom * gap_sl,
                tok_vol * chop,
                btc_av * btc_d10,
                btc_av * btc_d30,
                btc_av * btc_vol,
                btc_av * btc_mom,
                btc_av * btc_corr,
                btc_d10 * gap,
                btc_d30 * gap,
                btc_mom * gap_sl,
                btc_corr * mom,
                price * price,
                gap * gap,
                mom * mom,
            ]
        )
        out.append(ext)
    ext_names = names + [
        "price_x_time",
        "edge_x_time",
        "gap_x_time",
        "abs_gap_x_time",
        "mom_x_time",
        "gap_slope_x_time",
        "tok_vol_x_time",
        "chop_x_time",
        "price_x_gap",
        "edge_x_gap",
        "mom_x_gap",
        "mom_x_gap_slope",
        "tok_vol_x_chop",
        "btc_av_x_d10",
        "btc_av_x_d30",
        "btc_av_x_vol",
        "btc_av_x_mom",
        "btc_av_x_corr",
        "btc_d10_x_gap",
        "btc_d30_x_gap",
        "btc_mom_x_gap_slope",
        "btc_corr_x_mom",
        "price_sq",
        "gap_sq",
        "mom_sq",
    ]
    return out, ext_names


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
    pos_w = neg / pos
    neg_w = 1.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        norm = 0.0
        for i in train_idx:
            xi = x[i]
            yi = y[i]
            p = sigmoid(b + sum(w[j] * xi[j] for j in range(d)))
            wt = pos_w if yi == 1 else neg_w
            e = (p - yi) * wt
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
            norm += wt
        if norm <= 0:
            break
        for j in range(d):
            gw[j] = gw[j] / norm + l2 * w[j]
            w[j] -= lr * gw[j]
        b -= lr * (gb / norm)
    return w, b


def predict(x: list[list[float]], w: list[float], b: float) -> list[float]:
    return [sigmoid(b + sum(w[j] * row[j] for j in range(len(w)))) for row in x]


def pick_top_by_market(
    samples: list[object], probs: list[float], idx: list[int]
) -> list[int]:
    by_market: dict[str, list[int]] = {}
    for i in idx:
        by_market.setdefault(str(samples[i].market), []).append(i)
    picks: list[int] = []
    for m, mids in by_market.items():
        _ = m
        picks.append(max(mids, key=lambda i: probs[i]))
    return picks


def precision_stats(
    samples: list[object], probs: list[float], idx: list[int], thr: float
) -> dict[str, float]:
    picks = pick_top_by_market(samples, probs, idx)
    sel = [i for i in picks if probs[i] >= thr]
    if not sel:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_market": 0.0,
        }
    wins = sum(1 for i in sel if int(samples[i].y_win) == 1)
    wr = wins / len(sel)
    returns = [
        ((0.99 / samples[i].price) - 1.0) if int(samples[i].y_win) == 1 else -1.0
        for i in sel
    ]
    markets = len(set(str(samples[i].market) for i in idx))
    return {
        "trades": float(len(sel)),
        "win_rate": wr,
        "avg_pnl_per_trade": sum(returns) / len(returns),
        "total_pnl": sum(returns),
        "avg_pnl_per_market": sum(returns) / max(1, markets),
    }


def choose_threshold_for_precision(
    samples: list[object],
    probs: list[float],
    calib_idx: list[int],
    target_precision: float,
) -> tuple[float, dict]:
    thresholds = [i / 100.0 for i in range(50, 100)]  # 0.50..0.99
    feasible: list[tuple[float, dict]] = []
    for thr in thresholds:
        st = precision_stats(samples, probs, calib_idx, thr)
        if st["trades"] <= 0:
            continue
        if st["win_rate"] >= target_precision:
            feasible.append((thr, st))
    if feasible:
        # maximize trade count then pnl/trade.
        best = max(feasible, key=lambda x: (x[1]["trades"], x[1]["avg_pnl_per_trade"]))
        return best
    # fallback: highest precision, then trades.
    rows = [
        (thr, precision_stats(samples, probs, calib_idx, thr)) for thr in thresholds
    ]
    rows = [r for r in rows if r[1]["trades"] > 0]
    if not rows:
        return 0.99, {
            "trades": 0.0,
            "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_market": 0.0,
        }
    return max(rows, key=lambda x: (x[1]["win_rate"], x[1]["trades"]))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    fit_mod = runpy.run_path("scripts/fit_late_profit_model.py")

    markets = fit_mod["load_ticks"](args.train_jsonl)
    samples, _ = fit_mod["build_samples"](
        markets=markets,
        entry_min_tr=args.entry_min_time_remaining,
        entry_max_tr=args.entry_max_time_remaining,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    x_raw_base = [s.x for s in samples]
    y = [int(s.y_win) for s in samples]
    x_raw, feature_names = build_extended_x(x_raw_base)
    market_ids = sorted(set(str(s.market) for s in samples))
    n_markets = len(market_ids)
    val_size = max(5, n_markets // (args.walkforward_folds + 1))

    hp_grid = [
        {"epochs": 700, "lr": 0.08, "l2": 0.003},
        {"epochs": 900, "lr": 0.08, "l2": 0.01},
        {"epochs": 1200, "lr": 0.06, "l2": 0.02},
        {"epochs": 1200, "lr": 0.08, "l2": 0.03},
        {"epochs": 1500, "lr": 0.05, "l2": 0.05},
    ]

    candidates: list[dict] = []
    for hp in hp_grid:
        folds = []
        for fold in range(args.walkforward_folds):
            train_end = val_size * (fold + 1)
            val_start = train_end
            val_end = min(n_markets, val_start + val_size)
            if val_start >= val_end:
                continue
            train_markets = market_ids[:train_end]
            cut = max(1, int(len(train_markets) * 0.8))
            fit_markets = set(train_markets[:cut])
            calib_markets = set(train_markets[cut:]) or set(train_markets)
            val_markets = set(market_ids[val_start:val_end])

            fit_idx = idx_for_markets(samples, fit_markets)
            calib_idx = idx_for_markets(samples, calib_markets)
            val_idx = idx_for_markets(samples, val_markets)
            if not fit_idx or not calib_idx or not val_idx:
                continue

            means, stds = fit_standardize(x_raw, fit_idx)
            x = apply_standardize(x_raw, means, stds)
            w, b = fit_logistic_weighted(
                x=x,
                y=y,
                train_idx=fit_idx,
                epochs=hp["epochs"],
                lr=hp["lr"],
                l2=hp["l2"],
            )
            probs = predict(x, w, b)
            thr, calib_stats = choose_threshold_for_precision(
                samples=samples,
                probs=probs,
                calib_idx=calib_idx,
                target_precision=args.target_precision,
            )
            val_stats = precision_stats(samples, probs, val_idx, thr)
            folds.append(
                {
                    "fold": fold + 1,
                    "threshold": thr,
                    "calib": calib_stats,
                    "val": val_stats,
                }
            )
        if not folds:
            continue
        val_prec = [f["val"]["win_rate"] for f in folds]
        val_trades = [f["val"]["trades"] for f in folds]
        val_pm = [f["val"]["avg_pnl_per_market"] for f in folds]
        score = (
            median(val_prec)
            + 0.15 * min(val_prec)
            + 0.02 * median(val_trades)
            + 0.10 * median(val_pm)
        )
        candidates.append(
            {
                "hp": hp,
                "folds": folds,
                "median_val_precision": median(val_prec),
                "min_val_precision": min(val_prec),
                "median_val_trades": median(val_trades),
                "median_val_avg_market": median(val_pm),
                "score": score,
            }
        )

    if not candidates:
        raise SystemExit("no viable model candidates")
    best = max(candidates, key=lambda c: c["score"])

    # Final fit/calibration on full training set.
    cut = max(1, int(len(market_ids) * 0.8))
    fit_markets = set(market_ids[:cut])
    calib_markets = set(market_ids[cut:]) or set(market_ids)
    fit_idx = idx_for_markets(samples, fit_markets)
    calib_idx = idx_for_markets(samples, calib_markets)

    means, stds = fit_standardize(x_raw, fit_idx)
    x = apply_standardize(x_raw, means, stds)
    w, b = fit_logistic_weighted(
        x=x,
        y=y,
        train_idx=fit_idx,
        epochs=best["hp"]["epochs"],
        lr=best["hp"]["lr"],
        l2=best["hp"]["l2"],
    )
    probs = predict(x, w, b)
    thr, calib_stats = choose_threshold_for_precision(
        samples=samples,
        probs=probs,
        calib_idx=calib_idx,
        target_precision=args.target_precision,
    )

    # Holdout evaluation.
    h_markets = fit_mod["load_ticks"]([args.holdout_jsonl])
    h_samples, _ = fit_mod["build_samples"](
        markets=h_markets,
        entry_min_tr=args.entry_min_time_remaining,
        entry_max_tr=args.entry_max_time_remaining,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    hx_raw, _ = build_extended_x([s.x for s in h_samples])
    hx = apply_standardize(hx_raw, means, stds)
    h_probs = predict(hx, w, b)
    hold_stats = precision_stats(
        samples=h_samples,
        probs=h_probs,
        idx=list(range(len(h_samples))),
        thr=thr,
    )

    out_model = {
        "dataset": "high_precision_one_s",
        "paths": args.train_jsonl,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "weights": w,
        "bias": b,
        "prob_threshold": thr,
        "target_precision": args.target_precision,
        "config": {
            "entry_min_time_remaining": args.entry_min_time_remaining,
            "entry_max_time_remaining": args.entry_max_time_remaining,
            "min_price": args.min_price,
            "max_price": args.max_price,
            "walkforward_folds": args.walkforward_folds,
            "hp": best["hp"],
        },
        "calibration_stats": calib_stats,
        "holdout_stats": hold_stats,
    }
    out_report = {
        "candidates": candidates,
        "selected": best,
        "calibration_stats": calib_stats,
        "holdout_stats": hold_stats,
        "holdout_markets": len(set(str(s.market) for s in h_samples)),
        "holdout_samples": len(h_samples),
    }

    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(out_model, f, indent=2)
    report_path = Path(args.output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    print(f"selected_hp: {best['hp']}")
    print(f"prob_threshold: {thr:.2f}")
    print(
        f"holdout trades={int(hold_stats['trades'])} "
        f"win={hold_stats['win_rate']:.2%} "
        f"avg_market={hold_stats['avg_pnl_per_market']:.4f}"
    )
    print(f"model_file: {model_path}")
    print(f"report_file: {report_path}")


if __name__ == "__main__":
    main()
