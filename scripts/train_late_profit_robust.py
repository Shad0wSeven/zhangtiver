#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train late-profit model with walk-forward validation to reduce overfit."
        )
    )
    parser.add_argument(
        "--train-jsonl",
        nargs="+",
        default=["data/btc_1s_pricesv2.jsonl", "data/btc_1s_pricesv3.jsonl"],
    )
    parser.add_argument(
        "--holdout-jsonl",
        default="data/btc_1s_prices.jsonl",
    )
    parser.add_argument("--entry-min-time-remaining", type=int, default=8)
    parser.add_argument("--entry-max-time-remaining", type=int, default=70)
    parser.add_argument("--min-price", type=float, default=0.55)
    parser.add_argument("--max-price", type=float, default=0.985)
    parser.add_argument("--walkforward-folds", type=int, default=5)
    parser.add_argument(
        "--output-model", default="backtester/late_profit_model_robust.json"
    )
    parser.add_argument(
        "--output-report", default="backtester/late_profit_model_robust_report.json"
    )
    return parser.parse_args()


def idx_for_markets(samples: list[object], market_set: set[str]) -> list[int]:
    return [i for i, s in enumerate(samples) if str(s.market) in market_set]


def choose_threshold(
    fit_mod: dict,
    samples: list[object],
    probs: list[float],
    calib_idx: list[int],
    thresholds: list[float],
) -> tuple[float, dict]:
    best_thr = thresholds[0]
    best_stats = fit_mod["policy_eval"](samples, probs, calib_idx, best_thr)
    best_score = best_stats["avg_pnl_per_market"] - 0.2 * max(
        0.0, 0.25 - best_stats["trade_rate"]
    )
    for thr in thresholds[1:]:
        st = fit_mod["policy_eval"](samples, probs, calib_idx, thr)
        score = st["avg_pnl_per_market"] - 0.2 * max(0.0, 0.25 - st["trade_rate"])
        if score > best_score:
            best_score = score
            best_thr = thr
            best_stats = st
    return best_thr, best_stats


def train_predict(
    fit_mod: dict,
    x_raw: list[list[float]],
    y: list[int],
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[list[float], list[float], list[float], float, list[float]]:
    means, stds = fit_mod["fit_standardize"](x_raw, train_idx)
    x = fit_mod["apply_standardize"](x_raw, means, stds)
    w, b = fit_mod["fit_logistic"](x, y, train_idx, epochs=epochs, lr=lr, l2=l2)
    probs = fit_mod["predict"](x, w, b)
    return means, stds, w, b, probs


def main() -> None:
    args = parse_args()
    fit_mod = runpy.run_path("scripts/fit_late_profit_model.py")

    markets = fit_mod["load_ticks"](args.train_jsonl)
    samples, feature_names = fit_mod["build_samples"](
        markets=markets,
        entry_min_tr=args.entry_min_time_remaining,
        entry_max_tr=args.entry_max_time_remaining,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    if len(samples) < 500:
        raise SystemExit(f"not enough training samples: {len(samples)}")

    x_raw = [s.x for s in samples]
    y = [s.y_win for s in samples]
    market_ids = sorted(set(str(s.market) for s in samples))
    n_markets = len(market_ids)
    if n_markets < args.walkforward_folds + 2:
        raise SystemExit("not enough markets for requested walk-forward folds")

    val_size = max(5, n_markets // (args.walkforward_folds + 1))
    thresholds = [i / 1000.0 for i in range(-20, 81, 5)]  # -0.02 .. 0.08

    hp_grid = [
        {"epochs": 400, "lr": 0.08, "l2": 0.001},
        {"epochs": 700, "lr": 0.08, "l2": 0.003},
        {"epochs": 700, "lr": 0.12, "l2": 0.003},
        {"epochs": 900, "lr": 0.10, "l2": 0.01},
        {"epochs": 1200, "lr": 0.08, "l2": 0.03},
    ]

    candidates: list[dict] = []
    for hp in hp_grid:
        fold_rows = []
        for fold in range(args.walkforward_folds):
            train_end = val_size * (fold + 1)
            val_start = train_end
            val_end = min(n_markets, val_start + val_size)
            if val_start >= val_end:
                continue

            train_markets = market_ids[:train_end]
            # Inner split for threshold calibration: fit on early 80%, pick threshold on late 20%.
            cut = max(1, int(len(train_markets) * 0.8))
            fit_markets = set(train_markets[:cut])
            calib_markets = set(train_markets[cut:])
            if not calib_markets:
                calib_markets = set(train_markets)

            val_markets = set(market_ids[val_start:val_end])
            fit_idx = idx_for_markets(samples, fit_markets)
            calib_idx = idx_for_markets(samples, calib_markets)
            val_idx = idx_for_markets(samples, val_markets)
            if not fit_idx or not calib_idx or not val_idx:
                continue

            means, stds, w, b, probs = train_predict(
                fit_mod=fit_mod,
                x_raw=x_raw,
                y=y,
                train_idx=fit_idx,
                epochs=hp["epochs"],
                lr=hp["lr"],
                l2=hp["l2"],
            )

            thr, calib_stats = choose_threshold(
                fit_mod=fit_mod,
                samples=samples,
                probs=probs,
                calib_idx=calib_idx,
                thresholds=thresholds,
            )
            val_stats = fit_mod["policy_eval"](samples, probs, val_idx, thr)
            fold_rows.append(
                {
                    "fold": fold + 1,
                    "train_markets": len(train_markets),
                    "val_markets": len(val_markets),
                    "threshold": thr,
                    "calib": calib_stats,
                    "val": val_stats,
                }
            )

        if not fold_rows:
            continue
        val_pm = [r["val"]["avg_pnl_per_market"] for r in fold_rows]
        val_tot = [r["val"]["total_pnl"] for r in fold_rows]
        val_trade = [r["val"]["trade_rate"] for r in fold_rows]
        candidate = {
            "hp": hp,
            "folds": fold_rows,
            "median_val_avg_pnl_per_market": median(val_pm),
            "min_val_avg_pnl_per_market": min(val_pm),
            "mean_val_trade_rate": sum(val_trade) / len(val_trade),
            "mean_val_total_pnl": sum(val_tot) / len(val_tot),
        }
        # Conservative score: robust across folds and not too sparse.
        candidate["score"] = (
            candidate["median_val_avg_pnl_per_market"]
            + 0.35 * candidate["min_val_avg_pnl_per_market"]
            - 0.08 * max(0.0, 0.25 - candidate["mean_val_trade_rate"])
        )
        candidates.append(candidate)

    if not candidates:
        raise SystemExit("no viable candidates")
    best = max(candidates, key=lambda c: c["score"])

    # Final training on full train-jsonl with an internal 80/20 split for threshold.
    full_markets = sorted(set(str(s.market) for s in samples))
    cut = max(1, int(len(full_markets) * 0.8))
    final_fit_markets = set(full_markets[:cut])
    final_calib_markets = set(full_markets[cut:])
    final_fit_idx = idx_for_markets(samples, final_fit_markets)
    final_calib_idx = idx_for_markets(samples, final_calib_markets)

    means, stds, w, b, calib_probs = train_predict(
        fit_mod=fit_mod,
        x_raw=x_raw,
        y=y,
        train_idx=final_fit_idx,
        epochs=best["hp"]["epochs"],
        lr=best["hp"]["lr"],
        l2=best["hp"]["l2"],
    )
    final_thr, final_calib_stats = choose_threshold(
        fit_mod=fit_mod,
        samples=samples,
        probs=calib_probs,
        calib_idx=final_calib_idx,
        thresholds=thresholds,
    )

    # Strict unseen holdout evaluation.
    hold_markets = fit_mod["load_ticks"]([args.holdout_jsonl])
    hold_samples, hold_features = fit_mod["build_samples"](
        markets=hold_markets,
        entry_min_tr=args.entry_min_time_remaining,
        entry_max_tr=args.entry_max_time_remaining,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    if hold_features != feature_names:
        raise SystemExit("holdout features mismatch")
    hx_raw = [s.x for s in hold_samples]
    hx = fit_mod["apply_standardize"](hx_raw, means, stds)
    hprobs = fit_mod["predict"](hx, w, b)
    hold_idx = list(range(len(hold_samples)))
    hold_stats = fit_mod["policy_eval"](hold_samples, hprobs, hold_idx, final_thr)

    model_out = {
        "dataset": "robust_one_s",
        "paths": args.train_jsonl,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "weights": w,
        "bias": b,
        "selected_ev_threshold": final_thr,
        "config": {
            "entry_min_time_remaining": args.entry_min_time_remaining,
            "entry_max_time_remaining": args.entry_max_time_remaining,
            "min_price": args.min_price,
            "max_price": args.max_price,
            "walkforward_folds": args.walkforward_folds,
            "hp": best["hp"],
            "settle_price": 0.99,
        },
        "final_calibration_stats": final_calib_stats,
        "holdout_policy": hold_stats,
    }

    report_out = {
        "candidates": candidates,
        "selected": best,
        "holdout_path": args.holdout_jsonl,
        "holdout_markets": len(set(str(s.market) for s in hold_samples)),
        "holdout_samples": len(hold_samples),
        "holdout_policy": hold_stats,
    }

    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_out, f, indent=2)
    report_path = Path(args.output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2)

    print(f"selected_hp: {best['hp']}")
    print(f"selected_score: {best['score']:.6f}")
    print(
        f"holdout avg_market={hold_stats['avg_pnl_per_market']:.4f} "
        f"total={hold_stats['total_pnl']:.4f} trades={int(hold_stats['trades'])} "
        f"win={hold_stats['win_rate']:.2%}"
    )
    print(f"model_file: {model_path}")
    print(f"report_file: {report_path}")


if __name__ == "__main__":
    main()
