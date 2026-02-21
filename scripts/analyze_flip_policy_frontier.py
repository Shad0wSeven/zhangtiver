#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import runpy

SETTLE_PRICE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze risk-management vs profit-extraction frontier for flip-risk "
            "policies using a saved flip ensemble model."
        )
    )
    parser.add_argument(
        "--model-file",
        default="backtester/flip_shape_ensemble_model.json",
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=["data/btc_1s_pricesv2.jsonl", "data/btc_1s_pricesv3.jsonl"],
    )
    parser.add_argument(
        "--checkpoints",
        default="40,30,20",
        help="Subset to analyze (seconds remaining), comma-separated.",
    )
    parser.add_argument(
        "--output-json",
        default="backtester/flip_policy_frontier.json",
    )
    return parser.parse_args()


def summarize(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "n": 0.0,
            "avg": 0.0,
            "total": 0.0,
            "win_rate": 0.0,
            "std": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
        }
    n = len(returns)
    avg = sum(returns) / n
    var = sum((x - avg) ** 2 for x in returns) / n
    std = math.sqrt(var)
    sharpe = avg / std if std > 1e-12 else 0.0
    win_rate = sum(1 for x in returns if x > 0) / n

    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > mdd:
            mdd = dd

    return {
        "n": float(n),
        "avg": avg,
        "total": sum(returns),
        "win_rate": win_rate,
        "std": std,
        "sharpe_like": sharpe,
        "max_drawdown": mdd,
    }


def main() -> None:
    args = parse_args()
    checkpoints = {int(x.strip()) for x in args.checkpoints.split(",") if x.strip()}
    fit_mod = runpy.run_path("scripts/fit_flip_shape_model.py")
    model = json.load(open(args.model_file, encoding="utf-8"))

    rows = fit_mod["load_rows"](args.jsonl)
    markets = fit_mod["group_markets"](rows)
    samples, feature_names = fit_mod["build_dataset"](
        markets,
        model["checkpoints"],
        model["lookback_s"],
        15,
        80,
    )
    train_idx, test_idx = fit_mod["split_by_market"](
        samples, model["meta"]["train_frac"]
    )

    means = model["means"]
    stds = model["stds"]
    x = [[(s.x[j] - means[j]) / stds[j] for j in range(len(s.x))] for s in samples]
    y = [s.y_flip for s in samples]
    probs = fit_mod["predict_proba"](model["ensemble"], x)

    leader_idx = feature_names.index("leader_price")
    eval_idx = [i for i in test_idx if samples[i].checkpoint_s in checkpoints]

    def trade_return(i: int, size_mult: float = 1.0) -> float:
        price = samples[i].x[leader_idx]
        base = ((SETTLE_PRICE / price) - 1.0) if y[i] == 0 else -1.0
        return base * size_mult

    base_returns = [trade_return(i) for i in eval_idx]
    base_stats = summarize(base_returns)
    base_stats["avoid_rate"] = 0.0

    frontier: list[dict] = []
    for thr in [x / 100.0 for x in range(20, 66, 5)]:
        for gamma in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
            returns: list[float] = []
            kept = 0
            for i in eval_idx:
                if probs[i] >= thr:
                    continue
                kept += 1
                size = max(0.05, min(1.0, (1.0 - probs[i]) ** gamma))
                returns.append(trade_return(i, size))
            stats = summarize(returns)
            stats["threshold"] = thr
            stats["gamma"] = gamma
            stats["avoid_rate"] = 1.0 - (kept / max(1, len(eval_idx)))
            frontier.append(stats)

    # Policy picks.
    balanced = max(
        (
            r
            for r in frontier
            if r["avoid_rate"] <= 0.30 and r["max_drawdown"] <= 1.5 and r["n"] > 0
        ),
        key=lambda r: (r["total"], r["sharpe_like"]),
        default=None,
    )
    defensive = max(
        (
            r
            for r in frontier
            if r["avoid_rate"] <= 0.35 and r["max_drawdown"] <= 1.0 and r["n"] > 0
        ),
        key=lambda r: (r["sharpe_like"], r["total"]),
        default=None,
    )
    aggressive = max(
        (r for r in frontier if r["avoid_rate"] <= 0.20 and r["n"] > 0),
        key=lambda r: (r["total"], r["sharpe_like"]),
        default=None,
    )

    out = {
        "model_file": args.model_file,
        "jsonl": args.jsonl,
        "eval_checkpoints": sorted(checkpoints),
        "test_samples": len(eval_idx),
        "test_flip_rate": (
            sum(y[i] for i in eval_idx) / max(1, len(eval_idx)) if eval_idx else 0.0
        ),
        "baseline": base_stats,
        "recommended": {
            "balanced": balanced,
            "defensive": defensive,
            "aggressive": aggressive,
        },
        "top_by_total": sorted(frontier, key=lambda r: r["total"], reverse=True)[:20],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"test_samples: {len(eval_idx)}")
    print(f"test_flip_rate: {out['test_flip_rate']:.4f}")
    print(
        f"baseline total={base_stats['total']:.4f} avg={base_stats['avg']:.4f} "
        f"sharpe={base_stats['sharpe_like']:.4f} mdd={base_stats['max_drawdown']:.4f}"
    )
    for key in ("balanced", "defensive", "aggressive"):
        row = out["recommended"][key]
        if row is None:
            print(f"{key}: none")
            continue
        print(
            f"{key}: thr={row['threshold']:.2f} gamma={row['gamma']:.1f} "
            f"avoid={row['avoid_rate']:.3f} total={row['total']:.4f} "
            f"avg={row['avg']:.4f} sharpe={row['sharpe_like']:.4f} "
            f"mdd={row['max_drawdown']:.4f}"
        )
    print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
