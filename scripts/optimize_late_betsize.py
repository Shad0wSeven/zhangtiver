#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

SETTLE_PRICE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize model-driven late-entry bet sizing on train split, then "
            "report out-of-sample bankroll performance from $100."
        )
    )
    parser.add_argument(
        "--model-file",
        default="backtester/late_profit_model_best.json",
    )
    parser.add_argument("--start-cash", type=float, default=100.0)
    parser.add_argument(
        "--output-json",
        default="backtester/late_betsize_optimization.json",
    )
    return parser.parse_args()


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_probs(x: list[list[float]], w: list[float], b: float) -> list[float]:
    out: list[float] = []
    for row in x:
        z = b + sum(w[j] * row[j] for j in range(len(w)))
        out.append(sigmoid(z))
    return out


def market_key(sample: object) -> str:
    return str(sample.market)


def kelly_fraction(p_win: float, price: float) -> float:
    if price <= 0 or price >= SETTLE_PRICE:
        return 0.0
    b = (SETTLE_PRICE / price) - 1.0  # net win per $1 staked
    if b <= 0:
        return 0.0
    q = 1.0 - p_win
    f = (b * p_win - q) / b
    return max(0.0, f)


def pick_one_trade_per_market(
    samples: list[object],
    probs: list[float],
    idx: list[int],
    ev_min: float,
) -> list[int]:
    by_market: dict[str, list[int]] = {}
    for i in idx:
        by_market.setdefault(market_key(samples[i]), []).append(i)

    picks: list[int] = []
    for m in sorted(by_market.keys()):
        best_i = None
        best_ev = -1e9
        for i in by_market[m]:
            s = samples[i]
            p = probs[i]
            px = float(s.price)
            ev = p * ((SETTLE_PRICE / px) - 1.0) + (1.0 - p) * (-1.0)
            if ev > best_ev:
                best_ev = ev
                best_i = i
        if best_i is None:
            continue
        if best_ev >= ev_min:
            picks.append(best_i)
    return picks


def simulate_bankroll(
    samples: list[object],
    probs: list[float],
    picks: list[int],
    start_cash: float,
    kelly_scale: float,
    max_frac: float,
    min_frac: float,
) -> dict[str, float]:
    cash = start_cash
    peak = start_cash
    max_dd = 0.0
    trades = 0
    wins = 0
    sum_frac = 0.0
    sum_bet = 0.0

    for i in picks:
        s = samples[i]
        p = probs[i]
        px = float(s.price)
        f_raw = kelly_fraction(p, px)
        f = max(0.0, min(max_frac, f_raw * kelly_scale))
        if f <= 0:
            continue
        if f < min_frac:
            continue

        stake = cash * f
        if stake <= 0:
            continue

        trades += 1
        sum_frac += f
        sum_bet += stake
        if int(s.y_win) == 1:
            wins += 1
            ret = (SETTLE_PRICE / px) - 1.0
        else:
            ret = -1.0

        cash += stake * ret
        if cash > peak:
            peak = cash
        dd = (peak - cash) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "start_cash": start_cash,
        "end_cash": cash,
        "net_pnl": cash - start_cash,
        "return_pct": ((cash / start_cash) - 1.0) * 100.0 if start_cash > 0 else 0.0,
        "trades": float(trades),
        "win_rate": (wins / trades) if trades > 0 else 0.0,
        "avg_bet_frac": (sum_frac / trades) if trades > 0 else 0.0,
        "avg_bet_usd": (sum_bet / trades) if trades > 0 else 0.0,
        "max_drawdown_pct": max_dd * 100.0,
    }


def main() -> None:
    args = parse_args()
    model = json.load(open(args.model_file, encoding="utf-8"))

    fit_mod = runpy.run_path("scripts/fit_late_profit_model.py")
    markets = fit_mod["load_ticks"](model["paths"])
    samples, feature_names = fit_mod["build_samples"](
        markets=markets,
        entry_min_tr=int(model["config"]["entry_min_time_remaining"]),
        entry_max_tr=int(model["config"]["entry_max_time_remaining"]),
        min_price=float(model["config"]["min_price"]),
        max_price=float(model["config"]["max_price"]),
    )
    if feature_names != model["feature_names"]:
        raise SystemExit("feature mismatch vs saved model")

    train_idx, test_idx = fit_mod["split_by_market"](
        samples, float(model["config"].get("train_frac", 0.7))
    )

    means = model["means"]
    stds = model["stds"]
    x = [[(s.x[j] - means[j]) / stds[j] for j in range(len(s.x))] for s in samples]
    probs = predict_probs(x, model["weights"], float(model["bias"]))

    ev_grid = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
    kelly_scale_grid = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25]
    max_frac_grid = [0.10, 0.15, 0.20, 0.25, 0.35]
    min_frac_grid = [0.0, 0.01]

    rows: list[dict] = []
    for ev_min in ev_grid:
        train_picks = pick_one_trade_per_market(
            samples, probs, train_idx, ev_min=ev_min
        )
        test_picks = pick_one_trade_per_market(samples, probs, test_idx, ev_min=ev_min)
        for k_scale in kelly_scale_grid:
            for max_frac in max_frac_grid:
                for min_frac in min_frac_grid:
                    train_stats = simulate_bankroll(
                        samples=samples,
                        probs=probs,
                        picks=train_picks,
                        start_cash=args.start_cash,
                        kelly_scale=k_scale,
                        max_frac=max_frac,
                        min_frac=min_frac,
                    )
                    row = {
                        "ev_min": ev_min,
                        "kelly_scale": k_scale,
                        "max_frac": max_frac,
                        "min_frac": min_frac,
                        "train": train_stats,
                    }
                    # strict optimization target: maximize terminal cash with DD guard.
                    score = train_stats["end_cash"]
                    if train_stats["max_drawdown_pct"] > 45.0:
                        score -= 1000.0
                    row["score"] = score
                    rows.append(row)

    best = max(rows, key=lambda r: r["score"])
    best_test_picks = pick_one_trade_per_market(
        samples, probs, test_idx, ev_min=best["ev_min"]
    )
    best_test = simulate_bankroll(
        samples=samples,
        probs=probs,
        picks=best_test_picks,
        start_cash=args.start_cash,
        kelly_scale=best["kelly_scale"],
        max_frac=best["max_frac"],
        min_frac=best["min_frac"],
    )

    # Also provide a more defensive profile.
    def feasible_defensive(r: dict) -> bool:
        tr = r["train"]
        return tr["max_drawdown_pct"] <= 25.0 and tr["end_cash"] >= args.start_cash

    defensive_candidates = [r for r in rows if feasible_defensive(r)]
    defensive = max(
        defensive_candidates,
        key=lambda r: (r["train"]["end_cash"], -r["train"]["max_drawdown_pct"]),
        default=None,
    )
    defensive_test = None
    if defensive is not None:
        defensive_test_picks = pick_one_trade_per_market(
            samples, probs, test_idx, ev_min=defensive["ev_min"]
        )
        defensive_test = simulate_bankroll(
            samples=samples,
            probs=probs,
            picks=defensive_test_picks,
            start_cash=args.start_cash,
            kelly_scale=defensive["kelly_scale"],
            max_frac=defensive["max_frac"],
            min_frac=defensive["min_frac"],
        )

    out = {
        "model_file": args.model_file,
        "start_cash": args.start_cash,
        "samples": len(samples),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "best_train_policy": best,
        "best_test_result": best_test,
        "defensive_train_policy": defensive,
        "defensive_test_result": defensive_test,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        "best_policy "
        f"ev_min={best['ev_min']:.3f} kelly_scale={best['kelly_scale']:.2f} "
        f"max_frac={best['max_frac']:.2f} min_frac={best['min_frac']:.2f}"
    )
    print(
        "test_result "
        f"end_cash={best_test['end_cash']:.2f} pnl={best_test['net_pnl']:.2f} "
        f"return={best_test['return_pct']:.2f}% trades={int(best_test['trades'])} "
        f"win_rate={best_test['win_rate']:.2%} mdd={best_test['max_drawdown_pct']:.2f}%"
    )
    if defensive is not None and defensive_test is not None:
        print(
            "defensive_policy "
            f"ev_min={defensive['ev_min']:.3f} kelly_scale={defensive['kelly_scale']:.2f} "
            f"max_frac={defensive['max_frac']:.2f} min_frac={defensive['min_frac']:.2f}"
        )
        print(
            "defensive_test "
            f"end_cash={defensive_test['end_cash']:.2f} pnl={defensive_test['net_pnl']:.2f} "
            f"return={defensive_test['return_pct']:.2f}% trades={int(defensive_test['trades'])} "
            f"win_rate={defensive_test['win_rate']:.2%} mdd={defensive_test['max_drawdown_pct']:.2f}%"
        )
    print(f"output_json: {out_path}")


if __name__ == "__main__":
    main()
