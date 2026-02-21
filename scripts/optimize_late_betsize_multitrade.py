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
            "Optimize high-turnover late-market execution (multi-trade per market) "
            "with model-driven sizing from a $100 bankroll."
        )
    )
    parser.add_argument(
        "--model-file",
        default="backtester/late_profit_model_best.json",
    )
    parser.add_argument("--start-cash", type=float, default=100.0)
    parser.add_argument(
        "--output-json",
        default="backtester/late_betsize_multitrade_optimization.json",
    )
    return parser.parse_args()


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def kelly_fraction(p_win: float, price: float) -> float:
    if price <= 0 or price >= SETTLE_PRICE:
        return 0.0
    b = (SETTLE_PRICE / price) - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p_win
    f = (b * p_win - q) / b
    return max(0.0, f)


def simulate_policy(
    samples: list[object],
    probs: list[float],
    idx: list[int],
    start_cash: float,
    ev_min: float,
    kelly_scale: float,
    max_frac: float,
    min_frac: float,
    max_trades_per_market: int,
    cooldown_s: int,
    market_budget_frac: float,
) -> dict[str, float]:
    # Group chronological candidates per market.
    by_market: dict[str, list[int]] = {}
    for i in idx:
        by_market.setdefault(str(samples[i].market), []).append(i)
    for m in by_market:
        by_market[m].sort(key=lambda j: (samples[j].ts, -probs[j]))

    cash = start_cash
    peak = start_cash
    max_dd = 0.0
    trades = 0
    wins = 0
    sum_frac = 0.0
    sum_bet = 0.0
    market_count = 0

    for market in sorted(by_market.keys()):
        market_count += 1
        cands = by_market[market]
        market_start_cash = cash
        market_budget = market_budget_frac * market_start_cash
        market_staked = 0.0
        used = 0
        last_trade_ts = -(10**12)
        # Keep one action per timestamp (pick best EV at that second).
        by_ts: dict[int, int] = {}
        for i in cands:
            s = samples[i]
            p = probs[i]
            px = float(s.price)
            ev = p * ((SETTLE_PRICE / px) - 1.0) + (1.0 - p) * (-1.0)
            if ev < ev_min:
                continue
            prev = by_ts.get(s.ts)
            if prev is None:
                by_ts[s.ts] = i
                continue
            sp = samples[prev]
            pp = probs[prev]
            ppx = float(sp.price)
            ev_prev = pp * ((SETTLE_PRICE / ppx) - 1.0) + (1.0 - pp) * (-1.0)
            if ev > ev_prev:
                by_ts[s.ts] = i
        chosen_ts = sorted(by_ts.keys())

        fills: list[tuple[float, float]] = []  # (stake, realized_ret)
        for ts in chosen_ts:
            if used >= max_trades_per_market:
                break
            if ts - last_trade_ts < cooldown_s:
                continue
            i = by_ts[ts]
            s = samples[i]
            p = probs[i]
            px = float(s.price)
            f = kelly_fraction(p, px) * kelly_scale
            f = max(0.0, min(max_frac, f))
            if f <= 0 or f < min_frac:
                continue

            stake = cash * f
            if stake <= 0:
                continue
            remaining_budget = max(0.0, market_budget - market_staked)
            if remaining_budget <= 0:
                break
            if stake > remaining_budget:
                stake = remaining_budget
            if stake <= 0:
                continue

            cash -= stake
            market_staked += stake
            used += 1
            last_trade_ts = ts
            trades += 1
            sum_frac += f
            sum_bet += stake

            realized = ((SETTLE_PRICE / px) - 1.0) if int(s.y_win) == 1 else -1.0
            if realized > 0:
                wins += 1
            fills.append((stake, realized))

        # Settle all fills for this market.
        for stake, realized in fills:
            cash += stake * (1.0 + realized)

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
        "markets": float(market_count),
        "trades": float(trades),
        "trades_per_market": trades / max(1, market_count),
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
        samples, float(model["config"]["train_frac"])
    )
    train_markets = sorted(set(str(samples[i].market) for i in train_idx))
    cut = max(1, int(len(train_markets) * 0.8))
    inner_markets = set(train_markets[:cut])
    val_markets = set(train_markets[cut:])
    inner_idx = [i for i in train_idx if str(samples[i].market) in inner_markets]
    val_idx = [i for i in train_idx if str(samples[i].market) in val_markets]
    means = model["means"]
    stds = model["stds"]
    x = [[(s.x[j] - means[j]) / stds[j] for j in range(len(s.x))] for s in samples]
    probs = []
    w = [float(v) for v in model["weights"]]
    b = float(model["bias"])
    for row in x:
        probs.append(sigmoid(b + sum(w[j] * row[j] for j in range(len(w)))))

    ev_grid = [-0.03, -0.02, -0.01, 0.0, 0.01]
    kelly_scale_grid = [0.5, 0.8, 1.0, 1.25, 1.5]
    max_frac_grid = [0.08, 0.10, 0.12, 0.15, 0.20]
    min_frac_grid = [0.0, 0.005]
    max_trades_grid = [1, 2, 3, 4, 6]
    cooldown_grid = [1, 3, 5, 8]
    market_budget_grid = [0.20, 0.35, 0.50, 0.75, 1.00]

    rows: list[dict] = []
    for ev_min in ev_grid:
        for ks in kelly_scale_grid:
            for max_frac in max_frac_grid:
                for min_frac in min_frac_grid:
                    for max_tr in max_trades_grid:
                        for cooldown in cooldown_grid:
                            for market_budget_frac in market_budget_grid:
                                train_stats = simulate_policy(
                                    samples=samples,
                                    probs=probs,
                                    idx=inner_idx,
                                    start_cash=args.start_cash,
                                    ev_min=ev_min,
                                    kelly_scale=ks,
                                    max_frac=max_frac,
                                    min_frac=min_frac,
                                    max_trades_per_market=max_tr,
                                    cooldown_s=cooldown,
                                    market_budget_frac=market_budget_frac,
                                )
                                val_stats = simulate_policy(
                                    samples=samples,
                                    probs=probs,
                                    idx=val_idx,
                                    start_cash=args.start_cash,
                                    ev_min=ev_min,
                                    kelly_scale=ks,
                                    max_frac=max_frac,
                                    min_frac=min_frac,
                                    max_trades_per_market=max_tr,
                                    cooldown_s=cooldown,
                                    market_budget_frac=market_budget_frac,
                                )
                                score = min(
                                    train_stats["end_cash"], val_stats["end_cash"]
                                )
                                if (
                                    train_stats["trades_per_market"] < 2.0
                                    or val_stats["trades_per_market"] < 2.0
                                ):
                                    score -= 300.0
                                if (
                                    train_stats["max_drawdown_pct"] > 40.0
                                    or val_stats["max_drawdown_pct"] > 40.0
                                ):
                                    score -= 1000.0
                                rows.append(
                                    {
                                        "ev_min": ev_min,
                                        "kelly_scale": ks,
                                        "max_frac": max_frac,
                                        "min_frac": min_frac,
                                        "max_trades_per_market": max_tr,
                                        "cooldown_s": cooldown,
                                        "market_budget_frac": market_budget_frac,
                                        "train": train_stats,
                                        "val": val_stats,
                                        "score": score,
                                    }
                                )

    best = max(rows, key=lambda r: r["score"])
    best_test = simulate_policy(
        samples=samples,
        probs=probs,
        idx=test_idx,
        start_cash=args.start_cash,
        ev_min=best["ev_min"],
        kelly_scale=best["kelly_scale"],
        max_frac=best["max_frac"],
        min_frac=best["min_frac"],
        max_trades_per_market=best["max_trades_per_market"],
        cooldown_s=best["cooldown_s"],
        market_budget_frac=best["market_budget_frac"],
    )

    # High-turnover constrained pick (>=2.5 trades/market on both splits).
    high_turnover = [
        r
        for r in rows
        if r["train"]["trades_per_market"] >= 2.5
        and r["val"]["trades_per_market"] >= 2.5
        and r["train"]["max_drawdown_pct"] <= 45.0
        and r["val"]["max_drawdown_pct"] <= 45.0
    ]
    best_high = max(
        high_turnover,
        key=lambda r: min(r["train"]["end_cash"], r["val"]["end_cash"]),
        default=None,
    )
    best_high_test = None
    if best_high is not None:
        best_high_test = simulate_policy(
            samples=samples,
            probs=probs,
            idx=test_idx,
            start_cash=args.start_cash,
            ev_min=best_high["ev_min"],
            kelly_scale=best_high["kelly_scale"],
            max_frac=best_high["max_frac"],
            min_frac=best_high["min_frac"],
            max_trades_per_market=best_high["max_trades_per_market"],
            cooldown_s=best_high["cooldown_s"],
            market_budget_frac=best_high["market_budget_frac"],
        )

    out = {
        "model_file": args.model_file,
        "start_cash": args.start_cash,
        "samples": len(samples),
        "train_samples": len(train_idx),
        "inner_train_samples": len(inner_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "best_train_policy": best,
        "best_test_result": best_test,
        "best_high_turnover_train_policy": best_high,
        "best_high_turnover_test_result": best_high_test,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        "best_policy "
        f"ev_min={best['ev_min']:.3f} kelly_scale={best['kelly_scale']:.2f} "
        f"max_frac={best['max_frac']:.2f} max_trades={best['max_trades_per_market']} "
        f"cooldown={best['cooldown_s']} market_budget={best['market_budget_frac']:.2f}"
    )
    print(
        "best_test "
        f"end_cash={best_test['end_cash']:.2f} return={best_test['return_pct']:.2f}% "
        f"trades={int(best_test['trades'])} tpm={best_test['trades_per_market']:.2f} "
        f"win_rate={best_test['win_rate']:.2%} mdd={best_test['max_drawdown_pct']:.2f}%"
    )
    if best_high is not None and best_high_test is not None:
        print(
            "high_turnover_policy "
            f"ev_min={best_high['ev_min']:.3f} kelly_scale={best_high['kelly_scale']:.2f} "
            f"max_frac={best_high['max_frac']:.2f} max_trades={best_high['max_trades_per_market']} "
            f"cooldown={best_high['cooldown_s']} market_budget={best_high['market_budget_frac']:.2f}"
        )
        print(
            "high_turnover_test "
            f"end_cash={best_high_test['end_cash']:.2f} return={best_high_test['return_pct']:.2f}% "
            f"trades={int(best_high_test['trades'])} tpm={best_high_test['trades_per_market']:.2f} "
            f"win_rate={best_high_test['win_rate']:.2%} mdd={best_high_test['max_drawdown_pct']:.2f}%"
        )
    print(f"output_json: {out_path}")


if __name__ == "__main__":
    main()
