#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtester.data import expand_paths, is_complete_market, load_markets
from backtester.engine import BacktestEngine, EngineConfig
from backtester.gridsearch import load_grid, tune_strategy
from backtester.metrics import print_summary, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic backtester for livetrader strategy files + gridsearch"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    eval_p = sub.add_parser("eval", help="Evaluate one strategy")
    eval_p.add_argument("--strategy-file", required=True)
    eval_p.add_argument(
        "--jsonl",
        nargs="+",
        default=["btc_5m_prices-archive.jsonl", "btc_5m_prices.jsonl"],
    )
    eval_p.add_argument("--params-json", default="{}")
    eval_p.add_argument("--start-cash", type=float, default=1000.0)
    eval_p.add_argument("--default-bet-usd", type=float, default=10.0)
    eval_p.add_argument("--settle-win", type=float, default=0.99)
    eval_p.add_argument("--end-threshold-s", type=int, default=3)
    eval_p.add_argument("--show-markets", type=int, default=15)

    eval_all_p = sub.add_parser("eval-all", help="Evaluate many strategies")
    eval_all_p.add_argument(
        "--strategy-glob",
        nargs="+",
        default=["strategies/*.py"],
        help="Strategy file paths or globs",
    )
    eval_all_p.add_argument(
        "--include-private",
        action="store_true",
        help="Include strategy files prefixed with '_'",
    )
    eval_all_p.add_argument(
        "--jsonl",
        nargs="+",
        default=["btc_5m_prices-archive.jsonl", "btc_5m_prices.jsonl"],
    )
    eval_all_p.add_argument("--params-json", default="{}")
    eval_all_p.add_argument("--start-cash", type=float, default=1000.0)
    eval_all_p.add_argument("--default-bet-usd", type=float, default=10.0)
    eval_all_p.add_argument("--settle-win", type=float, default=0.99)
    eval_all_p.add_argument("--end-threshold-s", type=int, default=3)

    tune_p = sub.add_parser("tune", help="Gridsearch one strategy")
    tune_p.add_argument("--strategy-file", required=True)
    tune_p.add_argument("--grid-file", required=True)
    tune_p.add_argument(
        "--jsonl",
        nargs="+",
        default=["btc_5m_prices-archive.jsonl", "btc_5m_prices.jsonl"],
    )
    tune_p.add_argument(
        "--objective",
        default="total_pnl",
        choices=[
            "total_pnl",
            "avg_pnl",
            "win_rate",
            "sharpe_like",
            "profit_factor",
            "trade_count",
            "max_drawdown",
        ],
    )
    tune_p.add_argument("--top-k", type=int, default=10)
    tune_p.add_argument("--max-combos", type=int, default=0)
    tune_p.add_argument("--progress-every", type=int, default=0)
    tune_p.add_argument("--start-cash", type=float, default=1000.0)
    tune_p.add_argument("--default-bet-usd", type=float, default=10.0)
    tune_p.add_argument("--settle-win", type=float, default=0.99)
    tune_p.add_argument("--end-threshold-s", type=int, default=3)
    return parser.parse_args()


def build_engine(args: argparse.Namespace) -> BacktestEngine:
    return BacktestEngine(
        EngineConfig(
            window_start_cash=args.start_cash,
            default_bet_usd=args.default_bet_usd,
            settlement_win_price=args.settle_win,
            complete_end_threshold_s=args.end_threshold_s,
        )
    )


def cmd_eval(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    paths = expand_paths(args.jsonl)
    markets = load_markets(paths)
    params = json.loads(args.params_json)
    run = engine.run_strategy(args.strategy_file, markets, params=params)
    stats = summarize(run)

    print(f"strategy: {run['strategy_name']}")
    print(f"strategy_file: {run['strategy_file']}")
    print(f"params: {json.dumps(run['params'], sort_keys=True)}")
    print(f"files: {', '.join(paths)}")
    print_summary(stats)

    show_n = max(0, args.show_markets)
    if show_n > 0:
        print("market_samples:")
        for m in run["market_results"][:show_n]:
            print(
                f"  {m.market} winner={m.winner} pnl={m.pnl:.4f} "
                f"trades={m.trades} mdd={m.max_drawdown:.4f}"
            )


def cmd_eval_all(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    jsonl_paths = expand_paths(args.jsonl)
    markets = load_markets(jsonl_paths)
    strategy_paths = expand_paths(args.strategy_glob)
    if not args.include_private:
        strategy_paths = [p for p in strategy_paths if not Path(p).name.startswith("_")]
    strategy_paths = sorted(dict.fromkeys(strategy_paths))
    if not strategy_paths:
        raise SystemExit("no strategy files found")

    complete = sum(
        1
        for rows in markets.values()
        if is_complete_market(rows, engine.config.complete_end_threshold_s)
    )
    print(f"files: {', '.join(jsonl_paths)}")
    print(f"strategies: {len(strategy_paths)}")
    print(f"markets_total: {len(markets)}")
    print(f"markets_complete: {complete}")

    params = json.loads(args.params_json)
    rows: list[tuple[str, dict]] = []
    for strategy_file in strategy_paths:
        try:
            run = engine.run_strategy(strategy_file, markets, params=params)
            stats = summarize(run)
            rows.append((strategy_file, stats))
        except Exception as exc:
            print(f"error: {strategy_file} -> {exc}")

    rows.sort(key=lambda x: x[1]["total_pnl"], reverse=True)
    print("results:")
    for strategy_file, stats in rows:
        print(
            f"  {strategy_file}: total_pnl={stats['total_pnl']:.4f} "
            f"avg_pnl={stats['avg_pnl']:.4f} median_pnl={stats['median_pnl']:.4f} "
            f"win_rate={stats['win_rate']:.2f}% trades={stats['trade_count']} "
            f"markets={stats['markets']}"
        )


def cmd_tune(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    paths = expand_paths(args.jsonl)
    markets = load_markets(paths)
    grid = load_grid(args.grid_file)
    tuned = tune_strategy(
        engine=engine,
        strategy_file=args.strategy_file,
        markets=markets,
        param_grid=grid,
        objective=args.objective,
        top_k=args.top_k,
        max_combos=(args.max_combos if args.max_combos > 0 else None),
        progress_every=args.progress_every,
    )
    print(f"strategy_file: {args.strategy_file}")
    print(f"files: {', '.join(paths)}")
    print(f"objective: {tuned['objective']}")
    print(f"combos: {tuned['total_combos']}")
    print("best:")
    if tuned["best"] is None:
        print("  none")
        return
    print(f"  params: {json.dumps(tuned['best']['params'], sort_keys=True)}")
    best_stats = tuned["best"]["stats"]
    print(
        f"  total_pnl={best_stats['total_pnl']:.4f} avg_pnl={best_stats['avg_pnl']:.4f} "
        f"win_rate={best_stats['win_rate']:.2f}% pf={best_stats['profit_factor']}"
    )
    print("top:")
    for row in tuned["top"]:
        st = row["stats"]
        print(
            f"  {json.dumps(row['params'], sort_keys=True)} -> "
            f"total_pnl={st['total_pnl']:.4f}, avg_pnl={st['avg_pnl']:.4f}, "
            f"win_rate={st['win_rate']:.2f}%, mdd={st['max_drawdown']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "eval-all":
        cmd_eval_all(args)
    elif args.cmd == "tune":
        cmd_tune(args)


if __name__ == "__main__":
    main()
