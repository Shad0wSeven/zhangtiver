from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from backtester.engine import BacktestEngine
from backtester.metrics import summarize


def load_grid(grid_file: str) -> dict[str, list[Any]]:
    data = json.loads(Path(grid_file).read_text())
    params = data.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("grid json must contain object key: params")
    for key, values in params.items():
        if not isinstance(values, list) or len(values) == 0:
            raise RuntimeError(f"grid param {key} must be a non-empty list")
    return params


def iter_param_combos(param_grid: dict[str, list[Any]]):
    keys = sorted(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo)}


def split_markets_chronological(
    markets: dict[str, list], train_frac: float
) -> tuple[dict[str, list], dict[str, list]]:
    if not (0.0 < train_frac < 1.0):
        return dict(markets), {}
    ordered = sorted(
        markets.items(),
        key=lambda kv: (
            int(getattr(kv[1][0], "market_ts", 0)) if kv[1] else 0,
            kv[0],
        ),
    )
    cut = max(1, int(len(ordered) * train_frac))
    train_pairs = ordered[:cut]
    hold_pairs = ordered[cut:]
    if not hold_pairs:
        return dict(markets), {}
    return dict(train_pairs), dict(hold_pairs)


def tune_strategy(
    engine: BacktestEngine,
    strategy_file: str,
    markets: dict[str, list],
    param_grid: dict[str, list[Any]],
    objective: str = "total_pnl",
    top_k: int = 10,
    max_combos: int | None = None,
    progress_every: int = 0,
    holdout_markets: dict[str, list] | None = None,
    holdout_check_every: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_holdout_checked: dict[str, Any] | None = None
    for idx, params in enumerate(iter_param_combos(param_grid), start=1):
        if max_combos is not None and idx > max_combos:
            break
        run = engine.run_strategy(strategy_file, markets, params=params)
        stats = summarize(run)
        row: dict[str, Any] = {"params": params, "stats": stats}
        do_holdout = (
            bool(holdout_markets)
            and holdout_check_every > 0
            and (idx % holdout_check_every == 0)
        )
        if do_holdout and holdout_markets is not None:
            h_run = engine.run_strategy(strategy_file, holdout_markets, params=params)
            h_stats = summarize(h_run)
            row["holdout_stats"] = h_stats
            h_score = h_stats.get(objective, 0.0)
            if best_holdout_checked is None or h_score > best_holdout_checked[
                "holdout_stats"
            ].get(objective, 0.0):
                best_holdout_checked = row
        rows.append(row)
        if progress_every > 0 and idx % progress_every == 0:
            holdout_msg = ""
            if do_holdout and row.get("holdout_stats") is not None:
                holdout_msg = (
                    f" holdout_{objective}="
                    f"{row['holdout_stats'].get(objective, 0.0):.4f}"
                )
            print(
                f"[tune] {strategy_file} combos={idx} best_{objective}="
                f"{max((r['stats'].get(objective, 0.0) for r in rows), default=0.0):.4f}"
                f"{holdout_msg}",
                flush=True,
            )

    rows.sort(key=lambda x: x["stats"].get(objective, 0.0), reverse=True)
    return {
        "objective": objective,
        "best": rows[0] if rows else None,
        "top": rows[:top_k],
        "total_combos": len(rows),
        "holdout_checks": sum(1 for r in rows if r.get("holdout_stats") is not None),
        "best_holdout_checked": best_holdout_checked,
    }
