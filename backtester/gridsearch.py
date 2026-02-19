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


def tune_strategy(
    engine: BacktestEngine,
    strategy_file: str,
    markets: dict[str, list],
    param_grid: dict[str, list[Any]],
    objective: str = "total_pnl",
    top_k: int = 10,
    max_combos: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, params in enumerate(iter_param_combos(param_grid), start=1):
        if max_combos is not None and idx > max_combos:
            break
        run = engine.run_strategy(strategy_file, markets, params=params)
        stats = summarize(run)
        rows.append({"params": params, "stats": stats})
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"[tune] {strategy_file} combos={idx} best_{objective}="
                f"{max((r['stats'].get(objective, 0.0) for r in rows), default=0.0):.4f}",
                flush=True,
            )

    rows.sort(key=lambda x: x["stats"].get(objective, 0.0), reverse=True)
    return {
        "objective": objective,
        "best": rows[0] if rows else None,
        "top": rows[:top_k],
        "total_combos": len(rows),
    }
