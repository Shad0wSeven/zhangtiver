from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_strategies(strategy_dir: str) -> tuple[list[Any], list[str]]:
    path = Path(strategy_dir)
    loaded: list[Any] = []
    errors: list[str] = []

    if not path.exists():
        return [], [f"Strategy directory not found: {path}"]

    for file_path in sorted(path.glob("*.py")):
        if file_path.name.startswith("_"):
            continue

        module_name = f"livetrader_user_{file_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                errors.append(f"{file_path.name}: failed to load module spec")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            create_strategy = getattr(module, "create_strategy", None)
            if create_strategy is None:
                errors.append(f"{file_path.name}: missing create_strategy()")
                continue

            strategy = create_strategy()
            missing = [
                attr
                for attr in ("name", "color", "required_history", "on_tick")
                if not hasattr(strategy, attr)
            ]
            if missing:
                errors.append(f"{file_path.name}: missing attributes {missing}")
                continue

            loaded.append(strategy)
        except Exception as exc:
            errors.append(f"{file_path.name}: {exc}")

    return loaded, errors
