from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_CACHE: dict[str, ModuleType] = {}


def load_module(path: str) -> ModuleType:
    file_path = Path(path)
    cache_key = str(file_path.resolve())
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module_name = f"backtester_user_{file_path.stem}_{abs(hash(cache_key))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def instantiate_strategy(strategy_file: str, params: dict[str, Any] | None = None) -> Any:
    module = load_module(strategy_file)
    create_strategy = getattr(module, "create_strategy", None)
    if create_strategy is None:
        raise RuntimeError(f"{strategy_file}: missing create_strategy()")

    params = params or {}
    strategy = None
    try:
        sig = inspect.signature(create_strategy)
        if len(sig.parameters) == 0:
            strategy = create_strategy()
        else:
            strategy = create_strategy(params=params)
    except Exception:
        strategy = create_strategy()

    # For strategies that don't accept params in factory, apply attribute overrides.
    for key, value in params.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)

    for attr in ("name", "color", "required_history", "on_tick"):
        if not hasattr(strategy, attr):
            raise RuntimeError(f"{strategy_file}: strategy missing {attr}")
    return strategy
