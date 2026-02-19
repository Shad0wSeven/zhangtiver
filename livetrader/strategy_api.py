from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class Tick:
    timestamp: int
    timestamp_ms: int
    market_ts: int
    time_remaining: int
    up_bid: float
    up_ask: float
    up_mid: float
    down_bid: float
    down_ask: float
    down_mid: float


@dataclass
class Action:
    side: str
    token: str
    size: float = 1.0
    comment: str = ""


class Strategy(Protocol):
    name: str
    color: str
    required_history: int

    def on_tick(self, ctx: "StrategyContext") -> Action | list[Action] | None: ...

    def on_market_start(self, market_ts: int) -> None: ...

    def on_market_end(
        self, winner_token: str, ctx: "StrategyContext"
    ) -> str | None: ...


class StrategyContext:
    def __init__(
        self, history: Sequence[Tick], positions: dict[str, float], market_ts: int
    ):
        self._history = history
        self._positions = positions
        self.market_ts = market_ts

    def history(self, n: int | None = None) -> list[Tick]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history)[-n:]

    def latest(self) -> Tick | None:
        if not self._history:
            return None
        return self._history[-1]

    def position(self, token: str) -> float:
        return self._positions.get(token, 0.0)
