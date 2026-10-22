from __future__ import annotations

from dataclasses import dataclass, field

from livetrader.strategy_api import Tick


@dataclass
class Trade:
    market: str
    strategy: str
    side: str
    token: str
    timestamp_ms: int
    price: float
    qty: float
    notional: float
    fee: float
    time_remaining: int
    comment: str = ""


@dataclass
class MarketResult:
    market: str
    winner: str
    pnl: float
    trades: int
    end_cash: float
    max_drawdown: float
    fees: float
    duration_s: int


@dataclass
class StrategyState:
    strategy_name: str
    start_cash: float
    cash: float
    positions: dict[str, float] = field(
        default_factory=lambda: {"up": 0.0, "down": 0.0}
    )
    costs: dict[str, float] = field(default_factory=lambda: {"up": 0.0, "down": 0.0})
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    total_fees: float = 0.0

    def mark_to_market(self, tick: Tick) -> float:
        return (
            self.cash
            + self.positions["up"] * tick.up_mid
            + self.positions["down"] * tick.down_mid
        )
