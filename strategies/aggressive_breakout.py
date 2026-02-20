from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class AggressiveBreakoutStrategy:
    name = "Aggressive Breakout"
    color = "#ff6a00"
    required_history = 120

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.bid_min = float(params.get("bid_min", 0.78))
        self.strength_min = float(params.get("strength_min", 0.02))
        self.mom_gap_min = float(params.get("mom_gap_min", 0.006))
        self.spread_cap = float(params.get("spread_cap", 0.025))
        self.vol_cap = float(params.get("vol_cap_60s", 0.05))
        self.jump_cap = float(params.get("jump_cap", 0.025))
        self.late_seconds = int(params.get("late_seconds", 30))
        self.bet_pct = float(params.get("bet_pct", 0.30))

        self.last_ts_ms = 0
        self.mids: deque[float] = deque(maxlen=220)
        self.rets: deque[float] = deque(maxlen=220)
        self.entered = False

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.last_ts_ms = 0
        self.mids.clear()
        self.rets.clear()
        self.entered = False

    def _avg(self, n: int) -> float:
        if len(self.mids) < n:
            return 0.0
        arr = list(self.mids)[-n:]
        return sum(arr) / len(arr)

    def _vol(self, n: int) -> float:
        if len(self.rets) < n:
            return 0.0
        arr = list(self.rets)[-n:]
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return var**0.5

    def update(self, tick) -> None:
        if tick.timestamp_ms == self.last_ts_ms:
            return
        self.last_ts_ms = tick.timestamp_ms
        if self.mids:
            prev = self.mids[-1]
            if prev > 0:
                self.rets.append((tick.up_mid - prev) / prev)
        self.mids.append(tick.up_mid)

    def on_tick(self, ctx: StrategyContext):
        if self.entered:
            return None
        tick = ctx.latest()
        if tick is None:
            return None
        self.update(tick)

        if tick.time_remaining > self.late_seconds:
            return None
        if len(self.mids) < 25:
            return None
        if self._vol(30) > self.vol_cap:
            return None
        if self.rets and abs(self.rets[-1]) > self.jump_cap:
            return None

        short = self._avg(8)
        long = self._avg(30)
        mom_gap = short - long
        up_strength = tick.up_bid - tick.down_bid
        down_strength = tick.down_bid - tick.up_bid
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid

        if (
            mom_gap >= self.mom_gap_min
            and tick.up_bid >= self.bid_min
            and up_strength >= self.strength_min
            and up_spread <= self.spread_cap
        ):
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=self.bet_pct,
                comment=f"AB up gap={mom_gap:.3f}",
            )

        if (
            mom_gap <= -self.mom_gap_min
            and tick.down_bid >= self.bid_min
            and down_strength >= self.strength_min
            and down_spread <= self.spread_cap
        ):
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=self.bet_pct,
                comment=f"AB down gap={mom_gap:.3f}",
            )

        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"AggBreak winner={winner_token}"


def create_strategy(params: dict | None = None) -> AggressiveBreakoutStrategy:
    return AggressiveBreakoutStrategy(params=params)
