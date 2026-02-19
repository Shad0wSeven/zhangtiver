from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class AggressiveReversalStrategy:
    name = "Aggressive Reversal"
    color = "#d946ef"
    required_history = 120

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.extreme_gap = float(params.get("extreme_gap", 0.012))
        self.min_reversal = float(params.get("min_reversal", 0.004))
        self.vol_cap = float(params.get("vol_cap_60s", 0.10))
        self.bet_pct = float(params.get("bet_pct", 0.40))
        self.window_low = int(params.get("window_low", 4))
        self.window_high = int(params.get("window_high", 55))
        self.entered = False

        self.last_ts_ms = 0
        self.mids: deque[float] = deque(maxlen=260)
        self.rets: deque[float] = deque(maxlen=260)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.last_ts_ms = 0
        self.mids.clear()
        self.rets.clear()

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

        if not (self.window_low <= tick.time_remaining <= self.window_high):
            return None
        if len(self.mids) < 24:
            return None
        if self._vol(30) > self.vol_cap:
            return None

        short = self._avg(6)
        long = self._avg(24)
        gap = short - long

        # Reversal confirmation by last two returns turning opposite.
        if len(self.rets) < 3:
            return None
        r1, r2 = self.rets[-2], self.rets[-1]

        if (
            gap >= self.extreme_gap
            and r1 < -self.min_reversal
            and r2 < -self.min_reversal
        ):
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=self.bet_pct,
                comment=f"AR down gap={gap:.3f}",
            )

        if (
            gap <= -self.extreme_gap
            and r1 > self.min_reversal
            and r2 > self.min_reversal
        ):
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=self.bet_pct,
                comment=f"AR up gap={gap:.3f}",
            )

        if tick.time_remaining <= 6 and tick.up_bid >= 0.97:
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=max(self.bet_pct, 0.60),
                comment="AR hard edge",
            )
        if tick.time_remaining <= 6 and tick.down_bid >= 0.97:
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=max(self.bet_pct, 0.60),
                comment="AR hard edge",
            )

        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"AggRev winner={winner_token}"


def create_strategy(params: dict | None = None) -> AggressiveReversalStrategy:
    return AggressiveReversalStrategy(params=params)
