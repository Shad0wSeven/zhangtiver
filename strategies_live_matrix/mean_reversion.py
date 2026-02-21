from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class TrendPullbackStrategy:
    """
    Volatility-filtered trend pullback strategy:
    - Uses SMA shape to detect trend.
    - Buys only on small pullbacks during trend, not at peak momentum.
    - Avoids volatile/flip-heavy regimes.
    """

    name = "Trend Pullback"
    color = "#f4b400"
    required_history = 160

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.short_n = int(params.get("short_n", 8))
        self.long_n = int(params.get("long_n", 36))
        self.trend_gap_min = float(params.get("trend_gap_min", 0.004))
        self.pullback_min = float(params.get("pullback_min", 0.002))

        self.bid_strength_min = float(params.get("bid_strength_min", 0.01))
        self.spread_cap = float(params.get("spread_cap", 0.025))
        self.entry_price_cap = float(params.get("entry_price_cap", 0.97))

        self.vol_cap_60s = float(params.get("vol_cap_60s", 0.045))
        self.flip_rate_cap_60s = float(params.get("flip_rate_cap_60s", 0.28))
        self.jump_cap = float(params.get("jump_cap", 0.02))
        self.min_points = int(params.get("min_points", 12))

        self.early_pct = float(params.get("early_pct", 0.25))
        self.late_pct = float(params.get("late_pct", 0.40))
        self.late_seconds = int(params.get("late_seconds", 14))

        self.entered = False
        self.last_ts_ms = 0
        self.mids: deque[float] = deque(maxlen=400)
        self.rets: deque[float] = deque(maxlen=400)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.last_ts_ms = 0
        self.mids.clear()
        self.rets.clear()

    @staticmethod
    def _avg(values: deque[float], n: int) -> float:
        if len(values) < n or n <= 0:
            return 0.0
        arr = list(values)[-n:]
        return sum(arr) / len(arr)

    @staticmethod
    def _std(values: deque[float], n: int) -> float:
        if len(values) < n or n <= 1:
            return 0.0
        arr = list(values)[-n:]
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return var**0.5

    @staticmethod
    def _flip_rate(values: deque[float], n: int) -> float:
        if len(values) < n or n <= 2:
            return 0.0
        arr = list(values)[-n:]
        prev_sign = 0
        flips = 0
        cnt = 0
        for v in arr:
            sign = 1 if v > 0 else (-1 if v < 0 else 0)
            if sign == 0:
                continue
            cnt += 1
            if prev_sign != 0 and sign != prev_sign:
                flips += 1
            prev_sign = sign
        return flips / max(1, cnt - 1)

    def update_series(self, tick) -> None:
        if tick.timestamp_ms == self.last_ts_ms:
            return
        self.last_ts_ms = tick.timestamp_ms
        mid = tick.up_mid
        if self.mids:
            prev = self.mids[-1]
            if prev > 0:
                self.rets.append((mid - prev) / prev)
        self.mids.append(mid)

    def regime_ok(self) -> bool:
        if len(self.mids) < self.min_points or len(self.rets) < self.min_points - 1:
            return False
        vol = self._std(self.rets, 30)
        flips = self._flip_rate(self.rets, 30)
        return vol <= self.vol_cap_60s and flips <= self.flip_rate_cap_60s

    def on_tick(self, ctx: StrategyContext):
        if self.entered:
            return None
        tick = ctx.latest()
        if tick is None:
            return None
        self.update_series(tick)
        if self.rets and abs(self.rets[-1]) > self.jump_cap:
            return None

        in_window = tick.time_remaining <= self.late_seconds or (
            20 <= tick.time_remaining <= 50
        )
        if not in_window or not self.regime_ok():
            return None

        sma_short = self._avg(self.mids, self.short_n)
        sma_long = self._avg(self.mids, self.long_n)
        trend_gap = sma_short - sma_long
        pullback = sma_short - tick.up_mid

        up_strength = tick.up_bid - tick.down_bid
        down_strength = tick.down_bid - tick.up_bid
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        size = (
            self.early_pct if tick.time_remaining > self.late_seconds else self.late_pct
        )

        if (
            trend_gap >= self.trend_gap_min
            and pullback >= self.pullback_min
            and up_strength >= self.bid_strength_min
            and up_spread <= self.spread_cap
            and tick.up_ask <= self.entry_price_cap
        ):
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=size,
                comment=f"TP up gap={trend_gap:.3f} pb={pullback:.3f}",
            )

        if (
            trend_gap <= -self.trend_gap_min
            and -pullback >= self.pullback_min
            and down_strength >= self.bid_strength_min
            and down_spread <= self.spread_cap
            and tick.down_ask <= self.entry_price_cap
        ):
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=size,
                comment=f"TP down gap={trend_gap:.3f} pb={pullback:.3f}",
            )

        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"TrendPullback winner={winner_token}"


def create_strategy(params: dict | None = None) -> TrendPullbackStrategy:
    return TrendPullbackStrategy(params=params)
