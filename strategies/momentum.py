from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class LateDominanceStrategy:
    """
    Trend-shape strategy:
    - Uses SMA slope + acceleration + volatility/flip filters.
    - Skips highly unstable last-minute conditions.
    - Enters dominant side only if book structure and shape agree.
    """

    name = "Late Dominance"
    color = "#1e90ff"
    required_history = 140

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.bid_min = float(params.get("bid_min", 0.75))
        self.strength_min = float(params.get("strength_min", 0.01))
        self.spread_max = float(params.get("spread_max", 0.03))
        self.early_pct = float(params.get("early_pct", 0.25))
        self.late_pct = float(params.get("late_pct", 0.45))
        self.late_seconds = int(params.get("late_seconds", 20))

        self.vol_cap = float(params.get("vol_cap_60s", 0.07))
        self.flip_rate_cap = float(params.get("flip_rate_cap_60s", 0.35))
        self.momentum_gap_min = float(params.get("momentum_gap_min", 0.004))
        self.accel_min = float(params.get("accel_min", 0.0002))
        self.min_points = int(params.get("min_points", 12))

        self.entered = False
        self.last_ts_ms = 0
        self.mids: deque[float] = deque(maxlen=300)
        self.rets: deque[float] = deque(maxlen=300)
        self.mom_series: deque[float] = deque(maxlen=100)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.last_ts_ms = 0
        self.mids.clear()
        self.rets.clear()
        self.mom_series.clear()

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
        if len(self.mids) >= 30:
            short = self._avg(self.mids, 8)
            long = self._avg(self.mids, 30)
            self.mom_series.append(short - long)

    def regime_ok(self) -> bool:
        if len(self.mids) < self.min_points or len(self.rets) < self.min_points - 1:
            return False
        vol = self._std(self.rets, 30)
        flips = self._flip_rate(self.rets, 30)
        return vol <= self.vol_cap and flips <= self.flip_rate_cap

    def momentum_features(self) -> tuple[float, float]:
        if not self.mom_series:
            return 0.0, 0.0
        gap = self.mom_series[-1]
        accel = 0.0
        if len(self.mom_series) >= 2:
            accel = self.mom_series[-1] - self.mom_series[-2]
        return gap, accel

    def on_tick(self, ctx: StrategyContext):
        if self.entered:
            return None
        tick = ctx.latest()
        if tick is None:
            return None
        self.update_series(tick)

        in_window = tick.time_remaining <= self.late_seconds or (
            20 <= tick.time_remaining <= 50
        )
        if not in_window:
            return None
        if not self.regime_ok():
            return None

        gap, accel = self.momentum_features()
        if abs(gap) < self.momentum_gap_min:
            return None

        up_strength = tick.up_bid - tick.down_bid
        down_strength = tick.down_bid - tick.up_bid
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        size = (
            self.early_pct if tick.time_remaining > self.late_seconds else self.late_pct
        )

        # Late hard-dominance fallback when one side is already near-certain.
        if tick.time_remaining <= 8 and tick.up_bid >= 0.965 and up_spread <= 0.03:
            self.entered = True
            return Action(
                side="buy", token="up", size=max(size, 0.60), comment="DomUP hard edge"
            )
        if tick.time_remaining <= 8 and tick.down_bid >= 0.965 and down_spread <= 0.03:
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=max(size, 0.60),
                comment="DomDOWN hard edge",
            )

        if (
            gap > 0
            and accel >= -self.accel_min
            and tick.up_bid >= self.bid_min
            and up_strength >= self.strength_min
            and up_spread <= self.spread_max
        ):
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=size,
                comment=f"DomUP gap={gap:.3f} acc={accel:.3f}",
            )

        if (
            gap < 0
            and accel <= self.accel_min
            and tick.down_bid >= self.bid_min
            and down_strength >= self.strength_min
            and down_spread <= self.spread_max
        ):
            self.entered = True
            return Action(
                side="buy",
                token="down",
                size=size,
                comment=f"DomDOWN gap={gap:.3f} acc={accel:.3f}",
            )

        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"LateDom winner={winner_token}"


def create_strategy(params: dict | None = None) -> LateDominanceStrategy:
    return LateDominanceStrategy(params=params)
