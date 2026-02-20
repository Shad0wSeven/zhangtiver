from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class ActivePairScalperStrategy:
    """
    Trades the UP+DOWN pair around temporary mispricing.

    - Enters both legs when combined mid-price is cheap enough.
    - Exits both legs on reversion, timeout, or stop.
    - Can re-enter multiple times per market (cooldown-gated).
    """

    name = "Active Pair Scalper"
    color = "#00c2a8"
    required_history = 40

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.entry_sum_max = float(params.get("entry_sum_max", 0.992))
        self.exit_sum_min = float(params.get("exit_sum_min", 0.998))
        self.stop_sum_max = float(params.get("stop_sum_max", 0.975))
        self.max_spread_sum = float(params.get("max_spread_sum", 0.05))
        self.max_mid_imbalance = float(params.get("max_mid_imbalance", 0.18))
        self.jump_cap = float(params.get("jump_cap", 0.02))

        self.per_leg_usd = float(params.get("per_leg_usd", 20.0))
        self.max_hold_s = int(params.get("max_hold_s", 10))
        self.cooldown_s = int(params.get("cooldown_s", 1))
        self.window_low_s = int(params.get("window_low_s", 5))
        self.window_high_s = int(params.get("window_high_s", 140))

        self.recent_sum_n = int(params.get("recent_sum_n", 12))
        self.entry_z_min = float(params.get("entry_z_min", 0.2))

        self.entered = False
        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.sum_series: deque[float] = deque(maxlen=200)
        self.last_up_mid = 0.0

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.sum_series.clear()
        self.last_up_mid = 0.0

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) < 2:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, var**0.5

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None

        time_remaining = tick.time_remaining
        if not (self.window_low_s <= time_remaining <= self.window_high_s):
            return None

        sum_mid = tick.up_mid + tick.down_mid
        spread_sum = (tick.up_ask - tick.up_bid) + (tick.down_ask - tick.down_bid)
        mid_imbalance = abs(tick.up_mid - tick.down_mid)
        self.sum_series.append(sum_mid)
        if self.last_up_mid > 0:
            jump = abs((tick.up_mid - self.last_up_mid) / self.last_up_mid)
            if jump > self.jump_cap:
                self.last_up_mid = tick.up_mid
                return None
        self.last_up_mid = tick.up_mid

        up_pos = ctx.position("up")
        down_pos = ctx.position("down")
        have_pair = up_pos > 1e-9 and down_pos > 1e-9

        if not have_pair:
            self.entered = False

        if have_pair:
            held_s = max(0, tick.timestamp - self.entry_ts)
            should_exit = (
                sum_mid >= self.exit_sum_min
                or sum_mid <= self.stop_sum_max
                or held_s >= self.max_hold_s
            )
            if should_exit:
                self.last_exit_ts = tick.timestamp
                self.entered = False
                return [
                    Action(
                        side="sell",
                        token="up",
                        size=1.0,
                        comment=f"APS exit s={sum_mid:.3f}",
                    ),
                    Action(
                        side="sell",
                        token="down",
                        size=1.0,
                        comment=f"APS exit s={sum_mid:.3f}",
                    ),
                ]
            return None

        if self.entered:
            return None
        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None
        if spread_sum > self.max_spread_sum:
            return None
        if mid_imbalance > self.max_mid_imbalance:
            return None

        recent = list(self.sum_series)[-max(2, self.recent_sum_n) :]
        mean, std = self._mean_std(recent)
        z = ((mean - sum_mid) / std) if std > 1e-9 else 0.0
        if sum_mid <= self.entry_sum_max and z >= self.entry_z_min:
            self.entered = True
            self.entry_ts = tick.timestamp
            return [
                Action(
                    side="buy",
                    token="up",
                    size=self.per_leg_usd,
                    comment=f"APS in z={z:.2f}",
                ),
                Action(
                    side="buy",
                    token="down",
                    size=self.per_leg_usd,
                    comment=f"APS in z={z:.2f}",
                ),
            ]
        if sum_mid <= (self.entry_sum_max - 0.003):
            self.entered = True
            self.entry_ts = tick.timestamp
            return [
                Action(
                    side="buy",
                    token="up",
                    size=self.per_leg_usd,
                    comment=f"APS in deep s={sum_mid:.3f}",
                ),
                Action(
                    side="buy",
                    token="down",
                    size=self.per_leg_usd,
                    comment=f"APS in deep s={sum_mid:.3f}",
                ),
            ]
        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "APS done"


def create_strategy(params: dict | None = None) -> ActivePairScalperStrategy:
    return ActivePairScalperStrategy(params=params)
