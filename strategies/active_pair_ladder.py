from __future__ import annotations

from livetrader.strategy_api import Action, StrategyContext


class ActivePairLadderStrategy:
    """
    Multi-tranche pair strategy for active trading.

    - Buys UP+DOWN in tiers as combined price gets cheaper.
    - Scales out in tiers as combined price mean-reverts.
    - Uses stop/timeout to avoid holding stale positions.
    """

    name = "Active Pair Ladder"
    color = "#2a9d8f"
    required_history = 30

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.entry_1 = float(params.get("entry_1", 0.955))
        self.entry_2 = float(params.get("entry_2", 0.935))
        self.entry_3 = float(params.get("entry_3", 0.915))
        self.exit_1 = float(params.get("exit_1", 0.975))
        self.exit_2 = float(params.get("exit_2", 0.985))
        self.stop_sum = float(params.get("stop_sum", 0.87))
        self.max_spread_sum = float(params.get("max_spread_sum", 0.10))
        self.max_mid_imbalance = float(params.get("max_mid_imbalance", 0.36))

        self.per_leg_usd = float(params.get("per_leg_usd", 24.0))
        self.max_hold_s = int(params.get("max_hold_s", 16))
        self.cooldown_s = int(params.get("cooldown_s", 1))
        self.window_low_s = int(params.get("window_low_s", 5))
        self.window_high_s = int(params.get("window_high_s", 125))

        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.tiers_open = 0
        self.scaled_1 = False

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.tiers_open = 0
        self.scaled_1 = False

    def _can_trade_window(self, t: int) -> bool:
        return self.window_low_s <= t <= self.window_high_s

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None
        if not self._can_trade_window(tick.time_remaining):
            return None

        sum_mid = tick.up_mid + tick.down_mid
        spread_sum = (tick.up_ask - tick.up_bid) + (tick.down_ask - tick.down_bid)
        mid_imbalance = abs(tick.up_mid - tick.down_mid)
        if spread_sum > self.max_spread_sum or mid_imbalance > self.max_mid_imbalance:
            return None

        up_pos = ctx.position("up")
        down_pos = ctx.position("down")
        have_pair = up_pos > 1e-9 and down_pos > 1e-9

        actions: list[Action] = []
        if have_pair:
            held_s = max(0, tick.timestamp - self.entry_ts)
            if sum_mid <= self.stop_sum or held_s >= self.max_hold_s:
                self.last_exit_ts = tick.timestamp
                self.tiers_open = 0
                self.scaled_1 = False
                return [
                    Action(
                        side="sell",
                        token="up",
                        size=1.0,
                        comment=f"APL stop s={sum_mid:.3f}",
                    ),
                    Action(
                        side="sell",
                        token="down",
                        size=1.0,
                        comment=f"APL stop s={sum_mid:.3f}",
                    ),
                ]

            if not self.scaled_1 and sum_mid >= self.exit_1:
                self.scaled_1 = True
                actions.append(
                    Action(
                        side="sell",
                        token="up",
                        size=0.5,
                        comment=f"APL tp1 s={sum_mid:.3f}",
                    )
                )
                actions.append(
                    Action(
                        side="sell",
                        token="down",
                        size=0.5,
                        comment=f"APL tp1 s={sum_mid:.3f}",
                    )
                )

            if sum_mid >= self.exit_2:
                self.last_exit_ts = tick.timestamp
                self.tiers_open = 0
                self.scaled_1 = False
                actions.append(
                    Action(
                        side="sell",
                        token="up",
                        size=1.0,
                        comment=f"APL tp2 s={sum_mid:.3f}",
                    )
                )
                actions.append(
                    Action(
                        side="sell",
                        token="down",
                        size=1.0,
                        comment=f"APL tp2 s={sum_mid:.3f}",
                    )
                )

            return actions or None

        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None

        # Entry ladder only when flat.
        if self.tiers_open == 0 and sum_mid <= self.entry_1:
            self.tiers_open = 1
            self.entry_ts = tick.timestamp
            self.scaled_1 = False
            return [
                Action(
                    side="buy",
                    token="up",
                    size=self.per_leg_usd,
                    comment=f"APL in1 s={sum_mid:.3f}",
                ),
                Action(
                    side="buy",
                    token="down",
                    size=self.per_leg_usd,
                    comment=f"APL in1 s={sum_mid:.3f}",
                ),
            ]
        if self.tiers_open == 1 and sum_mid <= self.entry_2:
            self.tiers_open = 2
            return [
                Action(
                    side="buy",
                    token="up",
                    size=self.per_leg_usd,
                    comment=f"APL in2 s={sum_mid:.3f}",
                ),
                Action(
                    side="buy",
                    token="down",
                    size=self.per_leg_usd,
                    comment=f"APL in2 s={sum_mid:.3f}",
                ),
            ]
        if self.tiers_open == 2 and sum_mid <= self.entry_3:
            self.tiers_open = 3
            return [
                Action(
                    side="buy",
                    token="up",
                    size=self.per_leg_usd,
                    comment=f"APL in3 s={sum_mid:.3f}",
                ),
                Action(
                    side="buy",
                    token="down",
                    size=self.per_leg_usd,
                    comment=f"APL in3 s={sum_mid:.3f}",
                ),
            ]
        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "APL done"


def create_strategy(params: dict | None = None) -> ActivePairLadderStrategy:
    return ActivePairLadderStrategy(params=params)
