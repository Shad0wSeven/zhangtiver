from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class ActiveMomentumReversalStrategy:
    """
    Active directional trader with explicit exits and re-entry.

    - Enters on short-vs-long momentum spread + book strength.
    - Exits on take-profit, stop-loss, timeout, or momentum flip.
    - Re-enters after cooldown to keep trade count high.
    """

    name = "Active Momentum Reversal"
    color = "#ff8a00"
    required_history = 50

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.short_n = int(params.get("short_n", 6))
        self.long_n = int(params.get("long_n", 22))
        self.entry_gap = float(params.get("entry_gap", 0.004))
        self.flip_gap = float(params.get("flip_gap", 0.0015))
        self.strength_min = float(params.get("strength_min", 0.006))
        self.spread_cap = float(params.get("spread_cap", 0.045))

        self.bet_pct = float(params.get("bet_pct", 0.30))
        self.tp_move = float(params.get("tp_move", 0.012))
        self.sl_move = float(params.get("sl_move", 0.009))
        self.max_hold_s = int(params.get("max_hold_s", 18))
        self.cooldown_s = int(params.get("cooldown_s", 2))

        self.window_low_s = int(params.get("window_low_s", 5))
        self.window_high_s = int(params.get("window_high_s", 95))

        self.mids: deque[float] = deque(maxlen=240)
        self.side: str | None = None
        self.entry_mid = 0.0
        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.entered_once = False

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.mids.clear()
        self.side = None
        self.entry_mid = 0.0
        self.entry_ts = 0
        self.last_exit_ts = -10_000
        self.entered_once = False

    @staticmethod
    def _avg(values: deque[float], n: int) -> float:
        if n <= 0 or len(values) < n:
            return 0.0
        arr = list(values)[-n:]
        return sum(arr) / len(arr)

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None
        self.mids.append(tick.up_mid)

        if not (self.window_low_s <= tick.time_remaining <= self.window_high_s):
            return None
        if len(self.mids) < max(self.long_n, 10):
            return None

        short = self._avg(self.mids, self.short_n)
        long = self._avg(self.mids, self.long_n)
        gap = short - long

        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        if up_spread > self.spread_cap or down_spread > self.spread_cap:
            return None

        up_strength = tick.up_bid - tick.down_bid
        down_strength = tick.down_bid - tick.up_bid
        up_pos = ctx.position("up")
        down_pos = ctx.position("down")

        if self.side == "up" and up_pos > 1e-9:
            move = tick.up_mid - self.entry_mid
            held_s = max(0, tick.timestamp - self.entry_ts)
            if (
                move >= self.tp_move
                or move <= -self.sl_move
                or held_s >= self.max_hold_s
                or gap <= -self.flip_gap
            ):
                self.side = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token="up",
                    size=1.0,
                    comment=f"AMR up exit m={move:.3f}",
                )
            return None

        if self.side == "down" and down_pos > 1e-9:
            move = tick.down_mid - self.entry_mid
            held_s = max(0, tick.timestamp - self.entry_ts)
            if (
                move >= self.tp_move
                or move <= -self.sl_move
                or held_s >= self.max_hold_s
                or gap >= self.flip_gap
            ):
                self.side = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"AMR down exit m={move:.3f}",
                )
            return None

        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None

        if gap >= self.entry_gap and up_strength >= self.strength_min:
            self.side = "up"
            self.entry_mid = tick.up_mid
            self.entry_ts = tick.timestamp
            self.entered_once = True
            return Action(
                side="buy",
                token="up",
                size=self.bet_pct,
                comment=f"AMR up in g={gap:.3f}",
            )

        if gap <= -self.entry_gap and down_strength >= self.strength_min:
            self.side = "down"
            self.entry_mid = tick.down_mid
            self.entry_ts = tick.timestamp
            self.entered_once = True
            return Action(
                side="buy",
                token="down",
                size=self.bet_pct,
                comment=f"AMR down in g={gap:.3f}",
            )
        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "AMR done"


def create_strategy(params: dict | None = None) -> ActiveMomentumReversalStrategy:
    return ActiveMomentumReversalStrategy(params=params)
