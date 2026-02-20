from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class AggressiveReversalStrategy:
    name = "Aggressive Reversal"
    color = "#d946ef"
    required_history = 120

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.extreme_gap = float(params.get("extreme_gap", 0.015))
        self.min_reversal = float(params.get("min_reversal", 0.003))
        self.vol_cap = float(params.get("vol_cap_60s", 0.05))
        self.bet_pct = float(params.get("bet_pct", 0.22))
        self.window_low = int(params.get("window_low", 6))
        self.window_high = int(params.get("window_high", 80))
        self.max_spread = float(params.get("max_spread", 0.03))
        self.jump_cap = float(params.get("jump_cap", 0.025))
        self.tp_move = float(params.get("tp_move", 0.012))
        self.sl_move = float(params.get("sl_move", 0.009))
        self.max_hold_s = int(params.get("max_hold_s", 14))
        self.cooldown_s = int(params.get("cooldown_s", 2))
        self.side: str | None = None
        self.entry_mid = 0.0
        self.entry_ts = 0
        self.last_exit_ts = -10_000

        self.last_ts_ms = 0
        self.mids: deque[float] = deque(maxlen=260)
        self.rets: deque[float] = deque(maxlen=260)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.side = None
        self.entry_mid = 0.0
        self.entry_ts = 0
        self.last_exit_ts = -10_000
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
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        if up_spread > self.max_spread or down_spread > self.max_spread:
            return None
        if self.rets and abs(self.rets[-1]) > self.jump_cap:
            return None

        short = self._avg(6)
        long = self._avg(24)
        gap = short - long
        up_pos = ctx.position("up")
        down_pos = ctx.position("down")

        if self.side == "up" and up_pos > 1e-9:
            move = tick.up_mid - self.entry_mid
            held_s = max(0, tick.timestamp - self.entry_ts)
            if (
                move >= self.tp_move
                or move <= -self.sl_move
                or held_s >= self.max_hold_s
                or gap >= 0
            ):
                self.side = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token="up",
                    size=1.0,
                    comment=f"AR up exit m={move:.3f}",
                )
            return None
        if self.side == "down" and down_pos > 1e-9:
            move = tick.down_mid - self.entry_mid
            held_s = max(0, tick.timestamp - self.entry_ts)
            if (
                move >= self.tp_move
                or move <= -self.sl_move
                or held_s >= self.max_hold_s
                or gap <= 0
            ):
                self.side = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"AR down exit m={move:.3f}",
                )
            return None
        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None

        # Reversal confirmation by last two returns turning opposite.
        if len(self.rets) < 3:
            return None
        r1, r2 = self.rets[-2], self.rets[-1]

        if (
            gap >= self.extreme_gap
            and r1 < -self.min_reversal
            and r2 < -self.min_reversal
        ):
            self.side = "down"
            self.entry_mid = tick.down_mid
            self.entry_ts = tick.timestamp
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
            self.side = "up"
            self.entry_mid = tick.up_mid
            self.entry_ts = tick.timestamp
            return Action(
                side="buy",
                token="up",
                size=self.bet_pct,
                comment=f"AR up gap={gap:.3f}",
            )

        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"AggRev winner={winner_token}"


def create_strategy(params: dict | None = None) -> AggressiveReversalStrategy:
    return AggressiveReversalStrategy(params=params)
