from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class LateFlipHunterStrategy:
    """
    Follow confirmed late flips:
    - Detect when book dominance flips sides within a short recent window.
    - Enter only after a brief confirmation streak on the new side.
    - Intended to run alongside trend/momentum systems to capture reversals.
    """

    name = "Late Flip Hunter"
    color = "#00b894"
    required_history = 90

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.window_low_s = int(params.get("window_low_s", 6))
        self.window_high_s = int(params.get("window_high_s", 45))
        self.flip_lookback = int(params.get("flip_lookback", 14))
        self.confirm_ticks = int(params.get("confirm_ticks", 2))

        self.entry_gap_min = float(params.get("entry_gap_min", 0.015))
        self.entry_bid_min = float(params.get("entry_bid_min", 0.62))
        self.spread_cap = float(params.get("spread_cap", 0.03))
        self.jump_cap = float(params.get("jump_cap", 0.03))
        self.bet_pct = float(params.get("bet_pct", 0.20))

        self.entered = False
        self.last_ts_ms = 0
        self.last_up_mid = 0.0
        self.signs: deque[int] = deque(maxlen=220)
        self.gaps: deque[float] = deque(maxlen=220)
        self.rets: deque[float] = deque(maxlen=220)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.last_ts_ms = 0
        self.last_up_mid = 0.0
        self.signs.clear()
        self.gaps.clear()
        self.rets.clear()

    def _update(self, tick) -> None:
        if tick.timestamp_ms == self.last_ts_ms:
            return
        self.last_ts_ms = tick.timestamp_ms

        gap = tick.up_mid - tick.down_mid
        sign = 1 if gap > 0 else (-1 if gap < 0 else 0)
        self.signs.append(sign)
        self.gaps.append(gap)

        if self.last_up_mid > 0:
            self.rets.append((tick.up_mid - self.last_up_mid) / self.last_up_mid)
        self.last_up_mid = tick.up_mid

    def _is_recent_flip_confirmed(self) -> int:
        if len(self.signs) < max(self.confirm_ticks + 1, 4):
            return 0
        curr = self.signs[-1]
        if curr == 0:
            return 0

        tail = list(self.signs)[-self.confirm_ticks :]
        if any(s != curr for s in tail):
            return 0

        look = list(self.signs)[-self.flip_lookback :]
        return curr if any(s == -curr for s in look) else 0

    def on_tick(self, ctx: StrategyContext):
        if self.entered:
            return None
        tick = ctx.latest()
        if tick is None:
            return None
        self._update(tick)

        if not (self.window_low_s <= tick.time_remaining <= self.window_high_s):
            return None
        if self.rets and abs(self.rets[-1]) > self.jump_cap:
            return None

        flip_side = self._is_recent_flip_confirmed()
        if flip_side == 0:
            return None

        gap = abs(self.gaps[-1]) if self.gaps else 0.0
        if gap < self.entry_gap_min:
            return None

        if flip_side > 0:
            spread = tick.up_ask - tick.up_bid
            if tick.up_bid < self.entry_bid_min or spread > self.spread_cap:
                return None
            self.entered = True
            return Action(
                side="buy",
                token="up",
                size=self.bet_pct,
                comment=f"LFH up flip gap={gap:.3f}",
            )

        spread = tick.down_ask - tick.down_bid
        if tick.down_bid < self.entry_bid_min or spread > self.spread_cap:
            return None
        self.entered = True
        return Action(
            side="buy",
            token="down",
            size=self.bet_pct,
            comment=f"LFH down flip gap={gap:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = ctx
        return f"LateFlip winner={winner_token}"


def create_strategy(params: dict | None = None) -> LateFlipHunterStrategy:
    return LateFlipHunterStrategy(params=params)
