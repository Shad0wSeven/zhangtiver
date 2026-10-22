from __future__ import annotations

from livetrader.strategy_api import Action, StrategyContext


class AlwaysDownTPSLStrategy:
    """
    Baseline directional strategy:
    - Buys DOWN once in a configured late window.
    - Exits using take-profit / stop-loss / trailing-drop / timeout.
    """

    name = "Always Down TP/SL"
    color = "#ff4d6d"
    required_history = 1

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        # Broader entry defaults to increase market participation.
        self.entry_window_low_s = int(params.get("entry_window_low_s", 5))
        self.entry_window_high_s = int(params.get("entry_window_high_s", 60))
        self.entry_price_max = float(params.get("entry_price_max", 0.95))
        self.sum_asks_max = float(params.get("sum_asks_max", 1.00))
        self.bet_usd = float(params.get("bet_usd", 10.0))

        # Preferred exit: sell in high-confidence zone; can scale up with entry price.
        self.take_profit_price = float(params.get("take_profit_price", 0.90))
        self.take_profit_price_cap = float(params.get("take_profit_price_cap", 0.99))
        self.take_profit_entry_slope = float(params.get("take_profit_entry_slope", 0.8))
        self.take_profit_min_gain_cents = float(
            params.get("take_profit_min_gain_cents", 0.01)
        )
        # Optional extra TP by absolute gain from entry; disabled by default.
        self.take_profit_cents = float(params.get("take_profit_cents", 0.0))
        # Optional SL and trailing drop; disabled by default.
        self.stop_loss_cents = float(params.get("stop_loss_cents", 0.0))
        self.trailing_drop_cents = float(params.get("trailing_drop_cents", 0.0))
        self.min_hold_s = int(params.get("min_hold_s", 1))
        self.max_hold_s = int(params.get("max_hold_s", 120))
        self.force_exit_s = int(params.get("force_exit_s", 2))
        self.max_entries_per_market = int(params.get("max_entries_per_market", 1))

        self.entered = False
        self.entries_count = 0
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.entries_count = 0
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    def _dynamic_take_profit_price(self) -> float:
        # Raise TP above floor for higher entry prices to avoid exiting too early
        # on already-expensive entries.
        floor = self.take_profit_price
        excess = max(0.0, self.entry_price - floor)
        scaled = floor + self.take_profit_entry_slope * excess
        min_gain = self.entry_price + self.take_profit_min_gain_cents
        return min(self.take_profit_price_cap, max(floor, scaled, min_gain))

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None

        down_pos = ctx.position("down")
        if down_pos > 1e-9 and self.entered:
            px = tick.down_mid
            if px > self.peak_price:
                self.peak_price = px

            gain = px - self.entry_price
            loss = self.entry_price - px
            drop_from_peak = self.peak_price - px
            held_s = max(0, tick.timestamp - self.entry_ts)
            tp_px = self._dynamic_take_profit_price()

            if px >= tp_px:
                self.entered = False
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"ADTS tp_px {px:.3f}>={tp_px:.3f}",
                )
            if tick.time_remaining <= self.force_exit_s:
                self.entered = False
                return Action(side="sell", token="down", size=1.0, comment="ADTS time")
            if (
                self.take_profit_cents > 0
                and held_s >= self.min_hold_s
                and gain >= self.take_profit_cents
            ):
                self.entered = False
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"ADTS tp +{gain:.3f}",
                )
            if (
                self.stop_loss_cents > 0
                and held_s >= self.min_hold_s
                and loss >= self.stop_loss_cents
            ):
                self.entered = False
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"ADTS sl -{loss:.3f}",
                )
            if (
                self.trailing_drop_cents > 0
                and held_s >= self.min_hold_s
                and drop_from_peak >= self.trailing_drop_cents
            ):
                self.entered = False
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"ADTS trail -{drop_from_peak:.3f}",
                )
            if held_s >= self.max_hold_s:
                self.entered = False
                return Action(
                    side="sell",
                    token="down",
                    size=1.0,
                    comment=f"ADTS max_hold {held_s}s",
                )
            return None

        if self.entered:
            return None
        if self.entries_count >= self.max_entries_per_market:
            return None

        tr = tick.time_remaining
        if not (self.entry_window_low_s <= tr <= self.entry_window_high_s):
            return None
        if tick.down_ask > self.entry_price_max:
            return None
        if tick.up_ask + tick.down_ask > self.sum_asks_max:
            return None

        self.entered = True
        self.entries_count += 1
        self.entry_price = tick.down_ask
        self.peak_price = self.entry_price
        self.entry_ts = tick.timestamp
        return Action(
            side="buy",
            token="down",
            size=self.bet_usd,
            comment=f"ADTS in tr={tr} ask={tick.down_ask:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "ADTS done"


def create_strategy(params: dict | None = None) -> AlwaysDownTPSLStrategy:
    return AlwaysDownTPSLStrategy(params=params)
