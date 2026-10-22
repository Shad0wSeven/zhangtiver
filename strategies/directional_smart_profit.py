from __future__ import annotations

import math
from collections import deque

from livetrader.strategy_api import Action, StrategyContext


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


class DirectionalSmartProfitStrategy:
    """
    Directional EV strategy:
    - Scores UP and DOWN each tick using price edge + book strength + momentum.
    - Enters only when estimated EV and score clear thresholds.
    - Exits via dynamic take-profit, optional stop/trail, timeout, or signal flip.
    """

    name = "Directional Smart Profit"
    color = "#ff7f11"
    required_history = 40

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}

        self.settle_price = float(params.get("settle_price", 0.99))
        self.bet_usd = float(params.get("bet_usd", 10.0))

        self.entry_window_low_s = int(params.get("entry_window_low_s", 5))
        self.entry_window_high_s = int(params.get("entry_window_high_s", 70))
        self.max_entries_per_market = int(params.get("max_entries_per_market", 1))

        self.max_price = float(params.get("max_price", 0.96))
        self.max_sum_asks = float(params.get("max_sum_asks", 1.03))
        self.max_spread = float(params.get("max_spread", 0.05))

        self.short_n = int(params.get("short_n", 6))
        self.long_n = int(params.get("long_n", 24))

        self.w_edge = float(params.get("w_edge", 1.6))
        self.w_strength = float(params.get("w_strength", 2.0))
        self.w_momentum = float(params.get("w_momentum", 2.2))
        self.w_spread = float(params.get("w_spread", 1.2))
        self.w_overround = float(params.get("w_overround", 1.5))

        self.prob_scale = float(params.get("prob_scale", 4.0))
        self.prob_bias = float(params.get("prob_bias", 0.0))
        self.min_score = float(params.get("min_score", 0.01))
        self.ev_threshold = float(params.get("ev_threshold", 0.0))

        self.take_profit_price = float(params.get("take_profit_price", 0.90))
        self.take_profit_price_cap = float(params.get("take_profit_price_cap", 0.99))
        self.take_profit_entry_slope = float(params.get("take_profit_entry_slope", 0.9))
        self.take_profit_min_gain_cents = float(
            params.get("take_profit_min_gain_cents", 0.01)
        )
        self.stop_loss_cents = float(params.get("stop_loss_cents", 0.0))
        self.trailing_drop_cents = float(params.get("trailing_drop_cents", 0.0))
        self.min_hold_s = int(params.get("min_hold_s", 1))
        self.max_hold_s = int(params.get("max_hold_s", 90))
        self.force_exit_s = int(params.get("force_exit_s", 2))
        self.flip_exit_score_gap = float(params.get("flip_exit_score_gap", 0.03))

        self.up_hist: deque[float] = deque(maxlen=400)
        self.down_hist: deque[float] = deque(maxlen=400)

        self.entries_count = 0
        self.active_token: str | None = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.up_hist.clear()
        self.down_hist.clear()
        self.entries_count = 0
        self.active_token = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    @staticmethod
    def _avg(values: deque[float], n: int) -> float:
        if n <= 0 or len(values) < n:
            return 0.0
        arr = list(values)[-n:]
        return sum(arr) / len(arr)

    def _token_score(
        self,
        token: str,
        up_ask: float,
        down_ask: float,
        up_bid: float,
        down_bid: float,
        up_spread: float,
        down_spread: float,
    ) -> float:
        if token == "up":
            ask = up_ask
            spread = up_spread
            strength = up_bid - down_bid
            short = self._avg(self.up_hist, self.short_n)
            long = self._avg(self.up_hist, self.long_n)
        else:
            ask = down_ask
            spread = down_spread
            strength = down_bid - up_bid
            short = self._avg(self.down_hist, self.short_n)
            long = self._avg(self.down_hist, self.long_n)

        edge = self.settle_price - ask
        momentum = short - long
        overround = max(0.0, (up_ask + down_ask) - 1.0)
        return (
            self.w_edge * edge
            + self.w_strength * strength
            + self.w_momentum * momentum
            - self.w_spread * spread
            - self.w_overround * overround
        )

    def _score_to_prob(self, score: float) -> float:
        return _sigmoid(self.prob_bias + self.prob_scale * score)

    def _ev(self, p_win: float, ask: float) -> float:
        if ask <= 0 or ask >= self.settle_price:
            return -1.0
        return p_win * ((self.settle_price / ask) - 1.0) + (1.0 - p_win) * (-1.0)

    def _dynamic_take_profit_price(self) -> float:
        floor = self.take_profit_price
        excess = max(0.0, self.entry_price - floor)
        scaled = floor + self.take_profit_entry_slope * excess
        min_gain = self.entry_price + self.take_profit_min_gain_cents
        return min(self.take_profit_price_cap, max(floor, scaled, min_gain))

    def _current_scores(self, tick) -> tuple[float, float]:
        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        up_score = self._token_score(
            "up",
            tick.up_ask,
            tick.down_ask,
            tick.up_bid,
            tick.down_bid,
            up_spread,
            down_spread,
        )
        down_score = self._token_score(
            "down",
            tick.up_ask,
            tick.down_ask,
            tick.up_bid,
            tick.down_bid,
            up_spread,
            down_spread,
        )
        return up_score, down_score

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None

        self.up_hist.append(tick.up_mid)
        self.down_hist.append(tick.down_mid)

        # Manage open position.
        if self.active_token is not None:
            token = self.active_token
            pos = ctx.position(token)
            if pos <= 1e-9:
                self.active_token = None
                self.entry_price = 0.0
                self.peak_price = 0.0
                self.entry_ts = 0
                return None

            px = tick.up_mid if token == "up" else tick.down_mid
            if px > self.peak_price:
                self.peak_price = px
            gain = px - self.entry_price
            loss = self.entry_price - px
            drop_from_peak = self.peak_price - px
            held_s = max(0, tick.timestamp - self.entry_ts)
            tp_px = self._dynamic_take_profit_price()

            if px >= tp_px:
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP tp")
            if tick.time_remaining <= self.force_exit_s:
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP time")
            if (
                self.stop_loss_cents > 0
                and held_s >= self.min_hold_s
                and loss >= self.stop_loss_cents
            ):
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP sl")
            if (
                self.trailing_drop_cents > 0
                and held_s >= self.min_hold_s
                and drop_from_peak >= self.trailing_drop_cents
            ):
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP trail")
            if held_s >= self.max_hold_s:
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP hold")

            up_score, down_score = self._current_scores(tick)
            if token == "up" and (down_score - up_score) >= self.flip_exit_score_gap:
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP flip")
            if token == "down" and (up_score - down_score) >= self.flip_exit_score_gap:
                self.active_token = None
                return Action(side="sell", token=token, size=1.0, comment="DSP flip")
            return None

        # Entry gating.
        if self.entries_count >= self.max_entries_per_market:
            return None
        if not (
            self.entry_window_low_s <= tick.time_remaining <= self.entry_window_high_s
        ):
            return None

        up_spread = tick.up_ask - tick.up_bid
        down_spread = tick.down_ask - tick.down_bid
        if up_spread > self.max_spread or down_spread > self.max_spread:
            return None
        if tick.up_ask + tick.down_ask > self.max_sum_asks:
            return None
        if len(self.up_hist) < max(self.long_n, 12):
            return None

        up_score, down_score = self._current_scores(tick)
        token = "up" if up_score >= down_score else "down"
        score = up_score if token == "up" else down_score
        ask = tick.up_ask if token == "up" else tick.down_ask
        if ask > self.max_price:
            return None
        if score < self.min_score:
            return None

        p_win = self._score_to_prob(score)
        ev = self._ev(p_win, ask)
        if ev < self.ev_threshold:
            return None

        self.entries_count += 1
        self.active_token = token
        self.entry_price = ask
        self.peak_price = ask
        self.entry_ts = tick.timestamp
        return Action(
            side="buy",
            token=token,
            size=self.bet_usd,
            comment=f"DSP in {token} s={score:.3f} ev={ev:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "DSP done"


def create_strategy(params: dict | None = None) -> DirectionalSmartProfitStrategy:
    return DirectionalSmartProfitStrategy(params=params)
