from __future__ import annotations

from collections import deque

from livetrader.strategy_api import Action, StrategyContext


class SupportResistanceInverseStrategy:
    """
    Identify rolling support/resistance from recent pivots and trade the inverse:
    - Near support -> buy DOWN (inverse of classic support bounce trade).
    - Near resistance -> buy UP (inverse of classic resistance rejection trade).
    """

    name = "Support/Resistance Inverse"
    color = "#ff9f1c"
    required_history = 140

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self.window_low_s = int(params.get("window_low_s", 8))
        self.window_high_s = int(params.get("window_high_s", 110))
        self.lookback = int(params.get("lookback", 70))
        self.pivot_span = int(params.get("pivot_span", 3))
        self.touch_eps = float(params.get("touch_eps", 0.004))
        self.min_level_gap = float(params.get("min_level_gap", 0.03))
        self.level_bin = float(params.get("level_bin", 0.004))
        self.max_price = float(params.get("max_price", 0.94))
        self.min_price = float(params.get("min_price", 0.03))
        self.base_size = float(params.get("base_size", 0.10))
        self.max_entries_per_market = int(params.get("max_entries_per_market", 1))
        self.cooldown_s = int(params.get("cooldown_s", 4))
        # Fee-aware entry controls:
        # - avoid the high-fee center region near 0.50
        # - cap entry fee rate directly
        self.min_abs_from_mid = float(params.get("min_abs_from_mid", 0.16))
        self.entry_fee_rate_cap = float(params.get("entry_fee_rate_cap", 0.009))

        self.take_profit_cents = float(params.get("take_profit_cents", 0.03))
        self.stop_loss_cents = float(params.get("stop_loss_cents", 0.015))
        self.trailing_drop_cents = float(params.get("trailing_drop_cents", 0.016))
        self.min_hold_s = int(params.get("min_hold_s", 5))
        # For non-urgent exits (tp/trailing), prefer lower-fee sell zones.
        self.exit_min_abs_from_mid = float(params.get("exit_min_abs_from_mid", 0.10))
        self.exit_fee_rate_cap = float(params.get("exit_fee_rate_cap", 0.013))
        self.exit_time_cutoff_s = int(params.get("exit_time_cutoff_s", 2))

        self.active_token: str | None = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0
        self.entries_count = 0
        self.last_exit_ts = -10_000

        self.up_hist: deque[float] = deque(maxlen=500)
        self.limit_debug = "warming"
        self._zone = "none"

    @staticmethod
    def _fee_rate(price: float) -> float:
        table = [
            (0.01, 0.0000),
            (0.05, 0.0006),
            (0.10, 0.0020),
            (0.15, 0.0041),
            (0.20, 0.0064),
            (0.25, 0.0088),
            (0.30, 0.0110),
            (0.35, 0.0129),
            (0.40, 0.0144),
            (0.45, 0.0153),
            (0.50, 0.0156),
            (0.55, 0.0153),
            (0.60, 0.0144),
            (0.65, 0.0129),
            (0.70, 0.0110),
            (0.75, 0.0088),
            (0.80, 0.0064),
            (0.85, 0.0041),
            (0.90, 0.0020),
            (0.95, 0.0006),
            (0.99, 0.0000),
        ]
        p = max(table[0][0], min(table[-1][0], float(price)))
        for i in range(1, len(table)):
            p0, r0 = table[i - 1]
            p1, r1 = table[i]
            if p <= p1:
                if p1 <= p0:
                    return r1
                t = (p - p0) / (p1 - p0)
                return r0 + t * (r1 - r0)
        return table[-1][1]

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.active_token = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0
        self.entries_count = 0
        self.last_exit_ts = -10_000
        self.up_hist.clear()
        self.limit_debug = "warming"
        self._zone = "none"

    def _find_levels(self) -> tuple[float | None, float | None]:
        vals = list(self.up_hist)
        n = len(vals)
        k = self.pivot_span
        if n < (2 * k + 10):
            return None, None

        piv_lows: list[float] = []
        piv_highs: list[float] = []
        for i in range(k, n - k):
            v = vals[i]
            left = vals[i - k : i]
            right = vals[i + 1 : i + 1 + k]
            if all(v <= x for x in left + right):
                piv_lows.append(v)
            if all(v >= x for x in left + right):
                piv_highs.append(v)

        if not piv_lows or not piv_highs:
            return None, None

        def cluster_center(points: list[float], is_low: bool) -> float:
            # bucket pivots; pick densest bucket then representative edge.
            buckets: dict[int, list[float]] = {}
            b = max(1e-6, self.level_bin)
            for p in points:
                idx = int(round(p / b))
                buckets.setdefault(idx, []).append(p)
            best = max(
                buckets.values(),
                key=lambda xs: (len(xs), -abs(sum(xs) / len(xs) - vals[-1])),
            )
            return min(best) if is_low else max(best)

        support = cluster_center(piv_lows, is_low=True)
        resistance = cluster_center(piv_highs, is_low=False)
        if resistance - support < self.min_level_gap:
            return None, None
        return support, resistance

    def _zone_and_limits(
        self, tick, support: float, resistance: float
    ) -> tuple[str, float, float]:
        up = tick.up_mid
        d_sup = abs(up - support)
        d_res = abs(up - resistance)
        zone = "none"
        if d_sup <= self.touch_eps and up <= resistance:
            zone = "near_support"
        elif d_res <= self.touch_eps and up >= support:
            zone = "near_resistance"

        # Inverse limits:
        # near support -> prefer DOWN entry; near resistance -> prefer UP entry.
        up_limit = min(self.max_price, max(self.min_price, tick.up_ask))
        down_limit = min(self.max_price, max(self.min_price, tick.down_ask))
        if zone == "near_support":
            down_limit = min(self.max_price, tick.down_ask + 0.01)
            up_limit = max(self.min_price, tick.up_ask - 0.01)
        elif zone == "near_resistance":
            up_limit = min(self.max_price, tick.up_ask + 0.01)
            down_limit = max(self.min_price, tick.down_ask - 0.01)
        return zone, up_limit, down_limit

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None

        self.up_hist.append(tick.up_mid)
        hist = list(self.up_hist)[-self.lookback :]
        if len(hist) >= 5:
            self.up_hist = deque(hist, maxlen=500)

        # Manage open position.
        if self.active_token is not None:
            token = self.active_token
            if ctx.position(token) <= 1e-9:
                self.active_token = None
                self.entry_price = 0.0
                self.peak_price = 0.0
                self.entry_ts = 0
                return None

            px = tick.up_mid if token == "up" else tick.down_mid
            if px > self.peak_price:
                self.peak_price = px
            held_s = max(0, tick.timestamp - self.entry_ts)
            gain = px - self.entry_price
            loss = self.entry_price - px
            drop = self.peak_price - px
            exit_fr = self._fee_rate(px)
            favorable_exit_fee = (
                abs(px - 0.5) >= self.exit_min_abs_from_mid
                and exit_fr <= self.exit_fee_rate_cap
            )

            if tick.time_remaining <= self.exit_time_cutoff_s:
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(side="sell", token=token, size=1.0, comment="SRI time")
            if (
                held_s >= self.min_hold_s
                and gain >= self.take_profit_cents
                and favorable_exit_fee
            ):
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"SRI tp +{gain:.3f} fr={exit_fr:.4f}",
                )
            if held_s >= self.min_hold_s and loss >= self.stop_loss_cents:
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell", token=token, size=1.0, comment=f"SRI sl -{loss:.3f}"
                )
            if (
                held_s >= self.min_hold_s
                and drop >= self.trailing_drop_cents
                and favorable_exit_fee
            ):
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"SRI trail -{drop:.3f} fr={exit_fr:.4f}",
                )
            return None

        if not (self.window_low_s <= tick.time_remaining <= self.window_high_s):
            return None
        if self.entries_count >= self.max_entries_per_market:
            return None
        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None

        support, resistance = self._find_levels()
        if support is None or resistance is None:
            self.limit_debug = "warming levels"
            return None

        zone, up_limit, down_limit = self._zone_and_limits(tick, support, resistance)
        self._zone = zone
        self.limit_debug = (
            f"S={support:.3f} R={resistance:.3f} zone={zone} "
            f"up_lim={up_limit:.3f} down_lim={down_limit:.3f}"
        )

        if zone == "near_support" and tick.down_ask <= down_limit:
            fr = self._fee_rate(tick.down_ask)
            if (
                abs(tick.down_ask - 0.5) < self.min_abs_from_mid
                or fr > self.entry_fee_rate_cap
            ):
                self.limit_debug += f" skip_fee down_fr={fr:.4f}"
                return None
            self.active_token = "down"
            self.entry_price = tick.down_ask
            self.peak_price = self.entry_price
            self.entry_ts = tick.timestamp
            self.entries_count += 1
            return Action(
                side="buy",
                token="down",
                size=self.base_size,
                comment=f"SRI inverse@support {self.limit_debug}",
            )
        if zone == "near_resistance" and tick.up_ask <= up_limit:
            fr = self._fee_rate(tick.up_ask)
            if (
                abs(tick.up_ask - 0.5) < self.min_abs_from_mid
                or fr > self.entry_fee_rate_cap
            ):
                self.limit_debug += f" skip_fee up_fr={fr:.4f}"
                return None
            self.active_token = "up"
            self.entry_price = tick.up_ask
            self.peak_price = self.entry_price
            self.entry_ts = tick.timestamp
            self.entries_count += 1
            return Action(
                side="buy",
                token="up",
                size=self.base_size,
                comment=f"SRI inverse@resistance {self.limit_debug}",
            )
        return None

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return f"SRI {self.limit_debug}"


def create_strategy(params: dict | None = None) -> SupportResistanceInverseStrategy:
    return SupportResistanceInverseStrategy(params=params)
