from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

from livetrader.strategy_api import Action, StrategyContext

SETTLE_PRICE = 0.99


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    sx = n * (n - 1) / 2.0
    sy = sum(values)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * values[i] for i in range(n))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / den


def _flip_rate(values: list[float]) -> float:
    prev = 0
    flips = 0
    count = 0
    for v in values:
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s == 0:
            continue
        count += 1
        if prev != 0 and s != prev:
            flips += 1
        prev = s
    if count <= 1:
        return 0.0
    return flips / (count - 1)


class MLLateProfitStrategy:
    name = "ML Late Profit"
    color = "#8a63ff"
    required_history = 80

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        model_file = params.get("model_file", "backtester/late_profit_model_best.json")
        model = json.load(open(Path(model_file), encoding="utf-8"))

        self.weights = [float(x) for x in model["weights"]]
        self.bias = float(model["bias"])
        self.means = [float(x) for x in model["means"]]
        self.stds = [float(x) for x in model["stds"]]

        cfg = model.get("config", {})
        self.entry_min_tr = int(
            params.get(
                "entry_min_time_remaining", cfg.get("entry_min_time_remaining", 8)
            )
        )
        self.entry_max_tr = int(
            params.get(
                "entry_max_time_remaining", cfg.get("entry_max_time_remaining", 70)
            )
        )
        self.min_price = float(params.get("min_price", cfg.get("min_price", 0.55)))
        self.max_price = float(params.get("max_price", cfg.get("max_price", 0.985)))

        self.ev_threshold = float(params.get("ev_threshold", 0.0))
        self.base_size = float(params.get("base_size", 0.15))
        # Exit controls to capture late peaks before reversals.
        self.take_profit_cents = float(params.get("take_profit_cents", 0.015))
        self.trailing_drop_cents = float(params.get("trailing_drop_cents", 0.008))
        self.min_hold_s = int(params.get("min_hold_s", 1))
        self.exit_time_cutoff_s = int(params.get("exit_time_cutoff_s", 2))
        # Flip-aware exit: if reversal risk rises while holding, cut early.
        self.flip_exit_enabled = bool(params.get("flip_exit_enabled", True))
        self.flip_exit_min_hold_s = int(params.get("flip_exit_min_hold_s", 1))
        self.flip_exit_prob_gap = float(params.get("flip_exit_prob_gap", 0.08))
        self.flip_exit_opp_prob_min = float(params.get("flip_exit_opp_prob_min", 0.58))
        self.flip_exit_ev_gap = float(params.get("flip_exit_ev_gap", 0.025))
        self.flip_exit_gap_reversal = float(params.get("flip_exit_gap_reversal", 0.0))
        self.flip_exit_slope_max = float(params.get("flip_exit_slope_max", -0.0008))
        self.flip_exit_chop_min = float(params.get("flip_exit_chop_min", 0.10))

        self.use_safe_gating = bool(params.get("use_safe_gating", True))
        self.spread_cap = float(params.get("spread_cap", 0.03))
        self.jump_cap = float(params.get("jump_cap", 0.02))
        self.vol_cap = float(params.get("vol_cap_30", 0.06))
        self.flip_rate_cap = float(params.get("flip_rate_cap_30", 0.30))
        # Integrate the "always down" edge as a soft prior in token ranking.
        self.always_down_bias_enabled = bool(
            params.get("always_down_bias_enabled", True)
        )
        self.always_down_time_low_s = int(params.get("always_down_time_low_s", 5))
        self.always_down_time_high_s = int(params.get("always_down_time_high_s", 60))
        self.always_down_price_max = float(params.get("always_down_price_max", 0.95))
        self.always_down_sum_asks_max = float(
            params.get("always_down_sum_asks_max", 1.00)
        )
        self.always_down_ev_bonus = float(params.get("always_down_ev_bonus", 0.008))

        self.entered = False
        self.active_token: str | None = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0
        self.last_ts_ms = 0
        self.up_hist: deque[float] = deque(maxlen=360)
        self.down_hist: deque[float] = deque(maxlen=360)
        self.gap_hist: deque[float] = deque(maxlen=360)
        self.up_rets: deque[float] = deque(maxlen=360)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.active_token = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0
        self.last_ts_ms = 0
        self.up_hist.clear()
        self.down_hist.clear()
        self.gap_hist.clear()
        self.up_rets.clear()

    def _update(self, tick) -> None:
        if tick.timestamp_ms == self.last_ts_ms:
            return
        self.last_ts_ms = tick.timestamp_ms

        if self.up_hist:
            prev = self.up_hist[-1]
            if prev > 1e-12:
                self.up_rets.append((tick.up_mid - prev) / prev)

        self.up_hist.append(tick.up_mid)
        self.down_hist.append(tick.down_mid)
        self.gap_hist.append(tick.up_mid - tick.down_mid)

    def _features(
        self, tick, token: str, enforce_price_bounds: bool = True
    ) -> list[float] | None:
        if len(self.up_hist) < 35 or len(self.down_hist) < 35:
            return None

        sign = 1.0 if token == "up" else -1.0
        price = tick.up_ask if token == "up" else tick.down_ask
        other_price = tick.down_ask if token == "up" else tick.up_ask
        if enforce_price_bounds and (price < self.min_price or price > self.max_price):
            return None

        token_spread = (
            (tick.up_ask - tick.up_bid)
            if token == "up"
            else (tick.down_ask - tick.down_bid)
        )
        other_spread = (
            (tick.down_ask - tick.down_bid)
            if token == "up"
            else (tick.up_ask - tick.up_bid)
        )

        oriented_gap = sign * self.gap_hist[-1]
        abs_gap = abs(self.gap_hist[-1])
        sum_asks = tick.up_ask + tick.down_ask
        time_frac = tick.time_remaining / 300.0

        tok_hist = list(self.up_hist if token == "up" else self.down_hist)
        short = tok_hist[-8:]
        long = tok_hist[-30:]
        token_mom = (sum(short) / len(short)) - (sum(long) / len(long))

        gap_window = [sign * g for g in list(self.gap_hist)[-30:]]
        gap_sl = _slope(gap_window)

        tok_w = tok_hist[-30:]
        tok_ret: list[float] = []
        for j in range(1, len(tok_w)):
            if tok_w[j - 1] > 1e-12:
                tok_ret.append((tok_w[j] - tok_w[j - 1]) / tok_w[j - 1])
        tok_vol = _std(tok_ret)
        gap_chop = _flip_rate(gap_window)

        btc_available = 0.0
        btc_drift_10 = 0.0
        btc_drift_30 = 0.0
        btc_vol_30 = 0.0
        btc_mom = 0.0
        btc_tok_corr = 0.0

        return [
            price,
            SETTLE_PRICE - price,
            price - other_price,
            token_spread,
            other_spread,
            oriented_gap,
            abs_gap,
            sum_asks,
            time_frac,
            token_mom,
            gap_sl,
            tok_vol,
            gap_chop,
            btc_available,
            btc_drift_10,
            btc_drift_30,
            btc_vol_30,
            btc_mom,
            btc_tok_corr,
        ]

    def _p_win(self, x: list[float]) -> float:
        z = self.bias
        for j, v in enumerate(x):
            z += self.weights[j] * ((v - self.means[j]) / self.stds[j])
        return _sigmoid(z)

    def _ev(self, p_win: float, price: float) -> float:
        return p_win * ((SETTLE_PRICE / price) - 1.0) + (1.0 - p_win) * (-1.0)

    def _flip_exit_reason(self, tick, token: str, held_s: int) -> str | None:
        if not self.flip_exit_enabled or held_s < self.flip_exit_min_hold_s:
            return None
        other = "down" if token == "up" else "up"
        x_tok = self._features(tick, token, enforce_price_bounds=False)
        x_oth = self._features(tick, other, enforce_price_bounds=False)
        if x_tok is None or x_oth is None:
            return None

        p_tok = self._p_win(x_tok)
        p_oth = self._p_win(x_oth)
        tok_price = tick.up_ask if token == "up" else tick.down_ask
        oth_price = tick.down_ask if token == "up" else tick.up_ask
        tok_ev = self._ev(p_tok, tok_price)
        oth_ev = self._ev(p_oth, oth_price)

        if (
            p_oth - p_tok
        ) >= self.flip_exit_prob_gap and p_oth >= self.flip_exit_opp_prob_min:
            return f"ML flip-prob p_opp={p_oth:.3f} p_tok={p_tok:.3f}"
        if (oth_ev - tok_ev) >= self.flip_exit_ev_gap:
            return f"ML flip-ev ev_opp={oth_ev:.3f} ev_tok={tok_ev:.3f}"

        if len(self.gap_hist) >= 12:
            sign = 1.0 if token == "up" else -1.0
            og = [sign * g for g in list(self.gap_hist)[-12:]]
            og_now = og[-1]
            og_slope = _slope(og)
            og_flip = _flip_rate(og)
            if (
                og_now <= self.flip_exit_gap_reversal
                and og_slope <= self.flip_exit_slope_max
                and og_flip >= self.flip_exit_chop_min
            ):
                return (
                    f"ML flip-shape gap={og_now:.3f} sl={og_slope:.4f} "
                    f"chop={og_flip:.2f}"
                )
        return None

    def on_tick(self, ctx: StrategyContext):
        tick = ctx.latest()
        if tick is None:
            return None
        self._update(tick)

        # Exit management for existing position.
        if self.active_token is not None:
            token = self.active_token
            pos_qty = ctx.position(token)
            if pos_qty <= 1e-9:
                self.active_token = None
                self.entry_price = 0.0
                self.peak_price = 0.0
                self.entry_ts = 0
                return None

            px = tick.up_mid if token == "up" else tick.down_mid
            if px > self.peak_price:
                self.peak_price = px

            gain = px - self.entry_price
            drop_from_peak = self.peak_price - px
            held_s = max(0, tick.timestamp - self.entry_ts)

            if tick.time_remaining <= self.exit_time_cutoff_s:
                self.active_token = None
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"ML time-exit tr={tick.time_remaining}",
                )

            flip_reason = self._flip_exit_reason(tick=tick, token=token, held_s=held_s)
            if flip_reason is not None:
                self.active_token = None
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=flip_reason,
                )

            if held_s >= self.min_hold_s and gain >= self.take_profit_cents:
                self.active_token = None
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"ML tp +{gain:.3f}",
                )

            if held_s >= self.min_hold_s and drop_from_peak >= self.trailing_drop_cents:
                self.active_token = None
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"ML trail drop={drop_from_peak:.3f}",
                )

            return None

        if self.entered:
            return None

        if not (self.entry_min_tr <= tick.time_remaining <= self.entry_max_tr):
            return None

        if self.use_safe_gating:
            up_spread = tick.up_ask - tick.up_bid
            down_spread = tick.down_ask - tick.down_bid
            if up_spread > self.spread_cap or down_spread > self.spread_cap:
                return None

            if self.up_rets:
                if abs(self.up_rets[-1]) > self.jump_cap:
                    return None
                if len(self.up_rets) >= 10:
                    vol = _std(list(self.up_rets)[-30:])
                    fr = _flip_rate(list(self.up_rets)[-30:])
                    if vol > self.vol_cap or fr > self.flip_rate_cap:
                        return None

        best = None
        down_bias_active = (
            self.always_down_bias_enabled
            and self.always_down_time_low_s
            <= tick.time_remaining
            <= self.always_down_time_high_s
            and tick.down_ask <= self.always_down_price_max
            and (tick.up_ask + tick.down_ask) <= self.always_down_sum_asks_max
        )
        for token in ("up", "down"):
            x = self._features(tick, token)
            if x is None:
                continue
            price = tick.up_ask if token == "up" else tick.down_ask
            p = self._p_win(x)
            ev = self._ev(p, price)
            score = ev + (
                self.always_down_ev_bonus
                if (down_bias_active and token == "down")
                else 0.0
            )
            if best is None or score > best["score"]:
                best = {"token": token, "p": p, "ev": ev, "score": score}

        if best is None or best["ev"] < self.ev_threshold:
            return None

        self.entered = True
        self.active_token = str(best["token"])
        self.entry_price = tick.up_ask if self.active_token == "up" else tick.down_ask
        self.peak_price = self.entry_price
        self.entry_ts = tick.timestamp
        return Action(
            side="buy",
            token=self.active_token,
            size=self.base_size,
            comment=f"ML p={best['p']:.3f} ev={best['ev']:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "ML done"


def create_strategy(params: dict | None = None) -> MLLateProfitStrategy:
    return MLLateProfitStrategy(params=params)
