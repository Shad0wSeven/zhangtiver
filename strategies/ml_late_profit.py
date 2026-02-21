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
        self.entry_min_tr = int(params.get("entry_min_time_remaining", cfg.get("entry_min_time_remaining", 8)))
        self.entry_max_tr = int(params.get("entry_max_time_remaining", cfg.get("entry_max_time_remaining", 70)))
        self.min_price = float(params.get("min_price", cfg.get("min_price", 0.55)))
        self.max_price = float(params.get("max_price", cfg.get("max_price", 0.985)))

        self.ev_threshold = float(params.get("ev_threshold", 0.01))
        self.base_size = float(params.get("base_size", 0.22))

        self.use_safe_gating = bool(params.get("use_safe_gating", True))
        self.spread_cap = float(params.get("spread_cap", 0.03))
        self.jump_cap = float(params.get("jump_cap", 0.03))
        self.vol_cap = float(params.get("vol_cap_30", 0.06))
        self.flip_rate_cap = float(params.get("flip_rate_cap_30", 0.45))

        self.entered = False
        self.last_ts_ms = 0
        self.up_hist: deque[float] = deque(maxlen=360)
        self.down_hist: deque[float] = deque(maxlen=360)
        self.gap_hist: deque[float] = deque(maxlen=360)
        self.up_rets: deque[float] = deque(maxlen=360)

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
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

    def _features(self, tick, token: str) -> list[float] | None:
        if len(self.up_hist) < 35 or len(self.down_hist) < 35:
            return None

        sign = 1.0 if token == "up" else -1.0
        price = tick.up_ask if token == "up" else tick.down_ask
        other_price = tick.down_ask if token == "up" else tick.up_ask
        if price < self.min_price or price > self.max_price:
            return None

        token_spread = (tick.up_ask - tick.up_bid) if token == "up" else (tick.down_ask - tick.down_bid)
        other_spread = (tick.down_ask - tick.down_bid) if token == "up" else (tick.up_ask - tick.up_bid)

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

    def on_tick(self, ctx: StrategyContext):
        if self.entered:
            return None

        tick = ctx.latest()
        if tick is None:
            return None
        self._update(tick)

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
        for token in ("up", "down"):
            x = self._features(tick, token)
            if x is None:
                continue
            price = tick.up_ask if token == "up" else tick.down_ask
            p = self._p_win(x)
            ev = self._ev(p, price)
            if best is None or ev > best["ev"]:
                best = {"token": token, "p": p, "ev": ev}

        if best is None or best["ev"] < self.ev_threshold:
            return None

        self.entered = True
        return Action(
            side="buy",
            token=str(best["token"]),
            size=self.base_size,
            comment=f"ML p={best['p']:.3f} ev={best['ev']:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        return "ML done"


def create_strategy(params: dict | None = None) -> MLLateProfitStrategy:
    return MLLateProfitStrategy(params=params)
