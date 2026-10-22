from __future__ import annotations

import math
from pathlib import Path

from livetrader.strategy_api import Action, StrategyContext

SETTLE_PRICE = 0.99


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


class LSTMSafeEntryStrategy:
    name = "LSTM Safe Entry"
    color = "#118ab2"
    required_history = 40

    def __init__(self, params: dict | None = None) -> None:
        params = params or {}
        self._params = dict(params)
        self.model_file = str(params.get("model_file", "backtester/safe_entry_lstm.pt"))
        self.base_size = float(params.get("base_size", 0.15))
        self.ev_threshold = float(params.get("ev_threshold", 0.0))
        self.take_profit_cents = float(params.get("take_profit_cents", 0.03))
        self.trailing_drop_cents = float(params.get("trailing_drop_cents", 0.01))
        self.exit_time_cutoff_s = int(params.get("exit_time_cutoff_s", 2))
        self.min_hold_s = int(params.get("min_hold_s", 1))
        self.max_entries_per_market = int(params.get("max_entries_per_market", 1))
        self.cooldown_s = int(params.get("cooldown_s", 1))
        self.threshold_scale = float(params.get("threshold_scale", 0.72))
        self.min_prob_threshold = float(params.get("min_prob_threshold", 0.34))

        self._disabled_reason: str | None = None
        self._torch = None
        self._model = None
        self._cfg: dict = {}
        self._means: list[float] = []
        self._stds: list[float] = []
        self._thr = 0.65
        self._seq_len = 30
        self._entry_min_tr = 8
        self._entry_max_tr = 70
        self._min_price = 0.55
        self._max_price = 0.95
        self._hidden = 48
        self._layers = 1
        self._dropout = 0.1
        self._load_model()

        self.entered = False
        self.entries_count = 0
        self.last_exit_ts = -10_000
        self.active_token: str | None = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    def _load_model(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception as exc:
            self._disabled_reason = f"torch missing: {exc}"
            return

        ckpt_path = Path(self.model_file)
        if not ckpt_path.exists():
            self._disabled_reason = f"model missing: {ckpt_path}"
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        cfg = checkpoint.get("config", {})
        self._cfg = cfg
        self._means = [float(x) for x in checkpoint["means"]]
        self._stds = [
            float(x) if abs(float(x)) > 1e-9 else 1.0 for x in checkpoint["stds"]
        ]
        self._thr = float(cfg.get("selected_threshold", 0.65))
        self._seq_len = int(cfg.get("seq_len", 30))
        p = self._params
        self._entry_min_tr = int(
            p.get(
                "entry_min_time_remaining",
                min(5, int(cfg.get("entry_min_time_remaining", 8))),
            )
        )
        self._entry_max_tr = int(
            p.get(
                "entry_max_time_remaining",
                max(120, int(cfg.get("entry_max_time_remaining", 70))),
            )
        )
        self._min_price = float(
            p.get("min_price", min(0.45, float(cfg.get("min_price", 0.55))))
        )
        self._max_price = float(
            p.get("max_price", max(0.985, float(cfg.get("max_price", 0.95))))
        )
        self._hidden = int(cfg.get("hidden_size", 48))
        self._layers = int(cfg.get("layers", 1))
        self._dropout = float(cfg.get("dropout", 0.1))
        self.required_history = max(self.required_history, self._seq_len + 2)

        class _LSTMClassifier(nn.Module):
            def __init__(
                self, in_dim: int, hidden: int, layers: int, dropout: float
            ) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=(dropout if layers > 1 else 0.0),
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last).squeeze(1)

        model = _LSTMClassifier(
            in_dim=len(self._means),
            hidden=self._hidden,
            layers=self._layers,
            dropout=self._dropout,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._torch = torch
        self._model = model

    def on_market_start(self, market_ts: int) -> None:
        _ = market_ts
        self.entered = False
        self.entries_count = 0
        self.last_exit_ts = -10_000
        self.active_token = None
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_ts = 0

    @staticmethod
    def _ev(p_win: float, price: float) -> float:
        return p_win * ((SETTLE_PRICE / price) - 1.0) + (1.0 - p_win) * (-1.0)

    def _build_seq(self, hist, token: str):
        if len(hist) < self._seq_len:
            return None
        seq = hist[-self._seq_len :]
        out = []
        for i, t in enumerate(seq):
            p = seq[i - 1] if i > 0 else t
            up_ret = ((t.up_mid - p.up_mid) / p.up_mid) if p.up_mid > 1e-12 else 0.0
            prev_gap = p.up_mid - p.down_mid
            gap = t.up_mid - t.down_mid
            gap_ret = (
                ((gap - prev_gap) / abs(prev_gap)) if abs(prev_gap) > 1e-12 else 0.0
            )
            lo = max(0, i - 29)
            rw: list[float] = []
            for k in range(lo + 1, i + 1):
                a = seq[k - 1].up_mid
                b = seq[k].up_mid
                if a > 1e-12:
                    rw.append((b - a) / a)
            rv_up_30 = _std(rw) * math.sqrt(300.0)
            token_px = t.up_mid if token == "up" else t.down_mid
            token_px = min(0.999, max(0.001, token_px))
            tau = max(1.0 / 300.0, t.time_remaining / 300.0)
            sig = max(1e-4, rv_up_30)
            z = math.log(token_px / max(1e-6, 1.0 - token_px)) / (sig * math.sqrt(tau))
            phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            d_delta = phi / (sig * math.sqrt(tau))
            d_gamma = d_delta * (abs(z) + 1.0) / (sig * math.sqrt(tau))
            d_vega = phi * math.sqrt(tau)
            d_theta = -0.5 * phi / max(1e-6, math.sqrt(tau))
            row = [
                t.up_mid,
                t.down_mid,
                gap,
                (t.up_ask - t.up_bid),
                (t.down_ask - t.down_bid),
                up_ret,
                gap_ret,
                t.time_remaining / 300.0,
                token_px,
                rv_up_30,
                d_delta,
                d_gamma,
                d_vega,
                d_theta,
            ]
            if token == "down":
                row[2] = -row[2]
                row[6] = -row[6]
            out.append(
                [(row[j] - self._means[j]) / self._stds[j] for j in range(len(row))]
            )
        return out

    def _prob(self, seq) -> float:
        if self._model is None or self._torch is None:
            return 0.0
        x = self._torch.tensor([seq], dtype=self._torch.float32)
        with self._torch.no_grad():
            z = self._model(x)
            p = self._torch.sigmoid(z).item()
        return float(p)

    def on_tick(self, ctx: StrategyContext):
        if self._disabled_reason is not None:
            return None
        tick = ctx.latest()
        if tick is None:
            return None

        if self.active_token is not None:
            token = self.active_token
            pos_qty = ctx.position(token)
            if pos_qty <= 1e-9:
                self.entered = False
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
            drop_from_peak = self.peak_price - px

            if tick.time_remaining <= self.exit_time_cutoff_s:
                self.entered = False
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell", token=token, size=1.0, comment="LSTM time-exit"
                )
            if held_s >= self.min_hold_s and gain >= self.take_profit_cents:
                self.entered = False
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell", token=token, size=1.0, comment=f"LSTM tp +{gain:.3f}"
                )
            if held_s >= self.min_hold_s and drop_from_peak >= self.trailing_drop_cents:
                self.entered = False
                self.active_token = None
                self.last_exit_ts = tick.timestamp
                return Action(
                    side="sell",
                    token=token,
                    size=1.0,
                    comment=f"LSTM trail -{drop_from_peak:.3f}",
                )
            return None

        if self.entered:
            return None
        if self.entries_count >= self.max_entries_per_market:
            return None
        if tick.timestamp - self.last_exit_ts < self.cooldown_s:
            return None
        if not (self._entry_min_tr <= tick.time_remaining <= self._entry_max_tr):
            return None

        best = None
        hist = ctx.history(self._seq_len + 2)
        use_thr = max(self.min_prob_threshold, self._thr * self.threshold_scale)
        for token in ("up", "down"):
            price = tick.up_ask if token == "up" else tick.down_ask
            if price < self._min_price or price > self._max_price:
                continue
            seq = self._build_seq(hist, token)
            if seq is None:
                continue
            p = self._prob(seq)
            ev = self._ev(p, price)
            if p < use_thr or ev < self.ev_threshold:
                continue
            score = p + 0.25 * ev
            if best is None or score > best["score"]:
                best = {"token": token, "p": p, "ev": ev, "score": score}

        if best is None:
            return None
        self.entered = True
        self.entries_count += 1
        self.active_token = str(best["token"])
        self.entry_price = tick.up_ask if self.active_token == "up" else tick.down_ask
        self.peak_price = self.entry_price
        self.entry_ts = tick.timestamp
        return Action(
            side="buy",
            token=self.active_token,
            size=self.base_size,
            comment=f"LSTM p={best['p']:.3f} ev={best['ev']:.3f}",
        )

    def on_market_end(self, winner_token: str, ctx: StrategyContext):
        _ = winner_token
        _ = ctx
        if self._disabled_reason is not None:
            return f"LSTM disabled: {self._disabled_reason}"
        return "LSTM done"


def create_strategy(params: dict | None = None) -> LSTMSafeEntryStrategy:
    return LSTMSafeEntryStrategy(params=params)
