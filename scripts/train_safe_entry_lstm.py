#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for LSTM training. Install it first, then rerun.\n"
        f"Import error: {exc}"
    )

from backtester.data import load_markets


@dataclass
class Sample:
    market: str
    x: list[list[float]]
    y: int


class SeqDataset(Dataset):
    def __init__(self, xs: list[list[list[float]]], ys: list[int]) -> None:
        self.xs = xs
        self.ys = ys

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.xs[idx], dtype=torch.float32)
        y = torch.tensor(float(self.ys[idx]), dtype=torch.float32)
        return x, y


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LSTM to predict safe entry windows from sequence data."
    )
    p.add_argument(
        "--jsonl",
        nargs="+",
        default=[
            "btc_5m_prices-archive.jsonl",
            "btc_5m_prices.jsonl",
            "data/btc_1s_prices.jsonl",
            "data/btc_1s_pricesv2.jsonl",
            "data/btc_1s_pricesv3.jsonl",
            "data/btc_5m_prices_archive_q1s.jsonl",
            "data/btc_5m_prices_q1s.jsonl",
        ],
    )
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--entry-min-time-remaining", type=int, default=8)
    p.add_argument("--entry-max-time-remaining", type=int, default=70)
    p.add_argument("--min-price", type=float, default=0.55)
    p.add_argument("--max-price", type=float, default=0.95)
    p.add_argument("--horizon-s", type=int, default=35)
    p.add_argument(
        "--max-adverse-cents",
        type=float,
        default=0.02,
        help="Max allowed adverse excursion from entry mid for 'safe' label.",
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=48)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output-model", default="backtester/safe_entry_lstm.pt")
    p.add_argument("--output-report", default="backtester/safe_entry_lstm_report.json")
    return p.parse_args()


def safe_label(
    rows: list, idx: int, token: str, horizon_s: int, adverse_cap: float
) -> int:
    entry_mid = rows[idx].up_mid if token == "up" else rows[idx].down_mid
    start_ts = rows[idx].timestamp
    end_ts = start_ts + max(1, horizon_s)

    worst = entry_mid
    final = rows[idx]
    j = idx + 1
    while j < len(rows):
        r = rows[j]
        if r.timestamp > end_ts:
            break
        px = r.up_mid if token == "up" else r.down_mid
        worst = min(worst, px)
        final = r
        j += 1

    winner = "up" if rows[-1].up_mid >= rows[-1].down_mid else "down"
    win = int(token == winner)
    adverse = max(0.0, entry_mid - worst)
    return int(win == 1 and adverse <= adverse_cap and final.time_remaining > 0)


def build_samples(
    args: argparse.Namespace, markets: dict[str, list]
) -> tuple[list[Sample], list[str]]:
    names = [
        "up_mid",
        "down_mid",
        "gap",
        "up_spread",
        "down_spread",
        "up_ret",
        "gap_ret",
        "time_frac",
        "token_price",
        "rv_up_30",
        "digital_delta_proxy",
        "digital_gamma_proxy",
        "digital_vega_proxy",
        "digital_theta_proxy",
    ]
    out: list[Sample] = []
    for market in sorted(markets.keys()):
        rows = markets[market]
        if len(rows) < (args.seq_len + 6):
            continue
        for i in range(args.seq_len - 1, len(rows) - 2):
            t = rows[i]
            if not (
                args.entry_min_time_remaining
                <= t.time_remaining
                <= args.entry_max_time_remaining
            ):
                continue
            for token in ("up", "down"):
                price = t.up_ask if token == "up" else t.down_ask
                if price < args.min_price or price > args.max_price:
                    continue
                y = safe_label(
                    rows=rows,
                    idx=i,
                    token=token,
                    horizon_s=args.horizon_s,
                    adverse_cap=args.max_adverse_cents,
                )
                seq: list[list[float]] = []
                start = i - args.seq_len + 1
                for j in range(start, i + 1):
                    r = rows[j]
                    p = rows[j - 1] if j > 0 else r
                    up_ret = (
                        ((r.up_mid - p.up_mid) / p.up_mid) if p.up_mid > 1e-12 else 0.0
                    )
                    prev_gap = p.up_mid - p.down_mid
                    gap = r.up_mid - r.down_mid
                    gap_ret = (
                        ((gap - prev_gap) / abs(prev_gap))
                        if abs(prev_gap) > 1e-12
                        else 0.0
                    )
                    lo = max(start, j - 29)
                    rw: list[float] = []
                    for k in range(lo + 1, j + 1):
                        a = rows[k - 1].up_mid
                        b = rows[k].up_mid
                        if a > 1e-12:
                            rw.append((b - a) / a)
                    rv_up_30 = _std(rw) * math.sqrt(300.0)
                    token_px = r.up_mid if token == "up" else r.down_mid
                    token_px = min(0.999, max(0.001, token_px))
                    tau = max(1.0 / 300.0, r.time_remaining / 300.0)
                    sig = max(1e-4, rv_up_30)
                    z = math.log(token_px / max(1e-6, 1.0 - token_px)) / (
                        sig * math.sqrt(tau)
                    )
                    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                    d_delta = phi / (sig * math.sqrt(tau))
                    d_gamma = d_delta * (abs(z) + 1.0) / (sig * math.sqrt(tau))
                    d_vega = phi * math.sqrt(tau)
                    d_theta = -0.5 * phi / max(1e-6, math.sqrt(tau))
                    seq.append(
                        [
                            r.up_mid,
                            r.down_mid,
                            gap,
                            (r.up_ask - r.up_bid),
                            (r.down_ask - r.down_bid),
                            up_ret,
                            gap_ret,
                            r.time_remaining / 300.0,
                            token_px,
                            rv_up_30,
                            d_delta,
                            d_gamma,
                            d_vega,
                            d_theta,
                        ]
                    )
                # Orient gap features so model learns one policy for both directions.
                if token == "down":
                    for row in seq:
                        row[2] = -row[2]
                        row[6] = -row[6]
                out.append(Sample(market=market, x=seq, y=y))
    return out, names


def split_idx(samples: list[Sample], train_frac: float) -> tuple[list[int], list[int]]:
    markets = sorted(set(s.market for s in samples))
    if len(markets) <= 1:
        idx = list(range(len(samples)))
        return idx, idx
    cut = max(1, int(len(markets) * train_frac))
    train_markets = set(markets[:cut])
    tr, te = [], []
    for i, s in enumerate(samples):
        (tr if s.market in train_markets else te).append(i)
    if not te:
        te = tr[:]
    return tr, te


def fit_norm(
    xs: list[list[list[float]]], idx: list[int]
) -> tuple[list[float], list[float]]:
    d = len(xs[0][0])
    sums = [0.0] * d
    cnt = 0
    for i in idx:
        for step in xs[i]:
            cnt += 1
            for j in range(d):
                sums[j] += step[j]
    means = [s / max(1, cnt) for s in sums]
    var = [0.0] * d
    for i in idx:
        for step in xs[i]:
            for j in range(d):
                dv = step[j] - means[j]
                var[j] += dv * dv
    stds = [math.sqrt(v / max(1, cnt)) for v in var]
    stds = [s if s > 1e-9 else 1.0 for s in stds]
    return means, stds


def apply_norm(
    xs: list[list[list[float]]], means: list[float], stds: list[float]
) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for seq in xs:
        s2: list[list[float]] = []
        for row in seq:
            s2.append([(row[j] - means[j]) / stds[j] for j in range(len(row))])
        out.append(s2)
    return out


def predict_prob(model: nn.Module, dl: DataLoader, device: str) -> list[float]:
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            out.extend(float(p) for p in probs)
    return out


def metrics(y: list[int], p: list[float], thr: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for yi, pi in zip(y, p):
        pred = 1 if pi >= thr else 0
        if pred == 1 and yi == 1:
            tp += 1
        elif pred == 1 and yi == 0:
            fp += 1
        elif pred == 0 and yi == 0:
            tn += 1
        else:
            fn += 1
    n = max(1, len(y))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "samples": float(len(y)),
        "positive_rate": sum(y) / n,
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = [p for p in args.jsonl if Path(p).exists()]
    if not paths:
        raise SystemExit("no data files found")
    markets = load_markets(paths)
    samples, feat_names = build_samples(args, markets)
    if not samples:
        raise SystemExit("no samples built")

    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]

    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    tr_idx, te_idx = split_idx(samples, args.train_frac)
    means, stds = fit_norm(xs, tr_idx)
    xsn = apply_norm(xs, means, stds)

    tr_x = [xsn[i] for i in tr_idx]
    tr_y = [ys[i] for i in tr_idx]
    te_x = [xsn[i] for i in te_idx]
    te_y = [ys[i] for i in te_idx]

    tr_ds = SeqDataset(tr_x, tr_y)
    te_ds = SeqDataset(te_x, te_y)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMClassifier(
        in_dim=len(feat_names),
        hidden=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    pos = sum(tr_y)
    neg = max(1, len(tr_y) - pos)
    pos_w = float(neg / max(1, pos))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=args.device))
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        tot = 0.0
        n = 0
        for xb, yb in tr_dl:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            n += int(xb.size(0))
        train_losses.append(tot / max(1, n))

    p_tr = predict_prob(
        model, DataLoader(tr_ds, batch_size=args.batch_size), args.device
    )
    p_te = predict_prob(model, te_dl, args.device)

    thresholds = [i / 100.0 for i in range(35, 91, 5)]
    tr_rows = [{"threshold": t, **metrics(tr_y, p_tr, t)} for t in thresholds]
    best = max(tr_rows, key=lambda r: r["f1"])
    te_stats = metrics(te_y, p_te, best["threshold"])

    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feat_names,
            "means": means,
            "stds": stds,
            "config": {
                "seq_len": args.seq_len,
                "entry_min_time_remaining": args.entry_min_time_remaining,
                "entry_max_time_remaining": args.entry_max_time_remaining,
                "min_price": args.min_price,
                "max_price": args.max_price,
                "horizon_s": args.horizon_s,
                "max_adverse_cents": args.max_adverse_cents,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "selected_threshold": best["threshold"],
            },
        },
        out_model,
    )

    report = {
        "paths": paths,
        "samples_total": len(samples),
        "train_samples": len(tr_idx),
        "holdout_samples": len(te_idx),
        "train_loss_by_epoch": train_losses,
        "selected_threshold_by_train_f1": best,
        "holdout_stats_at_selected_threshold": te_stats,
    }
    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples_total: {len(samples)}")
    print(f"train_samples: {len(tr_idx)} holdout_samples: {len(te_idx)}")
    print(f"train_pos_rate: {sum(tr_y)/max(1,len(tr_y)):.4f}")
    print(
        "selected_thr_train: "
        f"{best['threshold']:.2f} f1={best['f1']:.4f} "
        f"prec={best['precision']:.4f} rec={best['recall']:.4f}"
    )
    print(
        "holdout: "
        f"f1={te_stats['f1']:.4f} prec={te_stats['precision']:.4f} "
        f"rec={te_stats['recall']:.4f} acc={te_stats['accuracy']:.4f}"
    )
    print(f"model_file: {out_model}")
    print(f"report_file: {out_report}")


if __name__ == "__main__":
    main()
