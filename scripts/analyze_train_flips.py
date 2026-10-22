#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze where flips happen on the training split only "
            "(chronological market split)."
        )
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
    p.add_argument("--checkpoints", default="40,30,20")
    p.add_argument("--lookback-s", type=int, default=45)
    p.add_argument("--min-window-rows", type=int, default=15)
    p.add_argument("--min-market-rows", type=int, default=80)
    p.add_argument("--train-frac", type=float, default=0.5)
    return p.parse_args()


def bucket(v: float, edges: list[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return f"[{edges[i]:.4f},{edges[i+1]:.4f})"
    return f"[{edges[-1]:.4f},+inf)"


def print_rates(title: str, rows: list[tuple[str, int, int]]) -> None:
    print(title)
    for key, total, flips in rows:
        rate = flips / total if total else 0.0
        print(f"  {key}: samples={total} flips={flips} flip_rate={rate:.4f}")


def main() -> None:
    args = parse_args()
    mod = runpy.run_path("scripts/fit_flip_shape_model.py")
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]

    rows = mod["load_rows"](args.jsonl)
    markets = mod["group_markets"](rows)
    samples, feature_names = mod["build_dataset"](
        market_rows=markets,
        checkpoints=checkpoints,
        lookback_s=args.lookback_s,
        min_window_rows=args.min_window_rows,
        min_market_rows=args.min_market_rows,
    )
    if not samples:
        raise SystemExit("no samples")

    train_idx, test_idx = mod["split_by_market"](samples, args.train_frac)
    _ = test_idx

    feat = {name: i for i, name in enumerate(feature_names)}
    i_cp = None
    i_price = feat["leader_price"]
    i_gap = feat["price_gap"]
    i_flip = feat["imbalance_flip_rate"]
    i_curv = feat["imbalance_curvature"]

    train = [samples[i] for i in train_idx]
    y = [int(s.y_flip) for s in train]

    print(f"markets_total={len(markets)}")
    print(f"samples_total={len(samples)} train_samples={len(train)}")
    print(f"train_flip_rate={sum(y)/max(1,len(y)):.4f}")

    by_cp: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_price: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_gap: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_chop: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_curv: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    price_edges = [0.0, 0.55, 0.65, 0.75, 0.85, 0.93]
    gap_edges = [0.0, 0.03, 0.06, 0.10, 0.16, 0.24]
    chop_edges = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
    curv_edges = [-0.0200, -0.0025, -0.0010, 0.0010, 0.0025, 0.0200]

    for s in train:
        cp = str(getattr(s, "checkpoint_s", "na")) if i_cp is None else "na"
        price = float(s.x[i_price])
        gap = abs(float(s.x[i_gap]))
        chop = float(s.x[i_flip])
        curv = float(s.x[i_curv])
        flip = int(s.y_flip)

        by_cp[cp][0] += 1
        by_cp[cp][1] += flip
        by_price[bucket(price, price_edges)][0] += 1
        by_price[bucket(price, price_edges)][1] += flip
        by_gap[bucket(gap, gap_edges)][0] += 1
        by_gap[bucket(gap, gap_edges)][1] += flip
        by_chop[bucket(chop, chop_edges)][0] += 1
        by_chop[bucket(chop, chop_edges)][1] += flip
        by_curv[bucket(curv, curv_edges)][0] += 1
        by_curv[bucket(curv, curv_edges)][1] += flip

    cp_rows = []
    for cp in sorted(by_cp.keys(), key=lambda x: int(x)):
        t, f = by_cp[cp]
        cp_rows.append((f"checkpoint={cp}s", t, f))
    print_rates("flip_rate_by_checkpoint", cp_rows)

    for name, d in [
        ("flip_rate_by_leader_price", by_price),
        ("flip_rate_by_abs_gap", by_gap),
        ("flip_rate_by_choppiness", by_chop),
        ("flip_rate_by_curvature", by_curv),
    ]:
        rows2 = []
        for k in sorted(d.keys()):
            t, f = d[k]
            rows2.append((k, t, f))
        print_rates(name, rows2)


if __name__ == "__main__":
    main()
