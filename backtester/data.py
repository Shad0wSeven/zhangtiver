from __future__ import annotations

import glob
import json
from pathlib import Path

from livetrader.strategy_api import Tick


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return [p for p in paths if Path(p).exists()]


def market_ts_from_slug(market: str) -> int:
    try:
        return int(market.rsplit("-", 1)[-1])
    except Exception:
        return 0


def load_markets(paths: list[str]) -> dict[str, list[Tick]]:
    by_market: dict[str, list[Tick]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    market = d.get("market")
                    if not market:
                        continue
                    if (
                        not d.get("up_bids")
                        or not d.get("up_asks")
                        or not d.get("down_bids")
                        or not d.get("down_asks")
                    ):
                        continue
                    up_bid = float(d["up_bids"][0][0])
                    up_ask = float(d["up_asks"][0][0])
                    down_bid = float(d["down_bids"][0][0])
                    down_ask = float(d["down_asks"][0][0])
                    ts_ms = int(d.get("timestamp_ms", int(d["timestamp"]) * 1000))
                    tick = Tick(
                        timestamp=ts_ms // 1000,
                        timestamp_ms=ts_ms,
                        market_ts=market_ts_from_slug(market),
                        time_remaining=int(d["time_remaining"]),
                        up_bid=up_bid,
                        up_ask=up_ask,
                        up_mid=(up_bid + up_ask) / 2.0,
                        down_bid=down_bid,
                        down_ask=down_ask,
                        down_mid=(down_bid + down_ask) / 2.0,
                    )
                    by_market.setdefault(market, []).append(tick)
                except Exception:
                    continue

    for market in by_market:
        by_market[market].sort(key=lambda t: t.timestamp_ms)
    return by_market


def is_complete_market(rows: list[Tick], end_threshold_s: int = 3) -> bool:
    return bool(rows) and min(t.time_remaining for t in rows) <= end_threshold_s


def market_winner(rows: list[Tick]) -> str:
    last = rows[-1]
    return "up" if last.up_mid >= last.down_mid else "down"
