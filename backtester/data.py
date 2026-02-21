from __future__ import annotations

import glob
import json
from datetime import datetime
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


def _parse_timestamp_seconds(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    try:
        return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())
    except Exception:
        pass
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _tick_from_orderbook_row(d: dict) -> tuple[str, Tick] | None:
    market = d.get("market")
    if not market:
        return None
    if (
        not d.get("up_bids")
        or not d.get("up_asks")
        or not d.get("down_bids")
        or not d.get("down_asks")
    ):
        return None
    up_bid = float(d["up_bids"][0][0])
    up_ask = float(d["up_asks"][0][0])
    down_bid = float(d["down_bids"][0][0])
    down_ask = float(d["down_asks"][0][0])
    ts_s = _parse_timestamp_seconds(d.get("timestamp"))
    ts_ms = int(d.get("timestamp_ms", (ts_s or 0) * 1000))
    if ts_ms <= 0:
        return None
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
    return market, tick


def _tick_from_simple_1s_row(d: dict) -> tuple[str, Tick] | None:
    if d.get("up") is None:
        return None
    up_mid = float(d["up"])
    down_mid = float(d.get("down", 1.0 - up_mid))
    ts_s = _parse_timestamp_seconds(d.get("timestamp"))
    if ts_s is None and d.get("timestamp_ms") is not None:
        ts_s = int(float(d["timestamp_ms"]) / 1000.0)
    if ts_s is None:
        return None
    market_ts = (ts_s // 300) * 300
    time_remaining = max(0, market_ts + 300 - ts_s)
    market = f"btc-updown-5m-{market_ts}"
    tick = Tick(
        timestamp=ts_s,
        timestamp_ms=ts_s * 1000,
        market_ts=market_ts,
        time_remaining=time_remaining,
        up_bid=up_mid,
        up_ask=up_mid,
        up_mid=up_mid,
        down_bid=down_mid,
        down_ask=down_mid,
        down_mid=down_mid,
    )
    return market, tick


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
                    parsed = _tick_from_orderbook_row(d)
                    if parsed is None:
                        parsed = _tick_from_simple_1s_row(d)
                    if parsed is None:
                        continue
                    market, tick = parsed
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
