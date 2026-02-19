#!/usr/bin/env python3
"""
BTC 5-minute second-level mid-price logger.

Writes one JSONL row per second for the current btc-updown-5m market using
the latest mid prices seen within each second, plus BTC/USD spot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone

try:
    import aiohttp
    import requests
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install aiohttp requests")
    sys.exit(1)


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
API_URL = "https://gamma-api.polymarket.com/markets"
BTC_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
ORDERBOOK_DEPTH_USD = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC 5-minute second-level bid/ask price logger"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="btc_5m_prices_1s.jsonl",
        help="Output JSONL file path",
    )
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_FILE = ARGS.output_file


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_jsonl(row: dict) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def get_current_market_timestamp() -> int:
    now = datetime.now(timezone.utc)
    base_minute = (now.minute // 5) * 5
    return int(now.replace(minute=base_minute, second=0, microsecond=0).timestamp())


def get_market_info(timestamp: int) -> dict | None:
    slug = f"btc-updown-5m-{timestamp}"
    url = f"{API_URL}?slug={slug}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
    except Exception as exc:
        log(f"market info error: {exc}")
    return None


def parse_orderbook_side(
    orders: list[dict[str, str]], is_bid: bool, target_volume_usd: float
) -> list[list[float]]:
    parsed: list[list[float]] = []
    for order in orders:
        try:
            parsed.append([float(order["price"]), float(order["size"])])
        except (KeyError, ValueError):
            continue

    parsed.sort(key=lambda row: row[0], reverse=is_bid)
    selected: list[list[float]] = []
    total = 0.0
    for price, size in parsed:
        selected.append([price, size])
        total += price * size
        if total >= target_volume_usd:
            break
    return selected


def resolve_market_tokens(market_info: dict) -> tuple[str, str] | None:
    clob_ids = market_info.get("clobTokenIds")
    if clob_ids:
        try:
            token_ids = json.loads(clob_ids)
            if len(token_ids) >= 2:
                return str(token_ids[0]), str(token_ids[1])
        except Exception:
            return None
        return None

    tokens = market_info.get("tokens", [])
    if len(tokens) < 2:
        return None

    up_token = None
    down_token = None
    for token in tokens:
        outcome = str(token.get("outcome", "")).lower()
        if "up" in outcome:
            up_token = token.get("token_id")
        elif "down" in outcome:
            down_token = token.get("token_id")
    if up_token and down_token:
        return str(up_token), str(down_token)
    return None


async def fetch_btc_spot_usd(session: aiohttp.ClientSession) -> float | None:
    try:
        async with session.get(BTC_TICKER_URL, timeout=2) as response:
            if response.status != 200:
                return None
            payload = await response.json()
            price = payload.get("price")
            if price is None:
                return None
            return float(price)
    except Exception:
        return None


def build_row(
    now_sec: int,
    up_book: dict[str, list[list[float]]],
    down_book: dict[str, list[list[float]]],
    btc_spot: float | None,
) -> dict:
    up_bid = up_book["bids"][0][0]
    up_ask = up_book["asks"][0][0]
    down_bid = down_book["bids"][0][0]
    down_ask = down_book["asks"][0][0]
    return {
        "timestamp": datetime.fromtimestamp(now_sec).strftime("%Y-%m-%d %H:%M:%S"),
        "up": (up_bid + up_ask) / 2.0,
        "down": (down_bid + down_ask) / 2.0,
        "btc": btc_spot,
    }


async def log_market(market_ts: int) -> None:
    market_slug = f"btc-updown-5m-{market_ts}"
    market_end = market_ts + 300
    log(f"starting market: {market_slug}")

    market_info = get_market_info(market_ts)
    if not market_info:
        log(f"market not found: {market_slug}")
        return

    tokens = resolve_market_tokens(market_info)
    if tokens is None:
        log("failed to resolve token ids")
        return
    up_token, down_token = tokens
    log(f"up token: {up_token}")
    log(f"down token: {down_token}")

    up_book: dict[str, list[list[float]]] = {"bids": [], "asks": []}
    down_book: dict[str, list[list[float]]] = {"bids": [], "asks": []}
    pending_sec: int | None = None
    pending_row: dict | None = None
    last_btc_sec: int | None = None
    last_btc_spot: float | None = None
    lines_written = 0

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL) as ws:
                token_ids = [up_token, down_token]
                await ws.send_json({"assets_ids": token_ids, "type": "market"})
                await ws.receive()
                await ws.send_json({"assets_ids": token_ids, "operation": "subscribe"})
                log("connected websocket")

                while int(time.time()) < market_end:
                    try:
                        msg = await ws.receive(timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                        continue

                    if not msg.data.startswith("{") and not msg.data.startswith("["):
                        continue

                    try:
                        raw = msg.json()
                    except Exception:
                        continue

                    events = raw if isinstance(raw, list) else [raw]
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        if event.get("event_type") != "book":
                            continue

                        asset_id = event.get("asset_id")
                        bids = parse_orderbook_side(
                            event.get("bids", []),
                            is_bid=True,
                            target_volume_usd=ORDERBOOK_DEPTH_USD,
                        )
                        asks = parse_orderbook_side(
                            event.get("asks", []),
                            is_bid=False,
                            target_volume_usd=ORDERBOOK_DEPTH_USD,
                        )

                        if asset_id == up_token:
                            up_book["bids"] = bids
                            up_book["asks"] = asks
                        elif asset_id == down_token:
                            down_book["bids"] = bids
                            down_book["asks"] = asks
                        else:
                            continue

                        if (
                            not up_book["bids"]
                            or not up_book["asks"]
                            or not down_book["bids"]
                            or not down_book["asks"]
                        ):
                            continue

                        now_sec = int(time.time())
                        if now_sec != last_btc_sec:
                            last_btc_spot = await fetch_btc_spot_usd(session)
                            last_btc_sec = now_sec

                        row = build_row(now_sec, up_book, down_book, last_btc_spot)

                        if pending_sec is None:
                            pending_sec = now_sec
                            pending_row = row
                            continue

                        if now_sec == pending_sec:
                            # Keep the latest snapshot for this second.
                            pending_row = row
                            continue

                        # New second started: flush the completed previous second.
                        if pending_row is not None:
                            write_jsonl(pending_row)
                            lines_written += 1
                            if lines_written % 20 == 0:
                                log(
                                    f"rows={lines_written} "
                                    f"last={pending_row['timestamp']} "
                                    f"up={pending_row['up']:.4f} down={pending_row['down']:.4f} "
                                    f"btc={pending_row['btc']}"
                                )

                        pending_sec = now_sec
                        pending_row = row
    except Exception as exc:
        log(f"market stream error: {exc}")

    # Flush the final second seen before market close.
    if pending_row is not None:
        write_jsonl(pending_row)
        lines_written += 1

    log(f"market ended: {market_slug}, rows written: {lines_written}")


async def main() -> None:
    log("BTC 5-minute second-level logger")
    log(f"output: {OUTPUT_FILE}")
    log("press Ctrl+C to stop")
    log("")

    current_market: int | None = None
    while True:
        try:
            market_ts = get_current_market_timestamp()
            if market_ts != current_market:
                current_market = market_ts
                await log_market(market_ts)
            else:
                await asyncio.sleep(0.2)
        except KeyboardInterrupt:
            log("stopping")
            break
        except Exception as exc:
            log(f"main loop error: {exc}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")
