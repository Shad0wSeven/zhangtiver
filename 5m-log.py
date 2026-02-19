#!/usr/bin/env python3
"""
Portable BTC 5-Minute Market Price Logger with Orderbook Depth

Continuously logs price data from Polymarket BTC 5-minute markets to JSONL.
Captures orderbook depth up to $1000 volume on each side.

Usage:
    python3 5m-log.py [output_file.jsonl] [--condensed]

Dependencies:
    pip install aiohttp requests

Output format (JSONL):
    {"timestamp": 1234567890, "market": "btc-updown-5m-1234567890",
     "up_bids": [[0.50, 100], [0.49, 200]], "up_asks": [[0.51, 150], [0.52, 250]],
     "down_bids": [[0.49, 120], [0.48, 180]], "down_asks": [[0.50, 140], [0.51, 220]],
     "time_remaining": 120}
"""

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


# Configuration
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
API_URL = "https://gamma-api.polymarket.com/markets"
ORDERBOOK_DEPTH_USD = 1000  # Capture up to $1000 of volume
CONDENSED_MIN_PRICE_MOVE = 0.005


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BTC 5-minute Polymarket orderbook logger"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="btc_5m_prices.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--condensed",
        action="store_true",
        help="Only write rows when meaningful prices change",
    )
    parser.add_argument(
        "--min-price-move",
        type=float,
        default=CONDENSED_MIN_PRICE_MOVE,
        help="Minimum absolute price move for --condensed mode",
    )
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_FILE = ARGS.output_file
CONDENSED_MODE = ARGS.condensed
MIN_PRICE_MOVE = max(0.0, ARGS.min_price_move)


def log(msg):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_current_market_timestamp():
    """Get current 5-minute market timestamp."""
    now = datetime.now(timezone.utc)
    current_minute = now.minute
    base_minute = (current_minute // 5) * 5
    market_ts = int(
        now.replace(minute=base_minute, second=0, microsecond=0).timestamp()
    )
    return market_ts


def get_market_info(timestamp):
    """Fetch market info from API."""
    try:
        slug = f"btc-updown-5m-{timestamp}"
        url = f"{API_URL}?slug={slug}"
        response = requests.get(url, timeout=2.0)

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        log(f"Error fetching market info: {e}")

    return None


def write_jsonl(data):
    """Append data to JSONL file."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


def parse_orderbook_side(orders, is_bid=True, target_volume_usd=1000):
    """
    Parse orderbook side until we hit target volume in USD.

    For bids: want highest prices first (best buy offers)
    For asks: want lowest prices first (best sell offers)

    Args:
        orders: List of {"price": str, "size": str} orders
        is_bid: True for bids (sort descending), False for asks (sort ascending)
        target_volume_usd: Target volume in USD

    Returns:
        List of [price, size] pairs sorted correctly
    """
    # Parse all orders first
    parsed = []
    for order in orders:
        try:
            price = float(order["price"])
            size = float(order["size"])
            parsed.append([price, size])
        except (KeyError, ValueError):
            continue

    # Sort correctly: bids descending (highest first), asks ascending (lowest first)
    parsed.sort(key=lambda x: x[0], reverse=is_bid)

    # Now collect up to target volume
    result = []
    total_volume = 0

    for price, size in parsed:
        result.append([price, size])
        total_volume += price * size

        if total_volume >= target_volume_usd:
            break

    return result


def build_price_snapshot(up_orderbook, down_orderbook):
    """Extract key prices used for condensed logging decisions."""
    up_bid = up_orderbook["bids"][0][0]
    up_ask = up_orderbook["asks"][0][0]
    down_bid = down_orderbook["bids"][0][0]
    down_ask = down_orderbook["asks"][0][0]
    return {
        "up_bid": up_bid,
        "up_ask": up_ask,
        "down_bid": down_bid,
        "down_ask": down_ask,
        "up_mid": (up_bid + up_ask) / 2.0,
        "down_mid": (down_bid + down_ask) / 2.0,
    }


def meaningful_price_change(current_prices, last_prices, min_move):
    """Return True if prices moved enough to be worth logging."""
    if last_prices is None:
        return True

    keys = ("up_bid", "up_ask", "down_bid", "down_ask", "up_mid", "down_mid")
    for key in keys:
        if abs(current_prices[key] - last_prices[key]) >= min_move:
            return True
    return False


def resolve_market_tokens(market_info):
    """Extract up/down token IDs from API response."""
    clob_token_ids = market_info.get("clobTokenIds")
    if clob_token_ids:
        try:
            token_ids = json.loads(clob_token_ids)
            if len(token_ids) >= 2:
                return token_ids[0], token_ids[1]
        except Exception:
            return None
        return None

    tokens = market_info.get("tokens", [])
    if len(tokens) < 2:
        return None

    up_token = None
    down_token = None
    for token in tokens:
        outcome = token.get("outcome", "").lower()
        if "up" in outcome:
            up_token = token.get("token_id")
        elif "down" in outcome:
            down_token = token.get("token_id")
    if up_token and down_token:
        return up_token, down_token
    return None


async def log_market(market_ts):
    """Log a single 5-minute market."""
    market_slug = f"btc-updown-5m-{market_ts}"
    market_end = market_ts + 300

    log(f"Starting market: {market_slug}")

    market_info = await asyncio.to_thread(get_market_info, market_ts)
    if not market_info:
        log(f"Market not found: {market_slug}")
        return []

    token_pair = resolve_market_tokens(market_info)
    if not token_pair:
        log("Could not find Up/Down tokens")
        return []
    up_token, down_token = token_pair
    log(f"Up token: {up_token}")
    log(f"Down token: {down_token}")

    up_orderbook = {"bids": [], "asks": []}
    down_orderbook = {"bids": [], "asks": []}

    updates_logged = 0
    last_logged_prices = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL) as ws:
                token_ids = [up_token, down_token]
                await ws.send_json({"assets_ids": token_ids, "type": "market"})
                await ws.receive()
                await ws.send_json({"assets_ids": token_ids, "operation": "subscribe"})
                log("Connected to websocket")

                while True:
                    now = int(time.time())
                    time_remaining = max(0, market_end - now)
                    if time_remaining <= 0:
                        print()
                        log(f"Market ended. Logged {updates_logged} updates")
                        break

                    try:
                        msg = await ws.receive(timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if not msg.data.startswith("{") and not msg.data.startswith(
                            "["
                        ):
                            continue
                        try:
                            raw = msg.json()
                        except Exception:
                            continue

                        events = raw if isinstance(raw, list) else [raw]
                        for data in events:
                            if not isinstance(data, dict):
                                continue
                            if data.get("event_type") != "book":
                                continue

                            asset_id = data.get("asset_id")
                            parsed_bids = parse_orderbook_side(
                                data.get("bids", []),
                                is_bid=True,
                                target_volume_usd=ORDERBOOK_DEPTH_USD,
                            )
                            parsed_asks = parse_orderbook_side(
                                data.get("asks", []),
                                is_bid=False,
                                target_volume_usd=ORDERBOOK_DEPTH_USD,
                            )

                            if asset_id == up_token:
                                up_orderbook["bids"] = parsed_bids
                                up_orderbook["asks"] = parsed_asks
                            elif asset_id == down_token:
                                down_orderbook["bids"] = parsed_bids
                                down_orderbook["asks"] = parsed_asks
                            else:
                                continue

                            now_ms = int(time.time() * 1000)
                            now = now_ms // 1000

                            if (
                                up_orderbook["bids"]
                                and up_orderbook["asks"]
                                and down_orderbook["bids"]
                                and down_orderbook["asks"]
                            ):
                                current_prices = build_price_snapshot(
                                    up_orderbook, down_orderbook
                                )
                                if CONDENSED_MODE and not meaningful_price_change(
                                    current_prices, last_logged_prices, MIN_PRICE_MOVE
                                ):
                                    pass
                                else:
                                    log_entry = {
                                        "timestamp": now,
                                        "timestamp_ms": now_ms,
                                        "market": market_slug,
                                        "up_bids": up_orderbook["bids"],
                                        "up_asks": up_orderbook["asks"],
                                        "down_bids": down_orderbook["bids"],
                                        "down_asks": down_orderbook["asks"],
                                        "time_remaining": max(0, market_end - now),
                                    }
                                    write_jsonl(log_entry)
                                    updates_logged += 1
                                    last_logged_prices = current_prices

                                    if updates_logged % 10 == 0:
                                        up_bid_depth = sum(
                                            p * s for p, s in up_orderbook["bids"]
                                        )
                                        up_ask_depth = sum(
                                            p * s for p, s in up_orderbook["asks"]
                                        )
                                        down_bid_depth = sum(
                                            p * s for p, s in down_orderbook["bids"]
                                        )
                                        down_ask_depth = sum(
                                            p * s for p, s in down_orderbook["asks"]
                                        )
                                        print(
                                            f"\r  Updates: {updates_logged} | Time: {max(0, market_end - now)}s | "
                                            f"Up: ${up_bid_depth:.0f}/${up_ask_depth:.0f} | "
                                            f"Down: ${down_bid_depth:.0f}/${down_ask_depth:.0f}",
                                            end="",
                                            flush=True,
                                        )

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        log("WebSocket error")
                        break

    except Exception as e:
        log(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return


async def main():
    """Main loop - continuously log markets."""
    log(f"BTC 5-Minute Orderbook Logger")
    log(f"Output file: {OUTPUT_FILE}")
    log(f"Orderbook depth: ${ORDERBOOK_DEPTH_USD} per side")
    if CONDENSED_MODE:
        log(f"Mode: condensed (min price move {MIN_PRICE_MOVE:.4f})")
    else:
        log("Mode: full (every update)")
    log(f"Press Ctrl+C to stop")
    log("")

    current_market_ts = None

    while True:
        try:
            # Get current market
            market_ts = get_current_market_timestamp()

            # Check if new market
            if market_ts != current_market_ts:
                current_market_ts = market_ts

                # Log this market
                await log_market(market_ts)
                await asyncio.sleep(0.05)
            else:
                # Still in same market, wait
                await asyncio.sleep(0.2)

        except KeyboardInterrupt:
            log("Stopping...")
            break
        except Exception as e:
            log(f"Error in main loop: {e}")
            import traceback

            traceback.print_exc()
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
