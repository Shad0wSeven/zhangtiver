#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

# Official Chainlink BTC/USD proxy addresses:
# - Ethereum mainnet: 0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c
# - Arbitrum one:      0x6ce185860a4963106506C203335A2910413708e9
DEFAULT_FEED_ADDRESS = "0x6ce185860a4963106506C203335A2910413708e9"
DEFAULT_RPC_HTTP = "https://arb1.arbitrum.io/rpc"
DEFAULT_OUT_PATH = "/tmp/chainlink_btcusd.json"

DECIMALS_SELECTOR = "0x313ce567"
LATEST_ROUND_SELECTOR = "0xfeaf968c"


def _decode_uint256(word_hex: str) -> int:
    return int(word_hex, 16)


def _decode_int256(word_hex: str) -> int:
    raw = int(word_hex, 16)
    if raw >= (1 << 255):
        raw -= 1 << 256
    return raw


def _split_words(result_hex: str) -> list[str]:
    data = result_hex[2:] if result_hex.startswith("0x") else result_hex
    if len(data) % 64 != 0:
        raise ValueError(f"invalid ABI word length: {len(data)}")
    return [data[i : i + 64] for i in range(0, len(data), 64)]


@dataclass
class FeedState:
    round_id: int = 0
    answer_raw: int = 0
    updated_at: int = 0
    price: float = 0.0
    decimals: int = 8
    ts_ms: int = 0
    round_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": "chainlink",
            "symbol": "BTC/USD",
            "round_id": self.round_id,
            "answer_raw": self.answer_raw,
            "decimals": self.decimals,
            "price": self.price,
            "updated_at": self.updated_at,
            "fetched_at_ms": self.ts_ms,
            "age_s": max(0, int(time.time()) - self.updated_at)
            if self.updated_at
            else None,
            "round_changed": self.round_changed,
        }


class ChainlinkFeeder:
    def __init__(
        self,
        rpc_http: str,
        feed_address: str,
        out_path: str,
        poll_interval_s: float,
        http_host: str,
        http_port: int,
        log_every_poll: bool,
    ) -> None:
        self.rpc_http = rpc_http
        self.feed_address = feed_address
        self.out_path = Path(out_path)
        self.poll_interval_s = poll_interval_s
        self.http_host = http_host
        self.http_port = http_port
        self.log_every_poll = log_every_poll
        self.state = FeedState()
        self._session: aiohttp.ClientSession | None = None
        self._app_runner: web.AppRunner | None = None
        self._last_emit_round = -1

    async def rpc_call(self, data: str) -> str:
        if self._session is None:
            raise RuntimeError("session not started")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": self.feed_address, "data": data}, "latest"],
        }
        async with self._session.post(self.rpc_http, json=payload, timeout=2.0) as resp:
            resp.raise_for_status()
            body = await resp.json()
        if "error" in body:
            raise RuntimeError(f"rpc error: {body['error']}")
        result = body.get("result")
        if not isinstance(result, str) or not result.startswith("0x"):
            raise RuntimeError(f"unexpected rpc result: {result}")
        return result

    async def fetch_decimals(self) -> int:
        res = await self.rpc_call(DECIMALS_SELECTOR)
        words = _split_words(res)
        if not words:
            raise RuntimeError("empty decimals response")
        return _decode_uint256(words[0])

    async def fetch_latest_round(self) -> FeedState:
        res = await self.rpc_call(LATEST_ROUND_SELECTOR)
        words = _split_words(res)
        if len(words) < 5:
            raise RuntimeError(
                f"latestRoundData response too short: {len(words)} words"
            )
        round_id = _decode_uint256(words[0])
        answer = _decode_int256(words[1])
        updated_at = _decode_uint256(words[3])
        if answer <= 0:
            raise RuntimeError(f"invalid answer: {answer}")
        price = answer / (10**self.state.decimals)
        return FeedState(
            round_id=round_id,
            answer_raw=answer,
            updated_at=updated_at,
            price=price,
            decimals=self.state.decimals,
            ts_ms=int(time.time() * 1000),
        )

    def write_snapshot(self) -> None:
        data = self.state.as_dict()
        tmp = self.out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(self.out_path)

    async def poll_loop(self) -> None:
        while True:
            try:
                new_state = await self.fetch_latest_round()
                changed = new_state.round_id != self._last_emit_round
                new_state.round_changed = changed
                self.state = new_state
                # Always refresh local timestamped snapshot so consumers can track fetch liveness.
                self.write_snapshot()
                if changed:
                    self._last_emit_round = new_state.round_id
                    print(
                        f"[chainlink] round_change round={new_state.round_id} "
                        f"price={new_state.price:.2f} "
                        f"updated_at={new_state.updated_at}",
                        flush=True,
                    )
                elif self.log_every_poll:
                    print(
                        f"[chainlink] polled round={new_state.round_id} "
                        f"price={new_state.price:.2f} fetched_at_ms={new_state.ts_ms}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[chainlink] poll error: {exc}", flush=True)
            await asyncio.sleep(self.poll_interval_s)

    async def handle_latest(self, _: web.Request) -> web.Response:
        return web.json_response(self.state.as_dict())

    async def start_http(self) -> None:
        app = web.Application()
        app.router.add_get("/latest", self.handle_latest)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.http_host, self.http_port)
        await site.start()
        self._app_runner = runner
        print(
            f"[chainlink] serving latest on http://{self.http_host}:{self.http_port}/latest",
            flush=True,
        )

    async def close_http(self) -> None:
        if self._app_runner is not None:
            await self._app_runner.cleanup()
            self._app_runner = None

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            self.state.decimals = await self.fetch_decimals()
            print(f"[chainlink] decimals={self.state.decimals}", flush=True)
            await self.start_http()
            try:
                await self.poll_loop()
            finally:
                await self.close_http()
                self._session = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-latency Chainlink BTC/USD feeder")
    p.add_argument(
        "--rpc-http",
        default=os.getenv("CHAINLINK_RPC_HTTP", DEFAULT_RPC_HTTP),
        help="JSON-RPC HTTP endpoint",
    )
    p.add_argument(
        "--feed-address",
        default=os.getenv("CHAINLINK_FEED_ADDRESS", DEFAULT_FEED_ADDRESS),
        help="Chainlink BTC/USD proxy contract address",
    )
    p.add_argument(
        "--out",
        default=os.getenv("CHAINLINK_OUT_PATH", DEFAULT_OUT_PATH),
        help="Atomic JSON snapshot output path",
    )
    p.add_argument(
        "--poll-ms",
        type=int,
        default=int(os.getenv("CHAINLINK_POLL_MS", "250")),
        help="Polling interval in milliseconds",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--log-every-poll",
        action="store_true",
        help="Log each poll, not only Chainlink round changes",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feeder = ChainlinkFeeder(
        rpc_http=args.rpc_http,
        feed_address=args.feed_address,
        out_path=args.out,
        poll_interval_s=max(0.05, args.poll_ms / 1000.0),
        http_host=args.host,
        http_port=args.port,
        log_every_poll=args.log_every_poll,
    )
    asyncio.run(feeder.run())


if __name__ == "__main__":
    main()
