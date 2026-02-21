from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import requests
from aiohttp import web

from livetrader.strategy_api import Action, StrategyContext, Tick
from livetrader.strategy_loader import load_strategies

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
API_URL = "https://gamma-api.polymarket.com/markets"
ORDERBOOK_DEPTH_USD = 1000
MAX_HISTORY = 6000
MAX_HISTORY_SECONDS = 900
MAX_STREAM_ROWS = 250
MAX_TRADE_HISTORY_ROWS = 250
TICK_QUEUE_MAX = 4000
UI_EVENT_QUEUE_MAX = 4000
UI_STATE_INTERVAL_S = 1.0
TICK_COALESCE_BACKLOG = 25
WINDOW_START_CASH = 100.0
DEFAULT_BET_USD = 10.0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Trader</title>
  <style>
    :root {
      --bg: #0f1216;
      --muted: #8b949e;
      --text: #e6edf3;
      --line: #2f3742;
      --good: #2ea043;
      --bad: #f85149;
      --up: #36c275;
      --down: #f85149;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      color: var(--text);
      background: radial-gradient(circle at 20% 0%, #1b2330 0%, var(--bg) 45%);
      height: 100vh;
      overflow: hidden;
    }
    .layout {
      display: grid;
      grid-template-rows: 50vh 50vh;
      height: 100vh;
    }
    .panel {
      border-bottom: 1px solid var(--line);
      padding: 10px;
      overflow: auto;
    }
    .title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .mid-quote {
      font-weight: 800;
      font-size: 15px;
      color: #eaf2ff;
      text-align: center;
      letter-spacing: 0.2px;
    }
    .market-row {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }
    .tile {
      border: 1px solid var(--line);
      background: #121821;
      padding: 8px;
    }
    .tile-title {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    #windowChart {
      width: 100%;
      height: 180px;
      display: block;
      background: #0c131d;
      border: 1px solid #253042;
    }
    .ob-grid {
      display: grid;
      gap: 8px;
    }
    .ob-side {
      border: 1px solid #2a3648;
      padding: 6px;
      background: #0f1721;
    }
    .ob-head {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 5px;
      color: #d5e2f8;
      font-weight: 700;
    }
    .ob-row {
      display: grid;
      grid-template-columns: 46px 1fr 56px;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      margin-bottom: 4px;
    }
    .ob-label { color: var(--muted); }
    .ob-bar {
      height: 12px;
      background: #1a2330;
      border: 1px solid #2a3648;
      position: relative;
      overflow: hidden;
    }
    .ob-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
    }
    .ob-fill.bid { background: linear-gradient(90deg, #1f6f46, #3ccf79); }
    .ob-fill.ask { background: linear-gradient(90deg, #7b2f36, #ed6a5f); }
    .ob-value { text-align: right; color: #d9e4f6; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .card {
      border: 1px solid var(--line);
      border-left: 8px solid #888;
      background: #121821;
      padding: 8px;
    }
    .card h3 {
      margin: 0 0 8px 0;
      font-size: 13px;
    }
    .meta {
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-line;
    }
    .subhead {
      font-size: 12px;
      color: var(--muted);
      margin: 4px 0;
    }
    #tradeHistoryTop {
      max-height: 150px;
      overflow: auto;
      border: 1px solid var(--line);
      padding: 6px;
      background: #0c131d;
      font-size: 12px;
      line-height: 1.4;
    }
    #stream {
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      max-height: calc(50vh - 48px);
      overflow: auto;
    }
    .line {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 1px 0;
      color: var(--text);
      word-break: break-word;
    }
    .line-left { min-width: 0; }
    .line-right { color: var(--muted); font-weight: 700; text-align: right; min-width: 46px; }
    .ts { font-weight: 800; margin-right: 8px; }
    .tick { color: #c5ddff; }
    .price-up { color: var(--up); font-weight: 700; }
    .price-down { color: var(--down); font-weight: 700; }
    .price-flat { color: #d0d7de; font-weight: 700; }
    .sys { color: #f0c674; }
    .buy { color: #000; font-weight: 700; }
    .sell { background: #000; color: #ddd; font-weight: 700; }
    .sold-up { color: var(--up); }
    .sold-down { color: var(--down); }
    .pos { color: #7ee787; }
    .neg { color: #ff7b72; }
    .opti-overlay {
      position: fixed;
      right: 10px;
      bottom: 10px;
      width: min(660px, 96vw);
      opacity: 0.75;
      pointer-events: none;
      z-index: 1000;
      filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.45));
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <div class="title">
        <div id="market">market: -</div>
        <div id="midQuote" class="mid-quote">--s | UP --/-- | DOWN --/--</div>
        <div id="clock">clients: 0</div>
      </div>
      <div class="market-row">
        <div class="tile">
          <div class="tile-title">Current Window Pricing + Strategy Trades</div>
          <canvas id="windowChart" width="960" height="180"></canvas>
        </div>
        <div class="tile">
          <div class="tile-title">Orderbook Depth (first $1000)</div>
          <div id="orderbookTile" class="ob-grid"></div>
        </div>
      </div>
      <div id="cards" class="cards"></div>
      <div class="subhead">Trade History (strategy actions)</div>
      <div id="tradeHistoryTop"></div>
    </section>
    <section class="panel">
      <div class="title"><div>Live Stream</div><div>buy=black text on strategy color | sell=strategy color on black</div></div>
      <div id="stream"></div>
    </section>
  </div>
  <img class="opti-overlay" src="__OPTI_SRC__" alt="opti" onerror="this.onerror=null;this.src='/opti.png';" />
  <script>
    const cardsEl = document.getElementById("cards");
    const streamEl = document.getElementById("stream");
    const tradeTopEl = document.getElementById("tradeHistoryTop");
    const marketEl = document.getElementById("market");
    const midQuoteEl = document.getElementById("midQuote");
    const clockEl = document.getElementById("clock");
    const orderbookTileEl = document.getElementById("orderbookTile");
    const chartEl = document.getElementById("windowChart");
    const chartCtx = chartEl.getContext("2d");
    const maxRows = 500;
    const maxTradeRows = 400;
    let prevTick = null;

    function fmt(n, d = 4) {
      if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
      return Number(n).toFixed(d);
    }

    function cls(v) {
      if (v > 0) return "pos";
      if (v < 0) return "neg";
      return "";
    }

    function renderCards(strategies) {
      cardsEl.innerHTML = "";
      for (const s of strategies) {
        const div = document.createElement("div");
        div.className = "card";
        div.style.borderLeftColor = s.color;
        div.innerHTML = `
          <h3>${s.name}</h3>
          <div class="meta">
Session PNL: <span class="${cls(s.session_pnl)}">$${fmt(s.session_pnl, 2)}</span>
Current Window: <span class="${cls(s.current_window_pnl)}">$${fmt(s.current_window_pnl, 2)}</span>
Unrealized: <span class="${cls(s.unrealized_pnl)}">$${fmt(s.unrealized_pnl, 2)}</span>
Cash: $${fmt(s.cash, 2)}
Window Buy Flow: $${fmt(s.window_buy_notional, 2)} | Sell Flow: $${fmt(s.window_sell_notional, 2)}
Window Capital Used: ${fmt(s.window_deployed_pct, 1)}% of $${fmt(s.window_start_cash, 0)}
Open UP: ${fmt(s.position_up, 2)} (avg $${fmt(s.avg_up_value, 4)}, value $${fmt(s.open_up_value, 2)})
Open DOWN: ${fmt(s.position_down, 2)} (avg $${fmt(s.avg_down_value, 4)}, value $${fmt(s.open_down_value, 2)})
Buys: ${s.buys} | Sells: ${s.sells} | Win sells: ${s.win_sells} | Loss sells: ${s.loss_sells}
Avg Profit/Trade: <span class="${cls(s.avg_profit_per_trade)}">$${fmt(s.avg_profit_per_trade, 4)}</span>
Avg Entry (all/win/loss): $${fmt(s.avg_entry_price, 4)} / $${fmt(s.avg_win_entry_price, 4)} / $${fmt(s.avg_loss_entry_price, 4)}
Avg Profit/Session: <span class="${cls(s.avg_profit_per_session)}">$${fmt(s.avg_profit_per_session, 4)}</span> (${s.closed_windows} closed)
Comment: ${s.last_comment || "-"}
          </div>`;
        cardsEl.appendChild(div);
      }
    }

    function renderOrderbook(s) {
      const maxDepth = Math.max(
        s.up_bid_depth_usd || 0,
        s.up_ask_depth_usd || 0,
        s.down_bid_depth_usd || 0,
        s.down_ask_depth_usd || 0,
        1
      );
      const row = (label, clsName, val) => `
        <div class="ob-row">
          <div class="ob-label">${label}</div>
          <div class="ob-bar"><div class="ob-fill ${clsName}" style="width:${Math.max(2, Math.round((val / maxDepth) * 100))}%"></div></div>
          <div class="ob-value">$${fmt(val, 0)}</div>
        </div>`;
      orderbookTileEl.innerHTML = `
        <div class="ob-side">
          <div class="ob-head"><span>UP</span><span>${fmt(s.up_bid)} / ${fmt(s.up_ask)}</span></div>
          ${row("BID", "bid", s.up_bid_depth_usd || 0)}
          ${row("ASK", "ask", s.up_ask_depth_usd || 0)}
        </div>
        <div class="ob-side">
          <div class="ob-head"><span>DOWN</span><span>${fmt(s.down_bid)} / ${fmt(s.down_ask)}</span></div>
          ${row("BID", "bid", s.down_bid_depth_usd || 0)}
          ${row("ASK", "ask", s.down_ask_depth_usd || 0)}
        </div>`;
    }

    function renderMidQuote(s) {
      if (!s) return;
      midQuoteEl.textContent = `${s.time_remaining ?? "--"}s | UP ${fmt(s.up_bid)} / ${fmt(s.up_ask)} | DOWN ${fmt(s.down_bid)} / ${fmt(s.down_ask)}`;
      renderOrderbook(s);
      renderChart(s.window_ticks || [], s.window_trades || [], s.window_markers || []);
    }

    function renderChart(ticks, trades, markers) {
      const w = chartEl.width;
      const h = chartEl.height;
      chartCtx.clearRect(0, 0, w, h);
      chartCtx.fillStyle = "#0c131d";
      chartCtx.fillRect(0, 0, w, h);
      if (!ticks.length) return;

      const minTs = ticks[0].timestamp_ms;
      const maxTs = ticks[ticks.length - 1].timestamp_ms || minTs + 1;
      const prices = [];
      for (const t of ticks) {
        prices.push(t.up_mid, t.down_mid);
      }
      const minP = Math.min(...prices);
      const maxP = Math.max(...prices);
      const spanP = Math.max(0.0001, maxP - minP);
      const x = (ts) => ((ts - minTs) / (maxTs - minTs || 1)) * (w - 16) + 8;
      const y = (p) => h - 10 - ((p - minP) / spanP) * (h - 20);

      chartCtx.strokeStyle = "#2a3648";
      chartCtx.lineWidth = 1;
      chartCtx.beginPath();
      chartCtx.moveTo(8, y(minP + spanP * 0.5));
      chartCtx.lineTo(w - 8, y(minP + spanP * 0.5));
      chartCtx.stroke();

      for (const markerTs of markers) {
        const px = x(markerTs);
        chartCtx.strokeStyle = "#ffffff";
        chartCtx.lineWidth = 1.5;
        chartCtx.beginPath();
        chartCtx.moveTo(px, 8);
        chartCtx.lineTo(px, h - 8);
        chartCtx.stroke();
        chartCtx.fillStyle = "#ffffff";
        chartCtx.font = "bold 10px monospace";
        chartCtx.fillText("NEW MARKET", Math.min(w - 78, px + 4), 14);
      }

      chartCtx.lineWidth = 2;
      chartCtx.strokeStyle = "#36c275";
      chartCtx.beginPath();
      ticks.forEach((t, i) => {
        const px = x(t.timestamp_ms);
        const py = y(t.up_mid);
        if (i === 0) chartCtx.moveTo(px, py);
        else chartCtx.lineTo(px, py);
      });
      chartCtx.stroke();

      chartCtx.strokeStyle = "#f85149";
      chartCtx.beginPath();
      ticks.forEach((t, i) => {
        const px = x(t.timestamp_ms);
        const py = y(t.down_mid);
        if (i === 0) chartCtx.moveTo(px, py);
        else chartCtx.lineTo(px, py);
      });
      chartCtx.stroke();

      const nearestTick = (ts) => {
        let best = ticks[0];
        let bestDt = Math.abs(ticks[0].timestamp_ms - ts);
        for (let i = 1; i < ticks.length; i++) {
          const dt = Math.abs(ticks[i].timestamp_ms - ts);
          if (dt < bestDt) {
            best = ticks[i];
            bestDt = dt;
          }
        }
        return best;
      };

      for (const tr of trades) {
        const px = x(tr.timestamp_ms);
        const nt = nearestTick(tr.timestamp_ms);
        const tradeOnUp = tr.token === "up";
        const py = y(tradeOnUp ? nt.up_mid : nt.down_mid);
        chartCtx.beginPath();
        chartCtx.arc(px, py, 3.8, 0, Math.PI * 2);
        if (tr.side === "buy") {
          chartCtx.fillStyle = tr.color || "#ffffff";
          chartCtx.fill();
          chartCtx.strokeStyle = "#0b1018";
          chartCtx.lineWidth = 1.0;
          chartCtx.stroke();
        } else {
          // Inverted marker: hollow circle (no fill) for sells.
          chartCtx.strokeStyle = tr.color || "#ffffff";
          chartCtx.lineWidth = 2.0;
          chartCtx.stroke();
        }
      }
    }

    function buildLine(evt, forTradeHistory = false) {
      const div = document.createElement("div");
      div.className = "line";
      const left = document.createElement("div");
      left.className = "line-left";
      const right = document.createElement("div");
      right.className = "line-right";
      const ts = `<span class="ts">${evt.time}</span>`;
      if (evt.type === "tick") {
        div.classList.add("tick");
        const upCls = !prevTick ? "price-flat" : (evt.up_mid > prevTick.up_mid ? "price-up" : (evt.up_mid < prevTick.up_mid ? "price-down" : "price-flat"));
        const downCls = !prevTick ? "price-flat" : (evt.down_mid > prevTick.down_mid ? "price-up" : (evt.down_mid < prevTick.down_mid ? "price-down" : "price-flat"));
        left.innerHTML = `${ts}UP <span class="${upCls}">${fmt(evt.up_mid)}</span> (${fmt(evt.up_bid)}/${fmt(evt.up_ask)})  DOWN <span class="${downCls}">${fmt(evt.down_mid)}</span> (${fmt(evt.down_bid)}/${fmt(evt.down_ask)})`;
        right.textContent = `${evt.time_remaining}s`;
        prevTick = evt;
      } else if (evt.type === "system") {
        div.classList.add("sys");
        left.innerHTML = `${ts}${evt.message}`;
      } else if (evt.type === "trade") {
        if (evt.side === "buy") {
          div.classList.add("buy");
          div.style.background = evt.color;
          left.innerHTML = `${ts}BUY ${evt.strategy} ${evt.token.toUpperCase()} x${fmt(evt.size, 2)} @ ${fmt(evt.price)} | ${evt.comment || ""}`;
        } else {
          div.classList.add("sell");
          div.style.color = evt.color;
          if (evt.token === "up") div.classList.add("sold-up");
          if (evt.token === "down") div.classList.add("sold-down");
          left.innerHTML = `${ts}SELL ${evt.strategy} ${evt.token.toUpperCase()} x${fmt(evt.size, 2)} @ ${fmt(evt.price)} | ${evt.comment || ""}`;
        }
      }
      div.appendChild(left);
      div.appendChild(right);
      if (forTradeHistory && evt.type === "tick") {
        return null;
      }
      return div;
    }

    function appendLine(evt, container, maxLen, forTradeHistory = false) {
      const div = buildLine(evt, forTradeHistory);
      if (!div) return;
      container.appendChild(div);
      while (container.children.length > maxLen) {
        container.removeChild(container.firstChild);
      }
      container.scrollTop = container.scrollHeight;
    }

    function renderTopTradeHistory(trades) {
      tradeTopEl.innerHTML = "";
      for (const evt of trades) appendLine(evt, tradeTopEl, maxTradeRows, true);
    }

    function connect() {
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.onopen = () => { clockEl.textContent = "clients: 1"; };
      ws.onclose = () => {
        clockEl.textContent = "clients: reconnecting";
        setTimeout(connect, 1000);
      };
      ws.onmessage = (msg) => {
        const payload = JSON.parse(msg.data);
        if (payload.type === "snapshot") {
          renderCards(payload.strategies);
          renderMidQuote(payload.top_stats);
          renderTopTradeHistory(payload.trade_history || []);
          prevTick = null;
          streamEl.innerHTML = "";
          for (const evt of payload.stream) appendLine(evt, streamEl, maxRows);
          marketEl.textContent = `market: ${payload.market || "-"}`;
          clockEl.textContent = `clients: ${payload.clients || 1}`;
          return;
        }
        if (payload.type === "event") {
          marketEl.textContent = `market: ${payload.market || "-"}`;
          clockEl.textContent = `clients: ${payload.clients}`;
          appendLine(payload.event, streamEl, maxRows);
          if (payload.event.type === "trade") appendLine(payload.event, tradeTopEl, maxTradeRows, true);
          return;
        }
        if (payload.type === "state") {
          marketEl.textContent = `market: ${payload.market || "-"}`;
          clockEl.textContent = `clients: ${payload.clients}`;
          renderCards(payload.strategies || []);
          renderMidQuote(payload.top_stats || null);
        }
      };
    }

    connect();
  </script>
</body>
</html>
"""


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def get_current_market_timestamp() -> int:
    now = datetime.now(timezone.utc)
    base_minute = (now.minute // 5) * 5
    return int(now.replace(minute=base_minute, second=0, microsecond=0).timestamp())


def get_market_info(timestamp: int) -> dict[str, Any] | None:
    slug = f"btc-updown-5m-{timestamp}"
    url = f"{API_URL}?slug={slug}"
    try:
        response = requests.get(url, timeout=2.0)
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


def token_ids_from_market_info(market_info: dict[str, Any]) -> tuple[str, str] | None:
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


@dataclass
class StrategyState:
    strategy: Any
    cash: float = WINDOW_START_CASH
    positions: dict[str, float] = field(
        default_factory=lambda: {"up": 0.0, "down": 0.0}
    )
    costs: dict[str, float] = field(default_factory=lambda: {"up": 0.0, "down": 0.0})
    buys: int = 0
    sells: int = 0
    win_sells: int = 0
    loss_sells: int = 0
    realized_pnl: float = 0.0
    session_realized_pnl: float = 0.0
    session_profit_closed_windows: float = 0.0
    window_buy_notional: float = 0.0
    window_sell_notional: float = 0.0
    last_comment: str = ""
    market_start_mtm: float = WINDOW_START_CASH
    sell_realized_pnl_sum: float = 0.0
    entry_price_sum: float = 0.0
    entry_price_count: int = 0
    win_entry_price_sum: float = 0.0
    win_entry_price_count: int = 0
    loss_entry_price_sum: float = 0.0
    loss_entry_price_count: int = 0
    closed_windows: int = 0

    @property
    def name(self) -> str:
        return str(self.strategy.name)

    @property
    def color(self) -> str:
        return str(self.strategy.color)


class LiveTraderServer:
    def __init__(self, strategy_dir: str):
        self.strategy_dir = strategy_dir
        self.opti_src = self._load_opti_src()
        self.history: deque[Tick] = deque(maxlen=MAX_HISTORY)
        self.stream: deque[dict[str, Any]] = deque(maxlen=MAX_STREAM_ROWS)
        self.trade_history: deque[dict[str, Any]] = deque(maxlen=MAX_TRADE_HISTORY_ROWS)
        self.ws_clients: set[web.WebSocketResponse] = set()
        self.states: list[StrategyState] = []
        self.current_market_ts: int | None = None
        self.latest_tick: Tick | None = None
        self.latest_books: dict[str, dict[str, list[list[float]]]] = {
            "up": {"bids": [], "asks": []},
            "down": {"bids": [], "asks": []},
        }
        self.market_change_markers: deque[int] = deque(maxlen=200)
        self.tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=TICK_QUEUE_MAX)
        self.ui_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=UI_EVENT_QUEUE_MAX
        )
        self.worker_tasks: list[asyncio.Task[Any]] = []
        self.dropped_ticks = 0
        self.dropped_ui_events = 0
        self.coalesced_ticks = 0

    @staticmethod
    def _load_opti_src() -> str:
        path = Path(__file__).with_name("opti.png")
        try:
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{payload}"
        except Exception:
            return "/opti.png"

    def state_mtm(self, state: StrategyState) -> float:
        up_px = self.latest_tick.up_mid if self.latest_tick else 0.0
        down_px = self.latest_tick.down_mid if self.latest_tick else 0.0
        return (
            state.cash
            + state.positions["up"] * up_px
            + state.positions["down"] * down_px
        )

    def state_unrealized(self, state: StrategyState) -> float:
        up_px = self.latest_tick.up_mid if self.latest_tick else 0.0
        down_px = self.latest_tick.down_mid if self.latest_tick else 0.0
        up_unr = state.positions["up"] * up_px - state.costs["up"]
        down_unr = state.positions["down"] * down_px - state.costs["down"]
        return up_unr + down_unr

    def top_stats(self) -> dict[str, Any]:
        up_mid = self.latest_tick.up_mid if self.latest_tick else None
        down_mid = self.latest_tick.down_mid if self.latest_tick else None
        up_bid = self.latest_tick.up_bid if self.latest_tick else None
        up_ask = self.latest_tick.up_ask if self.latest_tick else None
        down_bid = self.latest_tick.down_bid if self.latest_tick else None
        down_ask = self.latest_tick.down_ask if self.latest_tick else None
        if self.current_market_ts is not None:
            market_end = self.current_market_ts + 300
            time_remaining = max(0, market_end - int(time.time()))
        else:
            time_remaining = (
                self.latest_tick.time_remaining if self.latest_tick else None
            )

        up_bid_depth = sum(p * s for p, s in self.latest_books["up"]["bids"])
        up_ask_depth = sum(p * s for p, s in self.latest_books["up"]["asks"])
        down_bid_depth = sum(p * s for p, s in self.latest_books["down"]["bids"])
        down_ask_depth = sum(p * s for p, s in self.latest_books["down"]["asks"])

        window_ticks: list[dict[str, Any]] = []
        window_trades: list[dict[str, Any]] = []
        window_markers: list[int] = []
        if self.latest_tick:
            cutoff = self.latest_tick.timestamp - 300
            window = [tick for tick in self.history if tick.timestamp >= cutoff]
            window_ticks = [
                {
                    "timestamp_ms": tick.timestamp_ms,
                    "up_mid": tick.up_mid,
                    "down_mid": tick.down_mid,
                }
                for tick in window
            ]
            cutoff_ms = cutoff * 1000
            window_trades = [
                {
                    "timestamp_ms": evt.get("timestamp_ms"),
                    "price": evt.get("price"),
                    "side": evt.get("side"),
                    "token": evt.get("token"),
                    "color": evt.get("color"),
                }
                for evt in self.trade_history
                if evt.get("timestamp_ms") and evt.get("timestamp_ms") >= cutoff_ms
            ]
            window_markers = [
                ts_ms for ts_ms in self.market_change_markers if ts_ms >= cutoff_ms
            ]

        return {
            "time_remaining": time_remaining,
            "up_mid": up_mid,
            "up_bid": up_bid,
            "up_ask": up_ask,
            "down_mid": down_mid,
            "down_bid": down_bid,
            "down_ask": down_ask,
            "up_bid_depth_usd": up_bid_depth,
            "up_ask_depth_usd": up_ask_depth,
            "down_bid_depth_usd": down_bid_depth,
            "down_ask_depth_usd": down_ask_depth,
            "window_ticks": window_ticks,
            "window_trades": window_trades,
            "window_markers": window_markers,
        }

    def summary(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for state in self.states:
            up_px = self.latest_tick.up_mid if self.latest_tick else 0.0
            down_px = self.latest_tick.down_mid if self.latest_tick else 0.0
            mtm = self.state_mtm(state)
            avg_up_value = (
                state.costs["up"] / state.positions["up"]
                if state.positions["up"] > 0
                else 0.0
            )
            avg_down_value = (
                state.costs["down"] / state.positions["down"]
                if state.positions["down"] > 0
                else 0.0
            )
            current_window_pnl = mtm - WINDOW_START_CASH
            session_pnl = state.session_profit_closed_windows + current_window_pnl
            deployed_pct = (
                (state.window_buy_notional / WINDOW_START_CASH) * 100.0
                if WINDOW_START_CASH > 0
                else 0.0
            )
            avg_profit_per_trade = (
                state.sell_realized_pnl_sum / state.sells if state.sells > 0 else 0.0
            )
            avg_entry_price = (
                state.entry_price_sum / state.entry_price_count
                if state.entry_price_count > 0
                else 0.0
            )
            avg_win_entry_price = (
                state.win_entry_price_sum / state.win_entry_price_count
                if state.win_entry_price_count > 0
                else 0.0
            )
            avg_loss_entry_price = (
                state.loss_entry_price_sum / state.loss_entry_price_count
                if state.loss_entry_price_count > 0
                else 0.0
            )
            avg_profit_per_session = (
                state.session_profit_closed_windows / state.closed_windows
                if state.closed_windows > 0
                else 0.0
            )
            items.append(
                {
                    "name": state.name,
                    "color": state.color,
                    "session_pnl": session_pnl,
                    "current_window_pnl": current_window_pnl,
                    "unrealized_pnl": self.state_unrealized(state),
                    "pnl": mtm,
                    "window_pnl": mtm - state.market_start_mtm,
                    "cash": state.cash,
                    "position_up": state.positions["up"],
                    "position_down": state.positions["down"],
                    "avg_up_value": avg_up_value,
                    "avg_down_value": avg_down_value,
                    "open_up_value": state.positions["up"] * up_px,
                    "open_down_value": state.positions["down"] * down_px,
                    "buys": state.buys,
                    "sells": state.sells,
                    "win_sells": state.win_sells,
                    "loss_sells": state.loss_sells,
                    "window_buy_notional": state.window_buy_notional,
                    "window_sell_notional": state.window_sell_notional,
                    "window_deployed_pct": deployed_pct,
                    "window_start_cash": WINDOW_START_CASH,
                    "avg_profit_per_trade": avg_profit_per_trade,
                    "avg_entry_price": avg_entry_price,
                    "avg_win_entry_price": avg_win_entry_price,
                    "avg_loss_entry_price": avg_loss_entry_price,
                    "avg_profit_per_session": avg_profit_per_session,
                    "closed_windows": state.closed_windows,
                    "last_comment": state.last_comment,
                }
            )
        return items

    def queue_ui_event(self, event: dict[str, Any]) -> None:
        self.stream.append(event)
        if event.get("type") == "trade":
            self.trade_history.append(event)

        if self.ui_event_queue.full():
            try:
                self.ui_event_queue.get_nowait()
                self.ui_event_queue.task_done()
                self.dropped_ui_events += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.ui_event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_ui_events += 1

    def queue_tick(self, tick: Tick) -> None:
        if self.tick_queue.full():
            try:
                self.tick_queue.get_nowait()
                self.tick_queue.task_done()
                self.dropped_ticks += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.tick_queue.put_nowait(tick)
        except asyncio.QueueFull:
            self.dropped_ticks += 1

    async def push_payload(self, payload: dict[str, Any]) -> None:
        if not self.ws_clients:
            return
        payload = {
            **payload,
            "market": (
                f"btc-updown-5m-{self.current_market_ts}"
                if self.current_market_ts
                else payload.get("market")
            ),
            "clients": len(self.ws_clients),
        }
        raw = json.dumps(payload)
        dead: list[web.WebSocketResponse] = []
        for ws in self.ws_clients:
            try:
                await ws.send_str(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)

    async def ui_event_broadcast_loop(self) -> None:
        while True:
            event = await self.ui_event_queue.get()
            try:
                await self.push_payload({"type": "event", "event": event})
            finally:
                self.ui_event_queue.task_done()

    async def ui_state_broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(UI_STATE_INTERVAL_S)
            if not self.ws_clients:
                continue
            await self.push_payload(
                {
                    "type": "state",
                    "strategies": self.summary(),
                    "top_stats": self.top_stats(),
                }
            )

    async def strategy_tick_loop(self) -> None:
        while True:
            tick = await self.tick_queue.get()
            try:
                if self.tick_queue.qsize() >= TICK_COALESCE_BACKLOG:
                    while True:
                        try:
                            newer_tick = self.tick_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        self.tick_queue.task_done()
                        tick = newer_tick
                        self.coalesced_ticks += 1
                self.process_tick(tick)
            except Exception as exc:
                self.queue_ui_event(
                    {
                        "type": "system",
                        "time": self.to_clock(int(time.time() * 1000)),
                        "message": f"tick processor error: {exc}",
                    }
                )
            finally:
                self.tick_queue.task_done()

    def start_workers(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self.strategy_tick_loop()),
            asyncio.create_task(self.ui_event_broadcast_loop()),
            asyncio.create_task(self.ui_state_broadcast_loop()),
        ]

    async def stop_workers(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        for task in self.worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.worker_tasks = []

    async def handle_index(self, _: web.Request) -> web.Response:
        html = INDEX_HTML.replace("__OPTI_SRC__", self.opti_src)
        return web.Response(text=html, content_type="text/html")

    async def handle_opti(self, _: web.Request) -> web.Response:
        return web.FileResponse(Path(__file__).with_name("opti.png"))

    async def handle_ws(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.ws_clients.add(ws)

        await ws.send_str(
            json.dumps(
                {
                    "type": "snapshot",
                    "market": (
                        f"btc-updown-5m-{self.current_market_ts}"
                        if self.current_market_ts
                        else None
                    ),
                    "strategies": self.summary(),
                    "top_stats": self.top_stats(),
                    "trade_history": list(self.trade_history),
                    "stream": list(self.stream),
                }
            )
        )

        async for _ in ws:
            pass

        self.ws_clients.discard(ws)
        return ws

    def to_clock(self, ts_ms: int) -> str:
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S.%f")[:-3]

    def tick_from_books(
        self,
        market_ts: int,
        market_end: int,
        up_book: dict[str, list[list[float]]],
        down_book: dict[str, list[list[float]]],
        now_ms: int,
    ) -> Tick:
        now = now_ms // 1000
        up_bid = up_book["bids"][0][0]
        up_ask = up_book["asks"][0][0]
        down_bid = down_book["bids"][0][0]
        down_ask = down_book["asks"][0][0]
        return Tick(
            timestamp=now,
            timestamp_ms=now_ms,
            market_ts=market_ts,
            time_remaining=max(0, market_end - now),
            up_bid=up_bid,
            up_ask=up_ask,
            up_mid=(up_bid + up_ask) / 2.0,
            down_bid=down_bid,
            down_ask=down_ask,
            down_mid=(down_bid + down_ask) / 2.0,
        )

    def tick_from_books_with_fallback(
        self,
        market_ts: int,
        market_end: int,
        up_book: dict[str, list[list[float]]],
        down_book: dict[str, list[list[float]]],
        now_ms: int,
    ) -> Tick | None:
        def top(
            book: dict[str, list[list[float]]], side: str, default: float | None = None
        ) -> float | None:
            levels = book[side]
            if levels:
                return levels[0][0]
            return default

        up_bid = top(up_book, "bids")
        up_ask = top(up_book, "asks")
        down_bid = top(down_book, "bids")
        down_ask = top(down_book, "asks")

        # Fallback to complement market prices when one side thins out near close.
        if up_bid is None and down_ask is not None:
            up_bid = 1.0 - down_ask
        if up_ask is None and down_bid is not None:
            up_ask = 1.0 - down_bid
        if down_bid is None and up_ask is not None:
            down_bid = 1.0 - up_ask
        if down_ask is None and up_bid is not None:
            down_ask = 1.0 - up_bid

        if None in (up_bid, up_ask, down_bid, down_ask):
            return None

        # Keep quotes sane if crossed due to fallback/parsing timing.
        up_bid = max(0.0, min(1.0, float(up_bid)))
        up_ask = max(up_bid, min(1.0, float(up_ask)))
        down_bid = max(0.0, min(1.0, float(down_bid)))
        down_ask = max(down_bid, min(1.0, float(down_ask)))

        now = now_ms // 1000
        return Tick(
            timestamp=now,
            timestamp_ms=now_ms,
            market_ts=market_ts,
            time_remaining=max(0, market_end - now),
            up_bid=up_bid,
            up_ask=up_ask,
            up_mid=(up_bid + up_ask) / 2.0,
            down_bid=down_bid,
            down_ask=down_ask,
            down_mid=(down_bid + down_ask) / 2.0,
        )

    def normalize_actions(self, raw: Any) -> list[Action]:
        if raw is None:
            return []
        if isinstance(raw, Action):
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Action)]
        return []

    def apply_trade(
        self,
        state: StrategyState,
        action: Action,
        tick: Tick,
        price_override: float | None = None,
        time_override_ms: int | None = None,
    ) -> dict[str, Any] | None:
        side = action.side.lower().strip()
        token = action.token.lower().strip()
        size = float(action.size)
        if side not in {"buy", "sell"} or token not in {"up", "down"} or size <= 0:
            return None

        price = (
            price_override
            if price_override is not None
            else (tick.up_mid if token == "up" else tick.down_mid)
        )
        timestamp_ms = (
            time_override_ms if time_override_ms is not None else tick.timestamp_ms
        )

        if side == "buy":
            if size == 1.0:
                notional = DEFAULT_BET_USD
            elif 0 < size < 1:
                # Keep fractional sizing anchored to start cash so profits do not compound.
                notional = WINDOW_START_CASH * size
            else:
                notional = size
            notional = min(notional, state.cash)
            if notional <= 0:
                return None
            qty = notional / price if price > 0 else 0.0
            if qty <= 0:
                return None
            state.cash -= notional
            state.positions[token] += qty
            state.costs[token] += notional
            state.buys += 1
            state.window_buy_notional += notional
            size = qty
        else:
            if 0 < size <= 1:
                qty = state.positions[token] * size
            else:
                qty = size
            qty = min(qty, state.positions[token])
            if qty <= 0:
                return None
            prior_qty = state.positions[token]
            avg_cost = (state.costs[token] / prior_qty) if prior_qty > 0 else 0.0
            notional = price * qty
            realized = (price - avg_cost) * qty
            state.cash += notional
            state.positions[token] -= qty
            state.costs[token] -= avg_cost * qty
            if state.positions[token] <= 1e-9:
                state.positions[token] = 0.0
                state.costs[token] = 0.0
            state.sells += 1
            state.realized_pnl += realized
            state.session_realized_pnl += realized
            state.sell_realized_pnl_sum += realized
            state.window_sell_notional += notional
            state.entry_price_sum += avg_cost
            state.entry_price_count += 1
            if realized >= 0:
                state.win_sells += 1
                state.win_entry_price_sum += avg_cost
                state.win_entry_price_count += 1
            else:
                state.loss_sells += 1
                state.loss_entry_price_sum += avg_cost
                state.loss_entry_price_count += 1
            size = qty

        if action.comment:
            state.last_comment = action.comment

        return {
            "type": "trade",
            "time": self.to_clock(timestamp_ms),
            "timestamp_ms": timestamp_ms,
            "strategy": state.name,
            "color": state.color,
            "side": side,
            "token": token,
            "size": size,
            "price": price,
            "notional": price * size,
            "comment": action.comment,
        }

    def process_tick(self, tick: Tick) -> None:
        self.latest_tick = tick
        self.history.append(tick)
        # Keep only recent ticks to prevent long-session slowdown and memory growth.
        cutoff = tick.timestamp - MAX_HISTORY_SECONDS
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()
        self.queue_ui_event(
            {
                "type": "tick",
                "time": self.to_clock(tick.timestamp_ms),
                "timestamp_ms": tick.timestamp_ms,
                "up_bid": tick.up_bid,
                "up_ask": tick.up_ask,
                "up_mid": tick.up_mid,
                "down_bid": tick.down_bid,
                "down_ask": tick.down_ask,
                "down_mid": tick.down_mid,
                "time_remaining": tick.time_remaining,
            }
        )

        for state in self.states:
            ctx = StrategyContext(
                history=self.history,
                positions=dict(state.positions),
                market_ts=tick.market_ts,
            )
            try:
                actions = self.normalize_actions(state.strategy.on_tick(ctx))
                for action in actions:
                    evt = self.apply_trade(state, action, tick)
                    if evt:
                        self.queue_ui_event(evt)
            except Exception as exc:
                state.last_comment = f"error: {exc}"
                self.queue_ui_event(
                    {
                        "type": "system",
                        "time": self.to_clock(tick.timestamp_ms),
                        "message": f"strategy {state.name} failed: {exc}",
                    }
                )

    async def settle_market(self, market_ts: int) -> None:
        if not self.latest_tick:
            return
        winner = (
            "up" if self.latest_tick.up_mid >= self.latest_tick.down_mid else "down"
        )
        now_ms = int(time.time() * 1000)
        settlement_price = {
            "up": 0.99 if winner == "up" else 0.0,
            "down": 0.99 if winner == "down" else 0.0,
        }

        for state in self.states:
            for token in ("up", "down"):
                qty = state.positions[token]
                if qty <= 0:
                    continue
                px = settlement_price[token]
                evt = self.apply_trade(
                    state=state,
                    action=Action(
                        side="sell",
                        token=token,
                        size=qty,
                        comment=f"market settle {market_ts} winner={winner}",
                    ),
                    tick=self.latest_tick,
                    price_override=px,
                    time_override_ms=now_ms,
                )
                if evt:
                    self.queue_ui_event(evt)

            try:
                ctx = StrategyContext(
                    history=list(self.history),
                    positions=dict(state.positions),
                    market_ts=market_ts,
                )
                end_comment = state.strategy.on_market_end(winner, ctx)
                if end_comment:
                    state.last_comment = str(end_comment)
            except Exception:
                pass

        self.queue_ui_event(
            {
                "type": "system",
                "time": self.to_clock(now_ms),
                "message": f"market ended btc-updown-5m-{market_ts}, winner={winner}",
            }
        )

    async def run_market_stream(self, market_ts: int) -> None:
        market_end = market_ts + 300
        market_slug = f"btc-updown-5m-{market_ts}"

        info = None
        for _ in range(12):
            info = await asyncio.to_thread(get_market_info, market_ts)
            if info is not None:
                break
            await asyncio.sleep(0.5)
        if info is None:
            log(f"market not found after retries: {market_slug}")
            return

        tokens = token_ids_from_market_info(info)
        if tokens is None:
            log(f"invalid market token data: {market_slug}")
            return
        up_token, down_token = tokens
        log(f"market {market_slug} up={up_token} down={down_token}")

        up_book: dict[str, list[list[float]]] = {"bids": [], "asks": []}
        down_book: dict[str, list[list[float]]] = {"bids": [], "asks": []}
        last_heartbeat_sec = -1
        self.latest_books = {
            "up": {"bids": [], "asks": []},
            "down": {"bids": [], "asks": []},
        }

        while int(time.time()) < market_end:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL) as ws:
                        token_ids = [up_token, down_token]
                        await ws.send_json({"assets_ids": token_ids, "type": "market"})
                        await ws.receive()
                        await ws.send_json(
                            {"assets_ids": token_ids, "operation": "subscribe"}
                        )
                        log(f"connected {market_slug}")

                        while int(time.time()) < market_end:
                            try:
                                msg = await ws.receive(timeout=1.0)
                            except asyncio.TimeoutError:
                                now_ms = int(time.time() * 1000)
                                now_sec = now_ms // 1000
                                if now_sec != last_heartbeat_sec:
                                    last_heartbeat_sec = now_sec
                                    heartbeat_tick = self.tick_from_books_with_fallback(
                                        market_ts=market_ts,
                                        market_end=market_end,
                                        up_book=up_book,
                                        down_book=down_book,
                                        now_ms=now_ms,
                                    )
                                    if heartbeat_tick is not None:
                                        self.queue_tick(heartbeat_tick)
                                continue

                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (
                                    aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.CLOSING,
                                ):
                                    break
                                continue

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
                                bids = parse_orderbook_side(
                                    data.get("bids", []),
                                    is_bid=True,
                                    target_volume_usd=ORDERBOOK_DEPTH_USD,
                                )
                                asks = parse_orderbook_side(
                                    data.get("asks", []),
                                    is_bid=False,
                                    target_volume_usd=ORDERBOOK_DEPTH_USD,
                                )

                                if asset_id == up_token:
                                    if bids:
                                        up_book["bids"] = bids
                                    if asks:
                                        up_book["asks"] = asks
                                    self.latest_books["up"]["bids"] = bids
                                    self.latest_books["up"]["asks"] = asks
                                elif asset_id == down_token:
                                    if bids:
                                        down_book["bids"] = bids
                                    if asks:
                                        down_book["asks"] = asks
                                    self.latest_books["down"]["bids"] = bids
                                    self.latest_books["down"]["asks"] = asks
                                else:
                                    continue

                                now_ms = int(time.time() * 1000)
                                tick = self.tick_from_books_with_fallback(
                                    market_ts=market_ts,
                                    market_end=market_end,
                                    up_book=up_book,
                                    down_book=down_book,
                                    now_ms=now_ms,
                                )
                                if tick is not None:
                                    self.queue_tick(tick)
            except Exception as exc:
                log(f"ws error {market_slug}: {exc}")
                await asyncio.sleep(1)

        await self.tick_queue.join()
        if self.dropped_ticks > 0:
            log(f"dropped ticks during {market_slug}: {self.dropped_ticks}")
            self.dropped_ticks = 0
        if self.dropped_ui_events > 0:
            log(f"dropped ui events during {market_slug}: {self.dropped_ui_events}")
            self.dropped_ui_events = 0
        if self.coalesced_ticks > 0:
            log(f"coalesced stale ticks during {market_slug}: {self.coalesced_ticks}")
            self.coalesced_ticks = 0
        await self.settle_market(market_ts)

    async def market_loop(self) -> None:
        while True:
            prev_market_ts = self.current_market_ts
            market_ts = get_current_market_timestamp()
            self.current_market_ts = market_ts
            self.latest_tick = None

            for state in self.states:
                # Close prior window accounting, then restart each strategy with fresh
                # per-window bankroll.
                if prev_market_ts is not None:
                    state.session_profit_closed_windows += (
                        state.cash - WINDOW_START_CASH
                    )
                    state.closed_windows += 1
                state.cash = WINDOW_START_CASH
                state.positions = {"up": 0.0, "down": 0.0}
                state.costs = {"up": 0.0, "down": 0.0}
                state.realized_pnl = 0.0
                state.window_buy_notional = 0.0
                state.window_sell_notional = 0.0
                state.market_start_mtm = WINDOW_START_CASH

            for state in self.states:
                try:
                    state.strategy.on_market_start(market_ts)
                except Exception:
                    pass

            now_ms = int(time.time() * 1000)
            self.market_change_markers.append(now_ms)
            self.queue_ui_event(
                {
                    "type": "system",
                    "time": self.to_clock(now_ms),
                    "message": f"NEW MARKET btc-updown-5m-{market_ts}",
                }
            )
            await self.run_market_stream(market_ts)
            await asyncio.sleep(0.05)


async def start_server(
    host: str = "127.0.0.1", port: int = 8080, strategy_dir: str = "strategies"
) -> None:
    trader = LiveTraderServer(strategy_dir=strategy_dir)
    loaded, errors = load_strategies(strategy_dir)
    for strategy in loaded:
        trader.states.append(StrategyState(strategy=strategy))

    log(f"strategies loaded: {len(loaded)}")
    for strategy in loaded:
        log(f"- {strategy.name} ({strategy.color})")
    for err in errors:
        log(f"strategy load error: {err}")

    app = web.Application()
    app.router.add_get("/", trader.handle_index)
    app.router.add_get("/opti.png", trader.handle_opti)
    app.router.add_get("/ws", trader.handle_ws)

    async def on_startup(_: web.Application) -> None:
        trader.start_workers()
        app["market_task"] = asyncio.create_task(trader.market_loop())

    async def on_cleanup(_: web.Application) -> None:
        task = app["market_task"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await trader.stop_workers()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    log(f"UI running at http://{host}:{port}")

    while True:
        await asyncio.sleep(3600)


def run() -> None:
    asyncio.run(start_server())
