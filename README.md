## Live Trader (separate from `5m-log.py`)

This repo now includes a separate live-trader app that:

- streams Polymarket BTC 5m market prices live
- loads any number of strategies from external Python files
- gives each strategy shared price history (`ctx.history(n)`) and live price (`ctx.latest()`)
- executes buy/sell actions per tick
- auto-settles all open positions at market end (`0.99` winner / `0.00` loser)
- serves a UI:
  - top half: strategies + PnL + positions + comments
  - bottom half: live tick stream + color-coded buys/sells

### Run

```bash
uv run run_livetrader.py
```

Open: `http://127.0.0.1:8080`

### Strategy plugins

Strategy files live in `strategies/*.py` and must expose:

```python
def create_strategy():
    return MyStrategy()
```

Required strategy members:

- `name: str`
- `color: str` (used in UI highlighting)
- `required_history: int`
- `on_tick(ctx) -> Action | list[Action] | None`

Optional hooks:

- `on_market_start(market_ts: int)`
- `on_market_end(winner_token: str, ctx) -> str | None` (comment shown in UI)

Context helpers:

- `ctx.history(n)` returns last `n` ticks
- `ctx.latest()` returns current tick
- `ctx.position("up"|"down")` returns current open size for that strategy

Included examples:

- `strategies/mean_reversion.py`
- `strategies/momentum.py`
- `strategies/active_pair_scalper.py`
- `strategies/active_pair_ladder.py`

### Tune New Active Strategies

```bash
python3 -m backtester.cli tune \
  --strategy-file strategies/active_pair_scalper.py \
  --grid-file backtester/grids/active_pair_scalper_fast.json \
  --end-threshold-s 40 --max-combos 220 --progress-every 40 --top-k 8

python3 -m backtester.cli tune \
  --strategy-file strategies/active_pair_ladder.py \
  --grid-file backtester/grids/active_pair_ladder_fast.json \
  --end-threshold-s 40 --max-combos 220 --progress-every 40 --top-k 8
```

### Chainlink BTC/USD Feed (for strategy pull)

Run a local feeder that polls Chainlink on Arbitrum and publishes:
- snapshot file: `/tmp/chainlink_btcusd.json`
- local HTTP: `http://127.0.0.1:8765/latest`

```bash
python3 scripts/chainlink_btcusd_feed.py --poll-ms 250
```

If logs look "slow", note:
- Chainlink `updated_at` changes only when oracle rounds update (often 10-20s+ depending on deviation/heartbeat).
- The feeder can still poll faster; use `--log-every-poll` to confirm fetch cadence.

```bash
python3 scripts/chainlink_btcusd_feed.py --poll-ms 150 --log-every-poll
```

Override RPC/feed if needed:

```bash
CHAINLINK_RPC_HTTP="https://arb1.arbitrum.io/rpc" \
CHAINLINK_FEED_ADDRESS="0x6ce185860a4963106506C203335A2910413708e9" \
python3 scripts/chainlink_btcusd_feed.py --poll-ms 150
```

From strategies, read cached snapshots via:
- `livetrader/chainlink_client.py` (`ChainlinkSnapshotReader`)
