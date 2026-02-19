from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from livetrader.strategy_api import Action, StrategyContext, Tick

from backtester.data import is_complete_market, market_winner
from backtester.loader import instantiate_strategy
from backtester.types import MarketResult, StrategyState, Trade


@dataclass
class EngineConfig:
    window_start_cash: float = 1000.0
    default_bet_usd: float = 10.0
    settlement_win_price: float = 0.99
    complete_end_threshold_s: int = 3


class BacktestEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()

    @staticmethod
    def normalize_actions(raw: Any) -> list[Action]:
        if raw is None:
            return []
        if isinstance(raw, Action):
            return [raw]
        if isinstance(raw, list):
            return [a for a in raw if isinstance(a, Action)]
        return []

    def _apply_action(
        self,
        state: StrategyState,
        strategy_name: str,
        market: str,
        tick: Tick,
        action: Action,
    ) -> None:
        side = action.side.lower().strip()
        token = action.token.lower().strip()
        if side not in {"buy", "sell"} or token not in {"up", "down"}:
            return

        price = tick.up_mid if token == "up" else tick.down_mid
        if price <= 0:
            return

        size = float(action.size)
        if side == "buy":
            if size == 1.0:
                notional = self.config.default_bet_usd
            elif 0 < size < 1:
                notional = state.cash * size
            else:
                notional = size
            notional = min(notional, state.cash)
            if notional <= 0:
                return
            qty = notional / price
            state.cash -= notional
            state.positions[token] += qty
            state.costs[token] += notional
            state.trades.append(
                Trade(
                    market=market,
                    strategy=strategy_name,
                    side="buy",
                    token=token,
                    timestamp_ms=tick.timestamp_ms,
                    price=price,
                    qty=qty,
                    notional=notional,
                    time_remaining=tick.time_remaining,
                    comment=action.comment,
                )
            )
            return

        if 0 < size <= 1:
            qty = state.positions[token] * size
        else:
            qty = size
        qty = min(qty, state.positions[token])
        if qty <= 0:
            return
        avg_cost = state.costs[token] / state.positions[token] if state.positions[token] > 0 else 0.0
        notional = qty * price
        state.cash += notional
        state.positions[token] -= qty
        state.costs[token] -= qty * avg_cost
        if state.positions[token] <= 1e-9:
            state.positions[token] = 0.0
            state.costs[token] = 0.0
        state.trades.append(
            Trade(
                market=market,
                strategy=strategy_name,
                side="sell",
                token=token,
                timestamp_ms=tick.timestamp_ms,
                price=price,
                qty=qty,
                notional=notional,
                time_remaining=tick.time_remaining,
                comment=action.comment,
            )
        )

    def _settle(self, state: StrategyState, winner: str) -> None:
        for token in ("up", "down"):
            qty = state.positions[token]
            if qty <= 0:
                continue
            px = self.config.settlement_win_price if token == winner else 0.0
            state.cash += qty * px
            state.positions[token] = 0.0
            state.costs[token] = 0.0

    def run_strategy(
        self,
        strategy_file: str,
        markets: dict[str, list[Tick]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        strategy = instantiate_strategy(strategy_file, params=params)
        market_ids = sorted(markets.keys())
        market_results: list[MarketResult] = []
        all_trades: list[Trade] = []

        for market in market_ids:
            rows = markets[market]
            if not is_complete_market(rows, self.config.complete_end_threshold_s):
                continue
            state = StrategyState(
                strategy_name=strategy.name,
                start_cash=self.config.window_start_cash,
                cash=self.config.window_start_cash,
            )

            try:
                strategy.on_market_start(rows[0].market_ts)
            except Exception:
                pass

            history: list[Tick] = []
            max_equity = self.config.window_start_cash
            max_drawdown = 0.0
            for tick in rows:
                history.append(tick)
                ctx = StrategyContext(
                    history=history,
                    positions=dict(state.positions),
                    market_ts=tick.market_ts,
                )
                try:
                    actions = self.normalize_actions(strategy.on_tick(ctx))
                except Exception:
                    actions = []
                for action in actions:
                    self._apply_action(
                        state=state,
                        strategy_name=strategy.name,
                        market=market,
                        tick=tick,
                        action=action,
                    )
                equity = state.mark_to_market(tick)
                state.equity_curve.append(equity)
                if equity > max_equity:
                    max_equity = equity
                dd = max_equity - equity
                if dd > max_drawdown:
                    max_drawdown = dd

            winner = market_winner(rows)
            self._settle(state, winner)
            end_ctx = StrategyContext(
                history=history,
                positions=dict(state.positions),
                market_ts=rows[-1].market_ts,
            )
            try:
                strategy.on_market_end(winner, end_ctx)
            except Exception:
                pass

            pnl = state.cash - self.config.window_start_cash
            market_results.append(
                MarketResult(
                    market=market,
                    winner=winner,
                    pnl=pnl,
                    trades=len(state.trades),
                    end_cash=state.cash,
                    max_drawdown=max_drawdown,
                )
            )
            all_trades.extend(state.trades)

        return {
            "strategy_name": strategy.name,
            "strategy_file": strategy_file,
            "params": params or {},
            "market_results": market_results,
            "trades": all_trades,
        }
