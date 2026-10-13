import functools
from collections.abc import Callable
from datetime import timedelta

import pandas as pd

from evolutive_trading.application.ports import Forecaster
from evolutive_trading.domain.model import (
    Account,
    AgentResult,
    AgentStep,
    EquityPoint,
    Position,
    Run,
    SimulationResult,
    StepEvent,
    StepTrade,
    Trade,
)
from evolutive_trading.domain.strategies import STRATEGIES, DecisionContext

EXCHANGE_TZ = "America/New_York"
MARKET_OPEN = timedelta(hours=9, minutes=30)
MARKET_CLOSE = timedelta(hours=16)


def exchange_timestamp(day: pd.Timestamp, time_of_day: timedelta) -> str:
    return str((day + time_of_day).tz_localize(EXCHANGE_TZ).isoformat())


def execute_trades(
    account: Account,
    target_weights: dict[str, float],
    equity: float,
    open_prices: pd.Series,
    cost_rate: float,
    executed_at: str,
) -> tuple[Account, tuple[Trade, ...]]:
    cash, positions, trades = account.cash, dict(account.positions), []
    for symbol in sorted(set(positions) - set(target_weights)):
        position, price = positions.pop(symbol), open_prices[symbol]
        cash += position.shares * price * (1 - cost_rate)
        trades.append(Trade(executed_at, symbol, "sell", position.shares, price))
    for symbol in sorted(set(target_weights) - set(positions)):
        notional, price = equity * target_weights[symbol], open_prices[symbol]
        cash -= notional * (1 + cost_rate)
        positions[symbol] = Position(notional / price, price)
        trades.append(Trade(executed_at, symbol, "buy", notional / price, price))
    return Account(cash, positions), tuple(trades)


def run_simulation(
    run: Run,
    bars: pd.DataFrame,
    forecaster: Forecaster | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
) -> SimulationResult:
    selection = run.market
    symbols = tuple(sorted(selection.symbols))
    symbol_bars = bars[bars["symbol"].isin(symbols) & (bars["timestamp"] <= pd.Timestamp(selection.end))]
    open_prices = symbol_bars.pivot(index="timestamp", columns="symbol", values="open")
    close_prices = symbol_bars.pivot(index="timestamp", columns="symbol", values="close")
    missing_symbols = sorted(
        set(symbols) - set(close_prices.columns)
        | set(open_prices.columns[open_prices.isna().any()])
        | set(close_prices.columns[close_prices.isna().any()])
    )
    if missing_symbols:
        raise ValueError(f"Every symbol needs a bar on every day; missing bars for {missing_symbols}.")
    trading_days = close_prices.index
    step_indexes = range(trading_days.searchsorted(pd.Timestamp(selection.start)), len(trading_days) - 1)
    if not len(step_indexes):
        raise ValueError("The range needs at least two trading days from its start.")

    def expected_returns(history: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
        if forecaster is None:
            raise ValueError("This run needs a forecaster.")
        return forecaster.forecast(history, target_date).set_index("symbol")["exp_return"]

    cost_rate = run.cost_bps / 1e4
    agent_behaviours = [(agent, STRATEGIES[agent.strategy](agent.parameters)) for agent in run.agents]
    accounts_by_agent_id = {agent.id: Account(run.initial_cash, {}) for agent, _ in agent_behaviours}
    equity_curves_by_agent_id: dict[str, list[EquityPoint]] = {agent.id: [] for agent, _ in agent_behaviours}
    trades_by_agent_id: dict[str, list[Trade]] = {agent.id: [] for agent, _ in agent_behaviours}

    for step_index in step_indexes:
        today, tomorrow = trading_days[step_index], trading_days[step_index + 1]
        forecast = functools.cache(functools.partial(expected_returns, symbol_bars[symbol_bars["timestamp"] <= today], tomorrow))
        step_reports: list[AgentStep] = []
        for agent, behaviour in agent_behaviours:
            account = accounts_by_agent_id[agent.id]
            equity = account.value(close_prices.loc[today])
            target_weights = behaviour.decide(DecisionContext(symbols, close_prices.loc[:today], account.positions, forecast))
            accounts_by_agent_id[agent.id], trades = execute_trades(
                account, target_weights, equity, open_prices.loc[tomorrow], cost_rate, exchange_timestamp(tomorrow, MARKET_OPEN)
            )
            equity_curves_by_agent_id[agent.id].append(EquityPoint(exchange_timestamp(today, MARKET_CLOSE), equity))
            trades_by_agent_id[agent.id].extend(trades)
            step_reports.append(
                AgentStep(agent.id, equity, tuple(StepTrade(trade.symbol, trade.side, trade.quantity, trade.price) for trade in trades))
            )
        if on_step:
            on_step(StepEvent(exchange_timestamp(today, MARKET_CLOSE), tuple(step_reports)))

    agent_results = []
    for agent, _ in agent_behaviours:
        final_equity = accounts_by_agent_id[agent.id].value(close_prices.loc[trading_days[-1]])
        equity_curve = (*equity_curves_by_agent_id[agent.id], EquityPoint(exchange_timestamp(trading_days[-1], MARKET_CLOSE), final_equity))
        agent_results.append(
            AgentResult(agent.id, final_equity, final_equity / run.initial_cash - 1, equity_curve, tuple(trades_by_agent_id[agent.id]))
        )
    return SimulationResult(tuple(agent_results))
