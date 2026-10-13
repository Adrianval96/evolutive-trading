import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

ROOT = Path(__file__).parent
EXCHANGE_TZ = "America/New_York"
MARKET_OPEN = timedelta(hours=9, minutes=30)
MARKET_CLOSE = timedelta(hours=16)
INTERVAL = "1d"
LOOKBACK = 512
SAMPLE_COUNT = 20
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


class InvalidRunError(ValueError):
    pass


@dataclass(frozen=True)
class AgentSpec:
    strategy: str
    label: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketSelection:
    symbols: tuple[str, ...]
    interval: str
    start: str
    end: str


@dataclass(frozen=True)
class RunRequest:
    agents: tuple[AgentSpec, ...]
    market: MarketSelection
    initial_cash: float
    cost_bps: float = 0


@dataclass(frozen=True)
class Agent:
    id: str
    label: str
    strategy: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class Run:
    agents: tuple[Agent, ...]
    market: MarketSelection
    initial_cash: float
    cost_bps: float


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class Trade:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float


@dataclass(frozen=True)
class AgentResult:
    agent_id: str
    final_equity: float
    total_return: float
    equity: tuple[EquityPoint, ...]
    trades: tuple[Trade, ...]


@dataclass(frozen=True)
class SimulationResult:
    agents: tuple[AgentResult, ...]


# What on_step receives. Field names follow the StepEvent schema in openapi.yaml.
@dataclass(frozen=True)
class StepTrade:
    symbol: str
    side: str
    quantity: float
    price: float


@dataclass(frozen=True)
class AgentStep:
    agent_id: str
    equity: float
    trades: tuple[StepTrade, ...]


@dataclass(frozen=True)
class StepEvent:
    timestamp: str
    agents: tuple[AgentStep, ...]


@dataclass(frozen=True)
class Position:
    shares: float
    entry_price: float


@dataclass(frozen=True)
class Account:
    cash: float
    positions: dict[str, Position]

    def value(self, prices: pd.Series) -> float:
        return float(self.cash + sum(position.shares * prices[symbol] for symbol, position in self.positions.items()))


@dataclass(frozen=True)
class DecisionContext:
    symbols: tuple[str, ...]
    closes: pd.DataFrame
    positions: dict[str, Position]
    forecast: Callable[[], pd.Series]


class Forecaster(Protocol):
    def forecast(self, bars: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame: ...


class StrategyBehaviour(Protocol):
    def decide(self, context: DecisionContext) -> dict[str, float]: ...


def top_picks(expected_returns: pd.Series, top_k: int) -> set[str]:
    return set(expected_returns[expected_returns > 0].nlargest(top_k).index)


class BuyAndHold:
    def __init__(self, parameters: dict[str, float]) -> None:
        pass

    def decide(self, context: DecisionContext) -> dict[str, float]:
        return {symbol: 1 / len(context.symbols) for symbol in context.symbols}


class KronosTopPicks:
    def __init__(self, parameters: dict[str, float]) -> None:
        self.top_k = int(parameters["top_k"])

    def decide(self, context: DecisionContext) -> dict[str, float]:
        return dict.fromkeys(top_picks(context.forecast(), self.top_k), 1 / self.top_k)


class MovingAverageCrossover:
    def __init__(self, parameters: dict[str, float]) -> None:
        self.fast_window = int(parameters["fast"])
        self.slow_window = int(parameters["slow"])
        self.stop_loss = parameters["stop_loss"]
        self.entry_armed_by_symbol: dict[str, bool] = {}
        self.previous_signal_by_symbol: dict[str, bool] = {}

    def decide(self, context: DecisionContext) -> dict[str, float]:
        target_weights: dict[str, float] = {}
        for symbol in context.symbols:
            symbol_closes = context.closes[symbol]
            if len(symbol_closes) < self.slow_window:
                continue
            fast_above_slow = bool(symbol_closes.iloc[-self.fast_window :].mean() > symbol_closes.iloc[-self.slow_window :].mean())
            if fast_above_slow != self.previous_signal_by_symbol.get(symbol):
                self.entry_armed_by_symbol[symbol] = True
            self.previous_signal_by_symbol[symbol] = fast_above_slow
            position = context.positions.get(symbol)
            if position and self.stop_loss and symbol_closes.iloc[-1] < position.entry_price * (1 - self.stop_loss):
                self.entry_armed_by_symbol[symbol] = False
            if fast_above_slow and self.entry_armed_by_symbol[symbol]:
                target_weights[symbol] = 1 / len(context.symbols)
        return target_weights


@dataclass(frozen=True)
class Strategy:
    id: str
    parameters: tuple[Parameter, ...]
    behaviour_class: Callable[[dict[str, float]], StrategyBehaviour]


STRATEGIES = {
    strategy.id: strategy
    for strategy in (
        Strategy("buy-and-hold", (), BuyAndHold),
        Strategy("kronos-top5", (Parameter("top_k", "integer", 5, 1, 20),), KronosTopPicks),
        Strategy(
            "ma-crossover",
            (Parameter("fast", "integer", 10, 2, 100), Parameter("slow", "integer", 50, 5, 250), Parameter("stop_loss", "number", 0.05, 0, 0.5)),
            MovingAverageCrossover,
        ),
    )
}


def load_bars(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def available_dates(bars: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """First and last day with bars, per symbol."""
    spans = bars.groupby("symbol")["timestamp"].agg(["min", "max"])
    return {symbol: (span["min"].strftime("%Y-%m-%d"), span["max"].strftime("%Y-%m-%d")) for symbol, span in spans.iterrows()}


def normalize_agent(index: int, spec: AgentSpec) -> Agent:
    strategy = STRATEGIES.get(spec.strategy)
    if strategy is None:
        raise InvalidRunError(f'Unknown strategy "{spec.strategy}".')
    label = spec.label or strategy.id
    unknown_names = sorted(set(spec.parameters) - {parameter.name for parameter in strategy.parameters})
    if unknown_names:
        raise InvalidRunError(f'{label}: Unknown parameter "{unknown_names[0]}" for {strategy.id}.')
    parameters: dict[str, float] = {}
    for parameter in strategy.parameters:
        value = spec.parameters.get(parameter.name, parameter.default)
        if parameter.type == "integer":
            if value != int(value):
                raise InvalidRunError(f"{label}: {parameter.name} must be a whole number, got {value}.")
            value = int(value)
        if not parameter.min <= value <= parameter.max:
            raise InvalidRunError(f"{label}: {parameter.name} must be between {parameter.min} and {parameter.max}, got {value}.")
        parameters[parameter.name] = value
    if strategy.id == "ma-crossover" and parameters["slow"] <= parameters["fast"]:
        raise InvalidRunError(f"{label}: slow ({parameters['slow']}) must be greater than fast ({parameters['fast']}).")
    return Agent(f"agt_{index + 1}", label, strategy.id, parameters)


def check_market(selection: MarketSelection, available: dict[str, tuple[str, str]]) -> None:
    if selection.interval != INTERVAL:
        raise InvalidRunError(f'Unknown interval "{selection.interval}".')
    if selection.start >= selection.end:
        raise InvalidRunError("start must be before end.")
    for symbol in selection.symbols:
        if symbol not in available:
            raise InvalidRunError(f"{symbol} has no {INTERVAL} data.")
        first, last = available[symbol]
        if selection.start < first or selection.end > last:
            raise InvalidRunError(f"{symbol} has {INTERVAL} data from {first} to {last} only.")


def normalize_run(request: RunRequest, available: dict[str, tuple[str, str]]) -> Run:
    agents = tuple(normalize_agent(index, spec) for index, spec in enumerate(request.agents))
    check_market(request.market, available)
    return Run(agents, request.market, request.initial_cash, request.cost_bps)


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
    agent_behaviours = [(agent, STRATEGIES[agent.strategy].behaviour_class(agent.parameters)) for agent in run.agents]
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


class KronosForecaster:
    def __init__(self, lookback: int = LOOKBACK, sample_count: int = SAMPLE_COUNT) -> None:
        self.lookback, self.sample_count = lookback, sample_count

    @functools.cached_property
    def _predictor(self) -> Any:
        sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))
        from model import Kronos, KronosPredictor, KronosTokenizer

        return KronosPredictor(Kronos.from_pretrained(MODEL).eval(), KronosTokenizer.from_pretrained(TOKENIZER).eval(), max_context=self.lookback)

    def forecast(self, bars: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        history_by_symbol = {symbol: group.tail(self.lookback) for symbol, group in bars.groupby("symbol")}
        predictions = self._predictor.predict_batch(
            df_list=[symbol_history[["open", "high", "low", "close", "volume"]] for symbol_history in history_by_symbol.values()],
            x_timestamp_list=[symbol_history["timestamp"] for symbol_history in history_by_symbol.values()],
            y_timestamp_list=[pd.Series([target_date])] * len(history_by_symbol),
            pred_len=1,
            sample_count=self.sample_count,
            verbose=False,
        )
        forecasts = pd.DataFrame(
            {
                "symbol": list(history_by_symbol),
                "last_close": [symbol_history["close"].iloc[-1] for symbol_history in history_by_symbol.values()],
                "pred_close": [prediction["close"].iloc[0] for prediction in predictions],
            }
        )
        return forecasts.assign(exp_return=forecasts["pred_close"] / forecasts["last_close"] - 1)
