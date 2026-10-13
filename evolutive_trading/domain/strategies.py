from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from evolutive_trading.domain.catalog import STRATEGY_PARAMETERS, Parameter
from evolutive_trading.domain.model import Position


@dataclass(frozen=True)
class DecisionContext:
    symbols: tuple[str, ...]
    closes: pd.DataFrame
    positions: dict[str, Position]
    forecast: Callable[[], pd.Series]


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
    strategy_id: Strategy(strategy_id, STRATEGY_PARAMETERS[strategy_id], behaviour_class)
    for strategy_id, behaviour_class in (
        ("buy-and-hold", BuyAndHold),
        ("kronos-top5", KronosTopPicks),
        ("ma-crossover", MovingAverageCrossover),
    )
}
