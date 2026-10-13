from dataclasses import dataclass, field
from typing import Protocol


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


# What a simulation reports after each step. Field names follow the StepEvent schema in openapi.yaml.
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


class Prices(Protocol):
    """Price per symbol. A pandas Series satisfies it; the model does not import pandas."""

    def __getitem__(self, symbol: str, /) -> float: ...


@dataclass(frozen=True)
class Account:
    cash: float
    positions: dict[str, Position]

    def value(self, prices: Prices) -> float:
        return float(self.cash + sum(position.shares * prices[symbol] for symbol, position in self.positions.items()))
