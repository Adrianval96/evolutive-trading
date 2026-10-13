import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).parent
EXCHANGE_TZ = "America/New_York"
MARKET_OPEN = timedelta(hours=9, minutes=30)
MARKET_CLOSE = timedelta(hours=16)
LOOKBACK = 512
SAMPLE_COUNT = 20
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
INTERVALS = ({"id": "1d", "label": "Daily", "max_range_days": None},)


class InvalidRun(ValueError):
    pass


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
    steps_total: int

    def to_contract(self):
        return {"agents": [{
            "agent_id": a.agent_id, "final_equity": a.final_equity, "total_return": a.total_return,
            "equity": [asdict(p) for p in a.equity], "trades": [asdict(t) for t in a.trades],
        } for a in self.agents]}


@dataclass(frozen=True)
class Position:
    shares: float
    entry_price: float


@dataclass(frozen=True)
class Account:
    cash: float
    positions: dict

    def value(self, prices):
        return self.cash + sum(p.shares * prices[symbol] for symbol, p in self.positions.items())


@dataclass(frozen=True)
class FillResult:
    account: Account
    trades: tuple[Trade, ...]


@dataclass(frozen=True)
class Day:
    date: pd.Timestamp
    next_date: pd.Timestamp
    symbols: tuple[str, ...]
    closes: pd.DataFrame
    entries: dict
    forecast: Callable[[], pd.Series]


def top_picks(expected_returns, top_k):
    return set(expected_returns[expected_returns > 0].nlargest(top_k).index)


class BuyAndHold:
    def __init__(self, parameters):
        pass

    def decide(self, day):
        return {symbol: 1 / len(day.symbols) for symbol in day.symbols}


class KronosTop:
    def __init__(self, parameters):
        self.top_k = parameters["top_k"]

    def decide(self, day):
        return {symbol: 1 / self.top_k for symbol in top_picks(day.forecast(), self.top_k)}


class MovingAverageCrossover:
    def __init__(self, parameters):
        self.fast, self.slow, self.stop_loss = parameters["fast"], parameters["slow"], parameters["stop_loss"]
        self.armed, self.previous = {}, {}

    def decide(self, day):
        targets = {}
        for symbol in day.symbols:
            closes = day.closes[symbol]
            if len(closes) < self.slow:
                continue
            signal = closes.iloc[-self.fast:].mean() > closes.iloc[-self.slow:].mean()
            if signal != self.previous.get(symbol):
                self.armed[symbol] = True
            self.previous[symbol] = signal
            entry = day.entries.get(symbol)
            if entry is not None and self.stop_loss and closes.iloc[-1] < entry * (1 - self.stop_loss):
                self.armed[symbol] = False
            if signal and self.armed[symbol]:
                targets[symbol] = 1 / len(day.symbols)
        return targets


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    description: str
    parameters: tuple[Parameter, ...]
    policy: type


STRATEGIES = {s.id: s for s in (
    Strategy("buy-and-hold", "Buy and hold", "Buys every symbol in equal weight on the first day and never trades again.", (), BuyAndHold),
    Strategy("kronos-top5", "Kronos top picks", "Holds, in equal weight, the symbols with the best positive Kronos forecast for the next close.",
             (Parameter("top_k", "integer", 5, 1, 20),), KronosTop),
    Strategy("ma-crossover", "Moving-average crossover",
             "Holds a symbol while the fast average of its closes is above the slow one. After a stop-loss exit it waits for a fresh crossover.",
             (Parameter("fast", "integer", 10, 2, 100), Parameter("slow", "integer", 50, 5, 250), Parameter("stop_loss", "number", 0.05, 0, 0.5)),
             MovingAverageCrossover),
)}


def strategies():
    return [{"id": s.id, "name": s.name, "description": s.description, "parameters": [asdict(p) for p in s.parameters]}
            for s in sorted(STRATEGIES.values(), key=lambda s: s.id)]


def load_bars(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def market_info(bars):
    spans = bars.groupby("symbol")["timestamp"].agg(["min", "max"])
    return {
        "intervals": [dict(i) for i in INTERVALS],
        "symbols": [{"symbol": symbol, "name": symbol,
                     "available": {"1d": {"from": span["min"].strftime("%Y-%m-%d"), "to": span["max"].strftime("%Y-%m-%d")}}}
                    for symbol, span in spans.iterrows()],
    }


def normalize_agent(index, agent):
    strategy = STRATEGIES.get(agent["strategy"])
    if strategy is None:
        raise InvalidRun(f'Unknown strategy "{agent["strategy"]}".')
    label = agent.get("label") or strategy.id
    given = agent.get("parameters") or {}
    unknown = sorted(set(given) - {p.name for p in strategy.parameters})
    if unknown:
        raise InvalidRun(f'{label}: Unknown parameter "{unknown[0]}" for {strategy.id}.')
    parameters = {}
    for p in strategy.parameters:
        value = given.get(p.name, p.default)
        if p.type == "integer":
            if value != int(value):
                raise InvalidRun(f"{label}: {p.name} must be a whole number, got {value}.")
            value = int(value)
        if not p.min <= value <= p.max:
            raise InvalidRun(f"{label}: {p.name} must be between {p.min} and {p.max}, got {value}.")
        parameters[p.name] = value
    if strategy.id == "ma-crossover" and parameters["slow"] <= parameters["fast"]:
        raise InvalidRun(f"{label}: slow ({parameters['slow']}) must be greater than fast ({parameters['fast']}).")
    return {"id": f"agt_{index + 1}", "label": label, "strategy": strategy.id, "parameters": parameters}


def check_market(selection, market):
    interval = selection["interval"]
    if interval not in {i["id"] for i in market["intervals"]}:
        raise InvalidRun(f'Unknown interval "{interval}".')
    if selection["start"] >= selection["end"]:
        raise InvalidRun("start must be before end.")
    available = {s["symbol"]: s["available"] for s in market["symbols"]}
    for symbol in selection["symbols"]:
        span = available.get(symbol, {}).get(interval)
        if span is None:
            raise InvalidRun(f"{symbol} has no {interval} data.")
        if selection["start"] < span["from"] or selection["end"] > span["to"]:
            raise InvalidRun(f"{symbol} has {interval} data from {span['from']} to {span['to']} only.")


def normalize_run(request, market):
    agents = [normalize_agent(i, agent) for i, agent in enumerate(request["agents"])]
    selection = {key: request["market"][key] for key in ("symbols", "interval", "start", "end")}
    check_market(selection, market)
    return {"agents": agents, "market": selection, "initial_cash": request["initial_cash"], "cost_bps": request.get("cost_bps", 0)}


def at_close(day):
    return (day + MARKET_CLOSE).tz_localize(EXCHANGE_TZ).isoformat()


def at_open(day):
    return (day + MARKET_OPEN).tz_localize(EXCHANGE_TZ).isoformat()


def fill(account, targets, equity, prices, cost, when):
    cash, positions, trades = account.cash, dict(account.positions), []
    for symbol in sorted(set(positions) - set(targets)):
        position, price = positions.pop(symbol), prices[symbol]
        cash += position.shares * price * (1 - cost)
        trades.append(Trade(when, symbol, "sell", position.shares, price))
    for symbol in sorted(set(targets) - set(positions)):
        notional, price = equity * targets[symbol], prices[symbol]
        cash -= notional * (1 + cost)
        positions[symbol] = Position(notional / price, price)
        trades.append(Trade(when, symbol, "buy", notional / price, price))
    return FillResult(Account(cash, positions), tuple(trades))


def once(compute):
    cache = []

    def get():
        if not cache:
            cache.append(compute())
        return cache[0]

    return get


def run_simulation(run, bars, forecaster=None, on_step=None):
    selection = run["market"]
    symbols = tuple(sorted(selection["symbols"]))
    data = bars[bars["symbol"].isin(symbols) & (bars["timestamp"] <= pd.Timestamp(selection["end"]))]
    opens = data.pivot(index="timestamp", columns="symbol", values="open")
    closes = data.pivot(index="timestamp", columns="symbol", values="close")
    missing = sorted(set(symbols) - set(closes.columns)
                     | set(opens.columns[opens.isna().any()]) | set(closes.columns[closes.isna().any()]))
    if missing:
        raise ValueError(f"Every symbol needs a bar on every day; missing bars for {missing}.")
    days = closes.index
    steps = range(days.searchsorted(pd.Timestamp(selection["start"])), len(days) - 1)
    if not len(steps):
        raise ValueError("The range needs at least two trading days from its start.")

    cost = run.get("cost_bps", 0) / 1e4
    agents = [(a, STRATEGIES[a["strategy"]].policy(a["parameters"])) for a in run["agents"]]
    accounts = {a["id"]: Account(run["initial_cash"], {}) for a, _ in agents}
    curves = {a["id"]: [] for a, _ in agents}
    fills = {a["id"]: [] for a, _ in agents}

    for i in steps:
        today, tomorrow = days[i], days[i + 1]
        history = data[data["timestamp"] <= today]
        forecast = once(lambda h=history, t=tomorrow: forecaster.forecast(h, t).set_index("symbol")["exp_return"])
        reported = []
        for agent, policy in agents:
            account = accounts[agent["id"]]
            equity = account.value(closes.loc[today])
            entries = {symbol: p.entry_price for symbol, p in account.positions.items()}
            targets = policy.decide(Day(today, tomorrow, symbols, closes.loc[:today], entries, forecast))
            filled = fill(account, targets, equity, opens.loc[tomorrow], cost, at_open(tomorrow))
            accounts[agent["id"]] = filled.account
            curves[agent["id"]].append(EquityPoint(at_close(today), equity))
            fills[agent["id"]].extend(filled.trades)
            reported.append({"agent_id": agent["id"], "equity": equity, "trades": [
                {"symbol": t.symbol, "side": t.side, "quantity": t.quantity, "price": t.price} for t in filled.trades]})
        if on_step:
            on_step({"timestamp": at_close(today), "agents": reported})

    results = []
    for agent, _ in agents:
        final = accounts[agent["id"]].value(closes.loc[days[-1]])
        curve = (*curves[agent["id"]], EquityPoint(at_close(days[-1]), final))
        results.append(AgentResult(agent["id"], final, final / run["initial_cash"] - 1, curve, tuple(fills[agent["id"]])))
    return SimulationResult(tuple(results), len(steps))


class KronosForecaster:
    def __init__(self, lookback=LOOKBACK, sample_count=SAMPLE_COUNT):
        self.lookback, self.sample_count, self._predictor = lookback, sample_count, None

    def _load(self):
        if self._predictor is None:
            sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))
            from model import Kronos, KronosPredictor, KronosTokenizer

            self._predictor = KronosPredictor(
                Kronos.from_pretrained(MODEL).eval(), KronosTokenizer.from_pretrained(TOKENIZER).eval(), max_context=self.lookback)
        return self._predictor

    def forecast(self, bars, target_date):
        history = {symbol: group.tail(self.lookback) for symbol, group in bars.groupby("symbol")}
        preds = self._load().predict_batch(
            df_list=[h[["open", "high", "low", "close", "volume"]] for h in history.values()],
            x_timestamp_list=[h["timestamp"] for h in history.values()],
            y_timestamp_list=[pd.Series([target_date])] * len(history),
            pred_len=1,
            sample_count=self.sample_count,
            verbose=False,
        )
        forecasts = pd.DataFrame({
            "symbol": list(history),
            "last_close": [h["close"].iloc[-1] for h in history.values()],
            "pred_close": [p["close"].iloc[0] for p in preds],
        })
        return forecasts.assign(exp_return=forecasts["pred_close"] / forecasts["last_close"] - 1)
