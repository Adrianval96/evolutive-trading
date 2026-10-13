import json
from dataclasses import asdict

import pandas as pd
import pytest
from bars import make_bars
from contract import contract
from fakes import FakeForecaster

from evolutive_trading.application.simulate import run_simulation
from evolutive_trading.domain.model import Agent, AgentSpec, MarketSelection, Run, StepTrade

PRICES = {
    "A": [(10, 9), (10, 11), (12, 12), (12, 13)],
    "B": [(20, 20), (20, 20), (20, 21), (22, 24)],
}


def run_spec(*agents, symbols=("A", "B"), start="2026-01-05", end="2026-01-08", cost_bps=0):
    configured = tuple(Agent(f"agt_{index + 1}", spec.strategy, spec.strategy, spec.parameters) for index, spec in enumerate(agents))
    return Run(configured, MarketSelection(tuple(symbols), "1d", start, end), 1000, cost_bps)


def kronos(top_k):
    return AgentSpec("kronos-top5", parameters={"top_k": top_k})


def buy_and_hold():
    return AgentSpec("buy-and-hold")


def favouring(picks_by_day):
    def answer(history, target_date):
        today = history["timestamp"].max().strftime("%Y-%m-%d")
        favoured = picks_by_day[today]
        return pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01 if symbol in favoured else -0.01 for symbol in ["A", "B"]]})

    return FakeForecaster(answer)


def test_buy_and_hold_buys_every_symbol_once_and_never_trades_again():
    result = run_simulation(run_spec(buy_and_hold()), make_bars(PRICES))

    agent = result.agents[0]
    assert [(trade.symbol, trade.side, trade.quantity, trade.price) for trade in agent.trades] == [("A", "buy", 50, 10), ("B", "buy", 25, 20)]
    assert agent.final_equity == 50 * 13 + 25 * 24


def test_orders_fill_at_the_next_open_and_costs_are_charged_on_both_sides():
    forecaster = favouring({"2026-01-05": "A", "2026-01-06": "A", "2026-01-07": "B"})

    result = run_simulation(run_spec(kronos(1), cost_bps=10), make_bars(PRICES), forecaster)

    agent = result.agents[0]
    assert [(trade.timestamp, trade.symbol, trade.side, trade.quantity, trade.price) for trade in agent.trades] == [
        ("2026-01-06T09:30:00-05:00", "A", "buy", 100, 10),
        ("2026-01-08T09:30:00-05:00", "A", "sell", 100, 12),
        ("2026-01-08T09:30:00-05:00", "B", "buy", 54.5, 22),
    ]
    assert [equity_point.equity for equity_point in agent.equity[:3]] == pytest.approx([1000, 1099, 1199])
    assert agent.final_equity == pytest.approx(1305.601)
    assert agent.total_return == pytest.approx(0.305601)


def test_kept_positions_are_not_resized():
    forecaster = favouring({"2026-01-05": "A", "2026-01-06": "AB", "2026-01-07": "AB"})

    result = run_simulation(run_spec(kronos(2)), make_bars(PRICES), forecaster)

    trades = [(trade.symbol, trade.side, trade.quantity) for trade in result.agents[0].trades]
    assert trades == [("A", "buy", 50), ("B", "buy", 26.25)]


def test_the_forecaster_sees_only_history_up_to_each_close_and_runs_once_a_day_for_all_agents():
    forecaster = FakeForecaster(lambda history, target_date: pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01, 0.02]}))

    run_simulation(run_spec(kronos(1), kronos(2)), make_bars(PRICES), forecaster)

    assert [(history["timestamp"].max().day, pd.Timestamp(target_date).day) for history, target_date in forecaster.calls] == [(5, 6), (6, 7), (7, 8)]


def test_history_before_start_is_available_but_not_traded():
    forecaster = FakeForecaster(lambda history, target_date: pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01, -0.01]}))

    result = run_simulation(run_spec(kronos(1), start="2026-01-06"), make_bars(PRICES), forecaster)

    assert [history["timestamp"].min().day for history, _ in forecaster.calls] == [5, 5]
    assert len(result.agents[0].equity) == 3
    assert result.agents[0].trades[0].timestamp == "2026-01-07T09:30:00-05:00"


def test_crossover_enters_on_a_crossover_and_waits_for_a_fresh_one_after_a_stop():
    closes = [10, 10, 10, 10, 14, 15, 12.5, 13, 16, 16]
    opens = [10, 10, 10, 10, 10, 14, 15, 12.5, 13, 16]
    bars = make_bars({"S": list(zip(opens, closes, strict=True))})
    agent = AgentSpec("ma-crossover", parameters={"fast": 2, "slow": 4, "stop_loss": 0.1})

    result = run_simulation(run_spec(agent, symbols=("S",), end="2026-01-16"), bars)

    assert [(trade.timestamp[:10], trade.side, trade.price) for trade in result.agents[0].trades] == [
        ("2026-01-12", "buy", 14),
        ("2026-01-14", "sell", 12.5),
        ("2026-01-16", "buy", 16),
    ]


def test_agents_keep_independent_accounts():
    forecaster = favouring({"2026-01-05": "B", "2026-01-06": "A", "2026-01-07": "B"})

    alone = run_simulation(run_spec(buy_and_hold()), make_bars(PRICES))
    together = run_simulation(run_spec(buy_and_hold(), kronos(1)), make_bars(PRICES), forecaster)

    assert together.agents[0] == alone.agents[0]
    assert together.agents[1].final_equity != alone.agents[0].final_equity


def test_each_step_is_reported_as_it_happens():
    steps = []

    run_simulation(run_spec(buy_and_hold()), make_bars(PRICES), on_step=steps.append)

    assert len(steps) == 3
    for step in steps:
        contract("StepEvent", json.loads(json.dumps(asdict(step), default=float)))
    assert steps[0].timestamp == "2026-01-05T16:00:00-05:00"
    assert steps[0].agents[0].trades == (StepTrade("A", "buy", 50, 10), StepTrade("B", "buy", 25, 20))


def test_a_missing_bar_fails_loudly():
    bars = make_bars(PRICES).drop(index=5)

    with pytest.raises(ValueError, match=r"missing bars for \['B'\]"):
        run_simulation(run_spec(buy_and_hold()), bars)
