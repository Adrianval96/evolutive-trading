import json
import re
from dataclasses import asdict, replace

import pandas as pd
import pytest
from bars import make_bars
from contract import contract
from mockito import ANY, captor, mock, verify, when

import engine

PRICES = {
    "A": [(10, 9), (10, 11), (12, 12), (12, 13)],
    "B": [(20, 20), (20, 20), (20, 21), (22, 24)],
}


AVAILABLE = engine.available_dates(make_bars(PRICES))


def market(symbols=("A",), interval="1d", start="2026-01-05", end="2026-01-08"):
    return engine.MarketSelection(tuple(symbols), interval, start, end)


def request(*agents, selection=None, initial_cash=1000, cost_bps=0):
    return engine.RunRequest(tuple(agents), selection or market(), initial_cash, cost_bps)


def run_spec(*agents, symbols=("A", "B"), start="2026-01-05", end="2026-01-08", initial_cash=1000, cost_bps=0):
    configured = tuple(engine.Agent(f"agt_{index + 1}", spec.strategy, spec.strategy, spec.parameters) for index, spec in enumerate(agents))
    return engine.Run(configured, market(symbols, start=start, end=end), initial_cash, cost_bps)


def kronos(top_k):
    return engine.AgentSpec("kronos-top5", parameters={"top_k": top_k})


def buy_and_hold():
    return engine.AgentSpec("buy-and-hold")


def favouring(picks_by_day):
    forecaster = mock()

    def answer(history, target_date):
        today = history["timestamp"].max().strftime("%Y-%m-%d")
        favoured = picks_by_day[today]
        return pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01 if symbol in favoured else -0.01 for symbol in ["A", "B"]]})

    when(forecaster).forecast(...).thenAnswer(answer)
    return forecaster


def test_available_dates_lists_each_symbols_first_and_last_day():
    assert engine.available_dates(make_bars(PRICES)) == {"A": ("2026-01-05", "2026-01-08"), "B": ("2026-01-05", "2026-01-08")}


def test_normalize_fills_ids_labels_defaults_and_costs():
    run = engine.normalize_run(
        request(engine.AgentSpec("kronos-top5"), engine.AgentSpec("ma-crossover", "trend", {"fast": 5.0})),
        AVAILABLE,
    )

    assert run.agents == (
        engine.Agent("agt_1", "kronos-top5", "kronos-top5", {"top_k": 5}),
        engine.Agent("agt_2", "trend", "ma-crossover", {"fast": 5, "slow": 50, "stop_loss": 0.05}),
    )
    assert run.cost_bps == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"agents": (engine.AgentSpec("unknown-strategy"),)}, 'Unknown strategy "unknown-strategy"'),
        ({"agents": (kronos(0),)}, "top_k must be between 1 and 20"),
        ({"agents": (kronos(2.5),)}, "top_k must be a whole number"),
        ({"agents": (engine.AgentSpec("kronos-top5", parameters={"window": 3}),)}, 'Unknown parameter "window"'),
        ({"agents": (engine.AgentSpec("ma-crossover", parameters={"fast": 10, "slow": 5}),)}, "slow (5) must be greater than fast (10)"),
        ({"market": market(interval="5m")}, 'Unknown interval "5m"'),
        ({"market": market(symbols=("UNLISTED",))}, "UNLISTED has no 1d data"),
        ({"market": market(start="2026-01-08", end="2026-01-05")}, "start must be before end"),
        ({"market": market(start="2025-12-01")}, "A has 1d data from 2026-01-05 to 2026-01-08"),
    ],
)
def test_normalize_rejects_runs_the_engine_cannot_simulate(changes, message):
    with pytest.raises(engine.InvalidRunError, match=re.escape(message)):
        engine.normalize_run(replace(request(kronos(1)), **changes), AVAILABLE)


def test_buy_and_hold_buys_every_symbol_once_and_never_trades_again():
    result = engine.run_simulation(run_spec(buy_and_hold()), make_bars(PRICES))

    agent = result.agents[0]
    assert [(trade.symbol, trade.side, trade.quantity, trade.price) for trade in agent.trades] == [("A", "buy", 50, 10), ("B", "buy", 25, 20)]
    assert agent.final_equity == 50 * 13 + 25 * 24


def test_orders_fill_at_the_next_open_and_costs_are_charged_on_both_sides():
    forecaster = favouring({"2026-01-05": "A", "2026-01-06": "A", "2026-01-07": "B"})

    result = engine.run_simulation(run_spec(kronos(1), cost_bps=10), make_bars(PRICES), forecaster)

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

    result = engine.run_simulation(run_spec(kronos(2)), make_bars(PRICES), forecaster)

    trades = [(trade.symbol, trade.side, trade.quantity) for trade in result.agents[0].trades]
    assert trades == [("A", "buy", 50), ("B", "buy", 26.25)]


def test_the_forecaster_sees_only_history_up_to_each_close_and_runs_once_a_day_for_all_agents():
    seen = []
    forecaster = mock()

    def answer(history, target_date):
        seen.append((history["timestamp"].max(), pd.Timestamp(target_date)))
        return pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01, 0.02]})

    when(forecaster).forecast(...).thenAnswer(answer)

    engine.run_simulation(run_spec(kronos(1), kronos(2)), make_bars(PRICES), forecaster)

    verify(forecaster, times=3).forecast(...)
    assert [(today.day, target.day) for today, target in seen] == [(5, 6), (6, 7), (7, 8)]


def test_history_before_start_is_available_but_not_traded():
    seen = []
    forecaster = mock()
    when(forecaster).forecast(...).thenAnswer(
        lambda history, target_date: seen.append(history["timestamp"].min().day) or pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01, -0.01]})
    )

    result = engine.run_simulation(run_spec(kronos(1), start="2026-01-06"), make_bars(PRICES), forecaster)

    assert seen == [5, 5]
    assert len(result.agents[0].equity) == 3
    assert result.agents[0].trades[0].timestamp == "2026-01-07T09:30:00-05:00"


def test_crossover_enters_on_a_crossover_and_waits_for_a_fresh_one_after_a_stop():
    closes = [10, 10, 10, 10, 14, 15, 12.5, 13, 16, 16]
    opens = [10, 10, 10, 10, 10, 14, 15, 12.5, 13, 16]
    bars = make_bars({"S": list(zip(opens, closes, strict=True))})
    agent = engine.AgentSpec("ma-crossover", parameters={"fast": 2, "slow": 4, "stop_loss": 0.1})

    result = engine.run_simulation(run_spec(agent, symbols=("S",), end="2026-01-16"), bars)

    assert [(trade.timestamp[:10], trade.side, trade.price) for trade in result.agents[0].trades] == [
        ("2026-01-12", "buy", 14),
        ("2026-01-14", "sell", 12.5),
        ("2026-01-16", "buy", 16),
    ]


def test_agents_keep_independent_accounts():
    hold = buy_and_hold()
    forecaster = favouring({"2026-01-05": "B", "2026-01-06": "A", "2026-01-07": "B"})

    alone = engine.run_simulation(run_spec(hold), make_bars(PRICES))
    together = engine.run_simulation(run_spec(hold, kronos(1)), make_bars(PRICES), forecaster)

    assert together.agents[0] == alone.agents[0]
    assert together.agents[1].final_equity != alone.agents[0].final_equity


def test_each_step_is_reported_as_it_happens():
    progress = mock()

    engine.run_simulation(run_spec(buy_and_hold()), make_bars(PRICES), on_step=progress.step)

    steps = captor()
    verify(progress, times=3).step(steps)
    for step in steps.all_values:
        contract("StepEvent", json.loads(json.dumps(asdict(step), default=float)))
    assert steps.all_values[0].timestamp == "2026-01-05T16:00:00-05:00"
    assert steps.all_values[0].agents[0].trades == (
        engine.StepTrade("A", "buy", 50, 10),
        engine.StepTrade("B", "buy", 25, 20),
    )


def test_a_missing_bar_fails_loudly():
    bars = make_bars(PRICES).drop(index=5)

    with pytest.raises(ValueError, match=r"missing bars for \['B'\]"):
        engine.run_simulation(run_spec(buy_and_hold()), bars)


def test_the_kronos_forecaster_feeds_each_symbols_recent_history_and_returns_expected_returns():
    predictor, lengths = mock(), []

    def answer(df_list, **_):
        lengths.extend(len(symbol_history) for symbol_history in df_list)
        return [pd.DataFrame({"close": [symbol_history["close"].iloc[-1] * 1.1]}) for symbol_history in df_list]

    when(predictor).predict_batch(...).thenAnswer(answer)
    forecaster = engine.KronosForecaster(lookback=2, sample_count=3)
    forecaster._predictor = predictor

    forecasts = forecaster.forecast(make_bars(PRICES), pd.Timestamp("2026-01-09"))

    assert lengths == [2, 2]
    assert list(forecasts["symbol"]) == ["A", "B"]
    assert forecasts["exp_return"].tolist() == pytest.approx([0.1, 0.1])
    verify(predictor).predict_batch(df_list=ANY, x_timestamp_list=ANY, y_timestamp_list=ANY, pred_len=1, sample_count=3, verbose=False)
