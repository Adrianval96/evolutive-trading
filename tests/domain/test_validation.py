import re
from dataclasses import replace

import pytest
from bars import make_bars

from evolutive_trading.adapters.outbound.csv_bars import available_dates
from evolutive_trading.domain.model import Agent, AgentSpec, InvalidRunError, MarketSelection, RunRequest
from evolutive_trading.domain.validation import normalize_run

AVAILABLE = available_dates(make_bars({"A": [(10, 9), (10, 11), (12, 12), (12, 13)]}))


def market(symbols=("A",), interval="1d", start="2026-01-05", end="2026-01-08"):
    return MarketSelection(tuple(symbols), interval, start, end)


def request(*agents, selection=None, initial_cash=1000, cost_bps=0):
    return RunRequest(tuple(agents), selection or market(), initial_cash, cost_bps)


def kronos(top_k):
    return AgentSpec("kronos-top5", parameters={"top_k": top_k})


def test_normalize_fills_ids_labels_defaults_and_costs():
    run = normalize_run(request(AgentSpec("kronos-top5"), AgentSpec("ma-crossover", "trend", {"fast": 5.0})), AVAILABLE)

    assert run.agents == (
        Agent("agt_1", "kronos-top5", "kronos-top5", {"top_k": 5}),
        Agent("agt_2", "trend", "ma-crossover", {"fast": 5, "slow": 50, "stop_loss": 0.05}),
    )
    assert run.cost_bps == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"agents": (AgentSpec("unknown-strategy"),)}, 'Unknown strategy "unknown-strategy"'),
        ({"agents": (kronos(0),)}, "top_k must be between 1 and 20"),
        ({"agents": (kronos(2.5),)}, "top_k must be a whole number"),
        ({"agents": (AgentSpec("kronos-top5", parameters={"window": 3}),)}, 'Unknown parameter "window"'),
        ({"agents": (AgentSpec("ma-crossover", parameters={"fast": 10, "slow": 5}),)}, "slow (5) must be greater than fast (10)"),
        ({"market": market(interval="5m")}, 'Unknown interval "5m"'),
        ({"market": market(symbols=("UNLISTED",))}, "UNLISTED has no 1d data"),
        ({"market": market(start="2026-01-08", end="2026-01-05")}, "start must be before end"),
        ({"market": market(start="2025-12-01")}, "A has 1d data from 2026-01-05 to 2026-01-08"),
    ],
)
def test_normalize_rejects_runs_the_engine_cannot_simulate(changes, message):
    with pytest.raises(InvalidRunError, match=re.escape(message)):
        normalize_run(replace(request(kronos(1)), **changes), AVAILABLE)
