import json
import sqlite3

import pytest
from bars import make_bars
from contract import contract
from fastapi.testclient import TestClient

from evolutive_trading.bootstrap import create_app

PRICES = {"AAPL": [(10, 10)] * 30, "MSFT": [(20, 20)] * 30}
RUN = {
    "agents": [{"label": "trend", "strategy": "ma-crossover", "parameters": {"fast": 5, "slow": 20}}, {"strategy": "buy-and-hold"}],
    "market": {"symbols": ["AAPL", "MSFT"], "interval": "1d", "start": "2026-01-12", "end": "2026-02-13"},
    "initial_cash": 100000,
}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "runs.sqlite"


@pytest.fixture
def client(tmp_path, db_path):
    bars_path = tmp_path / "bars.csv"
    make_bars(PRICES).to_csv(bars_path, index=False)
    return TestClient(create_app(db_path=db_path, bars_path=bars_path))


def stored_runs(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM runs")]


def test_queues_a_run_and_answers_202(client, db_path):
    response = client.post("/v1/runs", json=RUN)

    assert response.status_code == 202
    body = response.json()
    contract("RunCreatedEnvelope", body)
    assert body["data"]["status"] == "queued"
    assert body["data"]["agents"] == [{"id": "agt_1", "label": "trend"}, {"id": "agt_2", "label": "buy-and-hold"}]

    [row] = stored_runs(db_path)
    assert row["id"] == body["data"]["id"]
    assert row["status"] == "queued"
    assert row["created_at"] == body["data"]["created_at"]
    stored = json.loads(row["request"])
    assert stored["agents"][0] == {
        "id": "agt_1",
        "label": "trend",
        "strategy": "ma-crossover",
        "parameters": {"fast": 5, "slow": 20, "stop_loss": 0.05},
    }


def test_keeps_the_costs_it_was_given(client, db_path):
    client.post("/v1/runs", json={**RUN, "cost_bps": 5})

    [row] = stored_runs(db_path)
    assert json.loads(row["request"])["cost_bps"] == 5


def test_each_run_gets_its_own_id(client, db_path):
    first = client.post("/v1/runs", json=RUN).json()["data"]["id"]
    second = client.post("/v1/runs", json=RUN).json()["data"]["id"]

    assert first != second
    assert {row["id"] for row in stored_runs(db_path)} == {first, second}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"agents": [{"strategy": "unknown-strategy"}]}, 'Unknown strategy "unknown-strategy"'),
        ({"agents": []}, "agents"),
        ({"cost_bps": -1}, "cost_bps"),
        ({"initial_cash": 0}, "initial_cash"),
        ({"leverage": 2}, "leverage"),
    ],
)
def test_rejects_invalid_runs_with_the_error_envelope(client, db_path, change, message):
    response = client.post("/v1/runs", json={**RUN, **change})

    assert response.status_code == 400
    body = response.json()
    contract("ErrorEnvelope", body)
    assert body["error"]["code"] == "invalid_parameters"
    assert message in body["error"]["message"]
    assert stored_runs(db_path) == []


def test_rejects_a_body_that_is_not_json(client, db_path):
    response = client.post("/v1/runs", content="not json", headers={"content-type": "application/json"})

    assert response.status_code == 400
    contract("ErrorEnvelope", response.json())
    assert stored_runs(db_path) == []
