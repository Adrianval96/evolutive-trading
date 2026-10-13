import pytest
from fakes import InMemoryRunRepository

from src.application.queue_run import QueueRun
from src.domain.model import AgentSpec, InvalidRunError, MarketSelection, RunRequest

AVAILABLE = {"AAPL": ("2026-01-05", "2026-02-27")}


def request(strategy="buy-and-hold"):
    return RunRequest((AgentSpec(strategy),), MarketSelection(("AAPL",), "1d", "2026-01-12", "2026-02-13"), 1000)


def test_a_valid_request_is_saved_once_as_queued_with_the_ids_it_returns():
    repository = InMemoryRunRepository()

    queued = QueueRun(repository, AVAILABLE).execute(request())

    assert repository.saved == [(queued.id, queued.created_at, queued.run)]
    assert queued.id.startswith("run_")
    assert [agent.id for agent in queued.run.agents] == ["agt_1"]


def test_an_invalid_request_saves_nothing():
    repository = InMemoryRunRepository()

    with pytest.raises(InvalidRunError):
        QueueRun(repository, AVAILABLE).execute(request(strategy="unknown-strategy"))

    assert repository.saved == []
