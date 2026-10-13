import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from evolutive_trading.application.ports import RunRepository
from evolutive_trading.domain.model import Run, RunRequest
from evolutive_trading.domain.validation import normalize_run


@dataclass(frozen=True)
class QueuedRun:
    id: str
    created_at: str
    run: Run


class QueueRun:
    def __init__(self, repository: RunRepository, available_dates: dict[str, tuple[str, str]]) -> None:
        self._repository = repository
        self._available_dates = available_dates

    def execute(self, request: RunRequest) -> QueuedRun:
        """Validate the request and store it as a queued run. Raises InvalidRunError, saving nothing."""
        run = normalize_run(request, self._available_dates)
        queued = QueuedRun(f"run_{uuid.uuid4().hex[:8]}", datetime.now(UTC).isoformat(timespec="seconds"), run)
        self._repository.save_queued(queued.id, queued.created_at, run)
        return queued
