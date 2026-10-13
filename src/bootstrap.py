import os
from pathlib import Path

from fastapi import FastAPI

from .adapters.api.app import create_api
from .adapters.market_data.csv_bars import available_dates, load_bars
from .adapters.storage.sqlite_run_repository import SqliteRunRepository
from .application.queue_run import QueueRun

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def create_app(db_path: str | Path | None = None, bars_path: str | Path | None = None) -> FastAPI:
    """Composition root: the only place that picks the concrete adapters."""
    database_file = Path(db_path or os.environ.get("STRATEGY_ARENA_DB") or REPOSITORY_ROOT / "data" / "runs.sqlite")
    bars_file = Path(bars_path or os.environ.get("STRATEGY_ARENA_BARS") or REPOSITORY_ROOT / "data" / "bars.csv")
    queue_run = QueueRun(SqliteRunRepository(database_file), available_dates(load_bars(bars_file)))
    return create_api(queue_run)
