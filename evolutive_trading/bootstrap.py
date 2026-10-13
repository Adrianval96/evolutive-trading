import os
from pathlib import Path

from fastapi import FastAPI

from evolutive_trading.adapters.inbound.http import create_api
from evolutive_trading.adapters.outbound.csv_bars import available_dates, load_bars
from evolutive_trading.adapters.outbound.sqlite_run_repository import SqliteRunRepository
from evolutive_trading.application.queue_run import QueueRun

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def create_app(db_path: str | Path | None = None, bars_path: str | Path | None = None) -> FastAPI:
    """Composition root: the only place that picks the concrete adapters."""
    database_file = Path(db_path or os.environ.get("EVOLUTIVE_DB") or REPOSITORY_ROOT / "data" / "runs.sqlite")
    bars_file = Path(bars_path or os.environ.get("EVOLUTIVE_BARS") or REPOSITORY_ROOT / "data" / "bars.csv")
    queue_run = QueueRun(SqliteRunRepository(database_file), available_dates(load_bars(bars_file)))
    return create_api(queue_run)
