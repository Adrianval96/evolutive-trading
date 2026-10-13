import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from ...domain.model import Run

SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    request TEXT NOT NULL
)"""


class SqliteRunRepository:
    def __init__(self, database_file: Path) -> None:
        self._database_file = database_file
        database_file.parent.mkdir(parents=True, exist_ok=True)
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_file)
        connection.execute(SCHEMA)
        return connection

    def save_queued(self, run_id: str, created_at: str, run: Run) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO runs (id, status, created_at, request) VALUES (?, 'queued', ?, ?)", (run_id, created_at, json.dumps(asdict(run)))
            )
