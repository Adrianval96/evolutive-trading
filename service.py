import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import engine

ROOT = Path(__file__).parent
SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    request TEXT NOT NULL
)"""


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None = None
    strategy: str
    parameters: dict[str, float] = Field(default_factory=dict)


class MarketSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list[str] = Field(min_length=1)
    interval: str
    start: date
    end: date


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: list[AgentSpec] = Field(min_length=1)
    market: MarketSelection
    initial_cash: float = Field(gt=0)
    cost_bps: float = Field(default=0, ge=0)


def envelope(data, status_code):
    return JSONResponse({"success": True, "data": data, "error": None, "meta": None}, status_code=status_code)


def failure(code, message, status_code):
    return JSONResponse({"success": False, "data": None, "error": {"code": code, "message": message}, "meta": None}, status_code=status_code)


def connect(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute(SCHEMA)
    return connection


def create_app(db_path=None, bars_path=None):
    db_path = Path(db_path or os.environ.get("EVOLUTIVE_DB", ROOT / "data" / "runs.sqlite"))
    bars_path = Path(bars_path or os.environ.get("EVOLUTIVE_BARS", ROOT / "data" / "bars.csv"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connect(db_path).close()
    market = engine.market_info(engine.load_bars(bars_path))
    app = FastAPI(title="evolutive-trading engine", version="1.0.0-draft")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        first = error.errors()[0]
        if first["type"] == "json_invalid":
            return failure("invalid_parameters", "The request body is not valid JSON.", 400)
        where = ".".join(str(part) for part in first["loc"] if part != "body")
        return failure("invalid_parameters", f"{where}: {first['msg']}", 400)

    @app.post("/v1/runs")
    def create_run(body: RunRequest):
        try:
            run = engine.normalize_run(body.model_dump(mode="json"), market)
        except engine.InvalidRun as error:
            return failure("invalid_parameters", str(error), 400)
        run_id, created_at = f"run_{uuid.uuid4().hex[:8]}", datetime.now(UTC).isoformat(timespec="seconds")
        with closing(connect(db_path)) as connection, connection:
            connection.execute("INSERT INTO runs (id, status, created_at, request) VALUES (?, 'queued', ?, ?)", (run_id, created_at, json.dumps(run)))
        return envelope({"id": run_id, "status": "queued", "created_at": created_at,
                         "agents": [{"id": agent["id"], "label": agent["label"]} for agent in run["agents"]]}, 202)

    return app
