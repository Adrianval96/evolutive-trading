from datetime import date
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ...application.queue_run import QueueRun
from ...domain import model


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


def to_domain_request(body: RunRequest) -> model.RunRequest:
    return model.RunRequest(
        agents=tuple(model.AgentSpec(agent.strategy, agent.label, agent.parameters) for agent in body.agents),
        market=model.MarketSelection(tuple(body.market.symbols), body.market.interval, body.market.start.isoformat(), body.market.end.isoformat()),
        initial_cash=body.initial_cash,
        cost_bps=body.cost_bps,
    )


def envelope(data: dict[str, Any], status_code: int) -> JSONResponse:
    return JSONResponse({"success": True, "data": data, "error": None, "meta": None}, status_code=status_code)


def failure(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"success": False, "data": None, "error": {"code": code, "message": message}, "meta": None}, status_code=status_code)


def create_api(queue_run: QueueRun) -> FastAPI:
    api = FastAPI(title="evolutive-trading engine", version="1.0.0-draft")

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0]
        if first["type"] == "json_invalid":
            return failure("invalid_parameters", "The request body is not valid JSON.", 400)
        where = ".".join(str(part) for part in first["loc"] if part != "body")
        return failure("invalid_parameters", f"{where}: {first['msg']}", 400)

    @api.post("/v1/runs")
    def create_run(body: RunRequest) -> JSONResponse:
        try:
            queued = queue_run.execute(to_domain_request(body))
        except model.InvalidRunError as error:
            return failure("invalid_parameters", str(error), 400)
        return envelope(
            {
                "id": queued.id,
                "status": "queued",
                "created_at": queued.created_at,
                "agents": [{"id": agent.id, "label": agent.label} for agent in queued.run.agents],
            },
            202,
        )

    return api
