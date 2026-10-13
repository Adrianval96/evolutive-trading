# Architecture

Hexagonal layout: the business rules sit in the middle and know nothing about HTTP, SQLite, CSV files or Kronos. Those are adapters that plug into ports.

```
evolutive_trading/
  domain/        model.py (dataclasses, Account, InvalidRunError), catalog.py (strategy parameters),
                 strategies.py (behaviours), validation.py (normalize_run)
  application/   ports.py (Forecaster, RunRepository), queue_run.py (QueueRun), simulate.py (run_simulation)
  adapters/
    inbound/     http.py            FastAPI route, pydantic request models, response envelope
    outbound/    csv_bars.py, kronos_forecaster.py, sqlite_run_repository.py
  bootstrap.py   create_app(): the only place that builds concrete adapters and wires them
trader.py        script and composition root for the Alpaca paper trader and the simulate / fetch / report commands
```

## Rules

1. **Dependency rule.** `domain` imports no other project layer and no framework. `application` imports only `domain`. Adapters import inward, never sideways: `inbound` must not import `outbound`. Only `bootstrap.py` and the root scripts know concrete adapters. `lint-imports` enforces this when you run it.
2. **Ports** are `typing.Protocol` classes in `application/ports.py`. Add one only when something calls it today, and give it only the methods that are called. `RunRepository` has `save_queued` and nothing else; the worker task adds what it needs.
3. **No wire-format dicts past the inbound adapter.** Parse into domain dataclasses at the edge and serialize at the edge.
4. **Tests.** Domain tests use plain objects. Use-case tests use in-memory fakes of the ports (`tests/fakes.py`). Adapter tests use the real thing: a real SQLite file, a real CSV. Do not mock classes we own.
5. **Names say what things are.** Single letters only for plain loop indexes. No placeholder values in fixtures.
6. **No code without a caller** in the same PR.
7. **Types.** Every function is annotated; `mypy --strict` runs on the package and `trader.py`.

## Known debt

- `domain/strategies.py` and `application/simulate.py` use pandas, because strategies read a closes DataFrame. `domain/model.py` and `domain/validation.py` do not, and a contract keeps it that way. `application/ports.py` imports pandas for the `Forecaster` signature.
- `trader.py` still re-implements the Kronos pick logic for live trading (`rebalance`) instead of sharing the strategy used by the simulator.
- `id` and `created_at` are generated inside `QueueRun`. A clock or id port is not added until a test needs to control them.
- The tests do not need torch or the Kronos submodule, so the real `KronosForecaster` is only tested through a fake predictor.
- There is no CI: the checks below run locally.

## Checks

```
ruff check . && ruff format --check . && mypy && lint-imports && pytest -q
```
