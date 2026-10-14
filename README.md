# strategy-arena

A daily trading simulator. Several agents, each running a strategy, trade the same past market side by side with the same starting cash and the same trading costs, so their results can be compared like for like. Simulation only: no real money.

The engine runs behind an HTTP service with a fixed v1 contract, so a web dashboard can start runs and show them. The contract is written in OpenAPI 3.1, one file per call, and kept outside this repo: put the files in `contracts/` and the API tests check every response against them. A separate script, `trader.py`, downloads market data from Alpaca and runs the Kronos strategy through the engine from the command line.

## What exists

| Piece | State |
|---|---|
| Simulation engine | Done. On each day's close every agent sees only the bars up to that day; its orders fill at the next day's open. Costs are charged per side on the traded amount. Kept positions are not resized. Bars before the start date are visible to strategies but not traded |
| Strategies | `buy-and-hold`; `ma-crossover` (fast/slow moving averages with a stop-loss that waits for a fresh crossover); `kronos-top5` (holds the stocks with the best positive [Kronos](https://github.com/shiyu-coder/Kronos) forecast for the next close) |
| HTTP service | `POST /v1/runs` validates a run and queues it in SQLite, answering `202`. Nothing executes queued runs yet |
| `trader.py` | Fetches daily bars from Alpaca and simulates the Kronos strategy from the command line |

## What we have learned

Kronos-small has shown no forecasting edge on data after its June 2024 pre-training cutoff. Predicting the direction of the next 5-minute bar, it was right 51.1% of the time. Ranking 20 large US stocks by the next day's forecast, its ranking was unrelated to what happened (rank correlation +0.005). Its daily top 5 replaced over half its holdings every day, and at 5 bps per side it ended at +3.1% against +29.1% for holding all 20. `kronos-top5` stays in as a model-driven baseline, not as a strategy that works.

## Roadmap

1. **Engine service.** A worker that executes queued runs, then the rest of the contract: `GET /v1/runs/{id}` (status, progress, results), `GET /v1/runs/{id}/events` (live progress as server-sent events), `GET /v1/strategies` and `GET /v1/market` (what a client can offer).
2. **Dashboard.** A separate Next.js app that configures runs, follows them live and compares agents' equity curves and trades.
3. **Strategies.** Open. Ideas still to test: using Kronos for position sizing or as a generator of possible futures rather than as a forecast, trading only when an expected move clears costs, and letting agents compete so the strongest are kept and varied.

## Quick start

No torch, Alpaca account or market data needed. Python 3.12:

```bash
git clone https://github.com/Adrianval96/strategy-arena.git
cd strategy-arena
python3.12 -m venv .venv.nosync
.venv.nosync/bin/pip install -r requirements-dev.txt
```

Tests and checks:

```bash
.venv.nosync/bin/python -m pytest
.venv.nosync/bin/ruff check . && .venv.nosync/bin/mypy && .venv.nosync/bin/lint-imports
```

Start the service on the sample data and queue a run:

```bash
BARS_PATH=examples/synthetic_bars.csv .venv.nosync/bin/uvicorn src.bootstrap:create_app --factory
```

```bash
curl -X POST localhost:8000/v1/runs -H 'content-type: application/json' -d '{"agents":[{"label":"trend","strategy":"ma-crossover"},{"strategy":"buy-and-hold"}],"market":{"symbols":["SYN01","SYN02","SYN03","SYN04"],"interval":"1d","start":"2026-09-01","end":"2026-09-30"},"initial_cash":100000,"cost_bps":5}'
```

Every response is a `{ success, data, error, meta }` envelope with snake_case fields; invalid runs get a `400` with a readable message. Runs are stored in `data/runs.sqlite` (`DB_PATH` to change it).

### Sample data

[`examples/synthetic_bars.csv`](examples/synthetic_bars.csv) holds made-up daily bars, not market data: 8 symbols (`SYN01` to `SYN08`) over 153 weekdays from 2026-03-02 to 2026-09-30, following random trends that change direction every few weeks. The 50-day moving average has history from May onwards, and the last week (22 to 30 September) has fast/slow crossovers in both directions on four symbols.

## Layout

Hexagonal: `src/domain` (the rules), `src/application` (use cases and ports), `src/adapters` (HTTP API, SQLite storage, CSV market data, the Kronos forecaster) and `src/bootstrap.py` (wiring). The dependency rules, checked by `lint-imports`, are in [`docs/architecture.md`](docs/architecture.md).

## Real market data and Kronos

Kronos needs torch (about 1 GB) and the model code, which is a git submodule:

```bash
git clone --recurse-submodules https://github.com/Adrianval96/strategy-arena.git
.venv.nosync/bin/pip install -r requirements.txt
cp .env.example .env
```

`.venv.nosync` keeps torch out of iCloud. Python 3.12 because Kronos pins `pandas==2.2.2`, which has no 3.13 wheels. For market data, create a free Alpaca account, switch to **Paper Trading**, generate API keys and paste them into `.env`. Then:

```bash
set -a; . ./.env; set +a
.venv.nosync/bin/python trader.py fetch 2023-01-01 2026-10-01 data/bars.csv
.venv.nosync/bin/python trader.py simulate data/bars.csv --cost-bps 5
```

`data/` is git-ignored: market data terms don't allow publishing the bars. By default the simulation starts after 512 days of history, the context Kronos reads; set `--start` and `--end` to choose the window and `--top-k` for the number of picks. Only results after June 2024 say anything about Kronos. `trader.py simulate examples/synthetic_bars.csv --start 2026-09-22` runs Kronos on the sample data instead.
