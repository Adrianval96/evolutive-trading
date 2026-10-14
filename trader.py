import argparse
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.adapters.forecasting.kronos_forecaster import LOOKBACK, KronosForecaster
from src.adapters.market_data.csv_bars import available_dates, load_bars
from src.application.simulate import run_simulation
from src.domain.model import AgentSpec, MarketSelection, RunRequest, StepEvent
from src.domain.validation import normalize_run

WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "LLY",
    "JNJ",
    "WMT",
    "PG",
    "MA",
    "HD",
    "COST",
    "NFLX",
]
TOP_K = 5
SIP_DELAY = timedelta(minutes=16)
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
INITIAL_CASH = 100_000.0


def daily_bars(start: datetime, end: datetime) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.models import BarSet
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    market_data_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    latest = datetime.now(UTC) - SIP_DELAY
    request = StockBarsRequest(
        symbol_or_symbols=WATCHLIST,
        timeframe=TimeFrame.Day,
        start=start,
        end=min(end, latest),
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    bar_set = market_data_client.get_stock_bars(request)
    assert isinstance(bar_set, BarSet)
    bars = bar_set.df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return bars


def simulate(path: Path, start: str | None, end: str | None, top_k: int, cost_bps: float) -> None:
    bars = load_bars(path)
    days = sorted(bars["timestamp"].unique())
    request = RunRequest(
        agents=(AgentSpec("kronos-top5", "kronos", {"top_k": top_k}),),
        market=MarketSelection(
            symbols=tuple(sorted(bars["symbol"].unique())),
            interval="1d",
            start=start or pd.Timestamp(days[min(LOOKBACK, len(days) - 2)]).strftime("%Y-%m-%d"),
            end=end or pd.Timestamp(days[-1]).strftime("%Y-%m-%d"),
        ),
        initial_cash=INITIAL_CASH,
        cost_bps=cost_bps,
    )
    run = normalize_run(request, available_dates(bars))

    def print_step(step: StepEvent) -> None:
        agent = step.agents[0]
        sides = {side: [trade.symbol for trade in agent.trades if trade.side == side] for side in ("sell", "buy")}
        print(f"{step.timestamp[:10]} equity {agent.equity:,.2f} | at next open, close: {sides['sell'] or '-'} | buy: {sides['buy'] or '-'}")

    result = run_simulation(run, bars, KronosForecaster(), on_step=print_step).agents[0]
    print(f"\n{INITIAL_CASH:,.2f} -> {result.final_equity:,.2f} ({result.total_return:+.2%})")


def fetch(start: datetime, end: datetime, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_bars(start, end)[BAR_COLUMNS].to_csv(path, index=False)
    print(f"Saved {path}")


def utc_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily bars from Alpaca and run the Kronos strategy through the simulation engine.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="save the watchlist's daily bars to a CSV")
    fetch_parser.add_argument("start", type=utc_date, help="first day, YYYY-MM-DD")
    fetch_parser.add_argument("end", type=utc_date, help="last day, YYYY-MM-DD")
    fetch_parser.add_argument("path", type=Path)
    simulate_parser = commands.add_parser("simulate", help="replay a bars CSV day by day with the Kronos strategy")
    simulate_parser.add_argument("path", type=Path)
    simulate_parser.add_argument("--start", help=f"first simulated day, YYYY-MM-DD (default: after {LOOKBACK} days of history)")
    simulate_parser.add_argument("--end", help="last day, YYYY-MM-DD (default: the last bar)")
    simulate_parser.add_argument("--top-k", type=int, default=TOP_K)
    simulate_parser.add_argument("--cost-bps", type=float, default=0, help="trading costs per side, in basis points")
    args = parser.parse_args()

    if args.command == "fetch":
        fetch(args.start, args.end, args.path)
    else:
        simulate(args.path, args.start, args.end, args.top_k, args.cost_bps)


if __name__ == "__main__":
    main()
