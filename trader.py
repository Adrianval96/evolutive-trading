import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from engine import LOOKBACK, KronosForecaster, RunRequest, StepEvent, load_bars, market_info, normalize_run, run_simulation, top_picks

ROOT = Path(__file__).parent

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
BENCHMARK = "SPY"
TOP_K = 5
HISTORY_DAYS = 800
SIP_DELAY = timedelta(minutes=16)
PREDICTIONS_CSV = ROOT / "predictions.csv"
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
INITIAL_CASH = 100_000.0


def alpaca_clients() -> tuple[Any, Any]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def daily_bars(market_data_client: Any, symbols: list[str], start: datetime, end: datetime | None = None) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    latest = datetime.now(UTC) - SIP_DELAY
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=min(end, latest) if end else latest,
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    bars = market_data_client.get_stock_bars(request).df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return bars


def rebalance(expected_returns: pd.Series, held: set[str], top_k: int) -> tuple[set[str], list[str], list[str]]:
    picks = top_picks(expected_returns, top_k)
    return picks, sorted(held - picks), sorted(picks - held)


def trade() -> None:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    trading, market_data_client = alpaca_clients()
    clock = trading.get_clock()
    if clock.is_open:
        sys.exit("Market is open. Run outside market hours so the last daily candle is complete.")
    target_date = pd.Timestamp(clock.next_open.date())

    forecasts = KronosForecaster().forecast(daily_bars(market_data_client, WATCHLIST, datetime.now(UTC) - timedelta(days=HISTORY_DAYS)), target_date)

    trading.cancel_orders()
    held = {position.symbol for position in trading.get_all_positions()}
    picks, to_close, to_buy = rebalance(forecasts.set_index("symbol")["exp_return"], held, TOP_K)

    for symbol in to_close:
        trading.close_position(symbol)
    notional = round(float(trading.get_account().equity) / TOP_K, 2)
    for symbol in to_buy:
        trading.submit_order(MarketOrderRequest(symbol=symbol, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))

    log = forecasts.assign(
        run_at=datetime.now(UTC).isoformat(timespec="seconds"),
        target_date=target_date.date(),
        picked=forecasts["symbol"].isin(picks),
    )
    log.to_csv(PREDICTIONS_CSV, mode="a", header=not PREDICTIONS_CSV.exists(), index=False)

    print(forecasts.sort_values("exp_return", ascending=False).to_string(index=False))
    print(f"\nTarget {target_date.date()} | close: {to_close or '-'} | buy ${notional} each: {to_buy or '-'}")


def simulate(path: Path, start: str | None, end: str | None, top_k: int, cost_bps: float) -> None:
    bars = load_bars(path)
    days = sorted(bars["timestamp"].unique())
    request: RunRequest = {
        "agents": [{"label": "kronos", "strategy": "kronos-top5", "parameters": {"top_k": top_k}}],
        "market": {
            "symbols": sorted(bars["symbol"].unique()),
            "interval": "1d",
            "start": start or pd.Timestamp(days[min(LOOKBACK, len(days) - 2)]).strftime("%Y-%m-%d"),
            "end": end or pd.Timestamp(days[-1]).strftime("%Y-%m-%d"),
        },
        "initial_cash": INITIAL_CASH,
        "cost_bps": cost_bps,
    }
    run = normalize_run(request, market_info(bars))

    def print_step(step: StepEvent) -> None:
        agent = step["agents"][0]
        sides = {side: [executed_trade["symbol"] for executed_trade in agent["trades"] if executed_trade["side"] == side] for side in ("sell", "buy")}
        print(f"{step['timestamp'][:10]} equity {agent['equity']:,.2f} | at next open, close: {sides['sell'] or '-'} | buy: {sides['buy'] or '-'}")

    result = run_simulation(run, bars, KronosForecaster(), on_step=print_step).agents[0]
    print(f"\n{INITIAL_CASH:,.2f} -> {result.final_equity:,.2f} ({result.total_return:+.2%})")


def fetch(start: datetime, end: datetime, path: Path) -> None:
    _, market_data_client = alpaca_clients()
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_bars(market_data_client, WATCHLIST, start, end)[BAR_COLUMNS].to_csv(path, index=False)
    print(f"Saved {path}")


def percentage_change(series: pd.Series) -> str:
    return f"{series.iloc[-1] / series.iloc[0] - 1:+.2%}"


def report() -> None:
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    trading, market_data_client = alpaca_clients()
    history = trading.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    equity = pd.Series(history.equity, index=pd.to_datetime(history.timestamp, unit="s")).dropna()
    equity = equity[equity > 0]
    benchmark = daily_bars(market_data_client, [BENCHMARK], equity.index[0])["close"]
    print(f"Since {equity.index[0].date()}: strategy {percentage_change(equity)} | {BENCHMARK} {percentage_change(benchmark)}")


def utc_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos daily paper trader. With no command, it trades the paper account.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("report", help="compare the paper account with SPY")
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

    if args.command == "report":
        report()
    elif args.command == "fetch":
        fetch(args.start, args.end, args.path)
    elif args.command == "simulate":
        simulate(args.path, args.start, args.end, args.top_k, args.cost_bps)
    else:
        trade()


if __name__ == "__main__":
    main()
