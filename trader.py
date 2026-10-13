import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "PG", "MA", "HD", "COST", "NFLX",
]
BENCHMARK = "SPY"
TOP_K = 5
LOOKBACK = 512
HISTORY_DAYS = 800
SAMPLE_COUNT = 20
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
SIP_DELAY = timedelta(minutes=16)
PREDICTIONS_CSV = ROOT / "predictions.csv"
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
INITIAL_CASH = 100_000.0


@dataclass(frozen=True)
class SimulationResult:
    final_equity: float
    days: pd.DataFrame


def clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def daily_bars(data, symbols, start, end=None):
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
    bars = data.get_stock_bars(request).df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return bars


def kronos_forecaster():
    sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))
    from model import Kronos, KronosPredictor, KronosTokenizer

    predictor = KronosPredictor(
        Kronos.from_pretrained(MODEL).eval(), KronosTokenizer.from_pretrained(TOKENIZER).eval(), max_context=LOOKBACK
    )

    def forecast(bars, target_date):
        history = {symbol: group.tail(LOOKBACK) for symbol, group in bars.groupby("symbol")}
        preds = predictor.predict_batch(
            df_list=[h[["open", "high", "low", "close", "volume"]] for h in history.values()],
            x_timestamp_list=[h["timestamp"] for h in history.values()],
            y_timestamp_list=[pd.Series([target_date])] * len(history),
            pred_len=1,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
        forecasts = pd.DataFrame({
            "symbol": list(history),
            "last_close": [h["close"].iloc[-1] for h in history.values()],
            "pred_close": [p["close"].iloc[0] for p in preds],
        })
        return forecasts.assign(exp_return=forecasts["pred_close"] / forecasts["last_close"] - 1)

    return forecast


def rebalance(expected_returns, held, top_k):
    picks = set(expected_returns[expected_returns > 0].nlargest(top_k).index)
    return picks, sorted(held - picks), sorted(picks - held)


def trade():
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    trading, data = clients()
    clock = trading.get_clock()
    if clock.is_open:
        sys.exit("Market is open. Run outside market hours so the last daily candle is complete.")
    target_date = pd.Timestamp(clock.next_open.date())

    forecast = kronos_forecaster()
    forecasts = forecast(daily_bars(data, WATCHLIST, datetime.now(UTC) - timedelta(days=HISTORY_DAYS)), target_date)

    trading.cancel_orders()
    held = {position.symbol for position in trading.get_all_positions()}
    picks, to_close, to_buy = rebalance(forecasts.set_index("symbol")["exp_return"], held, TOP_K)

    for symbol in to_close:
        trading.close_position(symbol)
    notional = round(float(trading.get_account().equity) / TOP_K, 2)
    for symbol in to_buy:
        trading.submit_order(
            MarketOrderRequest(symbol=symbol, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        )

    log = forecasts.assign(
        run_at=datetime.now(UTC).isoformat(timespec="seconds"),
        target_date=target_date.date(),
        picked=forecasts["symbol"].isin(picks),
    )
    log.to_csv(PREDICTIONS_CSV, mode="a", header=not PREDICTIONS_CSV.exists(), index=False)

    print(forecasts.sort_values("exp_return", ascending=False).to_string(index=False))
    print(f"\nTarget {target_date.date()} | close: {to_close or '-'} | buy ${notional} each: {to_buy or '-'}")


def simulate(bars, forecast, warmup=LOOKBACK, top_k=TOP_K, initial_cash=INITIAL_CASH):
    opens = bars.pivot(index="timestamp", columns="symbol", values="open")
    closes = bars.pivot(index="timestamp", columns="symbol", values="close")
    missing = sorted(set(opens.columns[opens.isna().any()]) | set(closes.columns[closes.isna().any()]))
    if missing:
        raise ValueError(f"Every symbol needs a bar on every day; missing bars for {missing}.")
    days = closes.index
    if not 1 <= warmup < len(days):
        raise ValueError(f"Warmup must be between 1 and {len(days) - 1} days, got {warmup}.")

    cash, shares, rows = initial_cash, {}, []
    for today, tomorrow in zip(days[warmup - 1 : -1], days[warmup:]):
        forecasts = forecast(bars[bars["timestamp"] <= today], tomorrow)
        _, to_close, to_buy = rebalance(forecasts.set_index("symbol")["exp_return"], set(shares), top_k)
        equity = cash + sum(n * closes.at[today, symbol] for symbol, n in shares.items())
        notional = equity / top_k
        for symbol in to_close:
            cash += shares.pop(symbol) * opens.at[tomorrow, symbol]
        for symbol in to_buy:
            shares[symbol] = notional / opens.at[tomorrow, symbol]
            cash -= notional
        rows.append({"date": today, "equity": equity, "sold": to_close, "bought": to_buy})
        print(f"{today.date()} equity {equity:,.2f} | at next open, close: {to_close or '-'} | buy: {to_buy or '-'}")

    final_equity = cash + sum(n * closes.at[days[-1], symbol] for symbol, n in shares.items())
    return SimulationResult(final_equity, pd.DataFrame(rows))


def fetch(start, end, path):
    _, data = clients()
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_bars(data, WATCHLIST, start, end)[BAR_COLUMNS].to_csv(path, index=False)
    print(f"Saved {path}")


def change(series):
    return f"{series.iloc[-1] / series.iloc[0] - 1:+.2%}"


def report():
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    trading, data = clients()
    history = trading.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    equity = pd.Series(history.equity, index=pd.to_datetime(history.timestamp, unit="s")).dropna()
    equity = equity[equity > 0]
    benchmark = daily_bars(data, [BENCHMARK], equity.index[0])["close"]
    print(f"Since {equity.index[0].date()}: strategy {change(equity)} | {BENCHMARK} {change(benchmark)}")


def utc_date(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def main():
    parser = argparse.ArgumentParser(description="Kronos daily paper trader. With no command, it trades the paper account.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("report", help="compare the paper account with SPY")
    fetch_parser = commands.add_parser("fetch", help="save the watchlist's daily bars to a CSV")
    fetch_parser.add_argument("start", type=utc_date, help="first day, YYYY-MM-DD")
    fetch_parser.add_argument("end", type=utc_date, help="last day, YYYY-MM-DD")
    fetch_parser.add_argument("path", type=Path)
    simulate_parser = commands.add_parser("simulate", help="replay a bars CSV day by day with Kronos")
    simulate_parser.add_argument("path", type=Path)
    simulate_parser.add_argument("--warmup", type=int, default=LOOKBACK, help="days of history before the first decision")
    args = parser.parse_args()

    if args.command == "report":
        report()
    elif args.command == "fetch":
        fetch(args.start, args.end, args.path)
    elif args.command == "simulate":
        result = simulate(pd.read_csv(args.path, parse_dates=["timestamp"]), kronos_forecaster(), warmup=args.warmup)
        print(f"\n{INITIAL_CASH:,.2f} -> {result.final_equity:,.2f} ({result.final_equity / INITIAL_CASH - 1:+.2%})")
    else:
        trade()


if __name__ == "__main__":
    main()
