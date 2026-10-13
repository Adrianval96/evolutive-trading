import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetPortfolioHistoryRequest, MarketOrderRequest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

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


def clients():
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def daily_bars(data, symbols, start):
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=datetime.now(UTC) - SIP_DELAY,
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    bars = data.get_stock_bars(request).df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return bars


def forecast(bars, target_date):
    predictor = KronosPredictor(
        Kronos.from_pretrained(MODEL).eval(), KronosTokenizer.from_pretrained(TOKENIZER).eval(), max_context=LOOKBACK
    )
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


def rebalance(expected_returns, held, top_k):
    picks = set(expected_returns[expected_returns > 0].nlargest(top_k).index)
    return picks, sorted(held - picks), sorted(picks - held)


def trade():
    trading, data = clients()
    clock = trading.get_clock()
    if clock.is_open:
        sys.exit("Market is open. Run outside market hours so the last daily candle is complete.")
    target_date = pd.Timestamp(clock.next_open.date())

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


def change(series):
    return f"{series.iloc[-1] / series.iloc[0] - 1:+.2%}"


def report():
    trading, data = clients()
    history = trading.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    equity = pd.Series(history.equity, index=pd.to_datetime(history.timestamp, unit="s")).dropna()
    equity = equity[equity > 0]
    benchmark = daily_bars(data, [BENCHMARK], equity.index[0])["close"]
    print(f"Since {equity.index[0].date()}: strategy {change(equity)} | {BENCHMARK} {change(benchmark)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        trade()
