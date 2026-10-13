import pandas as pd


def make_bars(prices, start="2026-01-05"):
    days = pd.bdate_range(start, periods=max(len(price_sequence) for price_sequence in prices.values()))
    return pd.DataFrame([
        {"symbol": symbol, "timestamp": days[day_index], "open": open_price, "high": max(open_price, close_price), "low": min(open_price, close_price), "close": close_price, "volume": 1000}
        for symbol, price_sequence in prices.items()
        for day_index, (open_price, close_price) in enumerate(price_sequence)
    ])
