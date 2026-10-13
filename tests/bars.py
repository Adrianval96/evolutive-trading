import pandas as pd


def make_bars(prices, start="2026-01-05"):
    days = pd.bdate_range(start, periods=max(len(seq) for seq in prices.values()))
    return pd.DataFrame([
        {"symbol": symbol, "timestamp": days[i], "open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1000}
        for symbol, seq in prices.items()
        for i, (o, c) in enumerate(seq)
    ])
