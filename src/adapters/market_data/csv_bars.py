from pathlib import Path

import pandas as pd


def load_bars(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def available_dates(bars: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """First and last day with bars, per symbol."""
    spans = bars.groupby("symbol")["timestamp"].agg(["min", "max"])
    return {symbol: (span["min"].strftime("%Y-%m-%d"), span["max"].strftime("%Y-%m-%d")) for symbol, span in spans.iterrows()}
