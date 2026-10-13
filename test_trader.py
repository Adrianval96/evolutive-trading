import math

import pandas as pd

from trader import rebalance, simulate


def test_rebalance():
    returns = pd.Series({"AAPL": 0.03, "MSFT": -0.01, "NVDA": 0.01, "AMZN": 0.02, "META": 0.0})

    picks, to_close, to_buy = rebalance(returns, held={"NVDA", "MSFT", "OLD"}, top_k=2)

    assert picks == {"AAPL", "AMZN"}
    assert to_close == ["MSFT", "NVDA", "OLD"]
    assert to_buy == ["AAPL", "AMZN"]

    picks, to_close, to_buy = rebalance(returns, held={"AAPL"}, top_k=10)

    assert picks == {"AAPL", "AMZN", "NVDA"}
    assert to_close == []
    assert to_buy == ["AMZN", "NVDA"]

    assert rebalance(pd.Series({"AAPL": -0.02}), held={"AAPL"}, top_k=5) == (set(), ["AAPL"], [])


def bars_from(prices):
    return pd.DataFrame([
        {"symbol": symbol, "timestamp": pd.Timestamp(day), "open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1000}
        for symbol, days in prices.items()
        for day, (o, c) in days.items()
    ])


PRICES = {
    "A": {"2026-01-05": (10, 9), "2026-01-06": (10, 11), "2026-01-07": (12, 12), "2026-01-08": (12, 13)},
    "B": {"2026-01-05": (20, 20), "2026-01-06": (20, 20), "2026-01-07": (20, 21), "2026-01-08": (22, 24)},
}


def test_simulate_decides_on_the_close_and_fills_at_the_next_open():
    favourite = {5: "A", 6: "A", 7: "B"}
    calls = []

    def forecast(seen, target_date):
        today = seen["timestamp"].max()
        calls.append((today, target_date))
        return pd.DataFrame({"symbol": ["A", "B"], "exp_return": [0.01, -0.01] if favourite[today.day] == "A" else [-0.01, 0.01]})

    result = simulate(bars_from(PRICES), forecast, warmup=1, top_k=1, initial_cash=1000)

    assert [(today.day, target.day) for today, target in calls] == [(5, 6), (6, 7), (7, 8)]
    assert result.days["equity"].tolist() == [1000, 1100, 1200]
    assert math.isclose(result.final_equity, 1200 / 22 * 24)


def test_simulate_refuses_missing_bars():
    bars = bars_from(PRICES)
    try:
        simulate(bars.drop(index=5), lambda seen, target: None, warmup=1)
    except ValueError as error:
        assert "['B']" in str(error)
    else:
        raise AssertionError("expected a ValueError for a missing bar")


if __name__ == "__main__":
    test_rebalance()
    test_simulate_decides_on_the_close_and_fills_at_the_next_open()
    test_simulate_refuses_missing_bars()
    print("ok")
