import pandas as pd

from trader import rebalance


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


if __name__ == "__main__":
    test_rebalance()
    print("ok")
