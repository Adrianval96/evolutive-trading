import pandas as pd
import pytest
from bars import make_bars
from fakes import FakePredictor

from src.adapters.forecasting.kronos_forecaster import KronosForecaster

PRICES = {
    "A": [(10, 9), (10, 11), (12, 12), (12, 13)],
    "B": [(20, 20), (20, 20), (20, 21), (22, 24)],
}


def test_the_kronos_forecaster_feeds_each_symbols_recent_history_and_returns_expected_returns():
    predictor = FakePredictor(close_multiplier=1.1)
    forecaster = KronosForecaster(lookback=2, sample_count=3)
    forecaster._predictor = predictor

    forecasts = forecaster.forecast(make_bars(PRICES), pd.Timestamp("2026-01-09"))

    assert predictor.history_lengths == [2, 2]
    assert predictor.sample_counts == [3]
    assert list(forecasts["symbol"]) == ["A", "B"]
    assert forecasts["exp_return"].tolist() == pytest.approx([0.1, 0.1])
