from collections.abc import Callable

import pandas as pd

from evolutive_trading.domain.model import Run


class FakeForecaster:
    """Answers with `answer(history, target_date)` and records every call it received."""

    def __init__(self, answer: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]) -> None:
        self._answer = answer
        self.calls: list[tuple[pd.DataFrame, pd.Timestamp]] = []

    def forecast(self, bars: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        self.calls.append((bars, target_date))
        return self._answer(bars, target_date)


class FakePredictor:
    """Stands in for the Kronos predictor. The signature is the one KronosForecaster calls."""

    def __init__(self, close_multiplier: float) -> None:
        self._close_multiplier = close_multiplier
        self.history_lengths: list[int] = []
        self.sample_counts: list[int] = []

    def predict_batch(
        self,
        *,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        sample_count: int,
        verbose: bool,
    ) -> list[pd.DataFrame]:
        self.history_lengths.extend(len(symbol_history) for symbol_history in df_list)
        self.sample_counts.append(sample_count)
        return [pd.DataFrame({"close": [symbol_history["close"].iloc[-1] * self._close_multiplier]}) for symbol_history in df_list]


class InMemoryRunRepository:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, Run]] = []

    def save_queued(self, run_id: str, created_at: str, run: Run) -> None:
        self.saved.append((run_id, created_at, run))
