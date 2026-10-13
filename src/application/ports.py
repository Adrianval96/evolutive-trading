from typing import Protocol

import pandas as pd

from ..domain.model import Run


class Forecaster(Protocol):
    """Expected next-close return per symbol. Returns a frame with `symbol` and `exp_return` columns."""

    def forecast(self, bars: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame: ...


class RunRepository(Protocol):
    def save_queued(self, run_id: str, created_at: str, run: Run) -> None: ...
