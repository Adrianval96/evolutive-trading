import functools
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LOOKBACK = 512
SAMPLE_COUNT = 20
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


class KronosForecaster:
    def __init__(self, lookback: int = LOOKBACK, sample_count: int = SAMPLE_COUNT) -> None:
        self.lookback, self.sample_count = lookback, sample_count

    @functools.cached_property
    def _predictor(self) -> Any:
        sys.path.insert(0, str(REPOSITORY_ROOT / "vendor" / "Kronos"))
        from model import Kronos, KronosPredictor, KronosTokenizer

        return KronosPredictor(Kronos.from_pretrained(MODEL).eval(), KronosTokenizer.from_pretrained(TOKENIZER).eval(), max_context=self.lookback)

    def forecast(self, bars: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        history_by_symbol = {symbol: group.tail(self.lookback) for symbol, group in bars.groupby("symbol")}
        predictions = self._predictor.predict_batch(
            df_list=[symbol_history[["open", "high", "low", "close", "volume"]] for symbol_history in history_by_symbol.values()],
            x_timestamp_list=[symbol_history["timestamp"] for symbol_history in history_by_symbol.values()],
            y_timestamp_list=[pd.Series([target_date])] * len(history_by_symbol),
            pred_len=1,
            sample_count=self.sample_count,
            verbose=False,
        )
        forecasts = pd.DataFrame(
            {
                "symbol": list(history_by_symbol),
                "last_close": [symbol_history["close"].iloc[-1] for symbol_history in history_by_symbol.values()],
                "pred_close": [prediction["close"].iloc[0] for prediction in predictions],
            }
        )
        return forecasts.assign(exp_return=forecasts["pred_close"] / forecasts["last_close"] - 1)
