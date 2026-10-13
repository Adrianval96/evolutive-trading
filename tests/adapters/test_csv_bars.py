from bars import make_bars

from src.adapters.market_data.csv_bars import available_dates

PRICES = {
    "A": [(10, 9), (10, 11), (12, 12), (12, 13)],
    "B": [(20, 20), (20, 20), (20, 21), (22, 24)],
}


def test_available_dates_lists_each_symbols_first_and_last_day():
    assert available_dates(make_bars(PRICES)) == {"A": ("2026-01-05", "2026-01-08"), "B": ("2026-01-05", "2026-01-08")}
