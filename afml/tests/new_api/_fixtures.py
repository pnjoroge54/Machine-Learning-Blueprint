"""
Shared test helpers for new_api tests.

load_tick_csv   : converts legacy tick CSV (Date and Time, Price, Volume)
                  to the new API format (bid=ask=price, DatetimeIndex).
                  Setting bid=ask=price ensures mid_price==price exactly,
                  so OHLC from make_bars() is numerically identical to the
                  values the old API produced on the same data.

make_synthetic_tick_df : small synthetic DataFrame for fast unit tests
                         that don't need to match legacy values.
"""

import os

import numpy as np
import pandas as pd

_TEST_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "test_data"))


def load_tick_csv(filename: str) -> pd.DataFrame:
    """Load a legacy tick CSV and return it in the new API format.

    Columns of the source CSV: Date and Time, Price, Volume.
    Returned DataFrame: bid=ask=price (zero spread), volume, DatetimeIndex.
    """
    path = os.path.join(_TEST_DATA_DIR, filename)
    raw = pd.read_csv(path)
    raw.index = pd.to_datetime(raw["Date and Time"])
    raw = raw.drop("Date and Time", axis=1)
    return pd.DataFrame(
        {
            "bid": raw["Price"].values,
            "ask": raw["Price"].values,
            "volume": raw["Volume"].values.astype(float),
        },
        index=raw.index,
    )


def make_synthetic_tick_df(n: int = 200, freq: str = "1s", tz: str = "UTC") -> pd.DataFrame:
    """Return a small synthetic tick DataFrame for fast unit tests."""
    assert n % 4 == 0, "n must be divisible by 4 for volume tile"
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
    return pd.DataFrame(
        {
            "bid": np.linspace(1.3000, 1.3200, n),
            "ask": np.linspace(1.3010, 1.3210, n),
            "volume": np.tile([10.0, 20.0, 15.0, 5.0], n // 4),
        },
        index=idx,
    )
