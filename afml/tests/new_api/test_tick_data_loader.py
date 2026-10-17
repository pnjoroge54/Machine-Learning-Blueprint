"""
Tests for afml/mt5/tick_data_loader.py.

All tests use unittest.mock to replace _load_data so no MT5 connection,
parquet files, or Dask installation are required.

The module-level `tick_data_loader = TickDataLoader(...)` call in
tick_data_loader.py invokes detect_distributed_frameworks(), which imports
dask. If dask is not installed this import fails; the guard below skips the
entire class in that case.
"""

import unittest
from unittest.mock import patch

import pandas as pd

try:
    from ...mt5.tick_data_loader import TickDataLoader

    _LOADER_AVAILABLE = True
except Exception:
    _LOADER_AVAILABLE = False
    TickDataLoader = None  # type: ignore[assignment,misc]


def _make_tick_df(n: int = 100, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"bid": 1.3, "ask": 1.302, "volume": 10.0}, index=idx)


@unittest.skipUnless(_LOADER_AVAILABLE, "TickDataLoader could not be imported")
class TestTickDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = TickDataLoader(max_cache_size_mb=5000, max_cached_symbols=20)

    # --- construction and initial state ---

    def test_init_does_not_raise(self):
        loader = TickDataLoader()
        self.assertIsNotNone(loader)

    def test_cache_info_keys(self):
        info = self.loader.get_cache_info()
        for key in ("total_cached_symbols", "total_cache_size_mb", "hit_rate", "cached_symbols"):
            self.assertIn(key, info)

    def test_cache_info_empty_state(self):
        info = self.loader.get_cache_info()
        self.assertEqual(info["hit_rate"], 0.0)
        self.assertEqual(info["cached_symbols"], [])
        self.assertEqual(info["total_cached_symbols"], 0)

    def test_clear_cache_on_empty_does_not_raise(self):
        self.loader.clear_cache()

    # --- cache miss ---

    def test_cache_miss_increments_counter(self):
        mock_df = _make_tick_df()
        with patch.object(self.loader, "_load_data", return_value=mock_df):
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-02", "ACC1")
        self.assertEqual(self.loader.cache_stats["misses"], 1)

    def test_cache_miss_calls_load_data(self):
        mock_df = _make_tick_df()
        with patch.object(self.loader, "_load_data", return_value=mock_df) as mock_load:
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-02", "ACC1")
        mock_load.assert_called_once()

    # --- cache hit ---

    def test_cache_hit_skips_second_load(self):
        mock_df = _make_tick_df(n=200)
        with patch.object(self.loader, "_load_data", return_value=mock_df) as mock_load:
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
        # Second call should be a cache hit — _load_data called only once
        mock_load.assert_called_once()
        self.assertEqual(self.loader.cache_stats["hits"], 1)

    def test_cache_hit_returns_dataframe(self):
        mock_df = _make_tick_df(n=200)
        with patch.object(self.loader, "_load_data", return_value=mock_df):
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
            result = self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
        self.assertIsInstance(result, pd.DataFrame)

    # --- partial cache ---

    def test_partial_cache_increments_counter(self):
        # Load a short range first, then request a wider range
        df_short = _make_tick_df(n=10, start="2024-01-01")
        df_ext = _make_tick_df(n=10, start="2024-01-06")

        call_num = [0]

        def mock_load(sym, start, end, acc):
            call_num[0] += 1
            return df_short if call_num[0] == 1 else df_ext

        with patch.object(self.loader, "_load_data", side_effect=mock_load):
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-05", "ACC1")
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-10", "ACC1")

        self.assertGreaterEqual(self.loader.cache_stats["partial_hits"], 1)

    # --- clear cache by symbol ---

    def test_clear_cache_by_symbol_removes_only_that_symbol(self):
        mock_df = _make_tick_df(n=200)
        with patch.object(self.loader, "_load_data", return_value=mock_df):
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
            self.loader.get_tick_data("GBPUSD", "2024-01-01", "2024-01-03", "ACC1")

        self.assertEqual(self.loader.get_cache_info()["total_cached_symbols"], 2)
        self.loader.clear_cache(symbol="EURUSD")
        info = self.loader.get_cache_info()
        self.assertEqual(info["total_cached_symbols"], 1)
        remaining = [item["symbol"] for item in info["cached_symbols"]]
        self.assertNotIn("EURUSD", remaining)
        self.assertIn("GBPUSD", remaining)

    def test_clear_all_cache(self):
        mock_df = _make_tick_df(n=200)
        with patch.object(self.loader, "_load_data", return_value=mock_df):
            self.loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-03", "ACC1")
        self.loader.clear_cache()
        self.assertEqual(self.loader.get_cache_info()["total_cached_symbols"], 0)

    # --- size-based eviction ---

    def test_size_eviction_clears_cache(self):
        tiny_loader = TickDataLoader(max_cache_size_mb=0.000001)
        mock_df = _make_tick_df(n=100)
        with patch.object(tiny_loader, "_load_data", return_value=mock_df):
            tiny_loader.get_tick_data("EURUSD", "2024-01-01", "2024-01-02", "ACC1")
        # After storing a ~KB DataFrame against a ~1-byte limit, eviction fires
        self.assertEqual(len(tiny_loader._cache), 0)
