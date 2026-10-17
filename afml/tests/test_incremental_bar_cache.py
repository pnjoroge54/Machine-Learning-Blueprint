"""
Tests for IncrementalBarCache and logging_config.

The critical test is the determinism invariant:

    from_scratch([t₀, t₂]) == extend(cached([t₀, t₁]), [t₁, t₂])

for each supported bar type.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


from afml.cache.incremental_bar_cache import (
    IncrementalBarCache,
    _detect_gap,
    _estimate_tick_freq,
    _extract_leftover_standard,
    _hash_bar_params,
)

# ---------------------------------------------------------------------------
# Synthetic tick data factory
# ---------------------------------------------------------------------------


def make_synthetic_ticks(
    n_ticks: int = 10_000,
    start: str = "2021-01-01",
    freq: str = "500ms",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic tick data with bid/ask/volume.

    Returns a DataFrame with DatetimeIndex and columns:
        bid, ask, volume
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=n_ticks, freq=freq)

    # Random walk for mid price
    returns = rng.normal(0, 0.0002, size=n_ticks)
    mid = 1.10000 + np.cumsum(returns)

    # Spread: 1-3 pips
    half_spread = rng.uniform(0.00005, 0.00015, size=n_ticks)

    volume = rng.poisson(lam=5, size=n_ticks).astype(float) + 1.0

    return pd.DataFrame(
        {
            "bid": mid - half_spread,
            "ask": mid + half_spread,
            "volume": volume,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Stub make_bars (self-contained, no afml dependency)
# ---------------------------------------------------------------------------


def stub_make_bars(
    tick_df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size=100,
    price: str = "mid_price",
    tick_num: bool = True,
    verbose: bool = False,
    exp_ticks_init=None,
    exp_imbalance_init: float = 0.1,
    ewm_span: int = 20,
    **kwargs,  # newer make_bars options (e.g. target_timeframe) are ignored
) -> pd.DataFrame:
    """Minimal bar construction for testing.

    Supports tick, volume, dollar, and time bars.
    Replicates the core logic from afml.data_structures.bars without
    the full dependency chain.
    """
    df = tick_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("tick_df must have DatetimeIndex")

    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    p = df[price] if price != "mid_price" else df["mid_price"]

    if bar_type == "time":
        if isinstance(bar_size, str):
            freq_map = {"M1": "1min", "M5": "5min", "H1": "1h"}
            freq = freq_map.get(bar_size, bar_size)
        else:
            freq = f"{bar_size}s"
        groups = df.resample(freq, closed="left", label="right")
        ohlc = groups[price if price != "mid_price" else "mid_price"].ohlc()
        ohlc["tick_volume"] = groups.size()
        ohlc["volume"] = groups["volume"].sum()
        ohlc = ohlc[ohlc["tick_volume"] > 0]
        if tick_num:
            ohlc["tick_num"] = ohlc["tick_volume"].cumsum()
        return ohlc

    df["_time"] = df.index

    if bar_type == "tick":
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type == "volume":
        cumsum = df["volume"].cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    elif bar_type == "dollar":
        cumsum = (df["volume"] * df["mid_price"]).cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"Unsupported bar_type: {bar_type}")

    grouped = df.groupby(bar_id)
    col = price if price != "mid_price" else "mid_price"
    ohlc = grouped[col].ohlc()
    ohlc["tick_volume"] = grouped.size()
    ohlc["volume"] = grouped["volume"].sum()
    ohlc["spread"] = grouped["spread"].mean()
    ohlc.index = grouped["_time"].last() + pd.Timedelta(microseconds=1)

    # Drop last incomplete bar
    if bar_type == "tick" and len(df) % bar_size > 0:
        ohlc = ohlc.iloc[:-1]
    elif bar_type in ("volume", "dollar"):
        # Drop the last bar which may be incomplete
        if len(ohlc) > 1:
            # Check if last bar is incomplete by comparing with avg bar size
            avg_ticks = ohlc["tick_volume"].iloc[:-1].mean()
            if ohlc["tick_volume"].iloc[-1] < avg_ticks * 0.3:
                ohlc = ohlc.iloc[:-1]

    if tick_num:
        ohlc["tick_num"] = ohlc["tick_volume"].cumsum()

    return ohlc


# ---------------------------------------------------------------------------
# Tests: utilities
# ---------------------------------------------------------------------------


class TestHashBarParams(unittest.TestCase):
    def test_deterministic(self):
        h1 = _hash_bar_params("tick", 100, "mid_price")
        h2 = _hash_bar_params("tick", 100, "mid_price")
        self.assertEqual(h1, h2)

    def test_changes_with_bar_size(self):
        h1 = _hash_bar_params("tick", 100, "mid_price")
        h2 = _hash_bar_params("tick", 200, "mid_price")
        self.assertNotEqual(h1, h2)

    def test_changes_with_bar_type(self):
        h1 = _hash_bar_params("tick", 100, "mid_price")
        h2 = _hash_bar_params("volume", 100, "mid_price")
        self.assertNotEqual(h1, h2)


class TestGapDetection(unittest.TestCase):
    def test_no_gap(self):
        t1 = pd.Timestamp("2021-01-01 12:00:00")
        t2 = pd.Timestamp("2021-01-01 12:00:01")
        freq = pd.Timedelta(seconds=1)
        self.assertFalse(_detect_gap(t1, t2, freq))

    def test_gap_detected(self):
        t1 = pd.Timestamp("2021-01-01 12:00:00")
        t2 = pd.Timestamp("2021-01-01 13:00:00")
        freq = pd.Timedelta(seconds=1)
        self.assertTrue(_detect_gap(t1, t2, freq))

    def test_custom_tolerance(self):
        t1 = pd.Timestamp("2021-01-01 12:00:00")
        t2 = pd.Timestamp("2021-01-01 12:00:05")
        freq = pd.Timedelta(seconds=1)
        self.assertFalse(_detect_gap(t1, t2, freq, tolerance_factor=10.0))
        self.assertTrue(_detect_gap(t1, t2, freq, tolerance_factor=2.0))


class TestEstimateTickFreq(unittest.TestCase):
    def test_regular_ticks(self):
        ticks = make_synthetic_ticks(n_ticks=100, freq="500ms")
        freq = _estimate_tick_freq(ticks)
        self.assertAlmostEqual(freq.total_seconds(), 0.5, places=1)

    def test_single_tick(self):
        ticks = make_synthetic_ticks(n_ticks=1)
        freq = _estimate_tick_freq(ticks)
        self.assertEqual(freq, pd.Timedelta(seconds=1))


class TestExtractLeftover(unittest.TestCase):
    def test_tick_bars_exact_fit(self):
        ticks = make_synthetic_ticks(n_ticks=500)
        ticks["mid_price"] = (ticks["bid"] + ticks["ask"]) / 2
        complete, leftover, accum = _extract_leftover_standard(ticks, "tick", 100, "mid_price")
        self.assertEqual(len(complete), 500)
        self.assertEqual(len(leftover), 0)
        self.assertEqual(accum, {})

    def test_tick_bars_with_remainder(self):
        ticks = make_synthetic_ticks(n_ticks=537)
        ticks["mid_price"] = (ticks["bid"] + ticks["ask"]) / 2
        complete, leftover, accum = _extract_leftover_standard(ticks, "tick", 100, "mid_price")
        self.assertEqual(len(complete), 500)
        self.assertEqual(len(leftover), 37)

    def test_time_bars_no_leftover(self):
        ticks = make_synthetic_ticks(n_ticks=100)
        ticks["mid_price"] = (ticks["bid"] + ticks["ask"]) / 2
        complete, leftover, accum = _extract_leftover_standard(ticks, "time", 60, "mid_price")
        self.assertEqual(len(complete), 100)
        self.assertEqual(len(leftover), 0)


# ---------------------------------------------------------------------------
# Tests: IncrementalBarCache
# ---------------------------------------------------------------------------


class TestIncrementalBarCache(unittest.TestCase):
    """Core cache tests including the determinism invariant."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = IncrementalBarCache(
            cache_dir=Path(self.tmpdir),
            make_bars_fn=stub_make_bars,
        )
        # Full tick dataset: 2021-01 through 2023-12
        self.all_ticks = make_synthetic_ticks(
            n_ticks=20_000,
            start="2021-01-01",
            freq="500ms",
            seed=42,
        )
        # Split point
        self.split_idx = 12_000
        self.ticks_part1 = self.all_ticks.iloc[: self.split_idx]
        self.ticks_part2 = self.all_ticks  # full range

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------

    def test_first_call_computes_from_scratch(self):
        bars = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        self.assertGreater(len(bars), 0)
        self.assertIn("open", bars.columns)
        self.assertIn("close", bars.columns)

    def test_second_call_returns_cached(self):
        bars1 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        bars2 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        pd.testing.assert_frame_equal(bars1, bars2)

    def test_cache_file_created(self):
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        pkl_files = list(Path(self.tmpdir).glob("bars_*.pkl"))
        self.assertEqual(len(pkl_files), 1)

    # ------------------------------------------------------------------
    # Determinism invariant (the critical tests)
    # ------------------------------------------------------------------

    def test_determinism_tick_bars(self):
        """from_scratch([t₀,t₂]) == extend(cached([t₀,t₁]), [t₁,t₂])"""
        bar_size = 100

        # Step 1: compute on part1, cache it
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=bar_size,
        )

        # Step 2: extend to full range
        extended = self.cache.get_or_extend(
            self.ticks_part2,
            bar_type="tick",
            bar_size=bar_size,
        )

        # Step 3: compute from scratch on full range
        from_scratch = stub_make_bars(
            self.ticks_part2,
            bar_type="tick",
            bar_size=bar_size,
        )

        # Compare OHLC columns (tick_num may differ due to offset logic)
        ohlc_cols = ["open", "high", "low", "close"]
        pd.testing.assert_frame_equal(
            extended[ohlc_cols].reset_index(drop=True),
            from_scratch[ohlc_cols].reset_index(drop=True),
            check_dtype=False,
            atol=1e-10,
        )

    def test_determinism_time_bars(self):
        """Time bars are trivially concatenable."""
        bar_size = "1min"

        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="time",
            bar_size=bar_size,
        )
        extended = self.cache.get_or_extend(
            self.ticks_part2,
            bar_type="time",
            bar_size=bar_size,
        )
        from_scratch = stub_make_bars(
            self.ticks_part2,
            bar_type="time",
            bar_size=bar_size,
        )

        ohlc_cols = ["open", "high", "low", "close"]
        pd.testing.assert_frame_equal(
            extended[ohlc_cols].reset_index(drop=True),
            from_scratch[ohlc_cols].reset_index(drop=True),
            check_dtype=False,
            atol=1e-10,
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_no_new_ticks(self):
        """Requesting same range twice returns cached result."""
        bars1 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        bars2 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        pd.testing.assert_frame_equal(bars1, bars2)

    def test_param_change_triggers_recompute(self):
        """Changing bar_size creates a separate cache entry."""
        bars_100 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        bars_200 = self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=200,
        )
        # Different bar sizes produce different bar counts
        self.assertNotEqual(len(bars_100), len(bars_200))
        # Two cache files should exist
        pkl_files = list(Path(self.tmpdir).glob("bars_*.pkl"))
        self.assertEqual(len(pkl_files), 2)

    def test_info_bar_triggers_full_recompute(self):
        """Information bars should warn and recompute fully.

        Skipped when using stub_make_bars (no info bar support).
        Requires real afml.data_structures.make_bars for integration test.
        """
        try:
            bars = self.cache.get_or_extend(
                self.ticks_part1,
                bar_type="tick_imbalance",
                exp_ticks_init=50,
            )
            self.assertGreater(len(bars), 0)
        except NotImplementedError:
            self.skipTest("stub_make_bars does not support info bars")

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def test_invalidate_all(self):
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=200,
        )
        count = self.cache.invalidate()
        self.assertEqual(count, 2)
        self.assertEqual(len(list(Path(self.tmpdir).glob("bars_*.pkl"))), 0)

    def test_invalidate_specific(self):
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=200,
        )
        count = self.cache.invalidate(
            bar_type="tick",
            bar_size=100,
            price="mid_price",
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(list(Path(self.tmpdir).glob("bars_*.pkl"))), 1)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def test_status_empty(self):
        status = self.cache.status()
        self.assertEqual(len(status), 0)

    def test_status_populated(self):
        self.cache.get_or_extend(
            self.ticks_part1,
            bar_type="tick",
            bar_size=100,
        )
        status = self.cache.status()
        self.assertEqual(len(status), 1)
        self.assertEqual(status.iloc[0]["bar_type"], "tick")
        self.assertGreater(status.iloc[0]["n_bars"], 0)


# ---------------------------------------------------------------------------
# Tests: logging_config
# ---------------------------------------------------------------------------


class TestLoggingConfig(unittest.TestCase):
    """Test that logging_config configures loguru exactly once."""

    def test_import_configures(self):
        from afml.logging_config import _CONFIGURED

        self.assertTrue(_CONFIGURED)

    def test_double_import_is_idempotent(self):
        from afml.logging_config import _CONSOLE_HANDLER_ID, configure_logging

        old_id = _CONSOLE_HANDLER_ID
        configure_logging()
        from afml.logging_config import _CONSOLE_HANDLER_ID as new_id

        self.assertEqual(old_id, new_id)

    def test_reconfigure_changes_level(self):
        from afml.logging_config import reconfigure_logging

        # Should not raise
        reconfigure_logging(level="DEBUG")
        reconfigure_logging(level="WARNING")
        reconfigure_logging(level="INFO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
