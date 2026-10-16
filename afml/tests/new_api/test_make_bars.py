"""
Tests for the new make_bars() API.

Two test classes:
  TestMakeBarsLegacyOHLC  - pins exact OHLC values against what the old API
                            produced on tick_data.csv (tick bars only; those
                            are unambiguous because bar_id = arange // N).
  TestMakeBarsSynthetic   - structural tests on a small synthetic DataFrame
                            covering all ten bar types plus all error paths.
"""

import unittest

import numpy as np
import pandas as pd

from ...data_structures import add_tick_sign_per_bar, make_bars
from ...data_structures.information_bars import _tick_rule
from ._fixtures import load_tick_csv, make_synthetic_tick_df

# ---------------------------------------------------------------------------
# Legacy regression class
# ---------------------------------------------------------------------------


class TestMakeBarsLegacyOHLC(unittest.TestCase):
    """
    Verifies that make_bars() reproduces the OHLC values the old
    get_tick_bars / get_volume_bars / get_dollar_bars asserted on
    tick_data.csv.
    """

    def setUp(self):
        # bid=ask=price → mid_price==price → same OHLC as the old API
        self.tick_df = load_tick_csv("tick_data.csv")

    # --- tick bars ---

    def test_tick_bars_bar_count(self):
        bars = make_bars(self.tick_df, "tick", bar_size=10)
        # 100 ticks / 10 per bar = exactly 10 complete bars
        self.assertEqual(len(bars), 10)

    def test_tick_bars_legacy_ohlc(self):
        bars = make_bars(self.tick_df, "tick", bar_size=10)
        first = bars.iloc[0]
        self.assertAlmostEqual(first["open"], 1205.0, places=2)
        self.assertAlmostEqual(first["high"], 1904.75, places=2)
        self.assertAlmostEqual(first["low"], 1005.0, places=2)
        self.assertAlmostEqual(first["close"], 1304.5, places=2)

    # --- volume bars (only open/high verified; close depends on cumsum boundary) ---

    def test_volume_bars_legacy_open_high(self):
        bars = make_bars(self.tick_df, "volume", bar_size=30)
        self.assertGreater(len(bars), 0)
        first = bars.iloc[0]
        self.assertAlmostEqual(first["open"], 1205.0, places=2)
        self.assertAlmostEqual(first["high"], 1904.75, places=2)
        self.assertAlmostEqual(first["low"], 1005.0, places=2)

    # --- dollar bars (only open/high verified) ---

    def test_dollar_bars_legacy_open_high(self):
        bars = make_bars(self.tick_df, "dollar", bar_size=100_000)
        self.assertGreater(len(bars), 0)
        first = bars.iloc[0]
        self.assertAlmostEqual(first["open"], 1205.0, places=2)
        self.assertAlmostEqual(first["high"], 1904.75, places=2)
        self.assertAlmostEqual(first["low"], 1005.0, places=2)

    # --- OHLC invariants on legacy data ---

    def test_tick_bars_ohlc_invariants(self):
        bars = make_bars(self.tick_df, "tick", bar_size=10)
        self.assertTrue((bars["high"] >= bars["low"]).all())
        self.assertTrue((bars["high"] >= bars["open"]).all())
        self.assertTrue((bars["high"] >= bars["close"]).all())
        self.assertTrue((bars["low"] <= bars["open"]).all())
        self.assertTrue((bars["low"] <= bars["close"]).all())


# ---------------------------------------------------------------------------
# Synthetic data test class
# ---------------------------------------------------------------------------


class TestMakeBarsSynthetic(unittest.TestCase):
    """
    Structural and error-path tests using a small synthetic tick DataFrame.
    """

    def setUp(self):
        self.tick_df = make_synthetic_tick_df(n=200)
        # Version without volume column for missing-volume error tests
        self.tick_df_no_volume = self.tick_df.drop(columns=["volume"])

    def _assert_ohlc_valid(self, bars: pd.DataFrame):
        self.assertIsInstance(bars.index, pd.DatetimeIndex)
        self.assertGreater(len(bars), 0)
        self.assertTrue((bars["high"] >= bars["low"]).all())
        self.assertTrue((bars["high"] >= bars["open"]).all())
        self.assertTrue((bars["high"] >= bars["close"]).all())
        self.assertTrue((bars["low"] <= bars["open"]).all())
        self.assertTrue((bars["low"] <= bars["close"]).all())

    # --- standard bar types ---

    def test_tick_bars_basic(self):
        bars = make_bars(self.tick_df, "tick", bar_size=20)
        self._assert_ohlc_valid(bars)
        for col in ("open", "high", "low", "close", "tick_volume"):
            self.assertIn(col, bars.columns)

    def test_volume_bars_basic(self):
        bars = make_bars(self.tick_df, "volume", bar_size=50)
        self._assert_ohlc_valid(bars)
        self.assertIn("volume", bars.columns)

    def test_dollar_bars_basic(self):
        bars = make_bars(self.tick_df, "dollar", bar_size=200)
        self._assert_ohlc_valid(bars)
        self.assertIn("volume", bars.columns)

    def test_dollar_bars_non_mid_price(self):
        # Regression: dollar bars are sized on volume × mid_price, so mid_price
        # must be available to the grouper even when the OHLC price column is
        # bid / ask / bid_ask. This used to raise KeyError('mid_price').
        for price in ("bid", "ask", "bid_ask"):
            with self.subTest(price=price):
                bars = make_bars(self.tick_df, "dollar", bar_size=200, price=price)
                self._assert_ohlc_valid(bars)
        # Bar boundaries are independent of the OHLC price column, so all
        # price modes must yield the same number of dollar bars.
        counts = {
            price: len(make_bars(self.tick_df, "dollar", bar_size=200, price=price))
            for price in ("mid_price", "bid", "ask", "bid_ask")
        }
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_spread_columns_present(self):
        for bar_type, size in [("tick", 20), ("volume", 50), ("dollar", 200)]:
            with self.subTest(bar_type=bar_type):
                bars = make_bars(self.tick_df, bar_type, bar_size=size)
                self.assertIn("spread", bars.columns)
                self.assertIn("spread_bps", bars.columns)

    # --- tick_num column ---

    def test_tick_num_column_present(self):
        bars = make_bars(self.tick_df, "tick", bar_size=20, tick_num=True)
        self.assertIn("tick_num", bars.columns)

    def test_tick_num_column_absent(self):
        bars = make_bars(self.tick_df, "tick", bar_size=20, tick_num=False)
        self.assertNotIn("tick_num", bars.columns)

    # --- OHLC invariants across all standard types ---

    def test_ohlc_invariants_all_standard_types(self):
        cases = [("tick", 20), ("volume", 50), ("dollar", 200)]
        for bar_type, size in cases:
            with self.subTest(bar_type=bar_type):
                bars = make_bars(self.tick_df, bar_type, bar_size=size)
                self._assert_ohlc_valid(bars)

    # --- information bar types ---

    def test_info_bar_tick_imbalance(self):
        # exp_imbalance_init=2.0 prevents density-abort on monotonic synthetic data
        bars = make_bars(self.tick_df, "tick_imbalance", exp_ticks_init=100, exp_imbalance_init=2.0)
        self._assert_ohlc_valid(bars)

    def test_info_bar_volume_imbalance(self):
        bars = make_bars(self.tick_df, "volume_imbalance", exp_ticks_init=20)
        self._assert_ohlc_valid(bars)

    def test_info_bar_dollar_imbalance(self):
        bars = make_bars(self.tick_df, "dollar_imbalance", exp_ticks_init=20)
        self._assert_ohlc_valid(bars)

    def test_info_bar_tick_runs(self):
        # exp_runs_buy_init=2.0 prevents density-abort on monotonic synthetic data
        bars = make_bars(
            self.tick_df,
            "tick_runs",
            exp_ticks_init=100,
            exp_runs_buy_init=2.0,
            exp_runs_sell_init=2.0,
        )
        self._assert_ohlc_valid(bars)

    def test_info_bar_volume_runs(self):
        bars = make_bars(self.tick_df, "volume_runs", exp_ticks_init=20)
        self._assert_ohlc_valid(bars)

    def test_info_bar_dollar_runs(self):
        bars = make_bars(self.tick_df, "dollar_runs", exp_ticks_init=20)
        self._assert_ohlc_valid(bars)

    # --- error paths ---

    def test_invalid_bar_type_raises(self):
        with self.assertRaises(NotImplementedError):
            make_bars(self.tick_df, "foo")

    def test_info_bar_no_params_raises(self):
        with self.assertRaises(ValueError):
            make_bars(self.tick_df, "tick_imbalance")

    def test_mixing_target_and_manual_raises(self):
        with self.assertRaises(ValueError):
            make_bars(
                self.tick_df,
                "tick_imbalance",
                target_timeframe="M15",
                exp_ticks_init=100,
            )

    def test_missing_volume_raises(self):
        with self.assertRaises(KeyError):
            make_bars(self.tick_df_no_volume, "volume_imbalance", exp_ticks_init=20)

    # --- bid_ask price mode ---

    def test_bid_ask_price_mode_tick_bars(self):
        bars = make_bars(self.tick_df, "tick", bar_size=20, price="bid_ask")
        self.assertGreater(len(bars), 0)
        # bid_ask mode produces bid_open, ask_open, ... as well as mid open/high/low/close
        for col in ("open", "high", "low", "close"):
            self.assertIn(col, bars.columns)

    # --- add_tick_sign ---

    def test_add_tick_sign(self):
        bars = make_bars(self.tick_df, "tick", bar_size=20, add_tick_sign=True)
        self.assertIn("tick_sign", bars.columns)
        valid_values = {"a", "b", "c", None, np.nan}
        for val in bars["tick_sign"]:
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                self.assertIn(val, {"a", "b", "c"})

    def test_tick_sign_aligns_when_volume_bar_id_skips(self):
        # A single tick whose volume spans several bar_sizes makes the internal
        # bar_id skip values, while OHLC keeps one dense row per non-empty bar.
        # tick_sign must align to those dense rows: no NaN holes, and identical
        # to the independent close-time reference path. Regression test.
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="1s")
        mid = 1.30 + 0.001 * np.sin(np.arange(n) / 4.0)
        vol = np.ones(n)
        vol[25] = 30.0  # spans ~3 bars of size 10 -> bar_id jumps, skipping ids
        tick_df = pd.DataFrame({"bid": mid, "ask": mid, "volume": vol}, index=idx)

        bars = make_bars(tick_df, "volume", bar_size=10, add_tick_sign=True)
        # Every volume bar has >=1 tick, so no label may be missing.
        self.assertTrue(bars["tick_sign"].notna().all())

        # Reference: assign membership from bar close times, not bar_id.
        ref_ticks = tick_df.copy()
        ref_ticks["mid_price"] = (ref_ticks["bid"] + ref_ticks["ask"]) / 2
        ref_ticks["tick_sign"] = _tick_rule(ref_ticks["mid_price"])
        ref = add_tick_sign_per_bar(
            make_bars(tick_df, "volume", bar_size=10, add_tick_sign=False), ref_ticks
        )
        self.assertEqual(
            list(bars["tick_sign"].astype(object)),
            list(ref["tick_sign"].astype(object)),
        )
