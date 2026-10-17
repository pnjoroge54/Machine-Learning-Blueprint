"""
Tests for afml/util/misc.py utility functions.
"""

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from ...util.misc import (
    count_max_decimals,
    crop_data_frame_in_batches,
    date_conversion,
    flatten_column_names,
    optimize_dtypes,
    set_resampling_freq,
)


class TestDateConversion(unittest.TestCase):
    def test_strings_return_utc_timestamps(self):
        start, end = date_conversion("2024-01-01", "2024-06-01")
        self.assertIsInstance(start, pd.Timestamp)
        self.assertIsInstance(end, pd.Timestamp)
        self.assertIsNotNone(start.tzinfo)
        self.assertEqual(str(start.tzinfo), "UTC")

    def test_datetime_objects(self):
        start, end = date_conversion(datetime(2024, 1, 1), datetime(2024, 6, 1))
        self.assertIsInstance(start, pd.Timestamp)
        self.assertIsNotNone(start.tzinfo)

    def test_wrong_order_raises(self):
        with self.assertRaises(ValueError):
            date_conversion("2024-06-01", "2024-01-01")

    def test_equal_dates_raise(self):
        with self.assertRaises(ValueError):
            date_conversion("2024-01-01", "2024-01-01")

    def test_none_start_raises(self):
        with self.assertRaises(ValueError):
            date_conversion(None, "2024-06-01")

    def test_none_end_raises(self):
        with self.assertRaises(ValueError):
            date_conversion("2024-01-01", None)

    def test_tz_aware_input_converted_to_utc(self):
        start_eastern = pd.Timestamp("2024-01-01", tz="US/Eastern")
        end_eastern = pd.Timestamp("2024-06-01", tz="US/Eastern")
        start, end = date_conversion(start_eastern, end_eastern)
        self.assertEqual(str(start.tzinfo), "UTC")
        self.assertEqual(str(end.tzinfo), "UTC")

    def test_order_preserved(self):
        start, end = date_conversion("2024-01-01", "2024-12-31")
        self.assertLess(start, end)


class TestSetResamplingFreq(unittest.TestCase):
    def test_d1_gives_business_day(self):
        self.assertEqual(set_resampling_freq("D1"), "B")

    def test_w1_gives_weekly_friday(self):
        self.assertEqual(set_resampling_freq("W1"), "W-FRI")

    def test_h4_gives_4h(self):
        self.assertEqual(set_resampling_freq("H4"), "4h")

    def test_m15_gives_15min(self):
        self.assertEqual(set_resampling_freq("M15"), "15min")

    def test_s30_gives_30s(self):
        self.assertEqual(set_resampling_freq("S30"), "30s")

    def test_h1_gives_1h(self):
        self.assertEqual(set_resampling_freq("H1"), "1h")

    def test_m1_gives_1min(self):
        self.assertEqual(set_resampling_freq("M1"), "1min")

    def test_lowercase_input(self):
        self.assertEqual(set_resampling_freq("d1"), "B")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            set_resampling_freq("X1")

    def test_unknown_prefix_raises(self):
        with self.assertRaises(ValueError):
            set_resampling_freq("Q4")


class TestCountMaxDecimals(unittest.TestCase):
    def test_three_decimal_places(self):
        result = count_max_decimals(np.array([1.5, 2.25, 3.125]))
        self.assertEqual(result, 3)

    def test_zero_decimal_places(self):
        result = count_max_decimals(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result, 0)

    def test_accepts_series(self):
        result = count_max_decimals(pd.Series([1.5, 2.25]))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 2)

    def test_nans_are_ignored(self):
        result = count_max_decimals(np.array([1.5, np.nan, 2.25]))
        self.assertEqual(result, 2)

    def test_single_value(self):
        result = count_max_decimals(np.array([0.1]))
        self.assertGreaterEqual(result, 1)

    def test_returns_int(self):
        result = count_max_decimals(np.array([1.5]))
        self.assertIsInstance(result, int)


class TestCropDataFrameInBatches(unittest.TestCase):
    def test_chunk_count(self):
        df = pd.DataFrame({"x": range(10)})
        chunks = crop_data_frame_in_batches(df, chunksize=3)
        self.assertEqual(len(chunks), 4)

    def test_chunk_sizes(self):
        df = pd.DataFrame({"x": range(10)})
        chunks = crop_data_frame_in_batches(df, chunksize=3)
        sizes = [len(c) for c in chunks]
        self.assertEqual(sizes, [3, 3, 3, 1])

    def test_exact_divisor(self):
        df = pd.DataFrame({"x": range(9)})
        chunks = crop_data_frame_in_batches(df, chunksize=3)
        self.assertEqual(len(chunks), 3)

    def test_returns_list_of_dataframes(self):
        df = pd.DataFrame({"x": range(5)})
        chunks = crop_data_frame_in_batches(df, chunksize=2)
        for chunk in chunks:
            self.assertIsInstance(chunk, pd.DataFrame)

    def test_concatenated_equals_original(self):
        df = pd.DataFrame({"x": range(10)})
        chunks = crop_data_frame_in_batches(df, chunksize=3)
        reconstructed = pd.concat(chunks)
        pd.testing.assert_frame_equal(df, reconstructed)


class TestFlattenColumnNames(unittest.TestCase):
    def test_multiindex_columns(self):
        df = pd.DataFrame(
            [[1, 2]],
            columns=pd.MultiIndex.from_tuples([("bid", "open"), ("ask", "open")]),
        )
        result = flatten_column_names(df)
        self.assertEqual(result, ["bid_open", "ask_open"])

    def test_flat_columns_unchanged(self):
        df = pd.DataFrame([[1, 2]], columns=["open", "close"])
        result = flatten_column_names(df)
        self.assertEqual(result, ["open", "close"])

    def test_returns_list(self):
        df = pd.DataFrame([[1]], columns=["x"])
        result = flatten_column_names(df)
        self.assertIsInstance(result, list)

    def test_three_level_tuple(self):
        df = pd.DataFrame(
            [[1]],
            columns=pd.MultiIndex.from_tuples([("a", "b", "c")]),
        )
        result = flatten_column_names(df)
        self.assertEqual(result, ["a_b_c"])


class TestOptimizeDtypes(unittest.TestCase):
    def test_integer_column_downcasted(self):
        df = pd.DataFrame({"small": np.array([0, 1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, verbose=False)
        self.assertLess(result["small"].dtype.itemsize, 8)

    def test_low_cardinality_object_becomes_category(self):
        df = pd.DataFrame({"label": ["a", "b", "a", "b"] * 10})
        result = optimize_dtypes(df, verbose=False)
        self.assertEqual(str(result["label"].dtype), "category")

    def test_returns_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = optimize_dtypes(df, verbose=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int64)})
        original_dtype = df["x"].dtype
        optimize_dtypes(df, verbose=False)
        self.assertEqual(df["x"].dtype, original_dtype)
