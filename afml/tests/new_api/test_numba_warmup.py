"""
Tests for afml/numba_warmup.py.

Uses _tick_rule_kernel as the test dispatcher since it takes a simple
1-D float64 array and has well-defined, predictable output.
"""

import unittest

import numpy as np

from ...data_structures.information_bars import _tick_rule_kernel
from ...numba_warmup import (
    _NUMBA_DUMMY_ARGS,
    lazy_warmup,
    register_numba_dummy,
    warmup_all,
)


class TestWarmupAll(unittest.TestCase):
    def test_returns_int(self):
        result = warmup_all(verbose=False)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_idempotent_second_call_returns_zero(self):
        warmup_all(verbose=False)
        second = warmup_all(verbose=False)
        self.assertEqual(second, 0)

    def test_verbose_does_not_raise(self):
        warmup_all(verbose=True)


class TestRegisterNumbaDummy(unittest.TestCase):
    _TEST_KEY = "_test_register_numba_dummy_unique_9876"

    def tearDown(self):
        _NUMBA_DUMMY_ARGS.pop(self._TEST_KEY, None)

    def test_stores_args_correctly(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        register_numba_dummy(self._TEST_KEY, (arr,))
        self.assertIn(self._TEST_KEY, _NUMBA_DUMMY_ARGS)
        stored_args, stored_kwargs = _NUMBA_DUMMY_ARGS[self._TEST_KEY]
        np.testing.assert_array_equal(stored_args[0], arr)
        self.assertEqual(stored_kwargs, {})

    def test_stores_kwargs_correctly(self):
        arr = np.array([1.0], dtype=np.float64)
        register_numba_dummy(self._TEST_KEY, (arr,), {"extra": True})
        _, stored_kwargs = _NUMBA_DUMMY_ARGS[self._TEST_KEY]
        self.assertEqual(stored_kwargs, {"extra": True})

    def test_overwrite_existing_entry(self):
        arr1 = np.array([1.0], dtype=np.float64)
        arr2 = np.array([2.0, 3.0], dtype=np.float64)
        register_numba_dummy(self._TEST_KEY, (arr1,))
        register_numba_dummy(self._TEST_KEY, (arr2,))
        stored_args, _ = _NUMBA_DUMMY_ARGS[self._TEST_KEY]
        np.testing.assert_array_equal(stored_args[0], arr2)


class TestLazyWarmup(unittest.TestCase):
    def test_returns_callable(self):
        wrapped = lazy_warmup(_tick_rule_kernel)
        self.assertTrue(callable(wrapped))

    def test_is_lazy_warmup_flag(self):
        wrapped = lazy_warmup(_tick_rule_kernel)
        self.assertTrue(wrapped._is_lazy_warmup)

    def test_numba_dispatcher_attribute(self):
        wrapped = lazy_warmup(_tick_rule_kernel)
        self.assertIs(wrapped._numba_dispatcher, _tick_rule_kernel)

    def test_wrapped_produces_same_result(self):
        prices = np.array([1.0, 1.1, 0.9, 1.0, 1.1], dtype=np.float64)
        wrapped = lazy_warmup(_tick_rule_kernel)
        result = wrapped(prices)
        expected = _tick_rule_kernel(prices)
        np.testing.assert_array_equal(result, expected)

    def test_tick_rule_first_element_is_positive(self):
        prices = np.array([1.0, 0.9, 1.1], dtype=np.float64)
        wrapped = lazy_warmup(_tick_rule_kernel)
        result = wrapped(prices)
        self.assertEqual(result[0], 1.0)

    def test_tick_rule_output_values_in_set(self):
        prices = np.linspace(1.0, 1.1, 10, dtype=np.float64)
        wrapped = lazy_warmup(_tick_rule_kernel)
        result = wrapped(prices)
        self.assertTrue(np.all((result == 1.0) | (result == -1.0)))
