"""
Integration tests for IncrementalBarCache against the REAL make_bars.

The legacy suite (afml/tests/test_incremental_bar_cache.py) exercises a
self-contained ``stub_make_bars`` in a sandbox and only compares OHLC
columns, so it never caught the ways the cache diverged from the production
bar builder. These tests wire the cache to the actual
``afml.data_structures.make_bars`` and assert the full determinism invariant

    from_scratch([t0, t2]) == extend(cached([t0, t1]), [t1, t2])

— OHLC, bar count, AND tick_num — for every standard bar type, plus the
specific regressions that were fixed:

  * dollar bars raised KeyError('mid_price') in _extract_leftover_standard;
  * time / volume bars came out off-by-one at the boundary;
  * tick_num double-counted the leftover ticks;
  * cache updates failed to persist on Windows (Path.rename over an existing
    file raises FileExistsError).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ...cache.incremental_bar_cache import (
    IncrementalBarCache,
    _extract_leftover_standard,
)
from ...data_structures import make_bars

_OHLC = ["open", "high", "low", "close"]


def _random_walk_ticks(n: int, seed: int = 0) -> pd.DataFrame:
    """Tick data with a random-walk mid and Poisson volume.

    A random walk (rather than a monotone line) ensures volume/dollar bars
    accumulate at an irregular cadence, so the cache boundary rarely lands on
    an exact cumulative-metric multiple — the case that used to break.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="500ms", name="time")
    mid = 1.10 + np.cumsum(rng.normal(0, 2e-4, n))
    half_spread = rng.uniform(5e-5, 15e-5, n)
    return pd.DataFrame(
        {
            "bid": mid - half_spread,
            "ask": mid + half_spread,
            "volume": rng.poisson(5, n).astype(float) + 1.0,
        },
        index=idx,
    )


class TestDeterminismInvariant(unittest.TestCase):
    """extend(cached(part1), part2) must equal make_bars(part2) exactly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = IncrementalBarCache(cache_dir=Path(self.tmpdir))
        # 8137 ticks; split at 5537 lands off any tick / clock / cumsum edge.
        self.all_ticks = _random_walk_ticks(8137, seed=0)
        self.part1 = self.all_ticks.iloc[:5537]
        self.part2 = self.all_ticks

    def _assert_invariant(self, bar_type, **kw):
        self.cache.invalidate()
        self.cache.get_or_extend(self.part1, bar_type=bar_type, **kw)
        extended = self.cache.get_or_extend(self.part2, bar_type=bar_type, **kw)
        scratch = make_bars(self.part2, bar_type=bar_type, **kw)

        self.assertEqual(len(extended), len(scratch), f"{bar_type}: bar count mismatch")
        np.testing.assert_allclose(
            extended[_OHLC].to_numpy(),
            scratch[_OHLC].to_numpy(),
            atol=1e-9,
            err_msg=f"{bar_type}: OHLC mismatch",
        )
        if "tick_num" in scratch.columns:
            np.testing.assert_array_equal(
                extended["tick_num"].to_numpy(),
                scratch["tick_num"].to_numpy(),
                err_msg=f"{bar_type}: tick_num mismatch",
            )
        return extended

    def test_tick_invariant(self):
        self._assert_invariant("tick", bar_size=100)

    def test_time_invariant(self):
        self._assert_invariant("time", bar_size="M1")

    def test_volume_invariant(self):
        self._assert_invariant("volume", bar_size=2000)

    def test_dollar_invariant(self):
        # Regression guard: the cache's leftover extraction used to raise
        # KeyError('mid_price') here.
        self._assert_invariant("dollar", bar_size=2200)

    def test_dollar_non_mid_price_invariant(self):
        # End-to-end with price='bid': the dollar metric is sized on mid_price
        # in both make_bars and the cache's leftover extraction, while OHLC is
        # taken from the bid column. Exercises both the make_bars dollar/bid
        # fix and the cache's mid_price-from-bid/ask synthesis.
        self._assert_invariant("dollar", bar_size=2200, price="bid")


class TestIncrementalPath(unittest.TestCase):
    """Tick bars take the genuine O(new ticks) extension path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = IncrementalBarCache(cache_dir=Path(self.tmpdir))
        self.all_ticks = _random_walk_ticks(8137, seed=1)

    def test_tick_leftover_offset_is_exact(self):
        # 5537 % 100 = 37 leftover ticks: the offset fix (subtract leftover)
        # is what keeps tick_num aligned through the boundary.
        part1 = self.all_ticks.iloc[:5537]
        self.cache.get_or_extend(part1, bar_type="tick", bar_size=100)
        status = self.cache.status()
        self.assertEqual(int(status.iloc[0]["leftover_ticks"]), 37)

        extended = self.cache.get_or_extend(self.all_ticks, bar_type="tick", bar_size=100)
        scratch = make_bars(self.all_ticks, bar_type="tick", bar_size=100)
        np.testing.assert_array_equal(
            extended["tick_num"].to_numpy(), scratch["tick_num"].to_numpy()
        )

    def test_persistence_survives_update(self):
        # Regression: on Windows the cache update used Path.rename, which
        # raises over an existing file, so extensions never persisted.
        part1 = self.all_ticks.iloc[:5537]
        self.cache.get_or_extend(part1, bar_type="tick", bar_size=100)
        extended = self.cache.get_or_extend(self.all_ticks, bar_type="tick", bar_size=100)

        # A fresh cache instance on the same dir must read back the extended bars.
        fresh = IncrementalBarCache(cache_dir=Path(self.tmpdir))
        reloaded = fresh.get_or_extend(self.all_ticks, bar_type="tick", bar_size=100)
        pd.testing.assert_frame_equal(reloaded, extended)


class TestExtractLeftoverStandard(unittest.TestCase):
    """Unit coverage for the leftover extraction that mirrors make_bars."""

    def setUp(self):
        self.ticks = _random_walk_ticks(1000, seed=2)

    def test_tick_remainder(self):
        complete, leftover, _ = _extract_leftover_standard(self.ticks, "tick", 100, "mid_price")
        self.assertEqual(len(complete), 1000)
        self.assertEqual(len(leftover), 0)
        complete, leftover, _ = _extract_leftover_standard(
            self.ticks.iloc[:937], "tick", 100, "mid_price"
        )
        self.assertEqual(len(leftover), 37)

    def test_dollar_uses_mid_price_without_mid_column(self):
        # Raw ticks (no 'mid_price'): must not raise and must split sensibly.
        self.assertNotIn("mid_price", self.ticks.columns)
        complete, leftover, accum = _extract_leftover_standard(
            self.ticks, "dollar", 2000, "mid_price"
        )
        self.assertEqual(len(complete) + len(leftover), len(self.ticks))
        self.assertGreater(len(leftover), 0)
        self.assertIn("partial_cumsum", accum)

    def test_volume_leftover_is_final_bar_group(self):
        # The leftover must be exactly the trailing bar_id group that make_bars
        # drops, so complete + leftover reconstructs the full tick set.
        bar_size = 2000
        complete, leftover, _ = _extract_leftover_standard(
            self.ticks, "volume", bar_size, "mid_price"
        )
        cumsum = self.ticks["volume"].cumsum()
        bar_id = (cumsum // bar_size).astype("int64").to_numpy()
        expected_leftover = int((bar_id == bar_id[-1]).sum())
        self.assertEqual(len(leftover), expected_leftover)


if __name__ == "__main__":
    unittest.main(verbosity=2)
