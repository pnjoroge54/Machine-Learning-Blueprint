"""
Regression tests for the incremental sequential-bootstrap sampler.

The production sampler (``seq_bootstrap`` →
``_seq_bootstrap_loop_incremental``) replaces the O(sample_length *
total_active_bars) reference loop (``_seq_bootstrap_loop``) with an
inverse-index driven loop that only rescans samples whose average-uniqueness
score actually changed after each draw.

The contract these tests pin down:

  TestInverseIndex
    - inverse_index_matches_bruteforce : the CSR bar -> samples index built by
      ``_build_inverse_index`` lists exactly the samples covering each bar.

  TestIncrementalEquivalence
    - identical_to_reference_random : across many randomized overlapping-label
      layouts the incremental loop yields draws bit-identical to the reference
      loop for the same pre-drawn uniforms.
    - identical_with_nonzero_initial_concurrency : equivalence still holds when
      the loop starts from a non-zero concurrency vector.
    - book_example_reproducible : the de Prado 6-bar / 3-sample example is
      reproducible across calls with a fixed seed.
    - edge_cases : empty-bar samples, a single sample, and
      sample_length > n_samples all run and stay equivalent.
"""

import unittest

import numpy as np

# Import through the real package path so the module identity matches numba's
# persistent (cache=True) overload cache.  Loading bootstrapping.py under a
# synthetic module name instead collides with that cache when a sibling test
# imports the genuine afml.sampling.bootstrapping in the same session.  The
# new_api conftest stubs the distributed-framework imports that would otherwise
# crash afml.__init__ on this machine.
from ...sampling import bootstrapping as bs


def _make_random_active(rng, n, n_bars, max_span):
    """Build a dict sample_id -> contiguous bar-index range with local overlap."""
    active = {}
    for i in range(n):
        s = int(rng.integers(0, n_bars))
        e = min(n_bars, s + int(rng.integers(1, max_span + 1)))
        active[i] = np.arange(s, e, dtype=np.int64)
    return active


def _reference_draws(flat, off, lengths, n_bars, uniforms, init_concurrency=None):
    concurrency = (
        np.zeros(n_bars, dtype=np.int64) if init_concurrency is None else init_concurrency.copy()
    )
    return bs._seq_bootstrap_loop(flat, off, lengths, concurrency, uniforms)


def _incremental_draws(flat, off, lengths, n_bars, uniforms, init_concurrency=None):
    concurrency = (
        np.zeros(n_bars, dtype=np.int64) if init_concurrency is None else init_concurrency.copy()
    )
    inv_off, inv_s = bs._build_inverse_index(flat, off, n_bars)
    return bs._seq_bootstrap_loop_incremental(
        flat, off, lengths, concurrency, uniforms, inv_off, inv_s
    )


class TestInverseIndex(unittest.TestCase):
    """The CSR inverse index must enumerate exactly the samples covering each bar."""

    def test_inverse_index_matches_bruteforce(self):
        rng = np.random.default_rng(123)
        active = _make_random_active(rng, n=40, n_bars=80, max_span=15)
        flat, off, lengths, _ = bs.pack_active_indices(active)
        n_bars = int(flat.max()) + 1

        inv_off, inv_s = bs._build_inverse_index(flat, off, n_bars)

        for bar in range(n_bars):
            got = sorted(inv_s[inv_off[bar] : inv_off[bar + 1]].tolist())
            expected = sorted(i for i in range(len(active)) if bar in set(active[i].tolist()))
            self.assertEqual(got, expected, f"inverse index mismatch at bar {bar}")


class TestIncrementalEquivalence(unittest.TestCase):
    """Incremental loop must match the reference loop bit-for-bit."""

    def test_identical_to_reference_random(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            n = int(rng.integers(5, 60))
            n_bars = int(rng.integers(10, 200))
            active = _make_random_active(rng, n, n_bars, max_span=25)
            flat, off, lengths, _ = bs.pack_active_indices(active)
            T = 0 if flat.size == 0 else int(flat.max()) + 1
            sample_length = int(rng.integers(1, 2 * n + 1))
            uniforms = (
                np.random.RandomState(int(rng.integers(0, 10_000)))
                .random_sample(sample_length)
                .astype(np.float64)
            )

            ref = _reference_draws(flat, off, lengths, T, uniforms)
            inc = _incremental_draws(flat, off, lengths, T, uniforms)
            self.assertTrue(
                np.array_equal(ref, inc),
                f"mismatch: n={n} n_bars={n_bars} sample_length={sample_length}",
            )

    def test_identical_with_nonzero_initial_concurrency(self):
        rng = np.random.default_rng(7)
        active = _make_random_active(rng, n=40, n_bars=120, max_span=20)
        flat, off, lengths, _ = bs.pack_active_indices(active)
        T = int(flat.max()) + 1
        init = rng.integers(0, 4, size=T).astype(np.int64)
        uniforms = np.random.RandomState(99).random_sample(50).astype(np.float64)

        ref = _reference_draws(flat, off, lengths, T, uniforms, init)
        inc = _incremental_draws(flat, off, lengths, T, uniforms, init)
        self.assertTrue(np.array_equal(ref, inc))

    def test_book_example_reproducible(self):
        active = {
            0: np.array([0, 1, 2], dtype=np.int64),
            1: np.array([2, 3], dtype=np.int64),
            2: np.array([4, 5], dtype=np.int64),
            3: np.empty(0, dtype=np.int64),
        }
        first = bs.seq_bootstrap(active, sample_length=6, random_seed=42)
        second = bs.seq_bootstrap(active, sample_length=6, random_seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

    def test_edge_cases(self):
        # All-empty samples -> uniform sampling, must still run and be equivalent.
        empty = {0: np.empty(0, dtype=np.int64), 1: np.empty(0, dtype=np.int64)}
        flat, off, lengths, _ = bs.pack_active_indices(empty)
        T = 0 if flat.size == 0 else int(flat.max()) + 1
        uniforms = np.random.RandomState(1).random_sample(4).astype(np.float64)
        self.assertTrue(
            np.array_equal(
                _reference_draws(flat, off, lengths, T, uniforms),
                _incremental_draws(flat, off, lengths, T, uniforms),
            )
        )

        # Single sample.
        single = {0: np.array([0, 1, 2], dtype=np.int64)}
        flat, off, lengths, _ = bs.pack_active_indices(single)
        T = int(flat.max()) + 1
        uniforms = np.random.RandomState(2).random_sample(5).astype(np.float64)
        inc = _incremental_draws(flat, off, lengths, T, uniforms)
        self.assertTrue(np.all(inc == 0))

        # sample_length greater than number of samples.
        rng = np.random.default_rng(5)
        active = _make_random_active(rng, n=8, n_bars=20, max_span=6)
        flat, off, lengths, _ = bs.pack_active_indices(active)
        T = int(flat.max()) + 1
        uniforms = np.random.RandomState(3).random_sample(40).astype(np.float64)
        self.assertTrue(
            np.array_equal(
                _reference_draws(flat, off, lengths, T, uniforms),
                _incremental_draws(flat, off, lengths, T, uniforms),
            )
        )


if __name__ == "__main__":
    unittest.main()
