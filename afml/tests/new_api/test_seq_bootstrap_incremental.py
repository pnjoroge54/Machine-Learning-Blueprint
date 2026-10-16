"""
Regression tests for the sequential-bootstrap sampler.

The incremental sampler these tests were written for (``_build_inverse_index``
and ``_seq_bootstrap_loop_incremental``) is not part of
``afml.sampling.bootstrapping``; only the reference ``seq_bootstrap`` path is,
so only its contract is pinned here:

  - book_example_reproducible : the de Prado 6-bar / 3-sample example is
    reproducible across calls with a fixed seed.
"""

import unittest

import numpy as np

# Import through the real package path so the module identity matches numba's
# persistent (cache=True) overload cache.
from ...sampling import bootstrapping as bs


class TestSeqBootstrap(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
