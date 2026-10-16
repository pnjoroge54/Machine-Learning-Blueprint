"""
Tests for caching of fit_calibrated_model().

fit_calibrated_model is the @cacheable helper extracted from
ModelDevelopmentPipeline.calibrate_model.  These tests verify the cache
contract directly, without running the full pipeline:

  TestFitCalibratedModelCache
    - cache_hit_skips_recompute : identical inputs are served from the
      persisted pickle on the second call (the inner CalibratorCV.fit runs
      exactly once across two calls) and return an equal result.
    - changed_inputs_recompute  : a different CV protocol produces a fresh key,
      so the computation runs again (cache miss).

The cache directory is redirected to a temp dir for the duration of each test
so the real AFML cache is never touched.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ...cache import unified_cache as uc
from ...calibration import calibration as calib_mod
from ...production.model_development import fit_calibrated_model


def _make_calibration_inputs(n=120, n_splits=3):
    """
    Build minimal, valid inputs for fit_calibrated_model.

    Bars are spaced 5 minutes apart with a 1-minute label span (t1), so labels
    do not overlap and PurgedKFold leaves every fold with training data.  y
    alternates 0/1 so both classes are present in any contiguous fold block.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["f0", "f1"], index=idx)
    y = pd.Series(np.arange(n) % 2, index=idx, name="bin")
    t1 = pd.Series(idx + pd.Timedelta(minutes=1), index=idx, name="t1")
    sample_weight = pd.Series(np.ones(n), index=idx)
    sample_weight_score = pd.Series(np.ones(n), index=idx)
    base_pipeline = LogisticRegression(max_iter=200)
    return dict(
        base_pipeline=base_pipeline,
        X=X,
        y=y,
        sample_weight=sample_weight,
        sample_weight_score=sample_weight_score,
        t1=t1,
        n_splits=n_splits,
        pct_embargo=0.0,
        calibration_diagnostic=False,
    )


class TestFitCalibratedModelCache(unittest.TestCase):
    def setUp(self):
        # Redirect the cache to a temp dir.  The @cacheable wrapper reads
        # CACHE_DIRS["base"] at call time, so mutating the dict in place is
        # enough to isolate this test from the real cache.
        self._tmp = Path(tempfile.mkdtemp(prefix="afml_cal_cache_"))
        self._orig_base = uc.CACHE_DIRS["base"]
        uc.CACHE_DIRS["base"] = self._tmp

        # Spy on the inner computation: CalibratorCV.fit runs only on a miss.
        self._orig_fit = calib_mod.CalibratorCV.fit
        self.fit_calls = 0

        def _counting_fit(inner_self, *a, **k):
            self.fit_calls += 1
            return self._orig_fit(inner_self, *a, **k)

        self._patcher = patch.object(calib_mod.CalibratorCV, "fit", _counting_fit)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        uc.CACHE_DIRS["base"] = self._orig_base
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_cache_hit_skips_recompute(self):
        kwargs = _make_calibration_inputs()

        r1 = fit_calibrated_model(**kwargs)
        # First call is a miss: it computed and persisted a pickle.
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(len(list(self._tmp.glob("*.pkl"))), 1)

        r2 = fit_calibrated_model(**kwargs)
        # Second call with identical inputs is served from cache — no recompute.
        self.assertEqual(self.fit_calls, 1)

        # Returned tuples are equivalent.
        calibrator1, oof_brier1, scores1 = r1
        calibrator2, oof_brier2, scores2 = r2
        self.assertEqual(type(calibrator1).__name__, "CalibratorCV")
        self.assertEqual(oof_brier1, oof_brier2)
        self.assertIsNone(scores1)
        self.assertIsNone(scores2)
        np.testing.assert_allclose(calibrator1.oof_probs_, calibrator2.oof_probs_, equal_nan=True)

    def test_changed_inputs_recompute(self):
        fit_calibrated_model(**_make_calibration_inputs(n_splits=3))
        self.assertEqual(self.fit_calls, 1)

        # A different CV protocol changes the key → cache miss → recompute.
        fit_calibrated_model(**_make_calibration_inputs(n_splits=4))
        self.assertEqual(self.fit_calls, 2)
        self.assertEqual(len(list(self._tmp.glob("*.pkl"))), 2)


if __name__ == "__main__":
    unittest.main()
