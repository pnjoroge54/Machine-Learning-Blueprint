"""
Tests for :mod:`afml.screening.permutation`.

Two classes of test live here.

Unit tests check the mechanics: OHLC invariants survive bar permutation, marginal
distributions are preserved, index schemes emit valid permutations, effective
sample size responds correctly to serial dependence.

Calibration tests check the thing that actually matters: does the scheme produce
a test of the advertised size? These are stochastic and slower. They assert that
``circular_shift`` holds nominal size on a smoothed feature against overlapping
labels, and that ``naive_shuffle`` does not. The second assertion is not
decoration. It is a regression guard: naive shuffling is the obvious thing to
reach for, it is wrong, and without a test that fails on it somebody will
reintroduce it.

Run the fast set with::

    pytest afml/tests/test_screening_permutation.py -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.screening.permutation import (
    bar_permutation,
    build_null,
    circular_shift_indices,
    effective_sample_size,
    naive_shuffle_indices,
    permutation_pvalue,
    stationary_bootstrap_indices,
    suggested_block_length,
)
from afml.screening.report import ScreeningResult, results_to_frame


# =============================================================================
# Fixtures and helpers
# =============================================================================
def make_bars(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV bars with realistic internal geometry."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    open_ = close * np.exp(rng.standard_normal(n) * 0.0005)
    wick = np.abs(rng.standard_normal((n, 2))) * 0.001
    high = np.maximum(open_, close) * np.exp(wick[:, 0])
    low = np.minimum(open_, close) * np.exp(-wick[:, 1])
    volume = rng.integers(100, 10_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) series, a stand-in for a smoothed indicator."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def overlapping_forward_return(n: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Rolling sum of forward innovations, a stand-in for a triple-barrier label."""
    e = rng.standard_normal(n + horizon)
    return np.convolve(e, np.ones(horizon), mode="valid")[:n]


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Test statistic. Any bivariate dependence measure exhibits the same size
    distortion, so the cheapest one is used here."""
    sx, sy = x.std(), y.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(abs(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))


# =============================================================================
# Bar permutation
# =============================================================================
class TestBarPermutation:
    def test_ohlc_invariants_hold(self):
        """High bounds the body above and low bounds it below, in every permuted bar.

        This is the reason the intra-bar triple is shuffled as a unit. Shuffling
        high, low and close independently would break these inequalities and
        manufacture range-based signal out of nothing.
        """
        bars = make_bars(400, seed=1)
        for perm in bar_permutation(bars, n_perm=20, seed=7):
            body_max = np.maximum(perm["open"], perm["close"])
            body_min = np.minimum(perm["open"], perm["close"])
            assert (perm["high"] >= body_max - 1e-12).all()
            assert (perm["low"] <= body_min + 1e-12).all()
            assert (perm["high"] >= perm["low"]).all()
            assert (perm[["open", "high", "low", "close"]] > 0).all().all()

    def test_marginal_distributions_preserved(self):
        """Gap and body returns are reordered, not altered."""
        bars = make_bars(400, seed=2)
        o, h = np.log(bars["open"].to_numpy()), np.log(bars["high"].to_numpy())
        lo, c = np.log(bars["low"].to_numpy()), np.log(bars["close"].to_numpy())
        ref_gaps = np.sort(o[1:] - c[:-1])
        ref_body_h = np.sort(h[1:] - o[1:])

        perm = next(bar_permutation(bars, n_perm=1, seed=3))
        po, ph = np.log(perm["open"].to_numpy()), np.log(perm["high"].to_numpy())
        pc = np.log(perm["close"].to_numpy())
        np.testing.assert_allclose(np.sort(po[1:] - pc[:-1]), ref_gaps, atol=1e-10)
        np.testing.assert_allclose(np.sort(ph[1:] - po[1:]), ref_body_h, atol=1e-10)

    def test_first_bar_anchored(self):
        bars = make_bars(200, seed=4)
        perm = next(bar_permutation(bars, n_perm=1, seed=5, anchor_first=True))
        pd.testing.assert_series_equal(
            perm.iloc[0][["open", "high", "low", "close"]],
            bars.iloc[0][["open", "high", "low", "close"]],
            check_names=False,
        )

    def test_total_drift_conserved_but_path_randomized(self):
        """Bar permutation conserves terminal price exactly and randomizes the path.

        Conservation is not incidental. The permuted close is
        ``anchor + cumsum(gap + body_close)`` over a reordering of the same
        summands, so the terminal value is invariant. The synthetic series is
        therefore a random reordering of the real drift, not a driftless walk.

        The practical consequence, worth stating plainly: this null does NOT test
        whether a strategy's profit came from market drift. A permanently long
        strategy earns the same terminal return on every permuted series and will
        never be rejected. Screen drift-capturing strategies against a separate
        buy-and-hold benchmark; bar permutation answers a narrower question,
        which is whether the TIMING carried information.
        """
        n = 600
        rng = np.random.default_rng(6)
        close = 100.0 * np.exp(np.cumsum(0.002 + rng.standard_normal(n) * 0.001))
        bars = pd.DataFrame(
            {"open": close * 0.9995, "high": close * 1.001, "low": close * 0.999, "close": close},
            index=pd.RangeIndex(n),
        )
        real_drift = np.log(close[-1] / close[0])
        paths = list(bar_permutation(bars, n_perm=50, seed=8))

        drifts = np.array([np.log(p["close"].iloc[-1] / p["close"].iloc[0]) for p in paths])
        np.testing.assert_allclose(drifts, real_drift, atol=1e-9)

        # The path between the endpoints must differ from the real one.
        real_path = np.log(close)
        deviations = [np.abs(np.log(p["close"].to_numpy()) - real_path).max() for p in paths]
        assert np.median(deviations) > 5 * np.std(np.diff(real_path))

    def test_volume_travels_with_body(self):
        bars = make_bars(300, seed=9)
        perm = next(bar_permutation(bars, n_perm=1, seed=10))
        assert "volume" in perm.columns
        np.testing.assert_allclose(
            np.sort(perm["volume"].to_numpy()), np.sort(bars["volume"].to_numpy())
        )

    def test_rejects_bad_input(self):
        with pytest.raises(KeyError, match="missing required column"):
            next(bar_permutation(pd.DataFrame({"close": [1.0, 2.0, 3.0]}), n_perm=1))
        with pytest.raises(ValueError, match="at least 3 bars"):
            next(bar_permutation(make_bars(2), n_perm=1))
        bad = make_bars(50)
        bad.iloc[10, bad.columns.get_loc("low")] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            next(bar_permutation(bad, n_perm=1))


# =============================================================================
# Index schemes
# =============================================================================
class TestIndexSchemes:
    @pytest.mark.parametrize("fn", [naive_shuffle_indices, circular_shift_indices])
    def test_are_true_permutations(self, fn):
        rng = np.random.default_rng(11)
        idx = fn(200, n_perm=25, rng=rng)
        assert idx.shape == (25, 200)
        for row in idx:
            assert np.array_equal(np.sort(row), np.arange(200))

    def test_circular_shift_preserves_autocorrelation(self):
        """The whole point of rotation: the target keeps its dependence structure."""
        rng = np.random.default_rng(12)
        y = ar1(2000, 0.95, rng)
        idx = circular_shift_indices(2000, n_perm=20, rng=rng)

        def lag1(v):
            v = v - v.mean()
            return np.dot(v[:-1], v[1:]) / np.dot(v, v)

        rotated = [lag1(y[row]) for row in idx]
        assert abs(np.mean(rotated) - lag1(y)) < 0.01

    def test_naive_shuffle_destroys_autocorrelation(self):
        rng = np.random.default_rng(13)
        y = ar1(2000, 0.95, rng)
        idx = naive_shuffle_indices(2000, n_perm=20, rng=rng)

        def lag1(v):
            v = v - v.mean()
            return np.dot(v[:-1], v[1:]) / np.dot(v, v)

        assert lag1(y) > 0.9
        assert abs(np.mean([lag1(y[row]) for row in idx])) < 0.05

    def test_stationary_bootstrap_resamples_with_replacement(self):
        rng = np.random.default_rng(14)
        idx = stationary_bootstrap_indices(500, mean_block=50.0, n_perm=10, rng=rng)
        assert idx.shape == (10, 500)
        assert idx.min() >= 0 and idx.max() < 500
        # Resampling, not reordering: some indices repeat, some are absent.
        assert any(len(np.unique(row)) < 500 for row in idx)

    def test_stationary_bootstrap_block_length_controls_dependence(self):
        rng = np.random.default_rng(15)
        y = ar1(4000, 0.98, rng)

        def lag1(v):
            v = v - v.mean()
            return np.dot(v[:-1], v[1:]) / np.dot(v, v)

        short = stationary_bootstrap_indices(4000, 5.0, 10, rng)
        long_ = stationary_bootstrap_indices(4000, 400.0, 10, rng)
        assert np.mean([lag1(y[r]) for r in short]) < np.mean([lag1(y[r]) for r in long_])

    def test_short_series_rejected(self):
        rng = np.random.default_rng(16)
        with pytest.raises(ValueError, match="too short"):
            circular_shift_indices(3, n_perm=5, rng=rng, min_shift=5)


# =============================================================================
# Diagnostics
# =============================================================================
class TestDiagnostics:
    def test_effective_sample_size_tracks_dependence(self):
        rng = np.random.default_rng(17)
        n = 5000
        iid = effective_sample_size(rng.standard_normal(n))
        smooth = effective_sample_size(ar1(n, 0.95, rng))
        assert 0.7 * n < iid <= n
        assert smooth < iid / 5

    def test_pvalue_is_never_zero(self):
        assert permutation_pvalue(1e9, np.zeros(999)) == pytest.approx(1 / 1000)

    def test_pvalue_bounds_and_tails(self):
        null = np.arange(100.0)
        assert 0.0 < permutation_pvalue(50.0, null) <= 1.0
        assert permutation_pvalue(-1.0, null) == 1.0
        assert permutation_pvalue(0.0, null, tail="left") == pytest.approx(2 / 101)
        with pytest.raises(ValueError, match="tail must be"):
            permutation_pvalue(0.0, null, tail="both")

    def test_suggested_block_length_from_label_span(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="h")
        t1 = pd.Series(index[100:400], index=index[:300])  # 100-bar spans
        assert suggested_block_length(t1, index, multiple=5.0) == pytest.approx(500.0)
        assert suggested_block_length(t1, index, multiple=0.01, floor=50) == 50.0
        with pytest.raises(ValueError, match="t1 is empty"):
            suggested_block_length(pd.Series(dtype="datetime64[ns]"), index)


# =============================================================================
# ScreeningResult contract
# =============================================================================
class TestScreeningResult:
    @staticmethod
    def _result(**kw) -> ScreeningResult:
        base = dict(
            name="f",
            statistic_name="s",
            statistic=0.5,
            null_mean=0.2,
            null_std=0.1,
            null_quantiles={0.95: 0.4},
            p_value=0.01,
            n_permutations=99,
            null_scheme="circular_shift",
            effective_sample_size=120.0,
            n_observations=1000,
        )
        base.update(kw)
        return ScreeningResult(**base)

    def test_cannot_be_coerced_to_a_bare_number(self):
        """The design invariant. A statistic must not escape its null."""
        r = self._result()
        with pytest.raises(TypeError):
            float(r)
        assert not hasattr(r, "value")

    def test_is_immutable(self):
        r = self._result()
        with pytest.raises(Exception):
            r.statistic = 99.0

    def test_excess_subtracts_the_null_mean(self):
        assert self._result().excess == pytest.approx(0.3)
        assert self._result().z_score == pytest.approx(3.0)

    def test_adjusted_p_takes_precedence(self):
        r = self._result(p_value=0.01)
        assert r.passes(alpha=0.05)
        assert not r.with_adjusted_p(0.40).passes(alpha=0.05)
        assert r.with_adjusted_p(0.40).passes(alpha=0.05, use_adjusted=False)

    def test_rejects_impossible_pvalue(self):
        with pytest.raises(ValueError, match="p_value must lie"):
            self._result(p_value=0.0)

    def test_report_frame_sorts_by_significance(self):
        frame = results_to_frame(
            [self._result(name="b", p_value=0.5), self._result(name="a", p_value=0.01)]
        )
        assert list(frame.index) == ["a", "b"]
        assert "null_q95" in frame.columns


# =============================================================================
# Calibration: the acceptance tests
# =============================================================================
def _false_positive_rate(
    scheme: str,
    n_trials: int,
    n_perm: int,
    phi: float,
    horizon: int,
    n: int,
    seed: int,
    **scheme_kwargs,
) -> float:
    """Rejection rate at alpha=0.05 when feature and target are independent.

    Feature is AR(1) with coefficient ``phi``; target is an ``horizon``-bar
    overlapping forward return. They are generated from independent innovations,
    so every rejection is a false positive.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    for trial in range(n_trials):
        x = ar1(n, phi, rng)
        y = overlapping_forward_return(n, horizon, rng)
        res = build_null(
            abs_corr,
            x,
            y,
            scheme=scheme,
            n_perm=n_perm,
            seed=int(rng.integers(0, 2**31)),
            scheme_kwargs=scheme_kwargs,
        )
        rejections += res.p_value < 0.05
    return rejections / n_trials


@pytest.mark.slow
def test_acceptance_circular_shift_holds_nominal_size():
    """ACCEPTANCE (plan Phase 1): size within [0.02, 0.08] at nominal 0.05.

    Configuration is the Blueprint default: a heavily smoothed indicator against
    an overlapping triple-barrier-style label.
    """
    fpr = _false_positive_rate(
        "circular_shift", n_trials=200, n_perm=200, phi=0.99, horizon=100, n=2000, seed=101
    )
    assert 0.02 <= fpr <= 0.08, f"circular_shift false-positive rate {fpr:.3f} outside [0.02, 0.08]"


@pytest.mark.slow
def test_acceptance_naive_shuffle_fails():
    """ACCEPTANCE (plan Phase 1): the naive scheme must fail the size bound.

    A regression guard, not a curiosity. If this test ever passes, somebody has
    changed the null machinery in a way that hides the defect this subpackage
    exists to prevent.
    """
    fpr = _false_positive_rate(
        "naive_shuffle", n_trials=100, n_perm=200, phi=0.99, horizon=100, n=2000, seed=202
    )
    assert fpr > 0.20, f"naive_shuffle false-positive rate {fpr:.3f} is implausibly well behaved"


@pytest.mark.slow
def test_acceptance_planted_signal_is_detected():
    """ACCEPTANCE (plan Phase 1): power against a planted relationship, p < 0.01.

    The feature carries a genuine but weak signal about the target, buried in
    the same serial dependence that defeats the naive scheme.
    """
    rng = np.random.default_rng(303)
    n, horizon = 2000, 20
    x = ar1(n, 0.9, rng)
    noise = overlapping_forward_return(n, horizon, rng)
    # Signal strength is set against the EFFECTIVE sample size, not n. Overlapping
    # labels leave roughly 50 independent observations here, so a correlation of
    # about 0.45 is the weakest relationship this configuration can resolve.
    y = 2.0 * x + noise
    res = build_null(abs_corr, x, y, scheme="circular_shift", n_perm=500, seed=1)
    assert res.effective_sample_size < n / 10
    assert res.p_value < 0.01, f"failed to detect a planted signal, p={res.p_value:.4f}"
    assert res.excess > 0
    assert res.recomputed_feature is False


@pytest.mark.slow
def test_size_is_correct_when_target_is_iid():
    """Sanity check: with an i.i.d. target every scheme should be well behaved.

    Establishes that the failure in ``test_acceptance_naive_shuffle_fails`` comes
    from the target's serial dependence and not from an error in the driver.
    """
    for scheme in ("naive_shuffle", "circular_shift"):
        fpr = _false_positive_rate(
            scheme, n_trials=100, n_perm=200, phi=0.0, horizon=1, n=2000, seed=404
        )
        assert fpr <= 0.12, f"{scheme} misbehaves on i.i.d. data: {fpr:.3f}"
