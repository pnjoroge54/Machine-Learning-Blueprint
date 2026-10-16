"""
Tests for Phase 5: permuting the development process.

The acceptance test asserts that a search-based development process produces a
positive spurious center on signal-free data, and that a search-free one does
not. The distinction is the whole point: the defect lives in the search, not in
the strategy.

``test_cache_isolation_catches_a_stale_cache`` simulates the failure mode most
likely to make a real study return a clean bill of health it has not earned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.cross_validation.process_permutation import (
    ProcessPermutationResult,
    assert_cache_isolation,
    permute_development_process,
    spurious_performance_curve,
)


# =============================================================================
# A miniature development process
# =============================================================================
def make_bars(n=3000, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    r = drift + rng.standard_normal(n) * 0.001
    close = 100.0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.standard_normal(n) * 0.0002)
    wick = np.abs(rng.standard_normal((n, 2))) * 0.0005
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * np.exp(wick[:, 0]),
            "low": np.minimum(open_, close) * np.exp(-wick[:, 1]),
            "close": close,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


ANN = np.sqrt(252 * 24)


def _backtest(close: np.ndarray, fast: int, slow: int, cost_bps: float) -> float:
    if fast >= slow or slow >= close.size // 4:
        return -10.0
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    ma_f = pd.Series(close).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(close).rolling(slow).mean().to_numpy()
    pos = np.nan_to_num(np.sign(ma_f - ma_s))
    turn = np.abs(np.diff(pos, prepend=0.0))
    pnl = pos[:-1] * ret[1:] - turn[:-1] * cost_bps * 1e-4
    sd = pnl.std()
    return -10.0 if sd == 0 else float(pnl.mean() / sd * ANN)


def searching_process(n_trials: int = 60, cost_bps: float = 1.0, seed: int = 0):
    """A development process that searches. This is the one that overfits."""
    rng = np.random.default_rng(seed)
    grid = [
        (int(f), int(s))
        for f, s in zip(rng.integers(2, 40, n_trials), rng.integers(20, 250, n_trials))
    ]

    def develop(bars: pd.DataFrame) -> float:
        close = bars["close"].to_numpy(dtype=float)
        return max(_backtest(close, f, s, cost_bps) for f, s in grid)

    return develop


def fixed_process(fast: int = 10, slow: int = 50, cost_bps: float = 1.0):
    """A development process with no search. Nothing is selected, so nothing is
    selected on noise."""

    def develop(bars: pd.DataFrame) -> float:
        return _backtest(bars["close"].to_numpy(dtype=float), fast, slow, cost_bps)

    return develop


# =============================================================================
# Cache isolation
# =============================================================================
class TestCacheIsolation:
    def test_passes_for_a_responsive_function(self):
        assert_cache_isolation(searching_process(20), make_bars(1000), seed=1)

    def test_catches_a_stale_cache(self):
        """The failure mode that makes a real study silently pass.

        A development function whose cache key ignores the bar content returns
        the same value for every permutation. The null then collapses onto the
        observed value and the near-zero deflated statistic reads as vindication.
        """
        real = make_bars(1000, seed=2)
        cached = float(searching_process(20)(real))

        def poisoned(_bars: pd.DataFrame) -> float:
            return cached  # served from a cache keyed on nothing that changed

        with pytest.raises(RuntimeError, match="not responding to its input"):
            assert_cache_isolation(poisoned, real, seed=2)

    def test_error_message_names_the_remedy(self):
        real = make_bars(600, seed=3)
        with pytest.raises(RuntimeError, match="permutation seed or a content hash"):
            assert_cache_isolation(lambda _b: 1.0, real)


# =============================================================================
# Mechanics
# =============================================================================
class TestPermuteDevelopmentProcess:
    def test_result_contract(self):
        bars = make_bars(1500, seed=4)
        out = permute_development_process(
            searching_process(30), bars, n_perm=10, seed=1, trial_budget=30
        )
        assert isinstance(out, ProcessPermutationResult)
        assert out.null_values.size == 10
        assert 0.0 < out.p_value <= 1.0
        assert out.deflated == pytest.approx(out.observed - out.spurious_center)
        assert out.trial_budget == 30
        assert "spurious_center" in out.to_dict()

    def test_failures_are_recorded_not_raised(self):
        bars = make_bars(800, seed=5)
        calls = {"n": 0}

        def flaky(frame: pd.DataFrame) -> float:
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise RuntimeError("simulated failure")
            return float(frame["close"].iloc[-1])

        out = permute_development_process(
            flaky, bars, n_perm=9, seed=2, check_cache_isolation=False
        )
        assert len(out.failures) > 0
        assert np.isfinite(out.spurious_center)

    def test_raise_mode_propagates(self):
        bars = make_bars(600, seed=6)
        with pytest.raises(RuntimeError, match="boom"):
            permute_development_process(
                lambda _f: (_ for _ in ()).throw(RuntimeError("boom")),
                bars,
                n_perm=3,
                on_error="raise",
                check_cache_isolation=False,
            )

    def test_total_failure_is_an_error(self):
        bars = make_bars(600, seed=7)

        def always_fails(_f):
            raise ValueError("no")

        with pytest.raises(RuntimeError, match="every permutation replicate failed"):
            permute_development_process(
                always_fails, bars, n_perm=4, observed=1.0, check_cache_isolation=False
            )

    def test_failure_on_real_bars_is_reported_clearly(self):
        """A process that cannot run on the real data is a bug to fix, not a null
        to measure."""
        bars = make_bars(600, seed=7)

        def always_fails(_f):
            raise ValueError("no")

        with pytest.raises(RuntimeError, match="failed on the real bars"):
            permute_development_process(always_fails, bars, n_perm=4, check_cache_isolation=False)

    def test_validation(self):
        bars = make_bars(600, seed=8)
        with pytest.raises(ValueError, match="on_error must be"):
            permute_development_process(fixed_process(), bars, on_error="ignore")
        with pytest.raises(ValueError, match="n_perm must be at least 2"):
            permute_development_process(fixed_process(), bars, n_perm=1)

    def test_observed_can_be_supplied(self):
        bars = make_bars(1000, seed=9)
        out = permute_development_process(
            fixed_process(), bars, n_perm=6, seed=3, observed=2.5, check_cache_isolation=False
        )
        assert out.observed == 2.5


# =============================================================================
# Budget scaling
# =============================================================================
class TestSpuriousPerformanceCurve:
    def test_spurious_performance_grows_with_budget(self):
        """The measurement that converts a trial count into a performance bar."""
        bars = make_bars(2500, seed=10)
        curve = spurious_performance_curve(
            lambda b: searching_process(b, seed=11),
            bars,
            budgets=(5, 25, 100),
            n_perm=8,
            seed=4,
        )
        assert list(curve.index) == [5, 25, 100]
        assert curve.loc[100, "spurious_median"] > curve.loc[5, "spurious_median"]
        assert set(curve.columns) >= {"spurious_median", "spurious_q90", "fraction_positive"}


# =============================================================================
# Acceptance
# =============================================================================
@pytest.mark.slow
def test_acceptance_search_manufactures_edge_from_noise():
    """ACCEPTANCE (plan Phase 5): a search-based process produces a positive
    spurious center on signal-free bars.

    Bar permutation destroys inter-bar predictability, so any positive
    performance the process reports is selection over noise. The prediction
    recorded before this phase was built: the center would be clearly positive.
    """
    bars = make_bars(4000, seed=20)
    out = permute_development_process(
        searching_process(100, seed=21), bars, n_perm=25, seed=5, trial_budget=100
    )
    assert out.spurious_center > 0.5, (
        f"spurious center {out.spurious_center:.3f}; a 100-trial search on "
        f"signal-free data should manufacture visible apparent edge"
    )
    assert not out.process_is_sound
    finite = out.null_values[np.isfinite(out.null_values)]
    assert np.mean(finite > 0) > 0.8


@pytest.mark.slow
def test_acceptance_search_free_process_is_sound():
    """ACCEPTANCE (plan Phase 5): a process with no search does not manufacture
    edge. The defect lives in the selection, not in the strategy family."""
    bars = make_bars(4000, seed=22)
    out = permute_development_process(fixed_process(), bars, n_perm=25, seed=6, trial_budget=1)
    assert out.spurious_center <= 0.5, (
        f"a search-free process produced a spurious center of "
        f"{out.spurious_center:.3f}; the null itself may be miscalibrated"
    )


@pytest.mark.slow
def test_acceptance_deflation_recovers_a_real_edge():
    """A process run on bars that genuinely trend should clear its own null.

    Establishes that the test has power: it does not simply reject everything.
    """
    trending = make_bars(4000, seed=23, drift=0.0004)
    out = permute_development_process(
        searching_process(60, seed=24), trending, n_perm=25, seed=7, trial_budget=60
    )
    assert out.observed > out.spurious_center
    assert out.deflated > 0.0
