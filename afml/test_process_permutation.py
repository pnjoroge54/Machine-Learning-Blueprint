"""
Tests for afml.cross_validation.process_permutation. Synthetic data only.

Run from MQL5/Files, the folder that contains the afml package:
    python -m pytest afml/test_process_permutation.py -q
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from afml.cross_validation.process_permutation import (
    ProcessPermutationResult, assert_cache_isolation, permute_development_process,
    quantile_ci, spurious_performance_curve,
)


def _bars(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Driftless random-walk OHLC bars with no exploitable structure."""
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 2e-4, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 5e-4, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 5e-4, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close}, index=idx)


def _sharpe(close: np.ndarray, fast: int, slow: int) -> float:
    ret = np.diff(np.log(close))
    ma_f = pd.Series(close).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(close).rolling(slow).mean().to_numpy()
    pos = np.nan_to_num(np.sign(ma_f - ma_s))[:-1]
    pnl = pos * ret - np.abs(np.diff(np.r_[0.0, pos])) * 1e-4
    return float(pnl.mean() / pnl.std() * np.sqrt(6048)) if pnl.std() > 0 else -10.0


def _search(n_trials: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    grid = [(int(f), int(s)) for f, s in zip(rng.integers(2, 20, n_trials), rng.integers(21, 120, n_trials))]
    return lambda bars: max(_sharpe(bars["close"].to_numpy(), f, s) for f, s in grid)


def _fixed(bars: pd.DataFrame) -> float:
    return _sharpe(bars["close"].to_numpy(), 10, 50)


# ── quantile_ci ───────────────────────────────────────────────────────────
def test_quantile_ci_order_statistics_and_coverage():
    lo, hi, cov = quantile_ci(np.arange(25.0), 0.5)
    assert (lo, hi) == (7.0, 17.0)                     # 8th and 18th order statistics
    assert cov >= 0.95


def test_quantile_ci_unbounded_q90_at_small_n():
    lo, hi, _ = quantile_ci(np.arange(25.0), 0.9)
    assert np.isfinite(lo) and hi == float("inf")


def test_quantile_ci_empirical_coverage():
    rng = np.random.default_rng(0)
    hits = sum(lo <= 0.0 <= hi for lo, hi, _ in (quantile_ci(rng.normal(size=60), 0.5) for _ in range(2000)))
    assert hits / 2000 >= 0.94


# ── result statistics ─────────────────────────────────────────────────────
def _result(observed: float, null: np.ndarray) -> ProcessPermutationResult:
    return ProcessPermutationResult(observed=observed, null_values=null, spurious_center=float(np.median(null)),
                                    p_value=(1 + np.sum(null >= observed)) / (1 + null.size), n_permutations=null.size)


def test_percentile_robust_z_and_excess():
    null = np.linspace(-1, 1, 101)
    r = _result(0.5, null)
    assert r.excess_over_null == pytest.approx(0.5)
    assert r.null_percentile == pytest.approx(75.5 / 101, abs=1e-12)
    mad = 1.4826 * np.median(np.abs(null - np.median(null)))
    assert r.robust_z == pytest.approx(0.5 / mad)


def test_same_excess_different_width_gives_different_z():
    narrow, wide = _result(0.2, np.linspace(-0.1, 0.1, 201)), _result(0.2, np.linspace(-2, 2, 201))
    assert narrow.excess_over_null == pytest.approx(wide.excess_over_null)
    assert narrow.robust_z > 5 * wide.robust_z


def test_selection_effect_uses_ci_not_point():
    rng = np.random.default_rng(3)
    tiny = _result(0.0, rng.normal(0.05, 1.0, 25))       # point median may be > 0, CI is not
    assert not tiny.selection_effect_exceeds(0.0)
    clear = _result(0.0, rng.normal(0.5, 0.1, 200))
    assert clear.selection_effect_exceeds(0.0)
    assert not clear.selection_effect_exceeds(tolerance=1.0)


# ── cache isolation ───────────────────────────────────────────────────────
def test_cache_isolation_catches_stale_value():
    bars = _bars()
    stale = float(_fixed(bars))
    with pytest.raises(RuntimeError, match="not responding to its input"):
        assert_cache_isolation(lambda _b: stale, bars, seed=2)


def test_cache_isolation_rejects_nondeterministic_process():
    bars = _bars()
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="same permuted"):
        assert_cache_isolation(lambda b: _fixed(b) + rng.normal(), bars, seed=2)


def test_cache_isolation_passes_responsive_deterministic_process():
    assert_cache_isolation(_fixed, _bars(), seed=2)


# ── end-to-end on synthetic noise ─────────────────────────────────────────
def test_search_null_sits_above_search_free_null():
    bars = _bars(seed=5)
    search = permute_development_process(_search(40), bars, n_perm=60, seed=1, trial_budget=40)
    fixed = permute_development_process(_fixed, bars, n_perm=60, seed=1, trial_budget=1)
    assert search.spurious_center > fixed.spurious_center
    assert search.selection_effect_exceeds(0.0)
    assert 1 / 61 <= search.p_value <= 1.0


def test_curve_columns_and_common_random_numbers():
    bars = _bars(seed=6)
    curve = spurious_performance_curve(_search, bars, budgets=(3, 30), n_perm=40, seed=4, include_observed=True)
    for col in ("median_ci_lower", "median_ci_upper", "q90_ci_upper", "observed", "null_percentile", "p_value"):
        assert col in curve.columns
    point = permute_development_process(_search(30), bars, n_perm=40, seed=4, check_cache_isolation=False)
    assert curve.loc[30, "spurious_median"] == pytest.approx(point.spurious_center)
    assert curve.loc[30, "median_ci_lower"] <= curve.loc[30, "spurious_median"] <= curve.loc[30, "median_ci_upper"]
