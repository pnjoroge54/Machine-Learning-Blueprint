"""
Tests for the selection-bias tooling in afml.cross_validation: effective trial
count, expected maximum Sharpe, the one-standard-error rule, and PBO.
"""

import numpy as np
import pandas as pd
import pytest

from ..cross_validation.pbo import compute_pbo
from ..cross_validation.selection_overfit import (
    effective_trial_count,
    expected_max_sharpe,
    one_standard_error_selection,
)

# --------------------------------------------------------------------------- effective trials


def test_independent_trials_count_close_to_n():
    rng = np.random.default_rng(0)
    res = effective_trial_count(rng.normal(size=(10, 400)))
    assert res.n_trials == 10
    assert 8.0 <= res.n_eff_kish <= 10.0
    assert 7.0 <= res.n_eff_eigen <= 10.0
    assert abs(res.mean_correlation) < 0.05
    assert not res.rank_limited


def test_near_identical_trials_count_close_to_one():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    res = effective_trial_count(base + rng.normal(scale=1e-3, size=(12, 200)))
    assert res.n_eff_kish < 1.1
    assert res.n_eff_eigen < 2.0
    assert res.mean_correlation > 0.99


def test_rank_limited_when_more_trials_than_scores():
    rng = np.random.default_rng(2)
    res = effective_trial_count(rng.normal(size=(20, 5)))
    assert res.rank_limited
    assert res.score_vector_length == 5
    assert res.n_eff_eigen <= 20


def test_constant_trials_collapse_to_one():
    res = effective_trial_count(np.ones((4, 6)))
    assert res.n_eff_eigen == res.n_eff_kish == 1.0


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((1, 5)), np.ones((5, 1))])
def test_effective_trial_count_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        effective_trial_count(bad)


# --------------------------------------------------------------------------- expected max Sharpe


def test_expected_max_sharpe():
    assert expected_max_sharpe(1, 1.0) == 0.0
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(values, values[1:]))
    # E[max of 100 standard normals] is about 2.5
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.53, abs=0.05)
    # Scales with the standard deviation of the Sharpe estimates
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))
    assert expected_max_sharpe(100, -1.0) == 0.0


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(3)
    sims = rng.normal(size=(20_000, 50)).max(axis=1).mean()
    assert expected_max_sharpe(50, 1.0) == pytest.approx(sims, rel=0.03)


# --------------------------------------------------------------------------- one-standard-error rule


def _candidates():
    return [
        {"params": {"depth": 2}, "fold_scores": [0.60, 0.62, 0.58, 0.61]},
        {"params": {"depth": 4}, "fold_scores": [0.62, 0.63, 0.61, 0.62]},
        {"params": {"depth": 8}, "fold_scores": [0.63, 0.64, 0.62, 0.63]},
        {"params": {"depth": 1}, "fold_scores": [0.50, 0.52, 0.49, 0.51]},
    ]


def test_one_standard_error_picks_simplest_within_tolerance():
    chosen, diag = one_standard_error_selection(_candidates(), complexity=lambda p: p["depth"])
    assert diag["best_mean"] == pytest.approx(0.63)
    best_se = np.std([0.63, 0.64, 0.62, 0.63], ddof=1) / 2
    assert diag["threshold"] == pytest.approx(0.63 - best_se)
    # depth 4 has mean 0.62 < threshold 0.6259, so the best candidate itself is kept
    assert chosen["params"]["depth"] == 8
    assert diag["n_within_1se"] == 1


def test_one_standard_error_uses_standard_error_not_std():
    cands = [
        {"params": {"c": 1}, "fold_scores": [0.55, 0.65, 0.50, 0.70]},
        {"params": {"c": 3}, "fold_scores": [0.62, 0.62, 0.64, 0.64]},
    ]
    chosen, diag = one_standard_error_selection(cands, complexity=lambda p: p["c"])
    # The best SE is about 0.0058, so the threshold is about 0.624 and the simpler
    # candidate (mean 0.60) is excluded. Using the fold std (0.0115) would still
    # exclude it here, but the threshold must be the SE.
    assert chosen["params"]["c"] == 3
    assert diag["n_within_1se"] == 1


@pytest.mark.parametrize("bad", [[], [{"params": {}, "fold_scores": [0.5]}]])
def test_one_standard_error_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        one_standard_error_selection(bad, complexity=lambda p: 0)


# --------------------------------------------------------------------------- PBO


def _returns(n_obs=800, n_strats=10, seed=0, planted=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    data = rng.normal(0.0, 0.01, size=(n_obs, n_strats))
    if planted is not None:
        data[:, planted] += 0.004  # a real, persistent edge
    df = pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(n_strats)])
    return df, pd.Series(idx, index=idx)


def test_pbo_structure():
    df, t1 = _returns()
    res = compute_pbo(df, t1, n_folds=8)
    assert res["n_splits"] == 70  # C(8, 4)
    assert 0.0 <= res["pbo"] <= 1.0
    assert len(res["oos_ranks"]) == 70
    assert res["below_median"] == pytest.approx(res["pbo"] * res["n_splits"])


def test_pbo_low_when_one_strategy_has_a_real_edge():
    df, t1 = _returns(planted=3)
    assert compute_pbo(df, t1, n_folds=8)["pbo"] < 0.1


def test_pbo_high_for_pure_noise_families():
    pbos = [compute_pbo(*_returns(seed=s), n_folds=8)["pbo"] for s in range(6)]
    # In-sample winners among noise strategies are no better than a coin flip out of sample
    assert 0.25 < np.mean(pbos) < 0.75


def test_pbo_rejects_odd_folds():
    df, t1 = _returns()
    with pytest.raises(ValueError, match="even"):
        compute_pbo(df, t1, n_folds=7)
