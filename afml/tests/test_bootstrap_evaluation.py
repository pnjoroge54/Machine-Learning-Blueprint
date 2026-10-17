"""
Tests for Phase 3: bootstrap evaluation of performance and drawdown.

The acceptance test is :func:`test_acceptance_bootstrap_exceeds_realized_drawdown`.
The rest establish that the mechanics behave, that block length does what it is
supposed to do, and that the prop-firm breach estimator answers a question the
realized path cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.backtest_statistics.bootstrap import (
    BootstrapResult,
    bootstrap_drawdown_distribution,
    bootstrap_statistic,
    breach_probability,
    max_drawdown,
    sequence_risk_ratio,
)
from afml.backtest_statistics.walkforward import walkforward_degradation


# =============================================================================
# Helpers
# =============================================================================
def trade_returns(n=400, seed=0, edge=0.0005, vol=0.01, rho=0.0):
    """Trade returns with optional serial dependence, as overlapping labels induce."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * vol
    if rho:
        for i in range(1, n):
            e[i] = rho * e[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    return pd.Series(edge + e, name="returns")


def sharpe(r):
    s = np.std(r, ddof=1)
    return 0.0 if s == 0 else float(np.mean(r) / s * np.sqrt(252))


# =============================================================================
# Drawdown mechanics
# =============================================================================
class TestMaxDrawdown:
    def test_monotone_gains_have_no_drawdown(self):
        assert max_drawdown(np.full(100, 0.01)) == 0.0

    def test_known_case(self):
        # +100%, then -50%: equity 1 -> 2 -> 1, drawdown 50%.
        assert max_drawdown(np.array([1.0, -0.5])) == pytest.approx(0.5)

    def test_ordering_changes_the_answer(self):
        """The premise of the whole module: the same trades, reordered, give a
        different worst case."""
        r = np.array([0.1, -0.2, 0.15, -0.25, 0.05])
        orders = [max_drawdown(r[list(p)]) for p in
                  [(0, 1, 2, 3, 4), (1, 3, 0, 2, 4), (3, 1, 4, 0, 2)]]
        assert max(orders) > min(orders) * 1.2

    def test_additive_mode(self):
        r = np.array([0.1, -0.2])
        assert max_drawdown(r, compound=False) == pytest.approx(0.2 / 1.1, rel=1e-6)

    def test_empty_input(self):
        assert max_drawdown(np.array([])) == 0.0


# =============================================================================
# Bootstrap statistic
# =============================================================================
class TestBootstrapStatistic:
    def test_result_contract(self):
        r = trade_returns(300, seed=1)
        out = bootstrap_statistic(r, sharpe, mean_block=5.0, n_boot=500, name="sharpe")
        assert isinstance(out, BootstrapResult)
        assert out.n_observations == 300
        assert out.lower_bound <= out.quantiles[0.50] <= out.quantiles[0.95]
        assert "q05" in out.to_dict()
        assert "sharpe" in repr(out)

    def test_lower_bound_is_below_the_point_estimate(self):
        """Masters' honest forward estimate: the fifth percentile, not the point."""
        r = trade_returns(500, seed=2, edge=0.002)
        out = bootstrap_statistic(r, sharpe, mean_block=5.0, n_boot=1000)
        assert out.lower_bound < out.observed

    def test_bootstrap_mean_tracks_the_observed_value(self):
        r = trade_returns(600, seed=3, edge=0.001)
        out = bootstrap_statistic(r, sharpe, mean_block=5.0, n_boot=2000)
        assert abs(out.bias) < 3 * out.std_error

    def test_rejects_short_samples_and_bad_blocks(self):
        with pytest.raises(ValueError, match="at least 20 trades"):
            bootstrap_statistic(trade_returns(10), sharpe, mean_block=2.0)
        with pytest.raises(ValueError, match="mean_block must be at least 1"):
            bootstrap_statistic(trade_returns(100), sharpe, mean_block=0.5)


# =============================================================================
# Drawdown distribution
# =============================================================================
class TestDrawdownDistribution:
    def test_distribution_shape(self):
        r = trade_returns(400, seed=4)
        dist = bootstrap_drawdown_distribution(r, mean_block=5.0, n_boot=1000)
        assert len(dist) == 1000
        assert dist.is_monotonic_increasing
        assert (dist >= 0).all() and (dist <= 1).all()

    def test_longer_blocks_reproduce_the_original_sequence(self):
        """Block length is not free. As it approaches the sample length the
        resampled paths converge on the realized one and the estimate collapses
        back to the single-path answer it was meant to replace."""
        r = trade_returns(400, seed=5, rho=0.3)
        realized = max_drawdown(r)
        short = bootstrap_drawdown_distribution(r, 2.0, n_boot=800).quantile(0.95)
        long_ = bootstrap_drawdown_distribution(r, 350.0, n_boot=800).quantile(0.95)
        assert abs(long_ - realized) < abs(short - realized)

    def test_iid_bootstrap_understates_dependent_returns(self):
        """With serially dependent trades, block length one discards the very
        clustering that produces deep drawdowns."""
        r = trade_returns(600, seed=6, rho=0.6, edge=0.0)
        q_iid = bootstrap_drawdown_distribution(r, 1.0, n_boot=1500).quantile(0.95)
        q_block = bootstrap_drawdown_distribution(r, 20.0, n_boot=1500).quantile(0.95)
        assert q_block > q_iid

    def test_sequence_risk_ratio_exceeds_one_on_a_lucky_path(self):
        r = trade_returns(400, seed=7, edge=0.001)
        assert sequence_risk_ratio(r, mean_block=5.0, n_boot=1000) > 1.0


# =============================================================================
# Prop-firm breach probability
# =============================================================================
class TestBreachProbability:
    def test_realized_path_can_survive_a_limit_the_distribution_breaches(self):
        """The commercial point. A history that never breached returns a breach
        probability of zero from the realized path by construction."""
        r = trade_returns(400, seed=8, edge=0.0006, vol=0.012)
        realized = max_drawdown(r)
        limit = realized * 1.15
        out = breach_probability(r, mean_block=5.0, max_drawdown_limit=limit, n_boot=2000)
        assert out["realized_max_drawdown"] < limit
        assert out["p_drawdown_breach"] > 0.05

    def test_daily_limit_requires_grouping(self):
        r = trade_returns(400, seed=9)
        with pytest.raises(ValueError, match="requires trades_per_day"):
            breach_probability(r, 5.0, max_drawdown_limit=0.1, daily_loss_limit=0.05)

    def test_daily_and_combined_probabilities(self):
        r = trade_returns(600, seed=10, edge=0.0, vol=0.015)
        out = breach_probability(r, mean_block=5.0, max_drawdown_limit=0.15,
                                 daily_loss_limit=0.04, trades_per_day=4, n_boot=1500)
        assert 0.0 <= out["p_daily_breach"] <= 1.0
        assert out["p_any_breach"] >= max(out["p_daily_breach"], out["p_drawdown_breach"])

    def test_tighter_limits_breach_more_often(self):
        r = trade_returns(500, seed=11, vol=0.012)
        loose = breach_probability(r, 5.0, max_drawdown_limit=0.30, n_boot=1500)
        tight = breach_probability(r, 5.0, max_drawdown_limit=0.08, n_boot=1500)
        assert tight["p_drawdown_breach"] > loose["p_drawdown_breach"]


# =============================================================================
# Walk-forward degradation
# =============================================================================
class TestWalkforwardDegradation:
    def test_detects_overfitting(self):
        out = walkforward_degradation(
            is_scores=[2.1, 1.9, 2.4, 2.0, 2.2, 1.8, 2.3, 2.1],
            oos_scores=[0.3, 0.1, 0.5, 0.2, 0.4, 0.0, 0.3, 0.2],
            n_boot=3000,
        )
        assert out.mean_degradation > 1.5
        assert out.overfits
        assert out.p_value < 0.05

    def test_clean_process_is_not_flagged(self):
        rng = np.random.default_rng(12)
        is_s = 1.0 + rng.standard_normal(10) * 0.3
        oos_s = 1.0 + rng.standard_normal(10) * 0.3
        out = walkforward_degradation(is_s, oos_s, n_boot=3000)
        assert not out.overfits

    def test_ratio_flagged_unreliable_on_negative_windows(self):
        """The defect in Masters' WFE: with a negative in-sample window the ratio
        is uninterpretable while the paired difference is not."""
        out = walkforward_degradation(
            is_scores=[1.5, -0.2, 1.8, 0.9],
            oos_scores=[0.4, -0.9, 0.6, 0.1],
            n_boot=2000,
        )
        assert not out.wfe_reliable
        assert np.isfinite(out.mean_degradation)
        assert "[ratio unreliable]" in repr(out)

    def test_ratio_explodes_where_the_difference_does_not(self):
        """Same degradation in performance units, wildly different WFE."""
        big = walkforward_degradation([2.0] * 6, [1.0] * 6, n_boot=500)
        small = walkforward_degradation([0.02] * 6, [0.01] * 6, n_boot=500)
        # Identical WFE, a hundredfold difference in actual degradation. The
        # ratio cannot distinguish a strategy losing a full Sharpe point from one
        # losing a hundredth of one.
        assert big.wfe_ratio == pytest.approx(small.wfe_ratio, abs=1e-9)
        assert big.mean_degradation == pytest.approx(100 * small.mean_degradation, rel=1e-9)
        assert big.mean_degradation == pytest.approx(1.0)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="must align"):
            walkforward_degradation([1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="at least 3 paired"):
            walkforward_degradation([1.0, 2.0], [0.5, 1.0])


# =============================================================================
# Acceptance
# =============================================================================
@pytest.mark.slow
def test_acceptance_bootstrap_exceeds_realized_drawdown():
    """ACCEPTANCE (plan Phase 3): the 95th-percentile bootstrap drawdown exceeds
    the realized maximum drawdown by a factor between 1.3 and 2.5.

    Below 1.1 the block length is too long and the resampling is reproducing the
    original ordering. Averaged over seeds so the assertion is about the method
    rather than one lucky path.
    """
    ratios = []
    for seed in range(25):
        r = trade_returns(500, seed=100 + seed, edge=0.0008, vol=0.012, rho=0.15)
        ratios.append(sequence_risk_ratio(r, mean_block=5.0, n_boot=2000, seed=seed))
    median = float(np.median(ratios))
    assert 1.3 <= median <= 2.5, f"sequence risk ratio {median:.2f} outside [1.3, 2.5]"


# =============================================================================
# Part 24 revision: drawdown basis, horizon, sign test, reliability threshold
# =============================================================================
class TestBreachBasisAndHorizon:
    def test_static_basis_ignores_losses_after_a_gain(self):
        """+20% then -15%: trailing loss is 15%, but equity never falls below the
        starting balance, so a static 10% floor is never touched."""
        r = np.r_[np.full(10, 0.0185), np.full(10, -0.0161), np.zeros(10)]
        trailing = breach_probability(r, mean_block=1e9, max_drawdown_limit=0.10,
                                      n_boot=50, drawdown_basis="peak")
        static = breach_probability(r, mean_block=1e9, max_drawdown_limit=0.10,
                                    n_boot=50, drawdown_basis="initial")
        assert trailing["realized_max_drawdown"] > 0.10
        assert static["realized_max_drawdown"] == 0.0

    def test_static_breach_never_exceeds_trailing(self):
        """Loss from the initial balance is bounded by loss from the running peak."""
        r = trade_returns(400, seed=3, edge=0.0, vol=0.01)
        kw = dict(mean_block=10.0, max_drawdown_limit=0.08, n_boot=2000, seed=1)
        assert (breach_probability(r, drawdown_basis="initial", **kw)["p_drawdown_breach"]
                <= breach_probability(r, drawdown_basis="peak", **kw)["p_drawdown_breach"])

    def test_breach_probability_grows_with_horizon(self):
        r = trade_returns(800, seed=4, edge=0.0, vol=0.01)
        kw = dict(mean_block=10.0, max_drawdown_limit=0.10, n_boot=2000, seed=2)
        short = breach_probability(r, horizon=100, **kw)["p_drawdown_breach"]
        full = breach_probability(r, **kw)["p_drawdown_breach"]
        assert short < full
        assert breach_probability(r, horizon=100, **kw)["horizon"] == 100

    def test_bad_arguments_raise(self):
        r = trade_returns(100, seed=5)
        with pytest.raises(ValueError):
            breach_probability(r, 5.0, 0.1, drawdown_basis="trailing")
        with pytest.raises(ValueError):
            breach_probability(r, 5.0, 0.1, horizon=101)


class TestDegradationSmallSample:
    def test_sign_test_six_of_six(self):
        is_s = np.array([0.10, 0.11, -0.004, 0.13, 0.07, 0.26])
        oos = is_s - np.array([0.18, 0.17, 0.03, 0.05, 0.21, 0.22])
        res = walkforward_degradation(is_s, oos, n_boot=2000, seed=0)
        assert res.n_degraded == 6
        assert res.sign_test_p == pytest.approx(1 / 64)
        # Every window degrades, so the bootstrap p-value sits at its floor.
        assert res.p_value == pytest.approx(1 / 2001)
        assert not res.wfe_reliable          # one non-positive IS window

    def test_reliability_needs_mean_is_away_from_zero(self):
        """All-positive IS windows whose mean is within noise of zero are flagged."""
        is_s = np.array([0.02, 0.30, 0.01, 0.25, 0.015, 0.005])
        oos = is_s * 0.5
        noisy = walkforward_degradation(is_s, oos, n_boot=500, min_is_tstat=5.0)
        loose = walkforward_degradation(is_s, oos, n_boot=500, min_is_tstat=1.0)
        assert not noisy.wfe_reliable
        assert loose.wfe_reliable
