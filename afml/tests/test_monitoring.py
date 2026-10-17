"""
Tests for Phase 4: live deterioration monitoring.

The two acceptance tests come from the plan: the SPRT must fire within 40 trades
of an injected regime break, and its false-retirement rate on a stationary series
must sit at or below the specified alpha.

:func:`test_naive_posterior_threshold_over_rejects` is the one that justifies the
module's existence. It measures what the compiled Masters reference recommends
and shows the rate it actually delivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.monitoring.deterioration import (
    WinRateSPRT,
    average_run_length,
    deterioration_probability,
    expected_sample_size,
)
from afml.monitoring.monitor import LiveStrategyMonitor
from afml.monitoring.profit_factor import (
    granular_profit_factor,
    profit_factor,
    profit_factor_trend,
)


# =============================================================================
# Incomplete beta posterior
# =============================================================================
class TestDeteriorationProbability:
    def test_masters_worked_example(self):
        """25 wins in 50 trades against a 60% historical rate."""
        p = deterioration_probability(25, 50, 0.60)
        assert 0.85 < p < 0.99

    def test_monotone_in_evidence(self):
        rising = [deterioration_probability(int(0.5 * n), n, 0.60) for n in (20, 50, 100, 200)]
        assert all(a < b for a, b in zip(rising, rising[1:]))

    def test_matches_the_prior_with_no_data(self):
        assert deterioration_probability(0, 0, 0.60) == pytest.approx(0.60)

    def test_validation(self):
        with pytest.raises(ValueError, match="0 <= k_wins"):
            deterioration_probability(51, 50, 0.6)
        with pytest.raises(ValueError, match="p0 must lie"):
            deterioration_probability(10, 50, 1.5)


# =============================================================================
# SPRT
# =============================================================================
class TestWinRateSPRT:
    def test_boundaries_follow_wald(self):
        s = WinRateSPRT(0.60, 0.50, alpha=0.05, beta=0.20, reflecting=False)
        assert s.upper == pytest.approx(np.log(0.80 / 0.05))
        assert s.lower == pytest.approx(np.log(0.20 / 0.95))

    def test_reflecting_mode_floors_at_zero(self):
        """Page's CUSUM. Health drives the statistic down but never below zero,
        so the detector never retires itself from duty."""
        s = WinRateSPRT(0.60, 0.50, reflecting=True)
        s.update_many([True] * 200)
        assert s.llr == 0.0
        assert s.decision == "continue"
        assert s.lower == 0.0

    def test_terminating_mode_stops_watching(self):
        """The defect the acceptance test found: Wald's SPRT reaches a decision
        and stops, so a break arriving later is never seen."""
        s = WinRateSPRT(0.60, 0.50, reflecting=False)
        s.update_many([True] * 200)
        assert s.decision == "accept"
        # Two hundred consecutive losses arrive and the verdict never changes.
        s.update_many([False] * 200)
        assert s.decision == "accept"

    def test_losses_push_toward_retirement(self):
        s = WinRateSPRT(0.60, 0.45)
        s.update_many([False] * 60)
        assert s.decision == "retire"

    def test_wins_confirm_health_in_terminating_mode(self):
        s = WinRateSPRT(0.60, 0.45, reflecting=False)
        s.update_many([True] * 60)
        assert s.decision == "accept"

    def test_decision_latches_but_recording_continues(self):
        """Retirement is irreversible; the chart is not truncated."""
        s = WinRateSPRT(0.60, 0.45)
        s.update_many([False] * 60)
        assert s.decision == "retire"
        n_at_decision = s.n_trades
        for _ in range(20):
            s.update(True)
        assert s.decision == "retire"
        assert s.n_trades == n_at_decision + 20
        assert len(s.history) == s.n_trades

    def test_progress_is_bounded_and_monotone_under_losses(self):
        s = WinRateSPRT(0.60, 0.45)
        progress = []
        for _ in range(20):
            s.update(False)
            progress.append(s.state.progress_to_retirement)
        assert all(0.0 <= p <= 1.0 for p in progress)
        assert progress[-1] > progress[0]

    def test_reset_clears_state(self):
        s = WinRateSPRT(0.60, 0.45)
        s.update_many([False] * 100)
        s.reset()
        assert s.decision == "continue" and s.n_trades == 0 and s.llr == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="0 < p1 < p0 < 1"):
            WinRateSPRT(0.5, 0.6)
        with pytest.raises(ValueError, match="alpha and beta"):
            WinRateSPRT(0.6, 0.5, alpha=0.0)

    def test_expected_sample_size_shrinks_with_separation(self):
        near = expected_sample_size(0.60, 0.57)
        far = expected_sample_size(0.60, 0.40)
        assert near > far > 0

    def test_arl_separates_health_from_degradation(self):
        """The quantity that replaces alpha under a reflecting barrier. A usable
        configuration detects degradation far faster than it false-alarms."""
        arl0 = average_run_length(0.60, 0.50, n_sim=250, horizon=6000, seed=1)
        arl1 = average_run_length(0.60, 0.50, true_p=0.50, n_sim=250, horizon=6000, seed=2)
        assert arl0 > 5 * arl1


# =============================================================================
# Profit factor
# =============================================================================
class TestProfitFactor:
    def test_basic_arithmetic(self):
        assert profit_factor([0.1, 0.1, -0.1]) == pytest.approx(2.0)
        assert profit_factor([0.1, 0.2]) == float("inf")
        assert np.isnan(profit_factor([]))

    def test_granular_drops_thin_periods(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        rng = np.random.default_rng(0)
        pf = granular_profit_factor(pd.Series(rng.standard_normal(200) * 0.01, index=idx),
                                    freq="W", min_trades=5)
        assert len(pf) > 20 and np.isfinite(pf).all()

    def test_granular_requires_datetime_index(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            granular_profit_factor(pd.Series([0.1, -0.1]))

    def test_trend_detects_decline(self):
        pf = pd.Series(np.exp(np.linspace(np.log(2.5), np.log(0.6), 30)))
        out = profit_factor_trend(pf, n_perm=1000)
        assert out.slope < 0 and out.deteriorating() and np.isfinite(out.half_life)

    def test_trend_null_is_centered(self):
        rng = np.random.default_rng(1)
        out = profit_factor_trend(pd.Series(np.exp(rng.standard_normal(40) * 0.4)), n_perm=2000)
        assert abs(out.null_mean) < 3 * out.null_std / np.sqrt(2000)
        assert not out.deteriorating()

    def test_log_scale_treats_gains_and_losses_symmetrically(self):
        """A doubling and a halving must produce equal and opposite slopes.

        On the raw scale they do not, and the regression is dominated by the
        high-profit-factor periods.
        """
        up = profit_factor_trend(pd.Series(np.exp(np.linspace(0, 2, 20))), n_perm=200)
        down = profit_factor_trend(pd.Series(np.exp(np.linspace(2, 0, 20))), n_perm=200)
        assert up.slope == pytest.approx(-down.slope, rel=1e-9)

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 6 periods"):
            profit_factor_trend(pd.Series([1.0, 2.0, 1.5]))


# =============================================================================
# Monitor
# =============================================================================
class TestLiveStrategyMonitor:
    @staticmethod
    def _feed(monitor, n, win_rate, seed=0, win=0.02, loss=-0.02, start="2024-01-01"):
        rng = np.random.default_rng(seed)
        stamps = pd.date_range(start, periods=n, freq="6h")
        for i in range(n):
            monitor.on_trade_closed(win if rng.random() < win_rate else loss, stamps[i])
        return monitor.status()

    def test_no_trades_reports_insufficient_data(self):
        assert LiveStrategyMonitor(0.60).status().status == "insufficient_data"

    def test_healthy_strategy_is_not_retired(self):
        m = LiveStrategyMonitor(0.60, degraded_win_rate=0.45)
        status = self._feed(m, 200, 0.60, seed=2)
        assert status.status != "retire"

    def test_degraded_strategy_is_retired(self):
        m = LiveStrategyMonitor(0.60, degraded_win_rate=0.45)
        status = self._feed(m, 300, 0.35, seed=3)
        assert status.status == "retire"
        assert any("SPRT" in r for r in status.reasons)

    def test_rejects_non_finite_returns(self):
        with pytest.raises(ValueError, match="must be finite"):
            LiveStrategyMonitor(0.60).on_trade_closed(np.nan)

    def test_reset_clears_history(self):
        m = LiveStrategyMonitor(0.60, degraded_win_rate=0.45)
        self._feed(m, 100, 0.30, seed=4)
        m.reset()
        assert m.status().status == "insufficient_data"

    def test_reporting_surfaces(self):
        m = LiveStrategyMonitor(0.60, degraded_win_rate=0.45)
        self._feed(m, 80, 0.55, seed=5)
        assert len(m.equity_curve()) == 80
        assert len(m.llr_path()) == 80
        assert "n_trades" in m.status().to_dict()

    def test_sprt_works_without_timestamps(self):
        m = LiveStrategyMonitor(0.60, degraded_win_rate=0.45)
        for _ in range(150):
            m.on_trade_closed(-0.02)
        assert m.status().status == "retire"
        assert m.status().pf_trend is None

    def test_degraded_default_is_rejected_for_low_win_rates(self):
        with pytest.raises(ValueError, match="degraded_win_rate resolved"):
            LiveStrategyMonitor(0.08)


# =============================================================================
# Calibration and acceptance
# =============================================================================
@pytest.mark.slow
def test_naive_posterior_threshold_over_rejects():
    """The reason the SPRT exists.

    Masters recommends computing the incomplete-beta posterior after every trade
    and retiring at a 95% threshold. On a strategy whose win rate never changes,
    that procedure should fire 5% of the time. It does not, because every look is
    another chance to cross.
    """
    fires = 0
    trials, horizon, p0 = 300, 300, 0.55
    for t in range(trials):
        rng = np.random.default_rng(2000 + t)
        outcomes = rng.random(horizon) < p0
        wins = 0
        for n, outcome in enumerate(outcomes, start=1):
            wins += int(outcome)
            if n >= 20 and deterioration_probability(wins, n, p0) > 0.95:
                fires += 1
                break
    rate = fires / trials
    assert rate > 0.15, (
        f"the naive sequential threshold fired at {rate:.3f}; expected well above "
        f"the nominal 0.05 it appears to promise"
    )


@pytest.mark.slow
def test_acceptance_sprt_detects_an_injected_break():
    """ACCEPTANCE (plan Phase 4): detect a regime break with a short delay.

    Healthy for 200 trades at a 60% win rate, then a ten-point drop, then 800
    trades to detect it. Measured as the median delay across seeds, since a
    sequential detector's delay is a random variable and one run proves nothing.

    This test is why the module defaults to a reflecting barrier. With Wald's
    terminating SPRT, 184 of 200 runs accepted the historical win rate by a
    median of trade 41 and stopped watching, so the break was never seen.
    """
    delays = []
    for seed in range(40):
        rng = np.random.default_rng(3000 + seed)
        detector = WinRateSPRT(0.60, 0.50, alpha=0.05, beta=0.20, reflecting=True)
        outcomes = np.r_[rng.random(200) < 0.60, rng.random(800) < 0.50]
        for i, outcome in enumerate(outcomes):
            if detector.update(bool(outcome)) == "retire":
                delays.append(max(0, i - 200))
                break
    assert len(delays) >= 34, f"only {len(delays)}/40 runs detected the break"
    median_delay = float(np.median(delays))
    assert median_delay <= 200, f"median detection delay {median_delay:.0f} trades is too long"


@pytest.mark.slow
def test_acceptance_sprt_false_retirement_rate():
    """ACCEPTANCE (plan Phase 4): the false-alarm horizon exceeds the deployment
    horizon by a wide margin.

    The plan asked for a false-retirement rate at or below alpha. Under a
    reflecting barrier that criterion is not well posed: over an unbounded
    horizon the boundary is crossed with probability one, so the rate depends
    entirely on how long you watch. The correct criterion is the average run
    length to false alarm, and it must be compared against the number of trades
    actually planned.

    Reference run: ARL0 = 776 trades healthy, ARL1 = 101 trades degraded.
    """
    arl0 = average_run_length(0.60, 0.50, alpha=0.05, beta=0.20,
                              n_sim=300, horizon=8000, seed=7)
    arl1 = average_run_length(0.60, 0.50, alpha=0.05, beta=0.20, true_p=0.50,
                              n_sim=300, horizon=8000, seed=8)
    assert arl0 > 400, f"ARL0 of {arl0:.0f} trades is too short to deploy against"
    assert arl1 < 200, f"ARL1 of {arl1:.0f} trades is too slow to be useful"
    assert arl0 / arl1 > 4.0, f"detection ratio {arl0 / arl1:.1f} is too weak"


# =============================================================================
# Part 25 revision: horizon-based false alarms, calibrated boundary,
# completed-period profit-factor checks
# =============================================================================
from afml.monitoring import calibrate_boundary, cusum_run_lengths, false_alarm_probability


class TestHorizonCalibration:
    def test_default_wald_boundary_does_not_control_false_alarms(self):
        """At 48.5% vs 38.5% the default boundary fires on far more than 5% of
        healthy 300-trade streams."""
        fa = false_alarm_probability(0.485, 0.385, horizon=300, n_sim=4000, seed=1)
        assert 0.25 < fa < 0.36

    def test_calibrated_boundary_meets_target_out_of_sample(self):
        h = calibrate_boundary(0.485, 0.385, horizon=300, max_false_alarm=0.05, n_sim=4000, seed=2)
        fa = false_alarm_probability(0.485, 0.385, horizon=300, boundary=h, n_sim=8000, seed=3)
        assert h > np.log(0.8 / 0.05)
        assert fa < 0.065

    def test_false_alarm_grows_with_horizon(self):
        short = false_alarm_probability(0.485, 0.385, horizon=100, n_sim=4000, seed=4)
        long = false_alarm_probability(0.485, 0.385, horizon=600, n_sim=4000, seed=4)
        assert short < long

    def test_run_lengths_are_censored_at_horizon_plus_one(self):
        rl = cusum_run_lengths(0.6, 0.5, horizon=50, boundary=50.0, n_sim=100)
        assert (rl == 51).all()

    def test_upper_boundary_override_and_validation(self):
        assert WinRateSPRT(0.5, 0.4, upper_boundary=4.0).upper == 4.0
        with pytest.raises(ValueError):
            WinRateSPRT(0.5, 0.4, upper_boundary=0.0)


class TestMonitorRevision:
    def test_retire_boundary_reaches_the_detector(self):
        m = LiveStrategyMonitor(0.485, 0.385, retire_boundary=4.31)
        assert m.sprt.upper == 4.31

    def test_healthy_cusum_reports_no_spurious_undecided_reason(self):
        m = LiveStrategyMonitor(0.485, 0.385)
        rng = np.random.default_rng(0)
        stamps = pd.date_range("2022-01-01", periods=400, freq="3D")
        for t in stamps:
            status = m.on_trade_closed(0.01 if rng.random() < 0.55 else -0.01, t)
        assert not any("undecided" in r for r in status.reasons)

    def test_trend_uses_completed_periods_only(self):
        m = LiveStrategyMonitor(0.5, 0.4, pf_freq="ME")
        rng = np.random.default_rng(1)
        stamps = pd.date_range("2022-01-01", "2022-10-20", freq="1D")
        for t in stamps:
            status = m.on_trade_closed(rng.normal(0.001, 0.01), t)
        # January to September are complete; October is still open.
        assert status.pf_trend is not None and status.pf_trend.n_periods == 9
