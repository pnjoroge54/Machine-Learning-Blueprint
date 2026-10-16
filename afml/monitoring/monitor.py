"""
Live strategy monitoring: the post-deployment surface the pipeline lacked.

Beyond its constructors, ``afml.strategies.learned_strategies.LearnedStrategy``
exposes ``generate_signals``, ``get_strategy_name`` and ``get_objective``. None of
them accepts a trade outcome.
Once a strategy is deployed there is no place for a trade outcome to go and no
object that knows whether the strategy is still working. Everything Masters says
about honest development applies with more force after deployment, where the
incentive to keep a losing strategy running is strongest and the evidence arrives
one trade at a time.

:class:`LiveStrategyMonitor` composes the two detectors, which see different
failures:

* :class:`~afml.monitoring.deterioration.WinRateSPRT` tracks how *often* the
  strategy wins. It fires on a hit-rate collapse.
* :func:`~afml.monitoring.profit_factor.profit_factor_trend` tracks how *much* it
  wins relative to what it loses. It fires when winners shrink or losers grow
  while the hit rate holds.

A strategy can fail either way, and neither detector sees the other's failure.
The monitor reports both. A win-rate retirement sets ``"retire"``; a
profit-factor decline sets ``"warning"``, because that test is evaluated once per
completed period and its error rate over a deployment depends on how many periods
are watched.

Deliberately absent: an automatic kill switch. The monitor returns a status and
the reasons behind it. Wiring that to order flow is a decision with capital at
stake and belongs in the execution layer, where the operator can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .deterioration import WinRateSPRT, cusum_run_lengths, deterioration_probability
from .profit_factor import TrendResult, granular_profit_factor, profit_factor_trend

__all__ = ["MonitorStatus", "LiveStrategyMonitor"]

Status = Literal["healthy", "warning", "retire", "insufficient_data"]


@dataclass
class MonitorStatus:
    """Current health assessment.

    :param status: (str) ``"healthy"``, ``"warning"``, ``"retire"`` or
        ``"insufficient_data"``.
    :param reasons: (list) Human-readable findings behind the status.
    :param n_trades: (int) Trades recorded.
    :param win_rate: (float) Observed win rate.
    :param sprt_decision: (str) Current SPRT decision.
    :param sprt_progress: (float) Fraction of the distance to the retirement
        boundary, in ``[0, 1]``.
    :param posterior_degraded: (float) Incomplete-beta posterior, for reporting.
    :param pf_trend: (TrendResult) Profit-factor trend, when computable.
    :param expected_trades_to_decision: (float) Simulated average run length to
        retirement under H1 (ARL1) at the configured boundary.
    """

    status: Status
    reasons: List[str] = field(default_factory=list)
    n_trades: int = 0
    win_rate: float = float("nan")
    sprt_decision: str = "continue"
    sprt_progress: float = 0.0
    posterior_degraded: float = float("nan")
    pf_trend: Optional[TrendResult] = None
    expected_trades_to_decision: float = float("nan")

    def to_dict(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "status": self.status,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "sprt_decision": self.sprt_decision,
            "sprt_progress": self.sprt_progress,
            "posterior_degraded": self.posterior_degraded,
            "expected_trades_to_decision": self.expected_trades_to_decision,
            "reasons": "; ".join(self.reasons),
        }
        if self.pf_trend is not None:
            row["pf_slope"] = self.pf_trend.slope
            row["pf_p_value"] = self.pf_trend.p_value
            row["pf_half_life"] = self.pf_trend.half_life
        return row

    def __repr__(self) -> str:
        return (
            f"MonitorStatus({self.status}, n={self.n_trades}, "
            f"win_rate={self.win_rate:.3f}, sprt_progress={self.sprt_progress:.2f})"
        )


class LiveStrategyMonitor:
    """Sequential health monitoring for a deployed strategy.

    :param historical_win_rate: (float) Win rate established in backtest, H0.
    :param degraded_win_rate: (float) Win rate worth detecting, H1. Defaults to
        ``historical_win_rate - 0.10``. State it before deployment.
    :param alpha: (float) Boundary parameter. The detector is a reflecting CUSUM,
        so this is not the probability of retiring a healthy strategy; that
        probability depends on the number of trades watched. See
        :func:`~afml.monitoring.deterioration.false_alarm_probability`.
    :param beta: (float) Boundary parameter, as for ``alpha``.
    :param retire_boundary: (float) Retirement boundary on the CUSUM statistic.
        Overrides ``alpha`` and ``beta``. Set it with
        :func:`~afml.monitoring.deterioration.calibrate_boundary` for the planned
        horizon and the false-alarm probability you accept.
    :param pf_freq: (str) Resampling frequency for granular profit factor.
    :param pf_min_periods: (int) Completed periods required before the trend is tested.
    :param pf_alpha: (float) Level for the profit-factor trend test, applied once
        per completed period. The default of 0.005 held false warnings to about 6%
        across three years of monthly checks in the Part 25 study (about 26% at
        0.05). That calibration is specific to one strategy and horizon; repeat it
        for yours.
    :param strategy_name: (str) Label carried into reports.

    Usage::

        boundary = calibrate_boundary(0.58, 0.48, horizon=300)
        monitor = LiveStrategyMonitor(historical_win_rate=0.58, retire_boundary=boundary)
        for trade in live_trades:
            status = monitor.on_trade_closed(trade.pnl, trade.close_time)
            if status.status == "retire":
                notify(status.reasons)
    """

    def __init__(
        self,
        historical_win_rate: float,
        degraded_win_rate: Optional[float] = None,
        alpha: float = 0.05,
        beta: float = 0.20,
        pf_freq: str = "W",
        pf_min_periods: int = 8,
        pf_alpha: float = 0.005,
        strategy_name: str = "strategy",
        retire_boundary: Optional[float] = None,
    ):
        p1 = degraded_win_rate if degraded_win_rate is not None else historical_win_rate - 0.10
        if p1 <= 0.0:
            raise ValueError(
                f"degraded_win_rate resolved to {p1}; supply it explicitly when the "
                f"historical rate is below 0.10"
            )
        self.strategy_name = strategy_name
        self.sprt = WinRateSPRT(historical_win_rate, p1, alpha=alpha, beta=beta,
                                upper_boundary=retire_boundary)
        self.pf_freq = pf_freq
        self.pf_min_periods = int(pf_min_periods)
        self.pf_alpha = float(pf_alpha)
        arl1 = cusum_run_lengths(historical_win_rate, p1, true_p=p1, horizon=5000,
                                 boundary=self.sprt.upper, n_sim=2000, seed=0)
        self.expected_horizon = float(arl1.mean())
        self._trend_cache: Optional[TrendResult] = None
        self._trend_periods = -1

        self._returns: List[float] = []
        self._timestamps: List[pd.Timestamp] = []

    # -- ingestion -----------------------------------------------------------
    def on_trade_closed(self, trade_return: float, timestamp=None) -> MonitorStatus:
        """Record a closed trade and return the updated status.

        :param trade_return: (float) Realized return. Sign determines win or loss;
            zero counts as a loss, since a scratch trade paid costs.
        :param timestamp: (datetime-like) Close time. Required for the
            profit-factor trend; the SPRT works without it.
        :return: (MonitorStatus)
        """
        value = float(trade_return)
        if not np.isfinite(value):
            raise ValueError(f"trade_return must be finite; got {trade_return}")
        self._returns.append(value)
        self._timestamps.append(pd.Timestamp(timestamp) if timestamp is not None else pd.NaT)
        self.sprt.update(value > 0.0)
        return self.status()

    def reset(self) -> None:
        """Clear all state. Use after a deliberate refit, never after a bad run."""
        self.sprt.reset()
        self._returns.clear()
        self._timestamps.clear()
        self._trend_cache = None
        self._trend_periods = -1

    # -- assessment ----------------------------------------------------------
    def status(self) -> MonitorStatus:
        """Assess health from everything recorded so far."""
        n = len(self._returns)
        if n == 0:
            return MonitorStatus(status="insufficient_data", reasons=["no trades recorded"])

        state = self.sprt.state
        wins = state.n_wins
        reasons: List[str] = []
        status: Status = "healthy"

        if state.decision == "retire":
            status = "retire"
            reasons.append(
                f"SPRT accepted the degraded hypothesis after {state.n_trades} trades "
                f"(win rate {state.observed_win_rate:.3f} against {self.sprt.p0:.3f})"
            )
        elif state.decision == "accept":
            reasons.append(f"SPRT confirmed the historical win rate after {state.n_trades} trades")
        elif state.progress_to_retirement > 0.6:
            status = "warning"
            reasons.append(
                f"SPRT is {state.progress_to_retirement:.0%} of the way to retirement"
            )

        trend = self._trend()
        if trend is not None and trend.deteriorating(self.pf_alpha):
            reasons.append(
                f"profit factor declining at {trend.slope:.4f} per period "
                f"(p={trend.p_value:.3f}, half-life {trend.half_life:.1f} periods)"
            )
            status = "retire" if status == "retire" else "warning"

        return MonitorStatus(
            status=status,
            reasons=reasons or ["no deterioration detected"],
            n_trades=n,
            win_rate=float(wins / n),
            sprt_decision=state.decision,
            sprt_progress=state.progress_to_retirement,
            posterior_degraded=deterioration_probability(wins, n, self.sprt.p0),
            pf_trend=trend,
            expected_trades_to_decision=self.expected_horizon,
        )

    def _trend(self) -> Optional[TrendResult]:
        """Profit-factor trend on completed periods, or ``None`` without enough history.

        The period containing the latest trade is excluded, so the test runs once
        per completed period rather than after every trade, and a partial period
        never enters the regression. The result is cached until a period completes.
        """
        stamps = pd.DatetimeIndex(self._timestamps)
        mask = ~stamps.isna()
        if not mask.any():
            return None
        series = pd.Series(np.asarray(self._returns)[mask], index=stamps[mask]).sort_index()
        codes = series.groupby(pd.Grouper(freq=self.pf_freq)).ngroup().to_numpy()
        completed = series[codes < codes.max()]
        n_completed = int(codes.max())
        if n_completed == self._trend_periods:
            return self._trend_cache
        self._trend_periods = n_completed
        self._trend_cache = None
        if completed.empty:
            return None
        try:
            pf = granular_profit_factor(completed, freq=self.pf_freq)
        except (TypeError, ValueError):
            return None
        if len(pf) < self.pf_min_periods:
            return None
        try:
            self._trend_cache = profit_factor_trend(pf, n_perm=1000)
        except ValueError:
            return None
        return self._trend_cache

    # -- reporting -----------------------------------------------------------
    def equity_curve(self) -> pd.Series:
        """Compounded equity from recorded trades."""
        if not self._returns:
            return pd.Series(dtype=float)
        index = pd.DatetimeIndex(self._timestamps)
        if index.isna().all():
            index = pd.RangeIndex(len(self._returns))
        return pd.Series(np.cumprod(1.0 + np.asarray(self._returns)), index=index, name="equity")

    def llr_path(self) -> pd.Series:
        """Cumulative log-likelihood ratio after each trade.

        The monitoring chart. Plot it with the two boundaries; the distance to the
        upper line is how close the strategy is to retirement, on a scale whose
        error rates are known.
        """
        return pd.Series(self.sprt.history, name="log_likelihood_ratio")

    def __repr__(self) -> str:
        return (
            f"LiveStrategyMonitor({self.strategy_name!r}, p0={self.sprt.p0}, "
            f"p1={self.sprt.p1}, n={len(self._returns)}, "
            f"status={self.status().status})"
        )
