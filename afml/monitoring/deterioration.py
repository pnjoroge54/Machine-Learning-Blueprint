"""
Detecting live strategy deterioration without retiring healthy strategies.

Masters poses the problem correctly. A strategy with a 60% historical win rate
delivers 25 wins in 50 live trades. Is that deterioration, or a bad streak
consistent with the strategy's own distribution? He answers with the regularized
incomplete beta function, computed as each trade arrives, with retirement
triggered when the posterior probability of degradation crosses 95%.

The statistic is right. The procedure around it is not.

Evaluating a fixed threshold repeatedly on accumulating data is a sequential
test. The probability that the statistic crosses the boundary *at some point*
over a sequence of trades is far above the nominal per-look rate, because every
look is another chance to cross. Applied as described, the procedure retires
healthy strategies at a rate that has nothing to do with the 5% it appears to
promise. Measured on a stream whose win rate never changes (48.5%, 300 trades,
uniform prior, threshold 0.95), the procedure fires at least once in about a
third of streams, while a single look at trade 300 fires in about 5%.

Two things follow. First, :func:`deterioration_probability` is provided for
reporting and plotting, and is explicitly not a trigger. Second, the trigger is
:class:`WinRateSPRT`. In terminating mode it is Wald's SPRT, whose error rates
hold (approximately, via Wald's bounds) however often it is inspected. In
reflecting mode it is Page's CUSUM, which has no false-alarm *probability* of its
own: the probability of a false alarm depends on how many trades are watched.
Use :func:`false_alarm_probability` for a planned horizon and
:func:`calibrate_boundary` to set the boundary that holds it at a chosen level.
A CUSUM run with the default Wald boundary is not automatically safer than the
naive threshold; at 48.5% against 38.5% it raises a false alarm in about 30% of
healthy 300-trade streams. Its advantage is that this rate is computable and
adjustable, and that at a matched false-alarm rate it detects a break sooner.

The SPRT requires something the posterior does not: a pre-specified alternative.
You must state the degraded win rate worth detecting before the strategy goes
live. That is a feature. "Retire when it looks bad" is not a decision rule, and
the discipline Masters advocates in his closing section is precisely the
discipline of fixing thresholds before seeing results.

References
----------
Wald, A. (1945). Sequential tests of statistical hypotheses.
Masters, T. (2020). *Permutation and Randomization Tests for Trading System
Development*, on the incomplete beta monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
from scipy.stats import beta as beta_dist

__all__ = [
    "SPRTState",
    "WinRateSPRT",
    "average_run_length",
    "calibrate_boundary",
    "cusum_run_lengths",
    "deterioration_probability",
    "expected_sample_size",
    "false_alarm_probability",
]

Decision = Literal["continue", "retire", "accept"]


def deterioration_probability(k_wins: int, n_trades: int, p0: float) -> float:
    """Posterior probability that the true win rate is below ``p0``.

    With a uniform prior, the posterior over the win rate after ``k`` wins in
    ``n`` trades is ``Beta(k + 1, n - k + 1)``, and the probability of interest is
    its CDF at ``p0``, the regularized incomplete beta function
    ``I_{p0}(k+1, n-k+1)``.

    :param k_wins: (int) Wins observed.
    :param n_trades: (int) Trades observed.
    :param p0: (float) Historical win rate.
    :return: (float) Posterior probability in ``[0, 1]``.

    .. warning:: Do not use a fixed threshold on this quantity as a retirement
       trigger. Recomputing it after every trade and firing the first time it
       exceeds 0.95 is a sequential test with no error control. Use
       :class:`WinRateSPRT`. This function is for reporting and for the
       monitoring chart.
    """
    if n_trades < 0 or not 0 <= k_wins <= n_trades:
        raise ValueError(f"need 0 <= k_wins <= n_trades; got {k_wins}/{n_trades}")
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie in (0, 1); got {p0}")
    return float(beta_dist.cdf(p0, k_wins + 1, n_trades - k_wins + 1))


@dataclass
class SPRTState:
    """Snapshot of a sequential test in progress.

    :param n_trades: (int) Trades observed.
    :param n_wins: (int) Wins observed.
    :param log_likelihood_ratio: (float) Cumulative log-likelihood ratio.
    :param decision: (str) ``"continue"``, ``"retire"`` or ``"accept"``.
    :param upper_boundary: (float) Threshold above which H1 is accepted.
    :param lower_boundary: (float) Threshold below which H0 is accepted.
    """

    n_trades: int
    n_wins: int
    log_likelihood_ratio: float
    decision: Decision
    upper_boundary: float
    lower_boundary: float

    @property
    def observed_win_rate(self) -> float:
        return float(self.n_wins / self.n_trades) if self.n_trades else float("nan")

    @property
    def progress_to_retirement(self) -> float:
        """Fraction of the distance from the lower to the upper boundary.

        A monitoring readout that a fixed threshold on the posterior cannot give:
        it is bounded, monotone in evidence, and comparable across strategies.
        """
        span = self.upper_boundary - self.lower_boundary
        if span <= 0:
            return float("nan")
        return float(np.clip((self.log_likelihood_ratio - self.lower_boundary) / span, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"SPRTState(n={self.n_trades}, wins={self.n_wins}, "
            f"win_rate={self.observed_win_rate:.3f}, llr={self.log_likelihood_ratio:.3f}, "
            f"progress={self.progress_to_retirement:.2f}, decision={self.decision})"
        )


class WinRateSPRT:
    """Sequential probability ratio test on live win and loss outcomes.

    Tests ``H0: win rate = p0`` against ``H1: win rate = p1`` with ``p1 < p0``.
    After each trade the cumulative log-likelihood ratio is compared against two
    boundaries. Crossing the upper boundary accepts H1 and triggers retirement;
    crossing the lower boundary accepts H0 and the strategy is confirmed healthy;
    between them, the test continues.

    Wald's boundaries are ``log((1 - beta) / alpha)`` and ``log(beta / (1 - alpha))``.
    In terminating mode the resulting error rates hold approximately (Wald's
    bounds) however often the statistic is inspected. In reflecting mode the
    boundary sets an average run length, not an error rate; see
    :func:`false_alarm_probability` and :func:`calibrate_boundary`.

    :param p0: (float) Historical win rate under H0.
    :param p1: (float) Degraded win rate worth detecting under H1. Must be below
        ``p0``. Choosing it is a business decision, not a statistical one: it is
        the smallest deterioration that would change what you do.
    :param alpha: (float) Type I error rate. Sets the upper boundary. Under
        ``reflecting=True`` it is a boundary parameter, not a false-alarm
        probability; see :func:`average_run_length`.
    :param beta: (float) Type II error rate. Sets the lower boundary.
    :param reflecting: (bool) ``True`` gives Page's CUSUM: the statistic is
        floored at zero and there is no terminal healthy decision, which is what
        continuous monitoring requires. ``False`` gives Wald's terminating SPRT,
        appropriate for a one-shot acceptance decision on a fixed batch of trades.
    :param upper_boundary: (float) Retirement boundary on the log-likelihood
        ratio. Overrides the Wald value derived from ``alpha`` and ``beta``; use
        the output of :func:`calibrate_boundary` for continuous monitoring.

    .. note:: In terminating mode the test can run a long time without deciding
       when the true win rate lies between ``p0`` and ``p1``. That is honest
       behavior, not a defect: the data do not distinguish the hypotheses. Pair it
       with :func:`expected_sample_size` so the horizon is known before
       deployment.
    """

    def __init__(self, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.20,
                 reflecting: bool = True, upper_boundary: Optional[float] = None):
        if not 0.0 < p1 < p0 < 1.0:
            raise ValueError(f"require 0 < p1 < p0 < 1; got p0={p0}, p1={p1}")
        if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("alpha and beta must lie in (0, 1)")

        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.reflecting = bool(reflecting)
        self.upper = float(np.log((1.0 - beta) / alpha)) if upper_boundary is None else float(upper_boundary)
        if self.upper <= 0.0:
            raise ValueError(f"upper_boundary must be positive; got {self.upper}")
        self.lower = 0.0 if self.reflecting else float(np.log(beta / (1.0 - alpha)))

        self._llr_win = float(np.log(p1 / p0))
        self._llr_loss = float(np.log((1.0 - p1) / (1.0 - p0)))

        self.llr = 0.0
        self.n_trades = 0
        self.n_wins = 0
        self.decision: Decision = "continue"
        self.history: List[float] = []

    def update(self, is_win: bool) -> Decision:
        """Record one trade outcome and return the current decision.

        :param is_win: (bool) Whether the trade was profitable.
        :return: (str) ``"continue"``, ``"retire"`` or ``"accept"``. Once a terminal
            decision is reached it latches and never reverts, but the statistic
            and the trade counts keep updating so the monitoring chart continues.
        """
        # The decision latches, but recording does not stop. A monitoring chart
        # needs the statistic to keep tracking past the retirement signal so the
        # operator can see whether it kept getting worse, and `llr_path` must stay
        # aligned with `equity_curve`.
        self.n_trades += 1
        self.n_wins += int(bool(is_win))
        increment = self._llr_win if is_win else self._llr_loss

        if self.reflecting:
            # Page's CUSUM: evidence of health drives the statistic toward zero
            # but never past it, so the detector keeps watching indefinitely.
            self.llr = max(0.0, self.llr + increment)
        else:
            self.llr += increment
        self.history.append(self.llr)

        if self.decision == "continue":
            if self.llr >= self.upper:
                self.decision = "retire"
            elif not self.reflecting and self.llr <= self.lower:
                self.decision = "accept"
        return self.decision

    def update_many(self, outcomes) -> Decision:
        """Feed a sequence of outcomes, stopping at the first decision.

        :param outcomes: (iterable) Boolean trade outcomes.
        :return: (str) Final decision.
        """
        for outcome in outcomes:
            if self.update(bool(outcome)) != "continue":
                break
        return self.decision

    def reset(self) -> None:
        """Clear the test. Use after a deliberate refit, never after a bad streak."""
        self.llr = 0.0
        self.n_trades = 0
        self.n_wins = 0
        self.decision = "continue"
        self.history.clear()

    @property
    def state(self) -> SPRTState:
        """Current :class:`SPRTState`."""
        return SPRTState(
            n_trades=self.n_trades,
            n_wins=self.n_wins,
            log_likelihood_ratio=self.llr,
            decision=self.decision,
            upper_boundary=self.upper,
            lower_boundary=self.lower,
        )

    def __repr__(self) -> str:
        mode = "cusum" if self.reflecting else "sprt"
        return (
            f"WinRateSPRT({mode}, p0={self.p0}, p1={self.p1}, alpha={self.alpha}, "
            f"beta={self.beta}, n={self.n_trades}, decision={self.decision})"
        )


def average_run_length(
    p0: float, p1: float, alpha: float = 0.05, beta: float = 0.20,
    true_p: Optional[float] = None, n_sim: int = 2000, horizon: int = 20_000,
    seed: int = 0, boundary: Optional[float] = None,
) -> float:
    """Mean trades before a reflecting detector crosses the retirement boundary.

    Under a reflecting barrier there is no false-alarm *probability* to quote,
    because over an unbounded horizon the boundary is crossed with probability
    one. ARL0, the run length when the strategy is healthy, is the quantity that
    replaces it, and ARL1, the run length once degradation begins, is the
    detection delay. A usable configuration has ARL0 far above the number of
    trades you plan to place and ARL1 far below it.

    :param p0: (float) Win rate under H0, used to build the detector.
    :param p1: (float) Win rate under H1, used to build the detector.
    :param alpha: (float) Type I parameter setting the boundary.
    :param beta: (float) Type II parameter setting the boundary.
    :param true_p: (float) True win rate of the simulated stream. Defaults to
        ``p0``, giving ARL0.
    :param n_sim: (int) Simulation runs.
    :param horizon: (int) Trades per run before giving up.
    :param seed: (int) Random seed.
    :param boundary: (float) Retirement boundary. Defaults to the Wald value.
    :return: (float) Mean run length. Runs that never fire contribute ``horizon``,
        so the result is a lower bound when many runs are censored.

    .. note:: The run length is roughly geometric, so a mean is a poor guide to
       the risk over a finite deployment: with ARL0 twice the planned trade count,
       a false alarm still occurs in roughly 40% of healthy deployments. Use
       :func:`false_alarm_probability` for the planned horizon.
    """
    p = float(p0 if true_p is None else true_p)
    rng = np.random.default_rng(seed)
    llr_win = float(np.log(p1 / p0))
    llr_loss = float(np.log((1.0 - p1) / (1.0 - p0)))
    boundary = float(np.log((1.0 - beta) / alpha)) if boundary is None else float(boundary)

    lengths = np.empty(n_sim, dtype=float)
    for run in range(n_sim):
        stat = 0.0
        fired = horizon
        outcomes = rng.random(horizon) < p
        for i, win in enumerate(outcomes, start=1):
            stat = max(0.0, stat + (llr_win if win else llr_loss))
            if stat >= boundary:
                fired = i
                break
        lengths[run] = fired
    return float(lengths.mean())


def expected_sample_size(
    p0: float, p1: float, alpha: float = 0.05, beta: float = 0.20, true_p: Optional[float] = None
) -> float:
    """Wald's approximate expected number of trades before a decision.

    :param p0: (float) Win rate under H0.
    :param p1: (float) Win rate under H1.
    :param alpha: (float) Type I error rate.
    :param beta: (float) Type II error rate.
    :param true_p: (float) True win rate. Defaults to ``p1``, the detection case.
    :return: (float) Expected trades to a decision.

    .. note:: The approximation ignores overshoot past the boundary and is
       unreliable when ``true_p`` sits between ``p0`` and ``p1``, where the
       expected sample size is largest and the test is least decisive. Use it to
       size the monitoring horizon, not to promise one.
    """
    p = float(p1 if true_p is None else true_p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"true_p must lie in (0, 1); got {p}")
    upper = np.log((1.0 - beta) / alpha)
    lower = np.log(beta / (1.0 - alpha))
    drift = p * np.log(p1 / p0) + (1.0 - p) * np.log((1.0 - p1) / (1.0 - p0))
    if abs(drift) < 1e-12:
        return float("inf")
    prob_upper = 1.0 - beta if p <= p1 else alpha
    return float(abs((prob_upper * upper + (1.0 - prob_upper) * lower) / drift))


def cusum_run_lengths(
    p0: float, p1: float, true_p: Optional[float] = None, horizon: int = 1000,
    boundary: Optional[float] = None, alpha: float = 0.05, beta: float = 0.20,
    n_sim: int = 20_000, seed: int = 0,
) -> np.ndarray:
    """Simulated run lengths of the reflecting detector, vectorized across runs.

    :param p0: (float) Win rate under H0, used to build the detector.
    :param p1: (float) Win rate under H1, used to build the detector.
    :param true_p: (float) True win rate of the simulated streams. Defaults to ``p0``.
    :param horizon: (int) Trades simulated per run.
    :param boundary: (float) Retirement boundary. Defaults to the Wald value.
    :param alpha: (float) Used only when ``boundary`` is ``None``.
    :param beta: (float) Used only when ``boundary`` is ``None``.
    :param n_sim: (int) Runs.
    :param seed: (int) Random seed. The same seed gives the same outcome streams
        for every boundary, so comparisons across boundaries share random numbers.
    :return: (np.ndarray) Trade index of the first crossing per run, or
        ``horizon + 1`` when the boundary was not crossed.
    """
    if not 0.0 < p1 < p0 < 1.0:
        raise ValueError(f"require 0 < p1 < p0 < 1; got p0={p0}, p1={p1}")
    p = float(p0 if true_p is None else true_p)
    h = float(np.log((1.0 - beta) / alpha)) if boundary is None else float(boundary)
    llr_win, llr_loss = float(np.log(p1 / p0)), float(np.log((1.0 - p1) / (1.0 - p0)))
    rng = np.random.default_rng(seed)
    stat = np.zeros(n_sim)
    lengths = np.full(n_sim, horizon + 1, dtype=np.int64)
    for t in range(1, int(horizon) + 1):
        wins = rng.random(n_sim) < p
        stat = np.maximum(0.0, stat + np.where(wins, llr_win, llr_loss))
        hit = (stat >= h) & (lengths > horizon)
        lengths[hit] = t
    return lengths


def false_alarm_probability(
    p0: float, p1: float, horizon: int, boundary: Optional[float] = None,
    alpha: float = 0.05, beta: float = 0.20, n_sim: int = 20_000, seed: int = 0,
) -> float:
    """Probability that a healthy strategy is retired within ``horizon`` trades.

    This is the quantity a deployment decision needs, and the one the default Wald
    boundary does not control under a reflecting barrier.

    :param p0: (float) Historical win rate (the true rate of the simulated streams).
    :param p1: (float) Degraded win rate the detector targets.
    :param horizon: (int) Trades planned.
    :param boundary: (float) Retirement boundary. Defaults to the Wald value.
    :param alpha: (float) Used only when ``boundary`` is ``None``.
    :param beta: (float) Used only when ``boundary`` is ``None``.
    :param n_sim: (int) Runs. The Monte Carlo standard error is about
        ``sqrt(q (1 - q) / n_sim)``.
    :param seed: (int) Random seed.
    :return: (float) Fraction of healthy runs that cross within the horizon.
    """
    rl = cusum_run_lengths(p0, p1, true_p=p0, horizon=horizon, boundary=boundary,
                           alpha=alpha, beta=beta, n_sim=n_sim, seed=seed)
    return float(np.mean(rl <= horizon))


def calibrate_boundary(
    p0: float, p1: float, horizon: int, max_false_alarm: float = 0.05,
    n_sim: int = 20_000, seed: int = 0, tol: float = 1e-3,
) -> float:
    """Smallest boundary whose false-alarm probability over ``horizon`` is at most the target.

    Bisection on the boundary with common random numbers, so the estimated
    false-alarm probability is monotone in the boundary across iterations.

    :param p0: (float) Historical win rate.
    :param p1: (float) Degraded win rate the detector targets.
    :param horizon: (int) Trades planned before the next deliberate review.
    :param max_false_alarm: (float) Target probability of retiring a healthy strategy.
    :param n_sim: (int) Runs per evaluation.
    :param seed: (int) Random seed shared by every evaluation.
    :param tol: (float) Boundary tolerance.
    :return: (float) Boundary to pass as ``upper_boundary``.
    """
    if not 0.0 < max_false_alarm < 1.0:
        raise ValueError("max_false_alarm must lie in (0, 1)")
    lo, hi = 1e-6, 1.0
    while false_alarm_probability(p0, p1, horizon, boundary=hi, n_sim=n_sim, seed=seed) > max_false_alarm:
        lo, hi = hi, hi * 2.0
        if hi > 1e3:
            raise RuntimeError("could not bracket a boundary; check the inputs")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if false_alarm_probability(p0, p1, horizon, boundary=mid, n_sim=n_sim, seed=seed) > max_false_alarm:
            lo = mid
        else:
            hi = mid
    return float(hi)
