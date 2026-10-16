"""
Granular profit factor as an early-warning metric.

Profit factor is gross profit over gross loss. Aggregated across an entire
evaluation period it hides temporal structure: a strategy that worked for
eighteen months and deteriorated for six can still show an aggregate figure above
1.0, and waiting for the aggregate to turn means absorbing six months of avoidable
losses.

Computing it at a fine granularity and testing the trend surfaces the
deterioration earlier. Two implementation choices are load-bearing.

**Regress the logarithm.** A profit factor of 2.0 and one of 0.5 are symmetric
departures from break-even (log 2 and -log 2), and on the raw scale they are not:
the first is 1.0 above break-even, the second 0.5 below. An untransformed
regression gives high-profit-factor periods more weight than losing periods.

**Permute the period order for the null.** Profit factor is a ratio of sums,
bounded below at zero and heavily right-skewed, so the ordinary least squares
t-distribution for the slope is not trustworthy. Permuting the order of the
granular periods gives the exact null distribution of the slope when the periods
are exchangeable, with no distributional assumption. Serial dependence across
periods breaks exchangeability and makes the test anti-conservative under
positive autocorrelation; check the lag-one autocorrelation of ``log(PF)``
before relying on the p-value.

**Each call is one test.** Recomputing the trend after every trade and acting on
the first p-value below ``alpha`` is the same repeated-look defect the module on
win rates describes. :class:`~afml.monitoring.monitor.LiveStrategyMonitor`
therefore tests only completed periods, at a stricter level.

This complements the win-rate SPRT in :mod:`afml.monitoring.deterioration`. The
two look at different things: the SPRT tracks how often the strategy wins,
profit-factor trend tracks how much it wins when it does. A strategy can hold its
win rate while its winners shrink, and only the second will see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["TrendResult", "profit_factor", "granular_profit_factor", "profit_factor_trend"]

_EPS = 1e-12


@dataclass(frozen=True)
class TrendResult:
    """Slope of log profit factor against time, with a permutation p-value.

    :param slope: (float) Slope of ``log(PF)`` per period. Negative means decline.
    :param intercept: (float) Fitted intercept.
    :param p_value: (float) One-sided permutation p-value against no decline.
    :param null_mean: (float) Mean slope under order permutation, near zero.
    :param null_std: (float) Standard deviation of the permuted slope.
    :param n_periods: (int) Periods used.
    :param n_permutations: (int) Replicates.
    :param half_life: (float) Periods for the fitted profit factor to halve, or
        ``inf`` when the slope is not negative.
    """

    slope: float
    intercept: float
    p_value: float
    null_mean: float
    null_std: float
    n_periods: int
    n_permutations: int
    half_life: float

    def deteriorating(self, alpha: float = 0.05) -> bool:
        """Whether the decline clears the permutation null at ``alpha``."""
        return bool(self.slope < 0.0 and self.p_value < alpha)

    def to_dict(self) -> Dict[str, object]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "n_periods": self.n_periods,
            "n_permutations": self.n_permutations,
            "half_life": self.half_life,
        }

    def __repr__(self) -> str:
        return (
            f"TrendResult(slope={self.slope:.5f}/period, p={self.p_value:.4f}, "
            f"half_life={self.half_life:.1f}, n={self.n_periods})"
        )


def profit_factor(returns) -> float:
    """Gross profit over gross loss.

    :param returns: (array) Trade or period returns.
    :return: (float) Profit factor. ``inf`` when there are no losses, ``0.0``
        when there are no gains, ``nan`` when there are no trades.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses <= _EPS:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def granular_profit_factor(
    returns: pd.Series, freq: str = "W", min_trades: int = 5
) -> pd.Series:
    """Profit factor computed within each calendar period.

    :param returns: (pd.Series) Trade returns indexed by close timestamp.
    :param freq: (str) Pandas resampling alias, e.g. ``"D"``, ``"W"``, ``"ME"``.
    :param min_trades: (int) Fewest trades for a period to be reported. Periods
        with too few trades produce a profit factor whose variance swamps any
        trend, so they are dropped rather than smoothed.
    :return: (pd.Series) Profit factor per period, periods without enough trades
        removed.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by a DatetimeIndex")
    grouped = returns.groupby(pd.Grouper(freq=freq))
    counts = grouped.count()
    values = grouped.apply(profit_factor)
    out = values[counts >= min_trades]
    return out[np.isfinite(out.to_numpy(dtype=float))].rename("profit_factor")


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    return 0.0 if denom == 0 else float(np.dot(xc, y - y.mean()) / denom)


def profit_factor_trend(
    pf: pd.Series, n_perm: int = 2000, seed: int = 0, floor: float = 0.05
) -> TrendResult:
    """Test whether granular profit factor is declining.

    :param pf: (pd.Series) Profit factor per period, in chronological order.
    :param n_perm: (int) Permutation replicates.
    :param seed: (int) Random seed.
    :param floor: (float) Lower clip applied before taking logarithms. A period
        with no winning trades has a profit factor of zero, whose logarithm is
        undefined; clipping keeps the period in the regression at a finite,
        clearly bad value rather than discarding the worst evidence available.
    :return: (TrendResult)
    """
    values = np.asarray(pf.to_numpy(dtype=float))
    values = values[np.isfinite(values)]
    if values.size < 6:
        raise ValueError(f"need at least 6 periods to fit a trend; got {values.size}")

    log_pf = np.log(np.clip(values, floor, None))
    x = np.arange(values.size, dtype=float)
    observed = _slope(log_pf, x)
    intercept = float(log_pf.mean() - observed * x.mean())

    rng = np.random.default_rng(seed)
    null = np.array([_slope(rng.permutation(log_pf), x) for _ in range(n_perm)])
    p_value = float((1.0 + np.sum(null <= observed)) / (1.0 + n_perm))

    half_life = float(np.log(0.5) / observed) if observed < -_EPS else float("inf")

    return TrendResult(
        slope=observed,
        intercept=intercept,
        p_value=p_value,
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)),
        n_periods=int(values.size),
        n_permutations=int(n_perm),
        half_life=half_life,
    )
