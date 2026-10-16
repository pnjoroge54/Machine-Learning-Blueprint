"""
Resampling the trade sequence to estimate performance and drawdown risk.

The single realized path through history is one draw. Its maximum drawdown
depends on the order in which wins and losses arrived, and that order is not a
property of the strategy. A different sequence of comparable trades produces a
different worst case, and nothing guarantees that the realized one was typical.

This module resamples the trade sequence to recover the distribution of
performance statistics across alternative orderings. The estimate that matters is
the upper tail of the drawdown distribution, which answers the question a single
out-of-sample path cannot: how bad is the drawdown from an unlucky sequence of
otherwise typical trades.

Scope, stated precisely because it is easy to overclaim. The stationary
bootstrap draws blocks of trades *with replacement*, so each resampled path has
the original length but its own mix of trades: it varies the ordering and the
sample composition, under the assumption that future trades come from the same
distribution and dependence structure as the realized ones. It captures sequence
risk and sampling variation. It does not capture regime change, and it cannot
tell you that next year's trades will be drawn from a different distribution. For that,
:class:`afml.backtester.cpcv_dynamic_backtest.CPCVDynamicBacktest` already
generates a distribution of performance paths across combinatorial folds, and the
two tools are complements rather than substitutes: CPCV varies which data the
model saw, bootstrap varies the order in which the results arrived.

Block length is an assumption about dependence, not a free parameter to tune.
Overlapping triple-barrier labels make consecutive trade returns dependent, and
the block must be long enough to contain that dependence; for overlapping labels,
:func:`afml.screening.permutation.suggested_block_length` gives a starting point
from the label span. For non-overlapping trades there is no overlap to measure and
that function returns its floor, so check serial dependence in returns and in
absolute returns directly and report the sensitivity of results to the block.

References
----------
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
Masters, T. (2020). *Permutation and Randomization Tests for Trading System
Development*, on bootstrap lower bounds as honest forward estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..screening.permutation import stationary_bootstrap_indices

__all__ = [
    "BootstrapResult",
    "max_drawdown",
    "bootstrap_statistic",
    "bootstrap_drawdown_distribution",
    "sequence_risk_ratio",
    "breach_probability",
]

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class BootstrapResult:
    """A statistic with the sampling distribution induced by resequencing.

    :param name: (str) Statistic name.
    :param observed: (float) Value on the realized ordering.
    :param mean: (float) Mean across bootstrap replicates.
    :param std_error: (float) Standard deviation across replicates.
    :param quantiles: (dict) Quantile level to bootstrap value.
    :param n_boot: (int) Replicates drawn.
    :param mean_block: (float) Mean geometric block length used.
    :param n_observations: (int) Trades resampled.
    :param metadata: (dict) Free-form extras.
    """

    name: str
    observed: float
    mean: float
    std_error: float
    quantiles: Dict[float, float]
    n_boot: int
    mean_block: float
    n_observations: int
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def lower_bound(self) -> float:
        """Fifth-percentile bootstrap value.

        Masters' honest forward estimate: report this rather than the point
        estimate, because the point estimate is the realized draw and the
        realized draw is not a forecast.
        """
        return self.quantiles.get(0.05, float("nan"))

    @property
    def bias(self) -> float:
        """Bootstrap mean minus observed. Large magnitude means the realized
        ordering was unrepresentative of its own trade distribution."""
        return self.mean - self.observed

    def to_dict(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "statistic": self.name,
            "observed": self.observed,
            "bootstrap_mean": self.mean,
            "std_error": self.std_error,
            "bias": self.bias,
            "n_boot": self.n_boot,
            "mean_block": self.mean_block,
            "n_observations": self.n_observations,
        }
        for level, value in sorted(self.quantiles.items()):
            row[f"q{int(round(level * 100)):02d}"] = value
        row.update(self.metadata)
        return row

    def __repr__(self) -> str:
        return (
            f"BootstrapResult({self.name}, observed={self.observed:.4f}, "
            f"q05={self.lower_bound:.4f}, q95={self.quantiles.get(0.95, float('nan')):.4f}, "
            f"block={self.mean_block:.0f})"
        )


def max_drawdown(returns, compound: bool = True) -> float:
    """Maximum peak-to-trough decline of the equity curve, as a positive fraction.

    :param returns: (array) Per-trade or per-period returns.
    :param compound: (bool) Compound returns multiplicatively. Set ``False`` for
        additive P&L already expressed in account fractions.
    :return: (float) Maximum drawdown in ``[0, 1]`` when compounding.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, 1.0 - equity / peak, 0.0)
    return float(np.nanmax(dd))


def _peak_loss(r: np.ndarray, compound: bool) -> float:
    """Largest loss from the running equity peak (trailing basis)."""
    return max_drawdown(r, compound=compound)


def _initial_loss(r: np.ndarray, compound: bool) -> float:
    """Largest loss below the starting balance (static basis), as a positive fraction."""
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    return float(max(0.0, 1.0 - equity.min()))


def _resample_matrix(
    r: np.ndarray, mean_block: float, n_boot: int, seed: int
) -> np.ndarray:
    """Draw ``n_boot`` resampled return sequences of the original length."""
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(r.size, mean_block=mean_block, n_perm=n_boot, rng=rng)
    return r[idx]


def bootstrap_statistic(
    returns,
    statistic: Callable[[np.ndarray], float],
    mean_block: float,
    n_boot: int = 10_000,
    seed: int = 0,
    name: str = "statistic",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> BootstrapResult:
    """Sampling distribution of any performance statistic under resequencing.

    :param returns: (array) Per-trade returns.
    :param statistic: (callable) Maps a return array to a scalar.
    :param mean_block: (float) Mean geometric block length. Use
        :func:`afml.screening.permutation.suggested_block_length`; setting this to
        1.0 assumes independent trades and understates the tails whenever labels
        overlap.
    :param n_boot: (int) Replicates.
    :param seed: (int) Random seed.
    :param name: (str) Statistic name recorded on the result.
    :param quantiles: (sequence) Quantile levels to report.
    :return: (BootstrapResult)
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        raise ValueError(f"need at least 20 trades to resample; got {r.size}")
    if mean_block < 1.0:
        raise ValueError("mean_block must be at least 1")

    samples = _resample_matrix(r, mean_block, n_boot, seed)
    values = np.array([statistic(row) for row in samples], dtype=float)

    return BootstrapResult(
        name=name,
        observed=float(statistic(r)),
        mean=float(np.nanmean(values)),
        std_error=float(np.nanstd(values, ddof=1)),
        quantiles={float(q): float(np.nanquantile(values, q)) for q in quantiles},
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        n_observations=int(r.size),
    )


def bootstrap_drawdown_distribution(
    returns,
    mean_block: float,
    n_boot: int = 10_000,
    seed: int = 0,
    compound: bool = True,
) -> pd.Series:
    """Maximum drawdowns across resampled trade orderings.

    :param returns: (array) Per-trade returns.
    :param mean_block: (float) Mean geometric block length.
    :param n_boot: (int) Replicates.
    :param seed: (int) Random seed.
    :param compound: (bool) Compound the equity curve.
    :return: (pd.Series) One maximum drawdown per replicate, sorted ascending.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        raise ValueError(f"need at least 20 trades to resample; got {r.size}")
    samples = _resample_matrix(r, mean_block, n_boot, seed)
    values = np.array([max_drawdown(row, compound=compound) for row in samples])
    return pd.Series(np.sort(values), name="max_drawdown")


def sequence_risk_ratio(
    returns,
    mean_block: float,
    n_boot: int = 10_000,
    seed: int = 0,
    quantile: float = 0.95,
    compound: bool = True,
) -> float:
    """How much worse an unlucky ordering is than the realized one.

    Ratio of the bootstrap drawdown at ``quantile`` to the realized maximum
    drawdown. Two cautions. First, the denominator is itself a single draw, so the
    ratio inherits its noise; report the bootstrap quantile alongside the ratio.
    Second, as ``mean_block`` approaches the series length, every replicate becomes
    a circular rotation of the realized sequence and the ratio drifts toward 1.0.
    A value near 1.0 is therefore not evidence of immunity to sequence risk until
    the block length has been checked.

    :param returns: (array) Per-trade returns.
    :param mean_block: (float) Mean geometric block length.
    :param n_boot: (int) Replicates.
    :param seed: (int) Random seed.
    :param quantile: (float) Tail quantile of the drawdown distribution.
    :param compound: (bool) Compound the equity curve.
    :return: (float) Ratio, or ``inf`` when the realized drawdown is zero.
    """
    realized = max_drawdown(returns, compound=compound)
    dist = bootstrap_drawdown_distribution(returns, mean_block, n_boot, seed, compound)
    tail = float(dist.quantile(quantile))
    return float("inf") if realized == 0 else float(tail / realized)


def breach_probability(
    returns,
    mean_block: float,
    max_drawdown_limit: float,
    daily_loss_limit: Optional[float] = None,
    trades_per_day: Optional[int] = None,
    n_boot: int = 10_000,
    seed: int = 0,
    compound: bool = True,
    drawdown_basis: str = "peak",
    horizon: Optional[int] = None,
) -> Dict[str, float]:
    """Probability of breaching account limits across resampled orderings.

    Prop-firm rules are conditions on a *path*, not on a distribution of returns.
    A strategy whose realized history never breached a limit can breach it with
    substantial probability under a different ordering of the same trades, and
    estimating that probability from the single realized path returns zero by
    construction whenever the realized path survived.

    :param returns: (array) Per-trade returns as account fractions.
    :param mean_block: (float) Mean geometric block length.
    :param max_drawdown_limit: (float) Maximum drawdown breach threshold, positive
        fraction, e.g. ``0.10`` for a 10% limit.
    :param daily_loss_limit: (float) Daily loss breach threshold, positive
        fraction. Requires ``trades_per_day``.
    :param trades_per_day: (int) Trades grouped into one trading day.
    :param n_boot: (int) Replicates.
    :param seed: (int) Random seed.
    :param compound: (bool) Compound the equity curve.
    :param drawdown_basis: (str) ``"peak"`` measures the loss from the running
        equity peak (a trailing limit); ``"initial"`` measures it from the starting
        balance (a static floor, as in FundedNext Stellar accounts). The realized
        statistic and the daily limit use the same basis.
    :param horizon: (int) Trades per resampled path. Defaults to the full sample.
        Breach probability grows with path length, so set this to the length of
        the evaluation being modeled.
    :return: (dict) ``p_drawdown_breach``, ``p_daily_breach`` when computable,
        ``p_any_breach``, ``realized_max_drawdown`` (on the chosen basis, over the
        full realized sample), ``horizon``, and ``n_boot``.

    .. note:: Equity is evaluated at trade exits only. Prop-firm limits are
       usually checked on floating equity, so intra-trade excursions make the
       true breach probability higher than this estimate.

    .. note:: Trades are grouped into days by count, not by timestamp. That is an
       approximation, and it is the right one for a resampled sequence: once the
       order is randomized the original calendar no longer applies. Where exact
       calendar grouping matters, resample days rather than trades.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        raise ValueError(f"need at least 20 trades to resample; got {r.size}")
    if daily_loss_limit is not None and not trades_per_day:
        raise ValueError("daily_loss_limit requires trades_per_day")
    if drawdown_basis not in ("peak", "initial"):
        raise ValueError("drawdown_basis must be 'peak' or 'initial'")
    h = r.size if horizon is None else int(horizon)
    if not 1 <= h <= r.size:
        raise ValueError(f"horizon must lie in [1, {r.size}]; got {h}")

    samples = _resample_matrix(r, mean_block, n_boot, seed)[:, :h]
    loss = _peak_loss if drawdown_basis == "peak" else _initial_loss
    dd_breach = np.array([loss(row, compound) >= max_drawdown_limit for row in samples])

    out = {
        "p_drawdown_breach": float(dd_breach.mean()),
        "realized_max_drawdown": loss(r, compound),
        "horizon": float(h),
        "n_boot": float(n_boot),
    }

    if daily_loss_limit is not None:
        n_days = samples.shape[1] // int(trades_per_day)
        if n_days < 1:
            raise ValueError("trades_per_day exceeds the number of trades")
        trimmed = samples[:, : n_days * int(trades_per_day)]
        daily = trimmed.reshape(samples.shape[0], n_days, int(trades_per_day)).sum(axis=2)
        daily_breach = (daily <= -abs(daily_loss_limit)).any(axis=1)
        out["p_daily_breach"] = float(daily_breach.mean())
        out["p_any_breach"] = float((dd_breach | daily_breach).mean())
    else:
        out["p_any_breach"] = out["p_drawdown_breach"]

    return out
