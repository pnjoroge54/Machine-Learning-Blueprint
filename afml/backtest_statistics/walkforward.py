"""
Walk-forward degradation as a paired difference, not a ratio.

Walk-Forward Efficiency (Pardo, 2008) is the ratio of out-of-sample to in-sample
performance, computed here as ``mean(OOS) / mean(IS)`` across windows, with values
near 1.0 read as evidence that the optimization is not overfitting. The intuition
is right; the statistic is a poor vehicle for it.

A ratio of means has no simple sampling distribution (its delta-method
approximation breaks down as the denominator approaches zero), becomes unstable
in that regime, and mixes signs when in-sample performance is negative in some
windows. A
strategy with mean IS Sharpe of 0.02 and mean OOS Sharpe of 0.01 reports the same
WFE of 0.5 as one with 2.0 and 1.0, and the two situations are not comparable.

The paired difference ``d_i = IS_i - OOS_i`` is a location parameter expressed in
performance units, signed and interpretable. Its uncertainty is estimated by block
bootstrap over windows, because adjacent walk-forward windows overlap in data and
are not independent. With few windows, the sign count (``n_degraded``) and its
one-sided sign-test p-value are reported as well, because a bootstrap over a
handful of values cannot be more informative than the values themselves.

Positive degradation says the in-sample scores were optimistic. It does not by
itself separate selection bias from a regime change between windows.

References
----------
Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*, 2nd ed.
Wiley.

The ratio is still reported, for continuity with the vocabulary and because it is
what readers expect. It should not drive decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, Optional, Sequence

import numpy as np

from ..screening.permutation import stationary_bootstrap_indices

__all__ = ["DegradationResult", "walkforward_degradation"]


@dataclass(frozen=True)
class DegradationResult:
    """In-sample to out-of-sample degradation across walk-forward windows.

    :param mean_is: (float) Mean in-sample score.
    :param mean_oos: (float) Mean out-of-sample score.
    :param mean_degradation: (float) Mean of ``IS - OOS``. Positive means the
        optimization was fitting noise.
    :param ci_low: (float) Lower bound of the bootstrap interval on degradation.
    :param ci_high: (float) Upper bound.
    :param p_value: (float) One-sided bootstrap p-value, ``(1 + #{means <= 0}) /
        (1 + n_boot)``. When every window degrades, no bootstrap mean can be at or
        below zero and this reaches its floor mechanically.
    :param n_windows: (int) Walk-forward windows.
    :param mean_block: (float) Block length used over windows.
    :param wfe_ratio: (float) Pardo's ratio of means, reported for continuity.
    :param wfe_reliable: (bool) ``False`` when any window has a non-positive IS
        score, or when the mean IS score is within ``min_is_tstat`` standard errors
        of zero. Under those conditions the ratio is unstable or mixes signs. The
        flag cannot repair the ratio's other weakness, which is that it is unitless.
    :param n_degraded: (int) Windows with ``IS > OOS``.
    :param sign_test_p: (float) One-sided sign-test p-value for ``n_degraded`` out
        of ``n_windows`` under the null that degradation is equally likely to be
        positive or negative in independent windows.
    """

    mean_is: float
    mean_oos: float
    mean_degradation: float
    ci_low: float
    ci_high: float
    p_value: float
    n_windows: int
    mean_block: float
    wfe_ratio: float
    wfe_reliable: bool
    n_degraded: int = 0
    sign_test_p: float = float("nan")

    @property
    def overfits(self) -> bool:
        """True when the bootstrap interval excludes zero degradation.

        Read this as "the in-sample scores were optimistic", not as a diagnosis of
        why: selection bias and a regime change both produce positive degradation.
        """
        return self.ci_low > 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_is": self.mean_is,
            "mean_oos": self.mean_oos,
            "mean_degradation": self.mean_degradation,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "overfits": self.overfits,
            "n_windows": self.n_windows,
            "mean_block": self.mean_block,
            "wfe_ratio": self.wfe_ratio,
            "wfe_reliable": self.wfe_reliable,
            "n_degraded": self.n_degraded,
            "sign_test_p": self.sign_test_p,
        }

    def __repr__(self) -> str:
        flag = "" if self.wfe_reliable else " [ratio unreliable]"
        return (
            f"DegradationResult(IS={self.mean_is:.3f}, OOS={self.mean_oos:.3f}, "
            f"degradation={self.mean_degradation:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}], p={self.p_value:.3f}, "
            f"degraded={self.n_degraded}/{self.n_windows}, "
            f"WFE={self.wfe_ratio:.2f}{flag})"
        )


def walkforward_degradation(
    is_scores: Sequence[float],
    oos_scores: Sequence[float],
    mean_block: Optional[float] = None,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.90,
    min_is_tstat: float = 2.0,
) -> DegradationResult:
    """Paired in-sample to out-of-sample degradation with a bootstrap interval.

    :param is_scores: (sequence) In-sample score per walk-forward window.
    :param oos_scores: (sequence) Out-of-sample score per window, same order.
    :param mean_block: (float) Mean block length over *windows*. Defaults to 2.0,
        since adjacent windows overlap in data (for example, one window's
        out-of-sample year is the next window's in-sample year). Set to 1.0 only
        when windows share no data at all.
    :param n_boot: (int) Bootstrap replicates.
    :param seed: (int) Random seed.
    :param ci: (float) Interval coverage, e.g. ``0.90`` for a 5th to 95th
        percentile interval.
    :param min_is_tstat: (float) Minimum t-statistic of the mean IS score for the
        ratio to be flagged reliable.
    :return: (DegradationResult)

    .. note:: The p-value is one-sided against the null of no degradation. It is
       a bootstrap probability, not an exact test, and with the handful of windows
       a typical walk-forward produces it should be read as a magnitude rather
       than a threshold. Six windows cannot resolve a small effect no matter how
       many bootstrap replicates are drawn.
    """
    is_arr = np.asarray(is_scores, dtype=float)
    oos_arr = np.asarray(oos_scores, dtype=float)
    if is_arr.shape != oos_arr.shape:
        raise ValueError(
            f"score arrays must align: {is_arr.shape} in-sample vs {oos_arr.shape} out-of-sample"
        )
    mask = np.isfinite(is_arr) & np.isfinite(oos_arr)
    is_arr, oos_arr = is_arr[mask], oos_arr[mask]
    n = is_arr.size
    if n < 3:
        raise ValueError(f"need at least 3 paired windows; got {n}")

    diff = is_arr - oos_arr
    block = 2.0 if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block=block, n_perm=n_boot, rng=rng)
    means = diff[idx].mean(axis=1)

    tail = (1.0 - ci) / 2.0
    mean_is = float(is_arr.mean())
    is_se = float(is_arr.std(ddof=1) / np.sqrt(n))
    is_tstat = abs(mean_is) / is_se if is_se > 0 else float("inf")
    reliable = bool(np.all(is_arr > 0) and abs(mean_is) > 1e-12 and is_tstat >= min_is_tstat)
    k = int(np.sum(diff > 0))
    sign_p = float(sum(comb(n, j) for j in range(k, n + 1)) / 2.0 ** n)

    return DegradationResult(
        mean_is=mean_is,
        mean_oos=float(oos_arr.mean()),
        mean_degradation=float(diff.mean()),
        ci_low=float(np.quantile(means, tail)),
        ci_high=float(np.quantile(means, 1.0 - tail)),
        p_value=float((1.0 + np.sum(means <= 0.0)) / (1.0 + n_boot)),
        n_windows=int(n),
        mean_block=block,
        wfe_ratio=float(oos_arr.mean() / mean_is) if abs(mean_is) > 1e-12 else float("nan"),
        wfe_reliable=reliable,
        n_degraded=k,
        sign_test_p=sign_p,
    )
