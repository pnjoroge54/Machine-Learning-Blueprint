"""
Permuting the development process, not the strategy.

Masters' third use of permutation testing asks a question the first two do not.
The first two ask whether a particular strategy has an edge. This one asks how
much apparent performance the *process* that produced the strategy reports when
its input has been stripped of inter-bar predictability. Run the entire
development procedure (feature generation, labeling, hyperparameter search,
model fit, backtest) on bar-permuted data, and read the distribution of what it
reports.

What the null is, precisely. :func:`afml.screening.permutation.bar_permutation`
shuffles inter-bar gaps and intra-bar bodies independently. It preserves the
marginal distributions of gaps and bodies, OHLC consistency, the bar count, and
the first and last price. It destroys every property that depends on bar order:
autocorrelation, volatility clustering, intraday and weekly seasonality, calendar
effects, and the sequence of gaps. The null is therefore "exchangeable bars", not
a general model of a signal-free market. A strategy family that exploits
volatility regimes or session structure is compared against a null that has none
of either, and the result must be read with that in mind.

Relation to the Deflated Sharpe Ratio. The two answer different questions and
are complementary. DSR corrects one reported Sharpe for selection over N trials,
non-normality and sample length, and needs an effective number of independent
trials; :func:`afml.cross_validation.selection_overfit.effective_trial_count`
estimates it. The permutation null measures, for one specific pipeline, data
set and budget, the distribution of what the whole process reports on
exchangeable bars. It needs no trial count, but it is only as general as the
permutation null and the pipeline configuration it was run on.

Reading the result. A positive null median is expected for any process that
selects a maximum over several trials: it is the size of the selection effect,
not by itself evidence of a defect. The informative quantities are where the
observed result sits within the null (``p_value``, ``null_percentile``,
``robust_z``), and whether the selection effect exceeds a tolerance the
researcher states in advance (:meth:`ProcessPermutationResult.selection_effect_exceeds`).

Reference measurement, one synthetic check only (driftless GBM, an MA crossover
with four searched parameters, 1 bp turnover cost, 30 replicates per cell):

=========  ==================  =================
budget     TPE median Sharpe   Random median
=========  ==================  =================
10         0.54                0.54
30         1.06                0.94
100        1.72                1.42
300        2.06                1.66
=========  ==================  =================

In that one configuration the null median rose with budget under both samplers,
and TPE exceeded random search. It is not evidence about other search spaces,
strategies or data.

Two things this module enforces, because both are easy to get wrong in ways that
make the test silently pass.

**Budget parity.** The permuted runs must use exactly the trial budget the real
run used. Reducing it to save compute changes the quantity being measured.

**Cache responsiveness.** ``FunctionVersionStore`` and the unified cache key on
function version and inputs. If a permuted bar frame does not change the cache
key at the top of the chain, cached artifacts computed from the real series are
served into permuted runs. :func:`assert_cache_isolation` catches the complete
version of this failure. It is a necessary check, not a sufficient one: see its
docstring for what it cannot detect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom

from ..screening.permutation import bar_permutation, permutation_pvalue

__all__ = [
    "ProcessPermutationResult",
    "quantile_ci",
    "assert_cache_isolation",
    "permute_development_process",
    "spurious_performance_curve",
]

DevelopFn = Callable[[pd.DataFrame], float]


def _frame_digest(frame: pd.DataFrame) -> str:
    """Stable content hash of a bar frame, for cache-isolation checks."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()[:16]


def quantile_ci(values: np.ndarray, q: float, level: float = 0.95) -> tuple[float, float, float]:
    """Distribution-free confidence interval for a quantile from order statistics.

    Uses the binomial argument: the number of draws below the true q-quantile is
    Binomial(n, q), so order statistics at the binomial ``(1 - level) / 2`` and
    ``(1 + level) / 2`` quantiles bracket it with coverage at least ``level``.
    When the required order statistic falls outside the sample (small n, extreme
    q), that bound is reported as -inf or +inf rather than silently clipped.

    :param values: (np.ndarray) Sample; non-finite values are dropped.
    :param q: (float) Quantile in (0, 1).
    :param level: (float) Nominal coverage.
    :return: (tuple) ``(lower, upper, exact_coverage)``.
    """
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        raise ValueError("no finite values")
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    a = (1.0 - level) / 2.0
    r = int(binom.ppf(a, n, q))            # 1-indexed lower order statistic
    s_ = int(binom.ppf(1.0 - a, n, q)) + 1  # 1-indexed upper order statistic
    lower = float(x[r - 1]) if r >= 1 else float("-inf")
    upper = float(x[s_ - 1]) if s_ <= n else float("inf")
    r_c, s_c = max(r, 1), min(s_, n + 1)
    coverage = float(binom.cdf(s_c - 1, n, q) - binom.cdf(r_c - 1, n, q))
    return lower, upper, coverage


@dataclass(frozen=True)
class ProcessPermutationResult:
    """Distribution of development-process output on bar-permuted data.

    :param observed: (float) Performance the process produced on the real bars.
    :param null_values: (np.ndarray) Performance on each permuted bar series.
    :param spurious_center: (float) Median of the finite null values: the
        typical performance the process reports on exchangeable bars.
    :param p_value: (float) One-sided, right-tail permutation p-value,
        ``(1 + #{null >= observed}) / (1 + n)``. Ties count as at least as
        extreme, the observed arrangement is counted as one of the ``n + 1``
        (Phipson and Smyth, 2010), and the smallest attainable value is
        ``1 / (n + 1)``.
    :param n_permutations: (int) Replicates requested.
    :param trial_budget: (int) Search budget used.
    :param failures: (list) Permutation indices where the process raised.
    :param metadata: (dict) Free-form extras.
    """

    observed: float
    null_values: np.ndarray
    spurious_center: float
    p_value: float
    n_permutations: int
    trial_budget: int | None = None
    failures: list[int] = field(default_factory=list)
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def _finite(self) -> np.ndarray:
        v = np.asarray(self.null_values, dtype=float)
        return v[np.isfinite(v)]

    @property
    def excess_over_null(self) -> float:
        """Observed performance minus the null median.

        A location shift only. It ignores the width and shape of the null, so
        read it together with ``p_value``, ``null_percentile`` and ``robust_z``,
        never on its own. It is not a Deflated Sharpe Ratio.
        """
        return float(self.observed - self.spurious_center)

    @property
    def null_percentile(self) -> float:
        """Mid-rank percentile of the observed value within the null, in [0, 1]."""
        v = self._finite
        return float((np.sum(v < self.observed) + 0.5 * np.sum(v == self.observed)) / v.size)

    @property
    def robust_z(self) -> float:
        """Excess over the null median in units of the null's scaled MAD."""
        v = self._finite
        mad = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        return float("nan") if mad == 0.0 else self.excess_over_null / mad

    def quantile(self, q: float) -> float:
        """Point estimate of a null quantile."""
        return float(np.quantile(self._finite, q))

    def quantile_ci(self, q: float, level: float = 0.95) -> tuple[float, float, float]:
        """Distribution-free CI for a null quantile; see :func:`quantile_ci`."""
        return quantile_ci(self._finite, q, level)

    def median_ci(self, level: float = 0.95) -> tuple[float, float, float]:
        """Distribution-free CI for the null median."""
        return quantile_ci(self._finite, 0.5, level)

    def selection_effect_exceeds(self, tolerance: float = 0.0, level: float = 0.95) -> bool:
        """Whether the selection effect is credibly larger than ``tolerance``.

        True when the lower bound of the ``level`` CI for the null median lies
        above ``tolerance``. A positive null median is expected whenever a
        process selects the best of several trials; this method asks whether it
        is larger than the researcher is prepared to accept, with the sampling
        uncertainty of the median taken into account. The tolerance must be
        fixed before the study is run.
        """
        return bool(self.median_ci(level)[0] > tolerance)

    def to_dict(self, level: float = 0.95) -> dict[str, object]:
        lo, hi, _ = self.median_ci(level)
        return {
            "observed": self.observed,
            "spurious_center": self.spurious_center,
            "median_ci_lower": lo,
            "median_ci_upper": hi,
            "excess_over_null": self.excess_over_null,
            "robust_z": self.robust_z,
            "null_percentile": self.null_percentile,
            "p_value": self.p_value,
            "null_q05": self.quantile(0.05),
            "null_q95": self.quantile(0.95),
            "n_permutations": self.n_permutations,
            "trial_budget": self.trial_budget,
            "n_failures": len(self.failures),
            **self.metadata,
        }

    def __repr__(self) -> str:
        lo, hi, _ = self.median_ci()
        return (
            f"ProcessPermutationResult(observed={self.observed:.3f}, "
            f"null_median={self.spurious_center:.3f} [95% CI {lo:.3f}, {hi:.3f}], "
            f"percentile={self.null_percentile:.2f}, p={self.p_value:.4f}, "
            f"n={self._finite.size})"
        )


def assert_cache_isolation(
    develop_fn: DevelopFn,
    bars: pd.DataFrame,
    seed: int = 0,
    require_determinism: bool = True,
) -> None:
    """Smoke test that the development function responds to its input.

    Two checks, in this order:

    1. Determinism. ``develop_fn`` is run twice on the same permuted frame and
       must return the same value. Without this, a stochastic pipeline passes
       check 2 vacuously: its outputs differ whether or not it reads the bars.
    2. Responsiveness. ``develop_fn`` is run on a second, different permutation
       and must return a different value.

    What this does not establish. It catches the complete failure, where a cache
    serves one stale result regardless of input. It cannot prove that every
    stage reads the permuted bars: a pipeline in which one stage is stale and
    another is not will still return different values on different inputs, and
    a coarse metric can coincide on genuinely different inputs. Stronger
    evidence comes from logging the content hash of the bars at the top of the
    feature chain and confirming zero cache hits across permuted replicates.

    :param develop_fn: (callable) Maps bars to a scalar performance measure.
    :param bars: (pd.DataFrame) OHLC bars.
    :param seed: (int) Seed for the two probe permutations.
    :param require_determinism: (bool) Run check 1. Disable only for a pipeline
        that is stochastic by design, and accept that check 2 is then weak.
    :raises RuntimeError: When the probe frames are identical, when the function
        is non-deterministic (with ``require_determinism``), or when it returns
        the same value on two different frames.
    """
    probes = list(bar_permutation(bars, n_perm=2, seed=seed))
    if _frame_digest(probes[0]) == _frame_digest(probes[1]):
        raise RuntimeError(
            "bar_permutation produced two identical frames; the permutation itself "
            "is not varying, so the study would measure nothing"
        )

    first = float(develop_fn(probes[0]))
    if require_determinism:
        repeat = float(develop_fn(probes[0]))
        if first != repeat:
            raise RuntimeError(
                f"develop_fn returned {first} and then {repeat} on the same permuted "
                f"bar series. A non-deterministic process makes the responsiveness "
                f"check uninformative, and mixes optimizer randomness into the null. "
                f"Fix the sampler and model seeds for the study."
            )
    second = float(develop_fn(probes[1]))
    if first == second:
        raise RuntimeError(
            f"develop_fn returned {first} on two different permuted bar series. "
            f"The function is not responding to its input. The usual cause is a "
            f"cache whose key does not incorporate the bar content: verify that "
            f"the permutation seed or a content hash enters the cache key at the "
            f"top of the feature-generation chain, or disable caching for this run."
        )


def permute_development_process(
    develop_fn: DevelopFn,
    bars: pd.DataFrame,
    n_perm: int = 100,
    seed: int = 0,
    observed: float | None = None,
    trial_budget: int | None = None,
    check_cache_isolation: bool = True,
    on_error: str = "record",
) -> ProcessPermutationResult:
    """Run an entire development process on permuted bars.

    :param develop_fn: (callable) Takes a bar frame and returns a scalar
        performance measure, typically the backtest Sharpe of whichever
        configuration the search selected. It must contain the *whole* process:
        feature generation, labeling, hyperparameter search, fit and backtest.
    :param bars: (pd.DataFrame) Real OHLC bars.
    :param n_perm: (int) Permutation replicates. Precision matters: at 25
        replicates the 95% CI for the null median spans the 8th to 18th order
        statistics, and the CI for the 90th percentile has no finite upper
        bound. Check ``result.median_ci()`` before interpreting the center.
    :param seed: (int) Seed for the bar permutations. It does not seed the
        search: ``develop_fn`` must fix its own sampler and model seeds, and use
        the same seed policy on real and permuted bars, so that differences
        between replicates come from the data and not from optimizer noise.
    :param observed: (float) Performance on the real bars. Computed by calling
        ``develop_fn(bars)`` when omitted.
    :param trial_budget: (int) Search budget, recorded on the result. The null
        depends on it, so a result is only comparable to a null at the same budget.
    :param check_cache_isolation: (bool) Run :func:`assert_cache_isolation` first.
    :param on_error: (str) ``"record"`` logs a failed replicate and continues,
        ``"raise"`` propagates.
    :return: (ProcessPermutationResult)

    .. warning:: ``develop_fn`` must accept **bars**, not precomputed features.
       A version taking a feature matrix would hold feature generation fixed and
       silently test only the model-fitting stage, which is the narrow slice this
       test exists to look past.

    .. note:: Expensive by design. A hundred replicates of a full Optuna study
       is a long run, and reducing the trial budget to save time changes the
       measurement. The null is specific to the pipeline version, data, period,
       cost model, search space, sampler, seed policy and budget; rerun it when
       any of these change, and parallelize across replicates rather than
       shrinking the budget.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    if n_perm < 2:
        raise ValueError(f"n_perm must be at least 2; got {n_perm}")

    if check_cache_isolation:
        assert_cache_isolation(develop_fn, bars, seed=seed)

    if observed is None:
        try:
            real = float(develop_fn(bars))
        except Exception as exc:
            raise RuntimeError(
                f"develop_fn failed on the real bars: {exc}. Fix the process before "
                f"measuring what it does to noise, or supply `observed` directly."
            ) from exc
    else:
        real = float(observed)

    values: list[float] = []
    failures: list[int] = []
    for i, permuted in enumerate(bar_permutation(bars, n_perm=n_perm, seed=seed)):
        try:
            values.append(float(develop_fn(permuted)))
        except Exception:
            if on_error == "raise":
                raise
            failures.append(i)
            values.append(float("nan"))

    null = np.asarray(values, dtype=float)
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        raise RuntimeError("every permutation replicate failed; nothing to compare against")

    return ProcessPermutationResult(
        observed=real,
        null_values=null,
        spurious_center=float(np.median(finite)),
        p_value=permutation_pvalue(real, finite),
        n_permutations=int(n_perm),
        trial_budget=trial_budget,
        failures=failures,
        metadata={"n_finite": float(finite.size)},
    )


def spurious_performance_curve(
    develop_fn_factory: Callable[[int], DevelopFn],
    bars: pd.DataFrame,
    budgets: Sequence[int],
    n_perm: int = 100,
    seed: int = 0,
    include_observed: bool = False,
    level: float = 0.95,
) -> pd.DataFrame:
    """Null performance as a function of search budget.

    For one fixed pipeline configuration, converts "we ran 200 trials" into the
    null median and 90th percentile that process reports on exchangeable bars at
    that budget, with distribution-free confidence intervals. The curve is valid
    only for the configuration it was computed on (pipeline version, data,
    period, cost model, search space, sampler, seed policy); it is not a
    constant to reuse after any of these change.

    The same ``seed`` is used at every budget, so every budget is evaluated on
    the same permuted series (common random numbers). Differences between
    budgets are then not confounded with differences between permutation draws.

    :param develop_fn_factory: (callable) Takes a trial budget and returns a
        development function using exactly that budget.
    :param bars: (pd.DataFrame) Real OHLC bars.
    :param budgets: (sequence) Trial budgets to evaluate.
    :param n_perm: (int) Replicates per budget.
    :param seed: (int) Seed for the bar permutations.
    :param include_observed: (bool) Also run each budget on the real bars and
        report the observed value, its null percentile and its p-value.
    :param level: (float) Coverage of the quantile confidence intervals.
    :return: (pd.DataFrame) Indexed by budget.
    """
    rows = []
    for budget in budgets:
        develop_fn = develop_fn_factory(int(budget))
        result = permute_development_process(
            develop_fn,
            bars,
            n_perm=n_perm,
            seed=seed,
            observed=None if include_observed else float("nan"),
            trial_budget=int(budget),
            check_cache_isolation=False,
        )
        med_lo, med_hi, _ = result.median_ci(level)
        q90_lo, q90_hi, _ = result.quantile_ci(0.90, level)
        finite = result._finite
        row = {
            "trial_budget": int(budget),
            "spurious_median": result.spurious_center,
            "median_ci_lower": med_lo,
            "median_ci_upper": med_hi,
            "spurious_q90": result.quantile(0.90),
            "q90_ci_lower": q90_lo,
            "q90_ci_upper": q90_hi,
            "fraction_positive": float(np.mean(finite > 0)),
            "n_finite": int(finite.size),
        }
        if include_observed:
            row.update(
                observed=result.observed,
                null_percentile=result.null_percentile,
                p_value=result.p_value,
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("trial_budget")
