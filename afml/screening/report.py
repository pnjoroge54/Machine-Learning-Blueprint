"""
Result containers for the screening subpackage.

The design invariant of this module is that a screening statistic is never
returned as a bare float. Every diagnostic in ``afml.screening`` returns a
:class:`ScreeningResult`, which carries the statistic together with the null
distribution it was compared against, the permutation scheme that generated
that null, and the effective sample size that governs how much the observation
is actually worth.

This is structural, not stylistic. A statistic detached from its null invites
threshold-based decisions, and thresholds calibrated by intuition are the
mechanism by which selection bias enters a research pipeline. ``ScreeningResult``
deliberately implements no ``__float__`` and exposes no ``.value`` alias, so a
caller cannot accidentally reduce it to a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.90, 0.95, 0.99)


@dataclass(frozen=True)
class ScreeningResult:
    """A screening statistic bound to the null distribution that calibrates it.

    :param name: (str) Identifier of the screened object, typically a feature name.
    :param statistic_name: (str) Name of the statistic, e.g. ``"normalized_mutual_information"``.
    :param statistic: (float) Observed value of the statistic on the real data.
    :param null_mean: (float) Mean of the statistic across the generated null.
    :param null_std: (float) Standard deviation of the statistic across the null.
    :param null_quantiles: (dict) Mapping of quantile level to null value.
    :param p_value: (float) One-sided permutation p-value, ``(1 + #{null >= obs}) / (1 + n_perm)``.
    :param n_permutations: (int) Number of null replicates generated.
    :param null_scheme: (str) Name of the scheme that generated the null.
    :param effective_sample_size: (float) Autocorrelation-adjusted sample size of the target.
    :param n_observations: (int) Raw number of observations.
    :param p_value_adjusted: (float) Multiple-testing-corrected p-value, populated by
        :func:`afml.screening.multiple_testing` in Phase 2. ``None`` until corrected.
    :param recomputed_feature: (bool) ``True`` when the feature was recomputed on each
        permuted bar series, ``False`` when the weaker target-resampling fallback was used.
    :param metadata: (dict) Free-form scheme parameters, e.g. block length.
    """

    name: str
    statistic_name: str
    statistic: float
    null_mean: float
    null_std: float
    null_quantiles: dict[float, float]
    p_value: float
    n_permutations: int
    null_scheme: str
    effective_sample_size: float
    n_observations: int
    p_value_adjusted: float | None = None
    recomputed_feature: bool = False
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.p_value <= 1.0:
            raise ValueError(f"p_value must lie in (0, 1]; got {self.p_value}")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be at least 1")

    @property
    def excess(self) -> float:
        """Observed statistic minus the null mean.

        The null mean is generally non-zero because of finite-sample estimation
        bias, so this quantity, and not the raw statistic, is the part of the
        observation that is not explained by the estimator itself.
        """
        return self.statistic - self.null_mean

    @property
    def z_score(self) -> float:
        """Standardized excess. Reported for ranking only.

        The null is rarely Gaussian, so this value must not be converted into a
        p-value. Use :attr:`p_value`.
        """
        if self.null_std == 0.0:
            return np.nan
        return self.excess / self.null_std

    def passes(self, alpha: float = 0.05, use_adjusted: bool = True) -> bool:
        """Whether the statistic clears the null at level ``alpha``.

        :param alpha: (float) Significance level.
        :param use_adjusted: (bool) Prefer the multiple-testing-corrected p-value
            when one is available. Defaults to ``True``.
        :return: (bool) Screen outcome.
        """
        p = (
            self.p_value_adjusted
            if (use_adjusted and self.p_value_adjusted is not None)
            else self.p_value
        )
        return p < alpha

    def with_adjusted_p(self, p_adjusted: float) -> ScreeningResult:
        """Return a copy carrying a multiple-testing-corrected p-value."""
        return ScreeningResult(**{**self.__dict__, "p_value_adjusted": float(p_adjusted)})

    def to_dict(self) -> dict[str, object]:
        """Flatten to a record suitable for a report row."""
        row: dict[str, object] = {
            "name": self.name,
            "statistic_name": self.statistic_name,
            "statistic": self.statistic,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "excess": self.excess,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "n_permutations": self.n_permutations,
            "null_scheme": self.null_scheme,
            "n_observations": self.n_observations,
            "effective_sample_size": self.effective_sample_size,
            "recomputed_feature": self.recomputed_feature,
        }
        for level, value in sorted(self.null_quantiles.items()):
            row[f"null_q{int(round(level * 100)):02d}"] = value
        row.update(self.metadata)
        return row

    def __repr__(self) -> str:
        return (
            f"ScreeningResult({self.name!r}, {self.statistic_name}={self.statistic:.5f}, "
            f"null_mean={self.null_mean:.5f}, p={self.p_value:.4f}, "
            f"scheme={self.null_scheme}, ess={self.effective_sample_size:.0f})"
        )


def results_to_frame(results) -> pd.DataFrame:
    """Assemble an iterable of :class:`ScreeningResult` into a report frame.

    :param results: (iterable) ScreeningResult objects.
    :return: (pd.DataFrame) One row per result, sorted by ascending p-value.
    """
    rows = [r.to_dict() for r in results]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("name")
    sort_key = "p_value_adjusted" if frame["p_value_adjusted"].notna().all() else "p_value"
    return frame.sort_values([sort_key, "statistic"], ascending=[True, False])
