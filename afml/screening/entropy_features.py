"""
Entropy diagnostics for numeric indicator series.

This module is distinct from :mod:`afml.microstructural_features.entropy`, and
the separation is deliberate. Those estimators take a ``message: str`` and serve
tick-rule and volume-encoded message sequences. Screening a numeric indicator
requires a different discretization contract: bins over the feature's own
empirical range, not the twenty-six-letter ASCII encoding in
``microstructural_features.encoding``. Sharing a namespace would invite the wrong
estimator.

Entropy answers one question and only one: does this feature vary enough across
market conditions to carry information at all? Low entropy proves a feature
cannot be informative. High entropy proves nothing. The screen is therefore
ordered first in the pipeline because it is O(N) and needs no null, and its
verdicts are one-directional: it can reject, never endorse.

Reference
---------
Masters, T. (2019). *Statistically Sound Indicators for Financial Market
Trading*, on entropy as a pre-screen for indicator quality.
Shannon, C. E. (1948). A mathematical theory of communication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..codependence.information import get_optimal_number_of_bins

__all__ = [
    "bin_counts",
    "entropy_screen",
    "indicator_entropy",
    "modal_occupancy",
]


def bin_counts(x: np.ndarray, n_bins: int, method: str = "uniform") -> np.ndarray:
    """Discretize a series and return occupancy counts.

    :param x: (np.ndarray) Feature values. Non-finite entries are dropped.
    :param n_bins: (int) Number of bins.
    :param method: (str) ``"uniform"`` for equal-width bins over the observed
        range, ``"quantile"`` for equal-frequency bins.
    :return: (np.ndarray) Counts per bin, length ``n_bins`` for uniform and at
        most ``n_bins`` for quantile, which collapses bins on ties.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(n_bins, dtype=float)

    if method == "uniform":
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            counts = np.zeros(n_bins, dtype=float)
            counts[0] = x.size
            return counts
        return np.histogram(x, bins=n_bins, range=(lo, hi))[0].astype(float)

    if method == "quantile":
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            return np.array([float(x.size)])
        return np.histogram(x, bins=edges)[0].astype(float)

    raise ValueError(f"method must be 'uniform' or 'quantile'; got {method!r}")


def indicator_entropy(x, n_bins: int | None = None, method: str = "uniform") -> float:
    """Normalized Shannon entropy of a discretized indicator, in ``[0, 1]``.

    Normalization is by ``log(n_bins)``, the entropy of the uniform distribution
    over the same bins, so values are comparable across features and bin counts.

    :param x: (pd.Series or np.ndarray) Feature values.
    :param n_bins: (int) Bin count. Defaults to the Hacine-Gharbi univariate rule
        already used elsewhere in ``afml``.
    :param method: (str) ``"uniform"`` or ``"quantile"``.
    :return: (float) Normalized entropy. ``0.0`` for a constant series.

    .. note:: Use ``"uniform"``. Quantile binning assigns equal frequency to every
       bin by construction, so entropy is 1.0 for any feature with enough distinct
       values, and the screen rejects nothing. That is why a naively implemented
       entropy filter appears to work: it passes everything. Uniform binning is
       what exposes the failure mode worth catching, which is a feature that
       saturates or sits in one state most of the time. Clipped z-scores, ADX
       regime flags and session dummies all look healthy under quantile bins and
       fail under uniform bins.
    """
    values = np.asarray(getattr(x, "values", x), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if n_bins is None:
        n_bins = get_optimal_number_of_bins(values.size)

    counts = bin_counts(values, n_bins, method=method)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(n_bins))


def modal_occupancy(x, n_bins: int | None = None, method: str = "uniform") -> float:
    """Fraction of observations falling in the single most populated bin.

    Complements entropy. Entropy aggregates the whole distribution; modal
    occupancy isolates the specific pathology of a feature that spends most of
    its life in one state while occasionally excursioning, which can retain
    moderate entropy while being useless as a conditioning variable.

    :param x: (pd.Series or np.ndarray) Feature values.
    :param n_bins: (int) Bin count. Defaults to the Hacine-Gharbi rule.
    :param method: (str) ``"uniform"`` or ``"quantile"``.
    :return: (float) Modal share in ``[0, 1]``.
    """
    values = np.asarray(getattr(x, "values", x), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    if n_bins is None:
        n_bins = get_optimal_number_of_bins(values.size)
    counts = bin_counts(values, n_bins, method=method)
    total = counts.sum()
    return 1.0 if total == 0 else float(counts.max() / total)


def entropy_screen(
    features: pd.DataFrame,
    min_entropy: float = 0.5,
    max_modal_occupancy: float = 0.5,
    n_bins: int | None = None,
    method: str = "uniform",
) -> pd.DataFrame:
    """Entropy and modal-occupancy diagnostics for a feature matrix.

    :param features: (pd.DataFrame) One column per candidate feature.
    :param min_entropy: (float) Normalized-entropy floor for a pass.
    :param max_modal_occupancy: (float) Modal-share ceiling for a pass.
    :param n_bins: (int) Bin count. Defaults to the Hacine-Gharbi rule per column.
    :param method: (str) ``"uniform"`` or ``"quantile"``.
    :return: (pd.DataFrame) Indexed by feature name with columns ``entropy``,
        ``modal_occupancy``, ``n_bins``, ``n_unique``, ``n_obs`` and ``passes``,
        sorted by ascending entropy so the worst offenders appear first.

    .. note:: The thresholds are conventions, not calibrated cutoffs, and this
       screen deliberately produces no p-value. Entropy has a null distribution
       like anything else, but a feature that fails here fails for a structural
       reason that does not need a hypothesis test: it does not vary. Treat the
       output as triage before the mutual-information screen, not as evidence.
    """
    rows = []
    for name in features.columns:
        col = features[name]
        values = np.asarray(col.values, dtype=float)
        values = values[np.isfinite(values)]
        bins = (
            n_bins
            if n_bins is not None
            else (get_optimal_number_of_bins(values.size) if values.size else 1)
        )
        ent = indicator_entropy(values, n_bins=bins, method=method)
        modal = modal_occupancy(values, n_bins=bins, method=method)
        rows.append(
            {
                "feature": name,
                "entropy": ent,
                "modal_occupancy": modal,
                "n_bins": int(bins),
                "n_unique": int(np.unique(values).size),
                "n_obs": int(values.size),
                "passes": bool(ent >= min_entropy and modal <= max_modal_occupancy),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("feature").sort_values("entropy")
