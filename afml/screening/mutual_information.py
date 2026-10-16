"""
Mutual-information screening of features against forward returns.

Mutual information measures the reduction in uncertainty about the target
achieved by observing the feature. Unlike correlation it detects non-monotone and
non-linear relationships, which is why Masters and Lopez de Prado both reach for
it, and unlike a fitted importance measure it requires no model.

It also does not do what the compiled Masters reference claims. Section 6.3 of
that document states that entropy-based screening "is therefore immune to the
overfitting concerns that affect importance measures derived from fitted models."
It is not, on two counts.

Estimation bias. Plug-in mutual information on binned data carries a positive
bias of roughly ``(Bx - 1)(By - 1) / (2 N ln 2)`` under independence. Measured on
2,000 observations with the Hacine-Gharbi bin rule, normalized MI averages 0.0108
when feature and target are independent by construction. It is never zero, so a
fixed threshold is not a screen.

Selection. Screening ``K`` features is a multiple-comparisons problem. The best
of fifty useless features scores 0.0150 against a single-feature null of 0.0108.

Both are handled here by comparing against a permutation null and correcting
across the family. The null is generated jointly, one shared permutation applied
to every feature per replicate, so that
:func:`afml.screening.multiple_testing.westfall_young_maxt` can exploit the
empirical correlation among features instead of assuming a worst case.

Two null modes are available, and they are not equivalent.

``bar_permute``
    Bars are permuted and every feature is *recomputed* on the synthetic series.
    Correct, because each feature's serial dependence regenerates naturally.
    Requires ``feature_fn``.

``circular_shift`` and ``stationary_bootstrap``
    The target is resampled and features are held fixed. A fallback. Well
    calibrated in the reference study, but it cannot capture how a feature's own
    construction interacts with the price path. Results record
    ``recomputed_feature=False`` so the weaker mode is visible in every report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..codependence.information import get_optimal_number_of_bins
from .multiple_testing import (
    benjamini_hochberg,
    holm_bonferroni,
    westfall_young_maxt,
)
from .permutation import (
    SCHEMES,
    bar_permutation,
    effective_sample_size,
    permutation_pvalue,
)
from .report import DEFAULT_QUANTILES, ScreeningResult

__all__ = [
    "mi_screen",
    "miller_madow_bias",
    "normalized_mutual_info",
]


def miller_madow_bias(contingency: np.ndarray) -> float:
    """First-order bias of the plug-in mutual-information estimator, in nats.

    :param contingency: (np.ndarray) Two-dimensional joint count table.
    :return: (float) Estimated bias, ``(m_xy - m_x - m_y + 1) / (2N)`` where each
        ``m`` counts occupied cells. Positive for the sparse-joint tables that
        arise under independence, which is the case the screen cares about. It
        turns negative for near-deterministic tables, where the plug-in estimator
        understates rather than overstates; that sign is correct, not a guard to
        clamp away.

    .. note:: Reduces the floor; it does not remove it. The correction is
       asymptotic and assumes independent observations, which overlapping labels
       violate. The permutation null remains mandatory.
    """
    contingency = np.asarray(contingency, dtype=float)
    n = contingency.sum()
    if n <= 0:
        return 0.0
    occupied_joint = int(np.sum(contingency > 0))
    occupied_x = int(np.sum(contingency.sum(axis=1) > 0))
    occupied_y = int(np.sum(contingency.sum(axis=0) > 0))
    return float((occupied_joint - occupied_x - occupied_y + 1) / (2.0 * n))


def normalized_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int | None = None,
    correction: str | None = "miller_madow",
) -> float:
    """Mutual information normalized by the smaller marginal entropy.

    :param x: (np.ndarray) Feature values.
    :param y: (np.ndarray) Target values.
    :param n_bins: (int) Bin count. Defaults to the Hacine-Gharbi bivariate rule.
    :param correction: (str) ``"miller_madow"`` or ``None``.
    :return: (float) Normalized mutual information, clipped at zero below.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if n_bins is None:
        with np.errstate(invalid="ignore"):
            rho = np.corrcoef(x, y)[0, 1]
        rho = 0.0 if not np.isfinite(rho) else rho
        n_bins = get_optimal_number_of_bins(x.size, corr_coef=rho)
    n_bins = max(2, int(n_bins))

    contingency = np.histogram2d(x, y, bins=n_bins)[0]
    total = contingency.sum()
    if total == 0:
        return 0.0

    joint = contingency / total
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])))

    if correction == "miller_madow":
        mi = mi - miller_madow_bias(contingency)

    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    denom = min(hx, hy)
    if denom <= 0:
        return 0.0
    return float(max(0.0, mi / denom))


def mi_screen(
    features: pd.DataFrame,
    target: pd.Series,
    scheme: str = "circular_shift",
    n_perm: int = 500,
    seed: int = 0,
    bars: pd.DataFrame | None = None,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    n_bins: int | None = None,
    correction: str | None = "miller_madow",
    scheme_kwargs: dict[str, float] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Screen a feature matrix against a target with a joint permutation null.

    :param features: (pd.DataFrame) One column per candidate feature, aligned
        with ``target``.
    :param target: (pd.Series) Forward return or label.
    :param scheme: (str) ``"bar_permute"`` to recompute features on permuted bars,
        or a key of :data:`afml.screening.permutation.SCHEMES` to resample the
        target instead.
    :param n_perm: (int) Permutation replicates.
    :param seed: (int) Random seed.
    :param bars: (pd.DataFrame) OHLC bars. Required for ``"bar_permute"``.
    :param feature_fn: (callable) Maps bars to a feature frame with the same
        columns as ``features``. Required for ``"bar_permute"``.
    :param n_bins: (int) Bin count. Defaults to the Hacine-Gharbi rule per pair.
    :param correction: (str) Bias correction passed to
        :func:`normalized_mutual_info`.
    :param scheme_kwargs: (dict) Extra scheme arguments, e.g. ``mean_block``.
    :param quantiles: (sequence) Null quantile levels to report.
    :param alpha: (float) Level used for the ``passes`` column.
    :return: (pd.DataFrame) One row per feature, sorted by adjusted p-value, with
        raw and adjusted p-values under Westfall-Young, Holm and
        Benjamini-Hochberg, plus the null summary and effective sample size.

    .. note:: The permutation is drawn once per replicate and applied to every
       feature. That is what makes the Westfall-Young adjustment valid without a
       dependence assumption, and it is why this function screens a matrix rather
       than being called in a loop over columns.
    """
    if features.empty:
        return pd.DataFrame()

    aligned = features.join(target.rename("__target__"), how="inner").dropna()
    if aligned.shape[0] < 64:
        raise ValueError(
            f"only {aligned.shape[0]} aligned observations; refusing to calibrate a null"
        )
    y = aligned["__target__"].to_numpy(dtype=float)
    names = list(features.columns)
    x_mat = aligned[names].to_numpy(dtype=float)
    n_obs, n_features = x_mat.shape

    def stat(col: np.ndarray, tgt: np.ndarray) -> float:
        return normalized_mutual_info(col, tgt, n_bins=n_bins, correction=correction)

    observed = np.array([stat(x_mat[:, j], y) for j in range(n_features)])
    null = np.empty((n_perm, n_features), dtype=float)
    rng = np.random.default_rng(seed)
    recomputed = False

    if scheme == "bar_permute":
        if bars is None or feature_fn is None:
            raise ValueError("scheme='bar_permute' requires both `bars` and `feature_fn`")
        recomputed = True
        for b, permuted in enumerate(bar_permutation(bars, n_perm=n_perm, seed=seed)):
            frame = feature_fn(permuted).reindex(columns=names)
            joined = frame.join(target.rename("__target__"), how="inner").dropna()
            yb = joined["__target__"].to_numpy(dtype=float)
            xb = joined[names].to_numpy(dtype=float)
            null[b] = [stat(xb[:, j], yb) for j in range(n_features)]
    else:
        if scheme not in SCHEMES:
            raise KeyError(f"unknown scheme {scheme!r}; available: {sorted(SCHEMES)}")
        kwargs = dict(scheme_kwargs or {})
        idx = SCHEMES[scheme](n_obs, n_perm=n_perm, rng=rng, **kwargs)
        for b, row in enumerate(idx):
            yb = y[row]
            null[b] = [stat(x_mat[:, j], yb) for j in range(n_features)]

    raw_p = np.array([permutation_pvalue(observed[j], null[:, j]) for j in range(n_features)])
    p_wy = westfall_young_maxt(observed, null)
    p_holm = holm_bonferroni(raw_p)
    p_bh = benjamini_hochberg(raw_p)
    ess = effective_sample_size(y)

    rows = []
    for j, name in enumerate(names):
        result = ScreeningResult(
            name=name,
            statistic_name="normalized_mutual_information",
            statistic=float(observed[j]),
            null_mean=float(np.nanmean(null[:, j])),
            null_std=float(np.nanstd(null[:, j], ddof=1)),
            null_quantiles={float(q): float(np.nanquantile(null[:, j], q)) for q in quantiles},
            p_value=float(raw_p[j]),
            n_permutations=int(n_perm),
            null_scheme=scheme,
            effective_sample_size=float(ess),
            n_observations=int(n_obs),
            p_value_adjusted=float(p_wy[j]),
            recomputed_feature=recomputed,
        )
        row = result.to_dict()
        row["p_westfall_young"] = float(p_wy[j])
        row["p_holm"] = float(p_holm[j])
        row["p_benjamini_hochberg"] = float(p_bh[j])
        row["passes"] = bool(p_wy[j] < alpha)
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("name")
    frame.attrs["null_matrix"] = null
    return frame.sort_values(["p_westfall_young", "statistic"], ascending=[True, False])
