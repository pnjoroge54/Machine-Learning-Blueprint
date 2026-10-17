"""
Null-generation schemes for feature and strategy screening.

Every diagnostic in ``afml.screening`` compares an observed statistic against a
reference distribution. This module generates that reference distribution. It is
the foundation of the subpackage: an entropy screen, a mutual-information screen
or a profit-factor trend test built on a miscalibrated null is not a filter, it
is a rubber stamp.

Three schemes are provided, and they are not interchangeable.

``bar_permutation``
    Masters' bar permutation. Decomposes each bar into an inter-bar gap return
    and an intra-bar body, shuffles the two components, and reassembles a
    synthetic OHLC series. Features are then *recomputed* on the permuted bars.
    This is the correct null for screening indicators, because the indicator's
    own serial dependence regenerates naturally instead of being destroyed.

``circular_shift``
    Rotates the target series by a random offset. Preserves the autocorrelation
    structure of both series exactly, up to a single wrap seam, and destroys any
    cross-dependence between them. Valid under stationarity, and by a wide margin
    the best-calibrated fallback when the feature cannot be recomputed.

``stationary_bootstrap``
    Politis and Romano (1994) resampling with geometric block lengths. Resamples
    rather than merely reorders, so it is the appropriate scheme for trade-return
    resampling in Phase 3. As an independence null between two serially dependent
    series it is measurably oversized; see ``notes/size_calibration``.

A fourth scheme, ``naive_shuffle``, is included solely so that regression tests
can assert that it fails. On a smoothed feature against overlapping labels it
produces a 100% false-positive rate at a nominal 5% level, because shuffling
destroys the target's autocorrelation and therefore generates the null of a
different, much narrower estimation problem.

References
----------
Masters, T. (2020). *Permutation and Randomization Tests for Trading System
Development.*
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the
American Statistical Association*, 89(428), 1303-1313.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from .report import DEFAULT_QUANTILES, ScreeningResult

__all__ = [
    "SCHEMES",
    "bar_permutation",
    "build_null",
    "circular_shift_indices",
    "effective_sample_size",
    "naive_shuffle_indices",
    "permutation_pvalue",
    "stationary_bootstrap_indices",
    "suggested_block_length",
]

_OHLC = ("open", "high", "low", "close")


# =============================================================================
# Diagnostics
# =============================================================================
def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Autocorrelation-adjusted sample size.

    Computes ``N / (1 + 2 * sum_k rho_k)`` with the sum truncated at the first
    non-positive autocorrelation, the initial-positive-sequence rule of Geyer
    (1992). For an i.i.d. series this returns approximately ``N``; for a smoothed
    indicator or an overlapping forward return it can be an order of magnitude
    smaller, which is precisely why the naive null fails.

    :param x: (np.ndarray) Input series.
    :param max_lag: (int) Maximum lag considered. Defaults to ``N // 4``.
    :return: (float) Effective sample size, clipped to ``[1, N]``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float(max(n, 1))
    if max_lag is None:
        max_lag = max(1, n // 4)

    centered = x - x.mean()
    denom = np.dot(centered, centered)
    if denom == 0.0:
        return float(n)

    total = 0.0
    for lag in range(1, max_lag + 1):
        rho = np.dot(centered[:-lag], centered[lag:]) / denom
        if rho <= 0.0:
            break
        total += rho
    return float(np.clip(n / (1.0 + 2.0 * total), 1.0, n))


def permutation_pvalue(observed: float, null_values: np.ndarray, tail: str = "right") -> float:
    """One-sided permutation p-value with the add-one correction.

    The ``+1`` in numerator and denominator counts the observed data as one of
    the possible arrangements. It guarantees a strictly positive p-value and
    keeps the test exact, which matters because a reported p-value of zero is
    always an artifact of a finite permutation budget.

    :param observed: (float) Observed statistic.
    :param null_values: (np.ndarray) Statistic under the null.
    :param tail: (str) ``"right"`` for larger-is-more-extreme, ``"left"`` otherwise.
    :return: (float) p-value in ``(0, 1]``.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        raise ValueError("null distribution is empty after removing non-finite values")
    if tail == "right":
        n_extreme = int(np.sum(null_values >= observed))
    elif tail == "left":
        n_extreme = int(np.sum(null_values <= observed))
    else:
        raise ValueError("tail must be 'right' or 'left'")
    return (1.0 + n_extreme) / (1.0 + null_values.size)


def suggested_block_length(
    t1: pd.Series,
    index: pd.Index | None = None,
    multiple: float = 5.0,
    floor: int = 50,
) -> float:
    """Derive a resampling block length from the label span.

    Masters advises a block "somewhat larger than the typical trade holding
    period". In the Blueprint that quantity is not a guess: it is the mean number
    of bars a triple-barrier label spans, which is already implicit in the
    ``t1`` series produced by :func:`afml.labeling.triple_barrier.get_events`.

    The default multiple of five is not a rule of thumb. It is the smallest value
    that achieved acceptable test size in the calibration study shipped with this
    module; see ``notes/size_calibration``.

    :param t1: (pd.Series) Label end times indexed by label start time.
    :param index: (pd.Index) Bar index used to convert timestamps into bar counts.
        Required when ``t1`` holds timestamps rather than integer positions.
    :param multiple: (float) Multiplier applied to the mean label span.
    :param floor: (int) Lower bound on the returned block length.
    :return: (float) Mean block length in bars.
    """
    t1 = t1.dropna()
    if t1.empty:
        raise ValueError("t1 is empty; cannot derive a block length")

    if index is not None:
        positions = pd.Index(index)
        start = positions.get_indexer(t1.index)
        end = positions.get_indexer(pd.Index(t1.values))
        valid = (start >= 0) & (end >= 0)
        if not valid.any():
            raise ValueError("no t1 entries could be located in the supplied index")
        spans = (end[valid] - start[valid]).astype(float)
    else:
        spans = np.asarray(t1.values, dtype=float) - np.asarray(t1.index, dtype=float)

    mean_span = float(np.mean(spans[spans > 0])) if np.any(spans > 0) else 1.0
    return float(max(floor, multiple * mean_span))


# =============================================================================
# Index-based schemes (target resampling)
# =============================================================================
def naive_shuffle_indices(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform permutations of ``0..n-1``.

    Provided for regression testing only. This scheme destroys the target's
    serial dependence and is invalid whenever the target is autocorrelated,
    which for overlapping labels is always.

    :param n: (int) Series length.
    :param n_perm: (int) Number of replicates.
    :param rng: (np.random.Generator) Random source.
    :return: (np.ndarray) Integer array of shape ``(n_perm, n)``.
    """
    return np.array([rng.permutation(n) for _ in range(n_perm)], dtype=np.int64)


def circular_shift_indices(
    n: int, n_perm: int, rng: np.random.Generator, min_shift: int = 1
) -> np.ndarray:
    """Random circular rotations of ``0..n-1``.

    A rotation preserves every lag-``k`` autocorrelation of the rotated series
    exactly, apart from the single wrap seam, while severing any alignment with
    a second series. Under the null of independence between two stationary
    series this yields an approximately exact randomization test.

    :param n: (int) Series length.
    :param n_perm: (int) Number of replicates.
    :param rng: (np.random.Generator) Random source.
    :param min_shift: (int) Smallest admissible rotation. Rotations near zero and
        near ``n`` leave the series almost aligned with itself and must be excluded.
    :return: (np.ndarray) Integer array of shape ``(n_perm, n)``.
    """
    if n <= 2 * min_shift + 1:
        raise ValueError(f"series of length {n} is too short for min_shift={min_shift}")
    base = np.arange(n, dtype=np.int64)
    shifts = rng.integers(min_shift, n - min_shift, size=n_perm)
    return np.array([np.roll(base, int(s)) for s in shifts], dtype=np.int64)


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices.

    Block lengths are geometric with mean ``mean_block``, which keeps the
    resampled series stationary. Fixed-length block bootstrap does not, because
    the position of an observation within its block becomes informative.

    :param n: (int) Series length.
    :param mean_block: (float) Mean geometric block length, in observations.
    :param n_perm: (int) Number of replicates.
    :param rng: (np.random.Generator) Random source.
    :return: (np.ndarray) Integer array of shape ``(n_perm, n)``, wrapped circularly.
    """
    if mean_block < 1.0:
        raise ValueError("mean_block must be at least 1")
    p_restart = 1.0 / float(mean_block)

    positions = np.arange(n, dtype=np.int64)
    out = np.empty((n_perm, n), dtype=np.int64)
    for r in range(n_perm):
        restarts = rng.random(n) < p_restart
        restarts[0] = True
        starts = rng.integers(0, n, size=n)
        # Index of the restart that opened the block each observation belongs to.
        block_start = np.maximum.accumulate(np.where(restarts, positions, 0))
        out[r] = (starts[block_start] + (positions - block_start)) % n
    return out


SCHEMES: dict[str, Callable[..., np.ndarray]] = {
    "circular_shift": circular_shift_indices,
    "stationary_bootstrap": stationary_bootstrap_indices,
    "naive_shuffle": naive_shuffle_indices,
}


# =============================================================================
# Bar permutation
# =============================================================================
def _normalize_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map OHLC column names case-insensitively and validate presence."""
    lookup = {str(c).strip().lower(): c for c in bars.columns}
    missing = [c for c in _OHLC if c not in lookup]
    if missing:
        raise KeyError(
            f"bars is missing required column(s) {missing}; "
            f"available columns are {list(bars.columns)}"
        )
    return bars, lookup


def bar_permutation(
    bars: pd.DataFrame,
    n_perm: int,
    seed: int = 0,
    anchor_first: bool = True,
    couple_volume_to_body: bool = True,
) -> Iterator[pd.DataFrame]:
    """Generate permuted OHLC bar series after Masters (2020).

    Each bar is decomposed in log space into

    * an inter-bar gap, ``log(O_t) - log(C_{t-1})``, and
    * an intra-bar body, the triple ``(log(H_t) - log(O_t), log(L_t) - log(O_t),
      log(C_t) - log(O_t))``.

    The gap sequence and the body sequence are shuffled with independent
    permutations, then the series is reassembled by compounding forward from the
    first bar's close.

    The body is shuffled as an intact triple, never component-wise. Shuffling
    high, low and close independently would break the ``H >= max(O, C)`` and
    ``L <= min(O, C)`` invariants and produce bars no exchange could emit, which
    downstream range and true-range features would then read as signal.

    What survives the permutation: the marginal distribution of gap returns, the
    marginal distribution of bar bodies, the relationship between a bar's range
    and its volume, and OHLC internal consistency. What is destroyed: all
    inter-bar predictability, which is the only thing an indicator screen is
    entitled to detect.

    The first bar is held fixed so the synthetic series starts at the true price
    level. This costs one observation and keeps price-level-dependent features
    such as fractional differentiation on the same scale as the real data.

    :param bars: (pd.DataFrame) OHLC bars, optionally with a ``volume`` column.
        Column names are matched case-insensitively.
    :param n_perm: (int) Number of permuted series to yield.
    :param seed: (int) Seed for the random source.
    :param anchor_first: (bool) Hold the first bar fixed. Defaults to ``True``.
    :param couple_volume_to_body: (bool) Carry volume with the intra-bar body
        permutation rather than permuting it separately. Defaults to ``True``.
    :return: (Iterator[pd.DataFrame]) Permuted bar frames sharing the input index.
    """
    bars, lookup = _normalize_bars(bars)
    n = len(bars)
    if n < 3:
        raise ValueError("bar permutation requires at least 3 bars")

    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.log(bars[lookup["open"]].to_numpy(dtype=float))
        h = np.log(bars[lookup["high"]].to_numpy(dtype=float))
        lo = np.log(bars[lookup["low"]].to_numpy(dtype=float))
        c = np.log(bars[lookup["close"]].to_numpy(dtype=float))
    if not np.all(np.isfinite([o, h, lo, c])):
        raise ValueError("bars contain non-positive or non-finite prices")

    vol_col = lookup.get("volume")
    volume = bars[vol_col].to_numpy(dtype=float) if vol_col is not None else None

    # The first bar anchors the price level; permutable material starts after it.
    start = 1 if anchor_first else 0
    gaps = o[1:] - c[:-1]  # inter-bar, length n - 1
    body_h = h[1:] - o[1:]
    body_l = lo[1:] - o[1:]
    body_c = c[1:] - o[1:]
    body_vol = volume[1:] if volume is not None else None

    if not anchor_first:
        # Bar zero has no predecessor close; give it a zero gap and let it be permuted.
        gaps = np.r_[0.0, gaps]
        body_h = np.r_[h[0] - o[0], body_h]
        body_l = np.r_[lo[0] - o[0], body_l]
        body_c = np.r_[c[0] - o[0], body_c]
        if body_vol is not None:
            body_vol = np.r_[volume[0], body_vol]

    m = gaps.size
    anchor_log_close = c[0] if anchor_first else o[0]
    rng = np.random.default_rng(seed)
    index = bars.index

    for _ in range(n_perm):
        gap_perm = rng.permutation(m)
        body_perm = rng.permutation(m)

        g = gaps[gap_perm]
        bh, bl, bc = body_h[body_perm], body_l[body_perm], body_c[body_perm]

        # close_i = anchor + cumsum(gap_i + body_close_i); open_i = close_{i-1} + gap_i.
        cum = np.cumsum(g + bc)
        seg_c = anchor_log_close + cum
        seg_o = anchor_log_close + np.r_[0.0, cum[:-1]] + g
        seg_h = seg_o + bh
        seg_l = seg_o + bl

        if anchor_first:
            new_o = np.r_[o[0], seg_o]
            new_h = np.r_[h[0], seg_h]
            new_l = np.r_[lo[0], seg_l]
            new_c = np.r_[c[0], seg_c]
        else:
            new_o, new_h, new_l, new_c = seg_o, seg_h, seg_l, seg_c

        out = pd.DataFrame(
            {
                "open": np.exp(new_o),
                "high": np.exp(new_h),
                "low": np.exp(new_l),
                "close": np.exp(new_c),
            },
            index=index,
        )
        if body_vol is not None:
            seg_v = body_vol[body_perm]
            out["volume"] = np.r_[volume[0], seg_v] if anchor_first else seg_v
        yield out


# =============================================================================
# Generic null driver
# =============================================================================
def build_null(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    feature: np.ndarray,
    target: np.ndarray,
    scheme: str = "circular_shift",
    n_perm: int = 500,
    seed: int = 0,
    name: str = "feature",
    statistic_name: str = "statistic",
    tail: str = "right",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    scheme_kwargs: dict[str, float] | None = None,
) -> ScreeningResult:
    """Calibrate a bivariate statistic against a resampled null.

    The target is resampled and the feature is held fixed. This is the fallback
    path, used when the feature cannot be recomputed on permuted bars; the
    returned result records ``recomputed_feature=False`` so the weaker mode is
    visible in every report.

    :param statistic_fn: (callable) Takes ``(feature, target)`` and returns a scalar.
    :param feature: (np.ndarray) Feature values.
    :param target: (np.ndarray) Target values, aligned with ``feature``.
    :param scheme: (str) Key into :data:`SCHEMES`.
    :param n_perm: (int) Number of null replicates.
    :param seed: (int) Seed for the random source.
    :param name: (str) Feature name recorded on the result.
    :param statistic_name: (str) Statistic name recorded on the result.
    :param tail: (str) ``"right"`` or ``"left"``.
    :param quantiles: (sequence) Null quantile levels to report.
    :param scheme_kwargs: (dict) Extra arguments for the scheme, e.g. ``mean_block``.
    :return: (ScreeningResult) Statistic bound to its null.
    """
    if scheme not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme!r}; available: {sorted(SCHEMES)}")

    feature = np.asarray(feature, dtype=float)
    target = np.asarray(target, dtype=float)
    if feature.shape != target.shape:
        raise ValueError(f"shape mismatch: feature {feature.shape} vs target {target.shape}")

    mask = np.isfinite(feature) & np.isfinite(target)
    feature, target = feature[mask], target[mask]
    n = feature.size
    if n < 32:
        raise ValueError(f"only {n} usable observations; refusing to calibrate a null")

    rng = np.random.default_rng(seed)
    kwargs = dict(scheme_kwargs or {})
    idx = SCHEMES[scheme](n, n_perm=n_perm, rng=rng, **kwargs)

    observed = float(statistic_fn(feature, target))
    null = np.array([statistic_fn(feature, target[row]) for row in idx], dtype=float)

    return ScreeningResult(
        name=name,
        statistic_name=statistic_name,
        statistic=observed,
        null_mean=float(np.nanmean(null)),
        null_std=float(np.nanstd(null, ddof=1)),
        null_quantiles={float(q): float(np.nanquantile(null, q)) for q in quantiles},
        p_value=permutation_pvalue(observed, null, tail=tail),
        n_permutations=int(n_perm),
        null_scheme=scheme,
        effective_sample_size=effective_sample_size(target),
        n_observations=int(n),
        recomputed_feature=False,
        metadata={k: float(v) for k, v in kwargs.items()},
    )
