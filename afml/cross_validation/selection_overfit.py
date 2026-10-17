"""
selection_overfit.py
────────────────────
Diagnostics for over-fitting *in model selection*, and the multiple-testing
correction that follows from it.

Scope relative to the rest of `afml.cross_validation`
-----------------------------------------------------
`nested_cv.py` removes the optimistic *bias* from the reported score by
performing hyperparameter search independently inside every outer fold. That
addresses Section 5 of Cawley and Talbot (2010). It does not address Section 4:
the search itself still over-fits, and nested CV only reports how much.

This module supplies the missing pieces:

1. `OffsetPurgedKFold`      — purged, embargoed CV with a boundary offset, so
                              the same sample admits R distinct legitimate
                              partitions. This is what makes the variance of
                              the selection criterion measurable at all.
2. `criterion_variance_audit` — estimates the noise floor of the model
                              selection criterion, the spread it must resolve
                              across candidates, and the expected selection
                              regret incurred by picking the argmax.
3. `selection_optimism_curve` — inner best-so-far score against honest
                              held-out score as the trial budget grows; the
                              financial-data analogue of Cawley and Talbot
                              Figure 2.
4. `effective_trial_count`  — converts N correlated HPO trials into the
                              effective number of independent trials that the
                              Deflated Sharpe Ratio requires.
5. `expected_max_sharpe`    — AFML (2018) Section 8.1 threshold, exposed so the
                              deflation shift from (4) can be quantified.
6. `one_standard_error_selection` — a 1-SE rule that uses the standard *error*
                              of the fold scores and an explicit complexity
                              ordering.
7. `HPOEarlyStopping`       — an Optuna callback that halts the search once the
                              incumbent stops improving by more than the noise
                              floor measured in (2).

References
----------
Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection
    and subsequent selection bias in performance evaluation. JMLR 11:2079-2107.
Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Ch. 7, 8, 12.
Li, J. and Ji, L. (2005). Adjusting multiple testing in multilocus analyses
    using the eigenvalues of a correlation matrix. Heredity 95:221-227.
Kish, L. (1965). Survey Sampling. Wiley.
Breiman et al. (1984); Masters (1995). One-standard-error rule.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator

__all__ = [
    "CriterionVarianceReport",
    "EffectiveTrialCount",
    "HPOEarlyStopping",
    "OffsetPurgedKFold",
    "aggregate_fold_params",
    "criterion_variance_audit",
    "effective_trial_count",
    "expected_max_sharpe",
    "neg_brier_scorer",
    "neg_log_loss_scorer",
    "one_standard_error_selection",
    "selection_optimism_curve",
    "top_m_ensemble",
]


# ═══════════════════════════════════════════════════════════════════════════
# Scorers
# ═══════════════════════════════════════════════════════════════════════════


def neg_log_loss_scorer(
    y_true: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Negative weighted log loss. Higher is better."""
    return -log_loss(y_true, proba, labels=[0, 1], sample_weight=sample_weight)


def neg_brier_scorer(
    y_true: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Negative weighted Brier score. Higher is better."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if sample_weight is None:
        return -float(np.mean((proba - y_true) ** 2))
    w = np.asarray(sample_weight, dtype=float)
    return -float(np.sum(w * (proba - y_true) ** 2) / np.sum(w))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Offset purged K-fold
# ═══════════════════════════════════════════════════════════════════════════


class OffsetPurgedKFold(BaseCrossValidator):
    """
    Purged, embargoed K-fold whose test blocks start at an arbitrary offset.

    `PurgedKFold` partitions the sample deterministically: the same data always
    yields the same folds, so the cross-validation score is a single number
    with no measurable dispersion. That makes the variance of the model
    selection criterion invisible, which is precisely the quantity Cawley and
    Talbot (2010, Section 4) identify as the driver of over-fitting in model
    selection.

    Shifting the fold boundaries by `offset` samples produces a different but
    equally legitimate purged partition of the same time series. Evaluating one
    fixed hyperparameter vector across R offsets gives R draws of the criterion,
    from which its standard deviation can be estimated.

    The offset is a boundary shift, not a shuffle: test blocks remain
    contiguous and time-ordered, and purging plus embargo are applied exactly
    as in AFML Snippet 7.3. Samples before the first boundary and after the
    last are dropped from the test coverage of that partition rather than
    wrapped around, which would break temporal order.

    Parameters
    ----------
    n_splits : int
        Number of contiguous test blocks.
    t1 : pd.Series
        Event end times, indexed by event start time. Must share `X`'s index.
    pct_embargo : float
        Embargo length as a fraction of the sample count.
    offset : int
        Number of samples by which the first test block boundary is advanced.
        `offset=0` reproduces the standard contiguous partition.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
        offset: int = 0,
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series of event end times")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.offset = offset

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _test_blocks(self, n_samples: int) -> list[tuple[int, int]]:
        """Contiguous [start, stop) test blocks, shifted by `self.offset`."""
        block = n_samples // self.n_splits
        if block == 0:
            raise ValueError("n_splits exceeds the number of samples")
        off = self.offset % block
        starts = np.arange(off, n_samples, block)
        blocks = []
        for s in starts:
            e = min(s + block, n_samples)
            if e - s >= max(2, block // 2):  # discard the ragged tail block
                blocks.append((int(s), int(e)))
        if off > 0:
            # The leading remainder is a legitimate test block when it is
            # large enough; otherwise it stays in the training pool.
            if off >= max(2, block // 2):
                blocks.insert(0, (0, int(off)))
        return blocks[: self.n_splits]

    def split(self, X, y=None, groups=None):
        if not X.index.equals(self.t1.index):
            raise ValueError("X.index and t1.index must be identical")

        n = X.shape[0]
        indices = np.arange(n)
        embargo = int(n * self.pct_embargo)
        t1_vals = self.t1.values
        t1_idx = self.t1.index.values

        for start, stop in self._test_blocks(n):
            test_idx = indices[start:stop]
            test_t0 = t1_idx[start]
            test_t1 = t1_vals[test_idx].max()

            # Left training block: events that closed strictly before the test
            # window opened. AFML Snippet 7.1 treats overlap as inclusive, so a
            # label closing exactly on the first test timestamp is a leak.
            left = indices[t1_vals < test_t0]
            left = left[left < start]

            # Right training block: events starting strictly after the last
            # test label closed, plus the embargo.
            right_start = int(np.searchsorted(t1_idx, test_t1, side="right"))
            right_start = min(right_start + embargo, n)
            right = indices[right_start:]

            train_idx = np.concatenate([left, right]).astype(int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            yield train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# 2. Criterion variance audit
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CriterionVarianceReport:
    """
    Result of `criterion_variance_audit`.

    Attributes
    ----------
    noise_floor : float
        Mean across candidates of the standard deviation of the criterion at a
        fixed hyperparameter vector, taken over the R partitions. This is the
        dispersion the search cannot see.
    candidate_spread : float
        Standard deviation across candidates of the criterion averaged over the
        R partitions. This is the signal the search is meant to resolve.
    resolution_ratio : float
        `candidate_spread / noise_floor`. Below 1.0 the search cannot separate
        candidates from sampling noise, and the argmax is largely arbitrary.
    selection_regret : float
        Expected shortfall, in criterion units, between the score of the
        candidate the search would pick on one partition and the score of the
        candidate that is best on average. This is over-fitting in model
        selection, measured directly.
    argmax_stability : float
        Fraction of the R partitions on which the search selects the candidate
        that is best on average.
    scores : pd.DataFrame
        (R x P) matrix of criterion values, partitions on rows.
    per_candidate : pd.DataFrame
        Per-candidate mean, standard deviation, and parameter dictionary.
    """

    noise_floor: float
    candidate_spread: float
    resolution_ratio: float
    selection_regret: float
    argmax_stability: float
    scores: pd.DataFrame
    per_candidate: pd.DataFrame
    n_resamples: int = 0
    n_candidates: int = 0

    def summary(self) -> str:
        return (
            f"criterion noise floor   : {self.noise_floor:.5f}\n"
            f"candidate spread        : {self.candidate_spread:.5f}\n"
            f"resolution ratio        : {self.resolution_ratio:.3f}\n"
            f"selection regret        : {self.selection_regret:.5f}\n"
            f"argmax stability        : {self.argmax_stability:.3f}\n"
            f"partitions x candidates : {self.n_resamples} x {self.n_candidates}"
        )


def _fit_score_fold(estimator, params, X, y, sw, tr, te, scorer) -> float:
    clf = clone(estimator).set_params(**params)
    try:
        clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=None if sw is None else sw.iloc[tr].values)
    except TypeError:
        clf.fit(X.iloc[tr], y.iloc[tr])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    return scorer(y.iloc[te].values, proba, None if sw is None else sw.iloc[te].values)


def criterion_variance_audit(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    param_candidates: Sequence[dict[str, Any]],
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    n_resamples: int = 8,
    scorer: Callable = neg_log_loss_scorer,
    return_fold_scores: bool = False,
) -> CriterionVarianceReport | tuple[CriterionVarianceReport, np.ndarray]:
    """
    Measure the variance of the model selection criterion and the regret that
    variance imposes on the argmax.

    For each of `n_resamples` fold-boundary offsets, every candidate is scored
    by purged, embargoed cross-validation. The dispersion of a candidate's
    score across offsets is the noise floor; the dispersion of candidate means
    is the signal. Their ratio decides whether hyperparameter search on this
    sample is estimation or noise mining.

    Parameters
    ----------
    estimator : sklearn estimator
        Cloned before every fit. Must expose `predict_proba`.
    X, y : pd.DataFrame, pd.Series
        Design matrix and binary labels sharing a DatetimeIndex with `t1`.
    t1 : pd.Series
        Event end times for purging.
    param_candidates : sequence of dict
        Hyperparameter vectors to compare. Two is the minimum that admits a
        spread; ten to thirty is a useful audit.
    sample_weight : pd.Series, optional
        Uniqueness or return-attribution weights, aligned to `X.index`.
    n_resamples : int
        Number of fold-boundary offsets. The offsets are spread evenly across
        one block width.
    return_fold_scores : bool
        If True, also return the (R, P, K) array of raw fold scores, which
        `effective_trial_count` consumes.

    Returns
    -------
    CriterionVarianceReport, optionally with the raw fold-score array.
    """
    if len(param_candidates) < 2:
        raise ValueError("param_candidates needs at least two entries to have a spread")
    if sample_weight is not None and not sample_weight.index.equals(X.index):
        raise ValueError("sample_weight must be aligned to X.index")

    block = X.shape[0] // n_splits
    offsets = np.linspace(0, block, n_resamples, endpoint=False).astype(int)

    n_r, n_p = len(offsets), len(param_candidates)
    means = np.full((n_r, n_p), np.nan)
    fold_store: list[list[list[float]]] = []

    for r, off in enumerate(offsets):
        cv = OffsetPurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo, offset=int(off))
        splits = list(cv.split(X, y))
        per_param_folds: list[list[float]] = []
        for p, params in enumerate(param_candidates):
            fold_scores = [
                _fit_score_fold(estimator, params, X, y, sample_weight, tr, te, scorer)
                for tr, te in splits
            ]
            per_param_folds.append(fold_scores)
            means[r, p] = float(np.mean(fold_scores))
        fold_store.append(per_param_folds)

    scores = pd.DataFrame(
        means, index=[f"offset_{o}" for o in offsets], columns=[f"cand_{i}" for i in range(n_p)]
    )

    per_cand_mean = scores.mean(axis=0)
    per_cand_std = scores.std(axis=0, ddof=1)

    noise_floor = float(per_cand_std.mean())
    candidate_spread = float(per_cand_mean.std(ddof=1))
    resolution = candidate_spread / noise_floor if noise_floor > 0 else np.inf

    # Selection regret: the search picks argmax on one partition; truth is
    # approximated by the mean across partitions.
    truth = per_cand_mean.values
    best_true = int(np.argmax(truth))
    picked = np.argmax(means, axis=1)
    regret = float(np.mean(truth[best_true] - truth[picked]))
    stability = float(np.mean(picked == best_true))

    per_candidate = pd.DataFrame(
        {
            "mean_score": per_cand_mean.values,
            "std_score": per_cand_std.values,
            "params": [str(p) for p in param_candidates],
        },
        index=scores.columns,
    )

    report = CriterionVarianceReport(
        noise_floor=noise_floor,
        candidate_spread=candidate_spread,
        resolution_ratio=float(resolution),
        selection_regret=regret,
        argmax_stability=stability,
        scores=scores,
        per_candidate=per_candidate,
        n_resamples=n_r,
        n_candidates=n_p,
    )

    if return_fold_scores:
        max_k = max(len(f) for pp in fold_store for f in pp)
        arr = np.full((n_r, n_p, max_k), np.nan)
        for r, pp in enumerate(fold_store):
            for p, fs in enumerate(pp):
                arr[r, p, : len(fs)] = fs
        return report, arr
    return report


# ═══════════════════════════════════════════════════════════════════════════
# 3. Selection optimism curve
# ═══════════════════════════════════════════════════════════════════════════


def selection_optimism_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    param_sampler: Callable[[np.random.Generator], dict[str, Any]],
    n_trials: int = 60,
    sample_weight: pd.Series | None = None,
    design_fraction: float = 0.7,
    n_splits: int = 4,
    pct_embargo: float = 0.01,
    n_offsets: int = 1,
    scorer: Callable = neg_log_loss_scorer,
    random_state: int = 0,
    return_fold_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Trace the inner selection criterion and an honest held-out score as the
    hyperparameter search budget grows.

    The sample is cut once, in time order, into a design set and a held-out
    set. The boundary is purged: design events whose label closes after the
    held-out set opens are dropped. Each trial draws a hyperparameter vector,
    scores it by purged CV inside the design set, refits on the whole design
    set, and scores the refit on the held-out set. The incumbent is the trial
    with the best inner score so far.

    The gap between `best_inner` and `incumbent_outer` is the selection
    optimism. Cawley and Talbot (2010, Figure 2) show it widening once the
    search exhausts the genuine signal in the criterion; the same curve on
    price data is what justifies stopping the search early.

    Parameters
    ----------
    n_offsets : int
        Number of fold-boundary offsets over which the inner score is averaged.
        Raising it lowers the variance of the selection criterion, which is the
        first remedy Cawley and Talbot (2010, Section 4.2) recommend. It also
        lengthens each trial's score vector, which `effective_trial_count`
        needs in order to avoid a rank-limited estimate.
    return_fold_scores : bool
        If True, also return an (n_trials, n_offsets * n_splits) array of raw
        fold scores for `effective_trial_count`.

    Returns
    -------
    pd.DataFrame with one row per trial and columns `trial`, `inner_score`,
    `outer_score`, `best_inner`, `incumbent_outer`, `optimism`, `params`,
    optionally with the fold-score array.
    """
    if not 0.3 <= design_fraction <= 0.9:
        raise ValueError("design_fraction should lie in [0.3, 0.9]")

    rng = np.random.default_rng(random_state)
    cut = int(len(X) * design_fraction)
    boundary = X.index[cut]

    # Purge the design set at the boundary: drop events still open at the cut.
    design_mask = np.zeros(len(X), dtype=bool)
    design_mask[:cut] = True
    design_mask &= (t1 <= boundary).values

    X_d, y_d, t1_d = X[design_mask], y[design_mask], t1[design_mask]
    X_h, y_h = X.iloc[cut:], y.iloc[cut:]
    sw_d = None if sample_weight is None else sample_weight[design_mask]
    sw_h = None if sample_weight is None else sample_weight.iloc[cut:]

    if len(X_d) < n_splits * 10:
        raise ValueError("design set too small after purging; lower design_fraction or n_splits")

    block = len(X_d) // n_splits
    offsets = np.linspace(0, block, max(1, n_offsets), endpoint=False).astype(int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for off in offsets:
        cv = OffsetPurgedKFold(n_splits=n_splits, t1=t1_d, pct_embargo=pct_embargo, offset=int(off))
        splits.extend(cv.split(X_d, y_d))

    rows = []
    fold_rows: list[list[float]] = []
    best_inner, incumbent_outer, incumbent_params = -np.inf, np.nan, None

    for i in range(n_trials):
        params = param_sampler(rng)

        fold_scores = [
            _fit_score_fold(estimator, params, X_d, y_d, sw_d, tr, te, scorer) for tr, te in splits
        ]
        fold_rows.append(fold_scores)
        inner = float(np.mean(fold_scores))

        clf = clone(estimator).set_params(**params)
        try:
            clf.fit(X_d, y_d, sample_weight=None if sw_d is None else sw_d.values)
        except TypeError:
            clf.fit(X_d, y_d)
        outer = scorer(
            y_h.values, clf.predict_proba(X_h)[:, 1], None if sw_h is None else sw_h.values
        )

        if inner > best_inner:
            best_inner, incumbent_outer, incumbent_params = inner, outer, params

        rows.append(
            {
                "trial": i,
                "inner_score": inner,
                "outer_score": outer,
                "best_inner": best_inner,
                "incumbent_outer": incumbent_outer,
                "optimism": best_inner - incumbent_outer,
                "params": str(incumbent_params),
            }
        )

    curve = pd.DataFrame(rows)
    if return_fold_scores:
        width = min(len(f) for f in fold_rows)
        return curve, np.array([f[:width] for f in fold_rows], dtype=float)
    return curve


# ═══════════════════════════════════════════════════════════════════════════
# 4. Effective trial count
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EffectiveTrialCount:
    """Result of `effective_trial_count`."""

    n_trials: int
    n_eff_eigen: float
    n_eff_kish: float
    mean_correlation: float
    eigenvalues: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    rank_limited: bool = False
    score_vector_length: int = 0

    def summary(self) -> str:
        flag = "  [rank-limited: see docstring]" if self.rank_limited else ""
        return (
            f"trials run              : {self.n_trials}\n"
            f"score vector length     : {self.score_vector_length}\n"
            f"mean pairwise corr      : {self.mean_correlation:.3f}\n"
            f"effective trials (eigen): {self.n_eff_eigen:.2f}{flag}\n"
            f"effective trials (Kish) : {self.n_eff_kish:.2f}"
        )


def effective_trial_count(score_matrix: np.ndarray) -> EffectiveTrialCount:
    """
    Convert N correlated hyperparameter trials into an effective number of
    independent trials for use in the Deflated Sharpe Ratio.

    The Deflated Sharpe Ratio deflates the observed Sharpe by the expected
    maximum of N independent trials. A Bayesian sampler does not produce
    independent trials: it concentrates draws in a region of hyperparameter
    space, so successive trials are scored on nearly the same errors and their
    score vectors are strongly correlated. Feeding the raw trial count into the
    deflation over-corrects; feeding one under-corrects.

    Two standard estimators are returned. The eigenvalue estimator of Li and Ji
    (2005) sums the integer and fractional parts of the eigenvalues of the
    trial correlation matrix. The Kish (1965) design-effect estimator assumes
    equicorrelation, `N / (1 + (N-1) * rho_bar)`, and is the conservative
    cross-check.

    Parameters
    ----------
    score_matrix : np.ndarray, shape (n_trials, n_scores)
        Per-trial score vectors. Each row is one trial; each column is one
        evaluation unit, normally one fold of one partition. Rows are
        correlated exactly when the trials make the same errors.

    Returns
    -------
    EffectiveTrialCount

    Notes
    -----
    A correlation matrix built from N vectors of length L has rank at most
    L - 1, so the eigenvalue estimator cannot report more than roughly L
    effective trials no matter how many trials were run. When `n_trials`
    exceeds the score vector length the result is flagged `rank_limited` and
    should be read as a floor, not an estimate. Lengthen the score vector by
    scoring each trial across several fold-boundary offsets, that is, by
    flattening the (R, K) grid produced by `criterion_variance_audit`.
    """
    S = np.asarray(score_matrix, dtype=float)
    if S.ndim != 2:
        raise ValueError("score_matrix must be 2-D (n_trials, n_scores)")
    n, length = S.shape
    if n < 2:
        raise ValueError("need at least two trials")
    if length < 2:
        raise ValueError("each trial needs at least two scores")

    # Drop trials with no variation; they carry no correlation information.
    keep = S.std(axis=1) > 1e-12
    if keep.sum() < 2:
        return EffectiveTrialCount(
            n_trials=n,
            n_eff_eigen=1.0,
            n_eff_kish=1.0,
            mean_correlation=1.0,
            eigenvalues=np.array([1.0]),
            rank_limited=True,
            score_vector_length=length,
        )
    Sk = S[keep]
    corr = np.corrcoef(Sk)
    corr = np.nan_to_num(corr, nan=0.0)
    m = corr.shape[0]

    off = corr[~np.eye(m, dtype=bool)]
    rho_bar = float(np.mean(off))

    eig = np.linalg.eigvalsh(corr)[::-1]
    eig = np.clip(eig, 0.0, None)
    # Li and Ji (2005): integer part plus fractional part of each eigenvalue.
    n_eff_eigen = float(np.sum((eig >= 1.0).astype(float) + (eig - np.floor(eig))))
    n_eff_eigen = float(np.clip(n_eff_eigen, 1.0, m))

    denom = 1.0 + (m - 1) * rho_bar
    n_eff_kish = float(np.clip(m / denom, 1.0, m)) if denom > 0 else float(m)

    return EffectiveTrialCount(
        n_trials=n,
        n_eff_eigen=n_eff_eigen,
        n_eff_kish=n_eff_kish,
        mean_correlation=rho_bar,
        eigenvalues=eig,
        rank_limited=bool(m > length),
        score_vector_length=length,
    )


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """
    Expected maximum Sharpe ratio across `n_trials` independent trials whose
    Sharpe estimates have variance `sr_variance`.

    AFML (2018) Section 8.1. This is the threshold the Deflated Sharpe Ratio
    tests against, exposed here so the effect of replacing the raw trial count
    with the effective count can be read directly in Sharpe units.
    """
    n = float(n_trials)
    if n <= 1.0:
        # With a single trial there is no maximum to take: under the null the
        # expected Sharpe is zero. The asymptotic expression below diverges here.
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    term = (1 - gamma) * norm.ppf(1 - 1.0 / n) + gamma * norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * term)


# ═══════════════════════════════════════════════════════════════════════════
# 5. One-standard-error selection
# ═══════════════════════════════════════════════════════════════════════════


def one_standard_error_selection(
    results: Sequence[dict[str, Any]],
    complexity: Callable[[dict[str, Any]], float],
    score_key: str = "fold_scores",
    param_key: str = "params",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Select the least complex candidate whose mean score is within one standard
    *error* of the best mean score.

    Two details separate this from a naive implementation.

    The tolerance is the standard error of the fold mean, `std / sqrt(k)`, not
    the standard deviation of the fold scores. Using the standard deviation
    inflates the tolerance by a factor of `sqrt(k)` and admits candidates that
    are genuinely worse, which turns a variance-reduction device into an
    unconditional preference for the simplest model in the grid.

    Complexity is supplied explicitly as a callable on the parameter dict.
    Falling back on the iteration order of a parameter grid does not work:
    `sklearn.model_selection.ParameterGrid` iterates in sorted-key order with
    the last key varying fastest, which has no relationship to model capacity.

    Parameters
    ----------
    results : sequence of dict
        Each entry holds a parameter dict under `param_key` and the per-fold
        scores (higher is better) under `score_key`.
    complexity : callable
        Maps a parameter dict to a scalar. Lower means simpler.

    Returns
    -------
    (selected_entry, diagnostics)
        `diagnostics` reports the best mean, the standard error, the tolerance
        threshold, and the number of candidates inside it.
    """
    if not results:
        raise ValueError("results is empty")

    table = []
    for r in results:
        fs = np.asarray(r[score_key], dtype=float)
        if fs.size < 2:
            raise ValueError("each candidate needs at least two fold scores")
        table.append(
            {
                "entry": r,
                "mean": float(fs.mean()),
                "se": float(fs.std(ddof=1) / np.sqrt(fs.size)),
                "complexity": float(complexity(r[param_key])),
            }
        )

    best = max(table, key=lambda d: d["mean"])
    threshold = best["mean"] - best["se"]
    within = [d for d in table if d["mean"] >= threshold]
    chosen = min(within, key=lambda d: d["complexity"])

    diagnostics = {
        "best_mean": best["mean"],
        "best_se": best["se"],
        "threshold": threshold,
        "n_within_1se": len(within),
        "n_candidates": len(table),
        "selected_mean": chosen["mean"],
        "selected_complexity": chosen["complexity"],
        "complexity_reduction": best["complexity"] - chosen["complexity"],
    }
    return chosen["entry"], diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# 6. Early stopping for the search itself
# ═══════════════════════════════════════════════════════════════════════════


class HPOEarlyStopping:
    """
    Optuna callback that stops a study once the incumbent has stopped improving
    by more than the noise floor of the selection criterion.

    Cawley and Talbot (2010, Section 4.4) note that the remedies for over-fitting
    in model selection are the remedies for over-fitting in training:
    regularization, early stopping, and averaging. Early stopping needs a
    threshold, and an arbitrary epsilon is the wrong one. The threshold that
    means something is the dispersion of the criterion itself: an improvement
    smaller than one standard deviation of the criterion at fixed
    hyperparameters is not evidence of a better model.

    Supply `noise_floor` from `criterion_variance_audit(...).noise_floor`.

    Parameters
    ----------
    noise_floor : float
        Standard deviation of the selection criterion at fixed hyperparameters.
    patience : int
        Number of consecutive trials without a qualifying improvement before
        the study is stopped.
    min_trials : int
        Trials that always run before stopping is considered.
    tolerance_multiple : float
        Improvement threshold expressed in noise floors. The default of 1.0
        requires an improvement of one standard deviation.

    Notes
    -----
    Attach with `study.optimize(objective, callbacks=[HPOEarlyStopping(...)])`.
    The callback calls `study.stop()`, which lets in-flight trials finish.
    """

    def __init__(
        self,
        noise_floor: float,
        patience: int = 15,
        min_trials: int = 20,
        tolerance_multiple: float = 1.0,
    ):
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive")
        if patience < 1 or min_trials < 1:
            raise ValueError("patience and min_trials must be positive")
        self.noise_floor = float(noise_floor)
        self.patience = int(patience)
        self.min_trials = int(min_trials)
        self.tolerance = float(tolerance_multiple) * float(noise_floor)
        self.best_value = -np.inf
        self.stale = 0
        self.stopped_at: int | None = None

    def __call__(self, study, trial) -> None:
        value = getattr(trial, "value", None)
        if value is None:
            return
        n_done = getattr(trial, "number", 0) + 1

        if value > self.best_value + self.tolerance:
            self.best_value = float(value)
            self.stale = 0
        else:
            self.stale += 1
            self.best_value = max(self.best_value, float(value))

        if n_done >= self.min_trials and self.stale >= self.patience:
            self.stopped_at = n_done
            warnings.warn(
                f"HPOEarlyStopping: no improvement above {self.tolerance:.5f} "
                f"in {self.stale} trials; stopping study at trial {n_done}.",
                stacklevel=2,
            )
            study.stop()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Deployment-time aggregation of hyperparameters
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_fold_params(
    fold_params: Sequence[dict[str, Any]],
    numeric_agg: str = "median",
) -> dict[str, Any]:
    """
    Collapse the per-outer-fold winning hyperparameters into one vector for the
    model that is actually deployed.

    Evaluation versus deployment
    ----------------------------
    Cawley and Talbot (2010, Section 5.2) show that tuning on a few folds,
    taking the median, and then scoring every fold with those frozen values
    produces an optimistically biased estimate, and one that favors different
    algorithms by different amounts. That finding is about *reported numbers*.

    Using the same aggregate to build the artifact you ship is a different act.
    Once the nested score has been reported honestly, no number is contaminated
    by choosing the deployed hyperparameters however you like. The aggregate is
    a variance-reduction device at that point, which is exactly what makes it a
    reasonable deployment choice and an unacceptable evaluation protocol.

    The rule this function encodes: aggregate after the score is reported, never
    before, and never re-score with the aggregate.

    Type preservation
    -----------------
    `statistics.median` of an even-length integer list returns the mean of the
    two central values. `median([3, 4])` is `3.5`, and `max_depth=3.5` raises
    inside scikit-learn. Booleans are a second trap: `isinstance(True, int)` is
    True in Python, so a naive numeric branch silently averages flags. Integer
    keys are therefore rounded back to `int`, booleans are resolved by majority,
    and anything non-numeric falls through to the modal value.

    Parameters
    ----------
    fold_params : sequence of dict
        One winning parameter dict per outer fold. All must share their keys.
    numeric_agg : {'median', 'mean'}
        Aggregator for numeric values.

    Returns
    -------
    dict with the original type of each value preserved.
    """
    if not fold_params:
        raise ValueError("fold_params is empty")
    keys = set(fold_params[0])
    for p in fold_params[1:]:
        if set(p) != keys:
            raise ValueError("all fold parameter dicts must share the same keys")
    if numeric_agg not in {"median", "mean"}:
        raise ValueError("numeric_agg must be 'median' or 'mean'")

    out: dict[str, Any] = {}
    for k in fold_params[0]:
        vals = [p[k] for p in fold_params]

        if all(isinstance(v, (bool, np.bool_)) for v in vals):
            out[k] = bool(sum(bool(v) for v in vals) * 2 > len(vals))
            continue

        if any(v is None for v in vals) or not all(
            isinstance(v, (int, float, np.integer, np.floating)) for v in vals
        ):
            # Categorical, string, or None-bearing: take the modal value and
            # break ties toward the first fold, which keeps the result stable.
            counts: dict[Any, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            out[k] = max(vals, key=lambda v: (counts[v], -vals.index(v)))
            continue

        agg = float(np.median(vals)) if numeric_agg == "median" else float(np.mean(vals))
        if all(isinstance(v, (int, np.integer)) for v in vals):
            out[k] = int(round(agg))
        else:
            out[k] = agg
    return out


def top_m_ensemble(
    estimator,
    scored_params: Sequence[tuple[float, dict[str, Any]]],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None = None,
    m: int = 5,
):
    """
    Fit the M best-scoring hyperparameter vectors and average their predicted
    probabilities.

    Averaging is the third remedy Cawley and Talbot (2010, Section 4.4) list
    alongside regularization and early stopping. Its logic here is direct: when
    the resolution ratio from `criterion_variance_audit` is below one, the
    ranking among the top candidates is mostly noise, so committing to the
    single argmax spends real capital on a coin flip. Averaging over the leaders
    keeps the region the search identified and discards the ordering it cannot
    justify.

    The cost is inference time and a model that no longer exports as one tree
    ensemble. For ONNX deployment, average the probabilities in the graph or
    accept M forward passes.

    Parameters
    ----------
    scored_params : sequence of (score, params)
        Higher score is better. Sorted internally.
    m : int
        Number of leaders to fit. Values above the count inside the 1-SE band
        add variance reduction at diminishing return.

    Returns
    -------
    An object exposing `predict_proba` and `predict`, holding the M fitted
    members in `.members_` and their scores in `.scores_`.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    ranked = sorted(scored_params, key=lambda t: t[0], reverse=True)[:m]
    if not ranked:
        raise ValueError("scored_params is empty")

    members, scores = [], []
    for score, params in ranked:
        clf = clone(estimator).set_params(**params)
        try:
            clf.fit(X, y, sample_weight=None if sample_weight is None else sample_weight.values)
        except TypeError:
            clf.fit(X, y)
        members.append(clf)
        scores.append(float(score))

    class _ProbabilityAverager:
        """Soft-voting wrapper over pre-fitted members."""

        def __init__(self, members_, scores_):
            self.members_ = members_
            self.scores_ = scores_

        def predict_proba(self, X_):
            return np.mean([mm.predict_proba(X_) for mm in self.members_], axis=0)

        def predict(self, X_):
            return (self.predict_proba(X_)[:, 1] >= 0.5).astype(int)

    return _ProbabilityAverager(members, scores)
