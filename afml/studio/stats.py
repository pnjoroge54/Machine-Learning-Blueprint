"""
Family statistics shown beside every score.

The plan names ``effective_trial_count``, ``expected_max_sharpe``,
``criterion_variance_audit`` and ``one_standard_error_selection``; afml now has them in
``afml.cross_validation.selection_overfit``.
This branch does not have them, so this module carries self-contained versions
with the same roles. ``resolve()`` prefers the afml implementations when they are
importable, so rewiring is a no-op once the newer code lands.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


@dataclass
class TrialCount:
    n_raw: int
    n_eigen: float
    n_kish: float
    n_obs: int
    rank_limited: bool

    def as_dict(self) -> Dict:
        return asdict(self)


def effective_trial_count(scores: np.ndarray) -> TrialCount:
    """
    Effective number of independent trials from a trials x observations score matrix
    (rows are trials, columns are fold scores or returns).

    * ``n_eigen``: participation ratio of the eigenvalues of the trial correlation
      matrix, ``(sum l)^2 / sum l^2``. Equals N for uncorrelated trials and 1 for
      identical ones.
    * ``n_kish``: Kish design-effect count, ``N / (1 + (N - 1) * mean_rho)``, with the
      mean off-diagonal correlation clipped at 0.

    The correlation matrix has rank at most ``n_obs - 1``, so when N exceeds the
    score-vector length the eigenvalue count is capped and the result is flagged
    ``rank_limited``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D trials x observations matrix")
    # Drop trials with missing scores or no variation; they carry no correlation info
    ok = np.isfinite(scores).all(axis=1)
    scores = scores[ok]
    n_raw, n_obs = scores.shape
    if n_raw <= 1:
        return TrialCount(n_raw, float(n_raw), float(n_raw), n_obs, False)

    # Trials whose scores never vary have no defined correlation; count only the rest
    varying = scores.std(axis=1) > 0
    if varying.sum() <= 1:
        return TrialCount(n_raw, 1.0, 1.0, n_obs, n_raw > n_obs)

    corr = np.corrcoef(scores[varying])
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    n_eigen = float(eig.sum() ** 2 / (eig**2).sum())

    m = corr.shape[0]
    mean_rho = (corr.sum() - m) / (m * (m - 1))
    n_kish = float(m / (1 + (m - 1) * max(mean_rho, 0.0)))

    return TrialCount(n_raw, n_eigen, n_kish, n_obs, n_raw > n_obs)


def expected_max_sharpe(n_trials: float, sr_variance: float, sr_mean: float = 0.0) -> float:
    """
    Expected maximum Sharpe ratio among ``n_trials`` independent trials with true
    Sharpe zero (the null threshold of the Deflated Sharpe Ratio, Bailey and
    Lopez de Prado 2014):

        E[max SR] = mean + sqrt(V) * ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)))
    """
    n = float(n_trials)
    if n < 1 or not np.isfinite(sr_variance) or sr_variance < 0:
        return float("nan")
    if n == 1:
        return float(sr_mean)
    z1 = sps.norm.ppf(1 - 1 / n)
    z2 = sps.norm.ppf(1 - 1 / (n * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def noise_floor(fold_scores: np.ndarray) -> float:
    """Standard error of a mean CV score from its fold scores."""
    s = np.asarray(fold_scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    return float(s.std(ddof=1) / np.sqrt(s.size))


def one_standard_error_selection(mean_scores: np.ndarray, std_errors: np.ndarray) -> int:
    """
    Index of the simplest candidate within one standard error of the best, assuming
    candidates are ordered from simplest to most complex.
    """
    mean_scores = np.asarray(mean_scores, dtype=float)
    best = int(np.nanargmax(mean_scores))
    threshold = mean_scores[best] - np.asarray(std_errors, dtype=float)[best]
    return int(np.flatnonzero(mean_scores >= threshold)[0])


# --------------------------------------------------------------------------- rewiring

_AFML_LOCATIONS = {
    "effective_trial_count": ["afml.cross_validation.selection_overfit"],
    "expected_max_sharpe": ["afml.cross_validation.selection_overfit"],
    "one_standard_error_selection": ["afml.cross_validation.selection_overfit"],
}


def resolve(name: str) -> Callable:
    """Return the afml implementation of ``name`` if one exists, else the local one."""
    for module in _AFML_LOCATIONS.get(name, []):
        try:
            fn = getattr(importlib.import_module(module), name, None)
        except ImportError:
            fn = None
        if fn is not None:
            return fn
    return globals()[name]


def family_summary(score_matrix: Optional[np.ndarray], best_sharpe: Optional[float] = None,
                   sharpe_variance: Optional[float] = None, n_raw: Optional[int] = None) -> Dict:
    """Numbers for a family's scoreboard row."""
    out: Dict = {"n_raw": n_raw or 0, "n_eigen": None, "n_kish": None,
                 "rank_limited": False, "null_threshold": None}
    if score_matrix is not None and len(score_matrix):
        tc = effective_trial_count(score_matrix)
        out.update(n_raw=max(out["n_raw"], tc.n_raw), n_eigen=tc.n_eigen,
                   n_kish=tc.n_kish, rank_limited=tc.rank_limited)
    if sharpe_variance is not None:
        n_eff = out["n_eigen"] or out["n_raw"]
        if n_eff:
            out["null_threshold"] = expected_max_sharpe(n_eff, sharpe_variance)
    if best_sharpe is not None:
        out["best_sharpe"] = best_sharpe
    return out
