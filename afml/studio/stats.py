"""
Family statistics shown beside every score.

The numbers come from ``afml.cross_validation.selection_overfit``: the effective trial
count (Li and Ji eigenvalue count and the Kish design effect), the expected maximum
under no skill, and the one-standard-error rule. This module adapts them to the
ledger: small or degenerate score matrices return counts instead of raising, and the
null threshold is shifted to the mean of the scores, because CV scores such as log
loss are not centered on zero the way Sharpe ratios are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from afml.cross_validation import selection_overfit as so


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
    Effective number of independent trials from a trials x folds score matrix,
    through ``selection_overfit.effective_trial_count``.

    Trials with missing scores are dropped. With fewer than two trials or two scores
    each there is no correlation to measure, so the raw count is returned. A matrix
    with more trials than scores per trial is ``rank_limited``: its effective count
    is a floor, not an estimate.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D trials x observations matrix")
    scores = scores[np.isfinite(scores).all(axis=1)]
    n_raw, n_obs = scores.shape
    if n_raw < 2 or n_obs < 2:
        return TrialCount(n_raw, float(n_raw), float(n_raw), n_obs, False)
    tc = so.effective_trial_count(scores)
    return TrialCount(n_raw, tc.n_eff_eigen, tc.n_eff_kish, n_obs, bool(n_raw > n_obs or tc.rank_limited))


def expected_max_sharpe(n_trials: float, sr_variance: float, sr_mean: float = 0.0) -> float:
    """
    Expected best score among ``n_trials`` independent trials with no skill
    (AFML Section 8.1), shifted to ``sr_mean``. Zero skill is the mean score, not
    zero, for criteria such as log loss.
    """
    if not np.isfinite(sr_variance) or sr_variance < 0 or n_trials < 1:
        return float("nan")
    return float(sr_mean + so.expected_max_sharpe(n_trials, sr_variance))


def noise_floor(fold_scores: np.ndarray) -> float:
    """Standard error of a mean CV score from its fold scores."""
    s = np.asarray(fold_scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    return float(s.std(ddof=1) / np.sqrt(s.size))


def one_standard_error_selection(results: Sequence[Dict[str, Any]],
                                 complexity: Callable[[Dict[str, Any]], float]) -> Tuple[Dict, Dict]:
    """
    The simplest candidate within one standard error of the best, through
    ``selection_overfit.one_standard_error_selection``. Each result holds ``params``
    and ``fold_scores``; ``complexity`` maps params to a number, lower being simpler.
    """
    usable: List[Dict[str, Any]] = [r for r in results
                                    if np.isfinite(np.asarray(r["fold_scores"], dtype=float)).sum() >= 2]
    return so.one_standard_error_selection(usable, complexity)


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
