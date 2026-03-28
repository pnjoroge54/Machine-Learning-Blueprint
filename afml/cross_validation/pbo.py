"""
Probability of Backtest Overfitting (PBO) — Bailey & Lopez de Prado (2014).

Uses CombinatorialPurgedCV with pct_embargo=0.0 and n_test_folds=n_folds//2
to generate the C(S, S/2) symmetric IS/OOS splits required by CSCV.
"""

from math import comb
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_pbo(
    returns_matrix: pd.DataFrame,
    t1: pd.Series,
    n_folds: int = 8,
    metric: Callable = None,
) -> Dict:
    """
    Compute the Probability of Backtest Overfitting (PBO).

    Parameters
    ----------
    returns_matrix : pd.DataFrame, shape (T, N)
        Columns are candidate strategies; rows are time-ordered observations.
        Values should be per-period returns (or any scalar performance measure
        that is comparable across strategies at a given time step).

    t1 : pd.Series
        Event end-times aligned with returns_matrix.index. Because CSCV is
        symmetric (no temporal arrow), pct_embargo is set to 0.0, so t1 is
        used only to satisfy the CombinatorialPurgedCV interface — you may
        pass t1 = pd.Series(index, index=index) as a neutral value.

    n_folds : int, default=8
        Number of equal subsets S. Must be even. C(S, S/2) splits are
        generated. Bailey & Lopez de Prado recommend 8–16.

    metric : callable(returns: pd.Series) -> float, optional
        Aggregates a column of returns into a scalar performance measure.
        Defaults to the Sharpe ratio (mean/std).

    Returns
    -------
    dict with keys:
        pbo          – float in [0, 1]: estimated probability of overfitting
        n_splits     – total number of IS/OOS splits evaluated (= C(S, S/2))
        below_median – number of splits where best-IS ranked below OOS median
        oos_ranks    – list of normalised OOS ranks for the best-IS strategy
                       (0 = best OOS, 1 = worst OOS)
        logit_sr     – logit-transformed OOS score (if in (0,1)) for best-IS strategy per split
    """
    from .combinatorial import CombinatorialPurgedCV

    if n_folds % 2 != 0:
        raise ValueError(f"n_folds must be even for symmetric CSCV; got {n_folds}.")

    n_test_folds = n_folds // 2          # ← symmetric split: half IS, half OOS

    if metric is None:
        def metric(r: pd.Series) -> float:
            """Sharpe ratio."""
            return r.mean() / r.std(ddof=1) if r.std() > 0 else 0.0

    # ── Build the CSCV generator ──────────────────────────────────────────
    # pct_embargo=0.0  → no embargo (CSCV is symmetric, not temporal)
    # n_test_folds=n_folds//2 → exactly half the folds form the OOS set
    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,   # = n_folds // 2
        t1=t1,
        pct_embargo=0.0,             # ← required for symmetric CSCV
    )

    n_strategies = returns_matrix.shape[1]
    below_median = 0
    oos_ranks    = []
    logit_sr     = []

    # ── Iterate over all C(n_folds, n_folds//2) symmetric splits ────────
    for train_idx, test_idx_list in cv.split(returns_matrix):
        test_idx = np.concatenate(test_idx_list)

        # IS half: compute metric for every strategy
        is_scores = returns_matrix.iloc[train_idx].apply(metric, axis=0)

        # OOS half: compute metric for every strategy
        oos_scores = returns_matrix.iloc[test_idx].apply(metric, axis=0)

        # Best IS strategy
        best_is_col = is_scores.idxmax()

        # Rank of that strategy OOS (0-based; 0 = best OOS performer)
        oos_sorted = oos_scores.rank(ascending=False)   # rank 1 = best
        oos_rank   = oos_sorted[best_is_col]               # 1-indexed

        # Normalise rank to [0, 1]: 0 = best OOS, 1 = worst OOS
        norm_rank = (oos_rank - 1) / (n_strategies - 1) if n_strategies > 1 else 0.0
        oos_ranks.append(float(norm_rank))

        # Below median? (norm_rank > 0.5 means worse than median)
        if norm_rank > 0.5:
            below_median += 1

        # Logit-transformed OOS score (if in (0,1)) for the best-IS strategy
        oos_sr = float(oos_scores[best_is_col])
        logit_sr.append(np.log(oos_sr / (1 - oos_sr)) if 0 < oos_sr < 1 else np.nan)

    n_splits = cv.n_splits
    pbo = below_median / n_splits

    return {
        "pbo":          pbo,
        "n_splits":     n_splits,
        "below_median": below_median,
        "oos_ranks":    oos_ranks,
        "logit_sr":     [v for v in logit_sr if not np.isnan(v)],
    }


# ── Example usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 50 strategy return streams over 500 time steps
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0001, 0.01, size=(500, 50)),
        index=dates,
        columns=[f"strategy_{i}" for i in range(50)],
    )

    # Neutral t1: event start == event end (no label horizon, no purging effect)
    t1_neutral = pd.Series(dates, index=dates)

    result = compute_pbo(
        returns_matrix=returns,
        t1=t1_neutral,
        n_folds=8,      # → n_test_folds=4, C(8,4)=70 splits
    )

    print(f"PBO  : {result['pbo']:.3f}")
    print(f"Splits evaluated : {result['n_splits']}")
    print(f"Below median (IS best → OOS below median): {result['below_median']}")

    # Interpret: if returns are pure noise, PBO ≈ 0.5
    # A strategy with genuine edge should push PBO toward 0
