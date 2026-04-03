"""
nested_cv.py
────────────
Unified Validation + Isotonic Calibration pipeline.

Merges the V-in-V three-zone architecture from your MQL5 article with
nested CV and OOF isotonic calibration.

Critical integration fixes vs. naive reimplementation
------------------------------------------------------
1. CombinatorialPurgedCV.split() yields (train_idx, test_idx_LIST) where
   test_idx_list is a LIST of arrays (one per test fold), not a flat array.
   All CPCV loops must call np.concatenate(test_idx_list) to get flat indices.

2. PurgedWalkForwardCV requires t1 as pd.Series with SAME index as X.
   When slicing a fold, t1 must be sliced with .iloc to preserve index
   alignment. This is validated inside PurgedWalkForwardCV.split().

3. min_train_size in PurgedWalkForwardCV is a FLOAT fraction (e.g. 0.1),
   NOT an integer. Passing an integer silently miscalculates the threshold.

4. CPCVAnalyzer requires close_prices: pd.Series for MtM Sharpe calculation.
   It is a required constructor argument, not optional.

5. CPCVAnalyzer.get_distribution_metrics() requires primary_sides: pd.Series
   for meta-labelling (+1/-1 direction signals). The simple distribution_summary()
   method we wrote previously does not exist in the actual implementation.

6. All internal indexing uses X.iloc[idx], y.iloc[idx] — afml convention.
   Using X[int_array] on a DataFrame raises KeyError.

7. sample_weight is pd.Series aligned to X.index throughout.
   ml_cross_val_score accepts separate sample_weight_train / sample_weight_score.

References
----------
Lopez de Prado (2018). AFML Ch. 4, 7, 12.
Masters (1995). Advanced Algorithms for Neural Networks. — 1-SE rule.
Your article: https://www.mql5.com/en/articles/21603
afml.cross_validation.cross_validation — PurgedKFold, PurgedWalkForwardCV
afml.cross_validation.combinatorial — CombinatorialPurgedCV, CPCVAnalyzer
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_is_fitted

# ── afml imports ──────────────────────────────────────────────────────────────
# Adjust paths to match your project layout.
from afml.cross_validation.cross_validation import (
    PurgedKFold,
    PurgedWalkForwardCV,
    ml_cross_val_score,
)
from afml.cross_validation.combinatorial import (
    CombinatorialPurgedCV,
    CPCVAnalyzer,
    optimal_folds_number,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Partition  (V-in-V three-zone architecture)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataPartition:
    """
    Enforces the strict three-layer V-in-V data partition from your article:

        [─── Outer Training (~60%) ───][─ Inner Val (~20%) ─][─ Final Test (~20%) ─]

    All slices preserve pd.DataFrame / pd.Series types and their original
    datetime indices — required for PurgedWalkForwardCV alignment checks.

    Raises RuntimeError if the Final Test Set is opened more than once,
    enforcing the Masters (1995) single-opening discipline.

    Reference: https://www.mql5.com/en/articles/21603 Part II
    """

    X_outer: pd.DataFrame
    y_outer: pd.Series
    sw_outer: pd.Series
    t1_outer: pd.Series

    X_inner_val: pd.DataFrame
    y_inner_val: pd.Series
    sw_inner_val: pd.Series

    X_final: pd.DataFrame
    y_final: pd.Series
    sw_final: pd.Series

    _final_opened: bool = field(default=False, init=False, repr=False)

    def open_final_test(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Open the Final Test Set. Raises RuntimeError on any subsequent call.

        Once opened, the result is committed. Any model adjustment after
        seeing this result invalidates the evaluation.
        Reference: Masters (1995); your article Section II.
        """
        if self._final_opened:
            raise RuntimeError(
                "The Final Test Set has already been opened. "
                "Any further model adjustment invalidates the evaluation.\n"
                "Reference: Masters (1995); "
                "https://www.mql5.com/en/articles/21603"
            )
        self._final_opened = True
        return self.X_final, self.y_final, self.sw_final


def partition_data(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series,
    inner_val_pct: float = 0.20,
    final_test_pct: float = 0.20,
) -> DataPartition:
    """
    Partition data into the three V-in-V zones in strict temporal order.

    Uses iloc throughout to preserve pd.DataFrame/Series types and their
    indices — critical for PurgedWalkForwardCV's index-alignment validation.

    Parameters
    ----------
    X, y, t1, sample_weight : pd.DataFrame / pd.Series  Full aligned dataset.
    inner_val_pct : float   Fraction reserved for inner validation.
    final_test_pct : float  Fraction reserved for the final test.
    """
    n = len(X)
    outer_end = int(n * (1.0 - inner_val_pct - final_test_pct))
    val_end = int(n * (1.0 - final_test_pct))

    return DataPartition(
        X_outer=X.iloc[:outer_end],
        y_outer=y.iloc[:outer_end],
        sw_outer=sample_weight.iloc[:outer_end],
        t1_outer=t1.iloc[:outer_end],
        X_inner_val=X.iloc[outer_end:val_end],
        y_inner_val=y.iloc[outer_end:val_end],
        sw_inner_val=sample_weight.iloc[outer_end:val_end],
        X_final=X.iloc[val_end:],
        y_final=y.iloc[val_end:],
        sw_final=sample_weight.iloc[val_end:],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inner CV Grid Search  (Masters 1-SE rule)
# ─────────────────────────────────────────────────────────────────────────────

_INNER_SCORERS = {
    'neg_brier': lambda y, p, w: -brier_score_loss(y, p, sample_weight=w),
    'neg_logloss': lambda y, p, w: -log_loss(y, p, sample_weight=w),
}


def inner_cv_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sw_train: pd.Series,
    t1_train: pd.Series,
    estimator,
    param_grid: Dict[str, List],
    n_inner_splits: int = 3,
    pct_embargo: float = 0.01,
    min_train_size: float = 0.1,
    scoring: str = 'neg_brier',
) -> Tuple[Dict[str, Any], float, List[Dict]]:
    """
    Anchored walk-forward grid search with Masters' 1-SE rule.

    For each hyperparameter combination, runs PurgedWalkForwardCV and
    averages fold scores. Among all combinations within 1 standard error
    of the best, returns the SIMPLEST (first in ParameterGrid order).

    Parameters
    ----------
    X_train, y_train, sw_train : pd.DataFrame/Series from outer training fold.
    t1_train : pd.Series  Event end-times with SAME index as X_train.
    min_train_size : float  Fraction (not int) passed to PurgedWalkForwardCV.
    scoring : 'neg_brier' | 'neg_logloss'

    Returns
    -------
    best_params, best_score, all_scores
    """
    if scoring not in _INNER_SCORERS:
        raise ValueError(f"scoring must be one of {list(_INNER_SCORERS.keys())}")
    scorer = _INNER_SCORERS[scoring]

    # PurgedWalkForwardCV requires t1 as pd.Series with same index as X —
    # t1_train is already the correctly-sliced series from partition_data.
    inner_cv = PurgedWalkForwardCV(
        n_splits=n_inner_splits,
        t1=t1_train,
        pct_embargo=pct_embargo,
        expanding_window=True,
        min_train_size=min_train_size,  # float fraction, NOT int
    )

    all_scores: List[Dict] = []

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            clf = clone(estimator)
            clf.set_params(**params)

            # iloc indexing — X_train is pd.DataFrame
            try:
                clf.fit(
                    X_train.iloc[tr_idx],
                    y_train.iloc[tr_idx],
                    sample_weight=sw_train.iloc[tr_idx].values,
                )
            except TypeError:
                clf.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])

            probs = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            fold_scores.append(
                scorer(
                    y_train.iloc[val_idx].values,
                    probs,
                    sw_train.iloc[val_idx].values,
                )
            )

        if not fold_scores:
            continue

        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        all_scores.append({
            'params': params,
            'mean_score': mean_s,
            'std_score': std_s,
        })

    if not all_scores:
        raise ValueError(
            "No valid inner CV folds produced. "
            "Reduce n_inner_splits or pct_embargo, or increase min_train_size."
        )

    # 1-SE rule: among params within 1 SE of best, prefer simplest (Masters 1995)
    best_entry = max(all_scores, key=lambda x: x['mean_score'])
    threshold = best_entry['mean_score'] - best_entry['std_score']
    within_1se = [s for s in all_scores if s['mean_score'] >= threshold]

    # ParameterGrid is ordered (coarser → finer), so within_1se[0] = simplest
    best_params = within_1se[0]['params']

    return best_params, best_entry['mean_score'], all_scores


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedValidationCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedValidationCalibrator(BaseEstimator, ClassifierMixin):
    """
    Unified nested CV + V-in-V + Isotonic Calibration pipeline.

    Supports two outer CV modes, switched via outer_cv_type:
    ─ 'walkforward' : PurgedWalkForwardCV (anchored expanding)
      Directly equivalent to your vin_v_anchored_walkforward but with
      automated hyperparameter search and OOF calibration added.
    ─ 'cpcv'        : CombinatorialPurgedCV (φ[N,k] path distribution)
      Requires close_prices for MtM Sharpe computation in CPCVAnalyzer.

    Both modes:
    ✓ Three-zone 60/20/20 V-in-V data partition (your article Part II)
    ✓ Anchored walk-forward inner loop (1-SE rule, Masters 1995)
    ✓ OOF isotonic calibration with sample weights (AFML Ch. 4)
    ✓ Single-open Final Test gate (raises RuntimeError on repeat)
    ✓ pd.DataFrame / pd.Series + iloc throughout (afml convention)
    ✓ t1 alignment validated by PurgedWalkForwardCV

    Parameters
    ----------
    estimator : unfitted sklearn classifier with predict_proba.
    param_grid : dict  Hyperparameter search space.
    n_outer_splits : int  Walk-forward outer splits (walkforward mode only).
    n_inner_splits : int  Walk-forward inner splits for hyperparam search.
    pct_embargo : float  Embargo fraction for both loops.
    min_train_size : float  Fraction passed to PurgedWalkForwardCV (NOT int).
    scoring : str  'neg_brier' | 'neg_logloss'
    inner_val_pct : float  Inner validation zone fraction.
    final_test_pct : float  Final test zone fraction.
    outer_cv_type : str  'walkforward' | 'cpcv'
    cpcv_n_folds : int  N for CPCV (cpcv mode only).
    cpcv_n_test_folds : int  k for CPCV (cpcv mode only).
    close_prices : pd.Series, optional
        Required when outer_cv_type='cpcv'. Passed to CPCVAnalyzer for
        MtM Sharpe calculation. Must cover the full outer training period.
    primary_sides : pd.Series, optional
        Required when outer_cv_type='cpcv'. +1/-1 direction signals for
        meta-labelling, passed to CPCVAnalyzer.get_distribution_metrics().
    """

    def __init__(
        self,
        estimator,
        param_grid: Dict[str, List],
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        pct_embargo: float = 0.01,
        min_train_size: float = 0.1,
        scoring: str = 'neg_brier',
        inner_val_pct: float = 0.20,
        final_test_pct: float = 0.20,
        outer_cv_type: str = 'walkforward',
        cpcv_n_folds: int = 6,
        cpcv_n_test_folds: int = 2,
        close_prices: Optional[pd.Series] = None,
        primary_sides: Optional[pd.Series] = None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.pct_embargo = pct_embargo
        self.min_train_size = min_train_size
        self.scoring = scoring
        self.inner_val_pct = inner_val_pct
        self.final_test_pct = final_test_pct
        self.outer_cv_type = outer_cv_type
        self.cpcv_n_folds = cpcv_n_folds
        self.cpcv_n_test_folds = cpcv_n_test_folds
        self.close_prices = close_prices
        self.primary_sides = primary_sides

    # ── Helpers ───────────────────────────────────────────────────────────

    def _make_outer_cv(
        self, t1: pd.Series,
    ) -> Union[PurgedWalkForwardCV, CombinatorialPurgedCV]:
        if self.outer_cv_type == 'cpcv':
            return CombinatorialPurgedCV(
                n_folds=self.cpcv_n_folds,
                n_test_folds=self.cpcv_n_test_folds,
                t1=t1,
                pct_embargo=self.pct_embargo,
            )
        return PurgedWalkForwardCV(
            n_splits=self.n_outer_splits,
            t1=t1,
            pct_embargo=self.pct_embargo,
            expanding_window=True,
            min_train_size=self.min_train_size,
        )

    def _fit_clf(self, estimator, X, y, sw, params):
        clf = clone(estimator)
        clf.set_params(**params)
        try:
            clf.fit(X, y, sample_weight=sw.values)
        except TypeError:
            clf.fit(X, y)
        return clf

    def _oof_for_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        sw_tr: pd.Series,
        t1_tr: pd.Series,
        params: dict,
    ) -> pd.Series:
        """
        Collect OOF predictions within an outer training fold using
        PurgedWalkForwardCV (anchored).

        t1_tr must have the same index as X_tr — ensured by iloc slicing
        in partition_data and the outer fold loop.
        """
        inner_cv = PurgedWalkForwardCV(
            n_splits=self.n_inner_splits,
            t1=t1_tr,  # same index as X_tr after iloc slice
            pct_embargo=self.pct_embargo,
            expanding_window=True,
            min_train_size=self.min_train_size,
        )
        inner_oof = pd.Series(np.nan, index=X_tr.index)

        for in_tr, in_val in inner_cv.split(X_tr, y_tr):
            clf_oof = clone(self.estimator)
            clf_oof.set_params(**params)
            try:
                clf_oof.fit(
                    X_tr.iloc[in_tr],
                    y_tr.iloc[in_tr],
                    sample_weight=sw_tr.iloc[in_tr].values,
                )
            except TypeError:
                clf_oof.fit(X_tr.iloc[in_tr], y_tr.iloc[in_tr])

            inner_oof.iloc[in_val] = clf_oof.predict_proba(
                X_tr.iloc[in_val]
            )[:, 1]

        return inner_oof

    # ── Main fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series,
        sample_weight: pd.Series,
    ) -> "UnifiedValidationCalibrator":
        """
        Run the full V-in-V nested CV pipeline.

        Parameters
        ----------
        X, y, t1, sample_weight : pd.DataFrame / pd.Series
            Full dataset. Partitioned internally. t1 must have same index as X.
        """
        if self.outer_cv_type == 'cpcv' and self.close_prices is None:
            raise ValueError(
                "close_prices is required when outer_cv_type='cpcv'. "
                "CPCVAnalyzer uses it for MtM Sharpe calculation."
            )

        # ── Partition into three V-in-V zones ─────────────────────────────
        self.partition_ = partition_data(
            X, y, t1, sample_weight,
            inner_val_pct=self.inner_val_pct,
            final_test_pct=self.final_test_pct,
        )
        dp = self.partition_

        # OOF storage aligned to outer training index
        oof_probs_raw = pd.Series(np.nan, index=dp.X_outer.index)
        outer_scores: List[Dict] = []
        best_params_per_fold: List[Dict] = []

        # ── Outer loop ────────────────────────────────────────────────────
        outer_cv = self._make_outer_cv(dp.t1_outer)

        _mode = (
            f"CPCV (N={self.cpcv_n_folds}, k={self.cpcv_n_test_folds}, "
            f"φ={outer_cv.n_test_paths} paths)"
            if self.outer_cv_type == 'cpcv'
            else f"Walk-Forward (n_splits={self.n_outer_splits}, anchored)"
        )
        print(f"\n{'█' * 60}")
        print(f"  OUTER LOOP — {_mode}")
        print(f"  Outer training: {len(dp.X_outer)} obs")
        print(f"{'█' * 60}")

        # CPCV yields (train_idx, test_idx_LIST) — list of arrays per test fold.
        # Walk-forward yields (train_idx, test_idx) — flat arrays.
        # We normalise by wrapping CPCV test_idx_list with np.concatenate.
        split_iter = outer_cv.split(dp.X_outer, dp.y_outer)

        for fold_num, split_output in enumerate(split_iter):
            if self.outer_cv_type == 'cpcv':
                tr_idx, test_idx_list = split_output
                te_idx = np.concatenate(test_idx_list)
            else:
                tr_idx, te_idx = split_output

            print(f"\n{'─' * 55}")
            print(f"  Outer fold {fold_num + 1}  |  "
                  f"Train: {len(tr_idx)}  |  Test: {len(te_idx)}")
            print(f"{'─' * 55}")

            # iloc slice — preserves pd.DataFrame/Series with original index
            X_tr = dp.X_outer.iloc[tr_idx]
            y_tr = dp.y_outer.iloc[tr_idx]
            sw_tr = dp.sw_outer.iloc[tr_idx]
            t1_tr = dp.t1_outer.iloc[tr_idx]

            X_te = dp.X_outer.iloc[te_idx]
            y_te = dp.y_outer.iloc[te_idx]
            sw_te = dp.sw_outer.iloc[te_idx]

            # ── Inner search ───────────────────────────────────────────────
            best_params, best_inner_score, _ = inner_cv_search(
                X_train=X_tr, y_train=y_tr, sw_train=sw_tr, t1_train=t1_tr,
                estimator=self.estimator,
                param_grid=self.param_grid,
                n_inner_splits=self.n_inner_splits,
                pct_embargo=self.pct_embargo,
                min_train_size=self.min_train_size,
                scoring=self.scoring,
            )
            best_params_per_fold.append(best_params)
            print(f"  Best params (1-SE): {best_params}  "
                  f"inner score={best_inner_score:.4f}")

            # ── Inner OOF for calibration ──────────────────────────────────
            inner_oof = self._oof_for_fold(X_tr, y_tr, sw_tr, t1_tr, best_params)

            # ── Isotonic calibrator on inner OOF ──────────────────────────
            valid = inner_oof.notna()
            if valid.sum() < 5:
                warnings.warn(
                    f"Fold {fold_num + 1}: only {valid.sum()} valid inner OOF obs."
                )
            cal = IsotonicRegression(out_of_bounds='clip', increasing=True)
            cal.fit(
                inner_oof[valid].values,
                y_tr[valid].values,
                sample_weight=sw_tr[valid].values,
            )

            # ── Refit on full outer training fold + score on outer test ────
            clf_f = self._fit_clf(self.estimator, X_tr, y_tr, sw_tr, best_params)
            raw_te = clf_f.predict_proba(X_te)[:, 1]
            cal_te = np.clip(cal.predict(raw_te), 0.0, 1.0)

            # Store raw OOF probs aligned to outer index
            oof_probs_raw.iloc[te_idx] = raw_te

            b_raw = brier_score_loss(y_te, raw_te, sample_weight=sw_te.values)
            b_cal = brier_score_loss(y_te, cal_te, sample_weight=sw_te.values)
            l_raw = log_loss(y_te, raw_te, sample_weight=sw_te.values)
            l_cal = log_loss(y_te, cal_te, sample_weight=sw_te.values)

            outer_scores.append({
                'fold': fold_num + 1,
                'params': best_params,
                'brier_raw': b_raw,
                'brier_cal': b_cal,
                'logloss_raw': l_raw,
                'logloss_cal': l_cal,
                'n_train': len(tr_idx),
                'n_test': len(te_idx),
            })
            print(f"  Brier  raw={b_raw:.4f}  cal={b_cal:.4f}")
            print(f"  LogLoss raw={l_raw:.4f}  cal={l_cal:.4f}")

        # ── CPCV: path Sharpe distribution via CPCVAnalyzer ──────────────
        if self.outer_cv_type == 'cpcv':
            print("\n  Running CPCVAnalyzer path analysis …")
            self.cpcv_analyzer_ = CPCVAnalyzer(
                estimator=clone(self.estimator),
                cv_gen=outer_cv,
                close_prices=self.close_prices,
            )
            self.cpcv_analyzer_.fit_predict(
                dp.X_outer, dp.y_outer,
                sample_weight=dp.sw_outer,
            )
            if self.primary_sides is not None:
                dist_metrics = self.cpcv_analyzer_.get_distribution_metrics(
                    primary_sides=self.primary_sides.loc[dp.X_outer.index],
                )
                self.cpcv_distribution_metrics_ = dist_metrics
                print(dist_metrics.to_string())
            else:
                warnings.warn(
                    "primary_sides not provided — skipping get_distribution_metrics(). "
                    "Pass primary_sides to UnifiedValidationCalibrator for full "
                    "CPCV path Sharpe analysis."
                )

        # ── Consensus params + final calibrator ──────────────────────────
        param_strs = [str(sorted(p.items())) for p in best_params_per_fold]
        consensus_str = Counter(param_strs).most_common(1)[0][0]
        consensus_params = best_params_per_fold[param_strs.index(consensus_str)]
        print(f"\n  Consensus params: {consensus_params}")

        # Fit on full outer zone + calibrate using full OOF
        clf_inner = self._fit_clf(
            self.estimator, dp.X_outer, dp.y_outer, dp.sw_outer, consensus_params
        )

        valid_all = oof_probs_raw.notna()
        self.calibrator_ = IsotonicRegression(
            out_of_bounds='clip', increasing=True
        )
        self.calibrator_.fit(
            oof_probs_raw[valid_all].values,
            dp.y_outer[valid_all].values,
            sample_weight=dp.sw_outer[valid_all].values,
        )

        # Inner validation checkpoint
        inner_raw = clf_inner.predict_proba(dp.X_inner_val)[:, 1]
        inner_cal = np.clip(self.calibrator_.predict(inner_raw), 0.0, 1.0)
        inner_b_r = brier_score_loss(
            dp.y_inner_val, inner_raw, sample_weight=dp.sw_inner_val.values
        )
        inner_b_c = brier_score_loss(
            dp.y_inner_val, inner_cal, sample_weight=dp.sw_inner_val.values
        )
        print(f"\n{'▶' * 55}")
        print(f"  INNER VALIDATION  (shortlisting checkpoint — do NOT retune)")
        print(f"  Brier  raw={inner_b_r:.4f}  cal={inner_b_c:.4f}")
        print(f"{'▶' * 55}")

        # Final model on Outer + Inner Val
        X_all = pd.concat([dp.X_outer, dp.X_inner_val])
        y_all = pd.concat([dp.y_outer, dp.y_inner_val])
        sw_all = pd.concat([dp.sw_outer, dp.sw_inner_val])

        self.estimator_ = self._fit_clf(
            self.estimator, X_all, y_all, sw_all, consensus_params
        )

        self.outer_scores_ = outer_scores
        self.best_params_per_fold_ = best_params_per_fold
        self.consensus_params_ = consensus_params
        self.inner_val_brier_ = {'raw': inner_b_r, 'cal': inner_b_c}
        self.oof_probs_raw_ = oof_probs_raw
        self.classes_ = np.unique(dp.y_outer.values)

        return self

    # ── Final Test ────────────────────────────────────────────────────────

    def evaluate_final_test(self) -> Dict:
        """
        Open and score the Final Test Set — EXACTLY ONCE.
        RuntimeError is raised on any subsequent call (Masters discipline).
        """
        check_is_fitted(self, ["estimator_", "calibrator_"])
        X_f, y_f, sw_f = self.partition_.open_final_test()

        raw = self.estimator_.predict_proba(X_f)[:, 1]
        cal = np.clip(self.calibrator_.predict(raw), 0.0, 1.0)

        result = {
            'brier_raw': brier_score_loss(y_f, raw, sample_weight=sw_f.values),
            'brier_cal': brier_score_loss(y_f, cal, sample_weight=sw_f.values),
            'logloss_raw': log_loss(y_f, raw, sample_weight=sw_f.values),
            'logloss_cal': log_loss(y_f, cal, sample_weight=sw_f.values),
            'raw_probs': raw,
            'cal_probs': cal,
            'y_true': y_f,
            'sample_weight': sw_f,
        }

        print(f"\n{'!' * 60}")
        print("  FINAL TEST SET — OPENED (evaluation now committed)")
        print(f"  Brier  raw={result['brier_raw']:.4f}  "
              f"cal={result['brier_cal']:.4f}")
        print(f"  LogLoss raw={result['logloss_raw']:.4f}  "
              f"cal={result['logloss_cal']:.4f}")
        print(f"{'!' * 60}")

        return result

    def outer_scores_summary(self) -> pd.DataFrame:
        check_is_fitted(self, ["outer_scores_"])
        return pd.DataFrame(self.outer_scores_)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["estimator_", "calibrator_"])
        raw = self.estimator_.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator_.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
