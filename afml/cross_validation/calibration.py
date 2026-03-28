"""
Probability calibration utilities for financial time series.

Provides CVIsotonicCalibrator (OOF isotonic regression with PurgedKFold)
and a comprehensive cross-validation analysis that directly compares
raw vs calibrated performance on PWA and Brier score.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import brier_score_loss

from .scoring import probability_weighted_accuracy   # your existing scoring module
# If you also need the other metrics, you can import them here too
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
from sklearn.model_selection import BaseCrossValidator


# ===================================================================
# UPDATED CVIsotonicCalibrator (replace the old version at the bottom of your file)
# ===================================================================
class CVIsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """
    Probability calibrator using Isotonic Regression fitted on
    out-of-fold (OOF) predictions from PurgedKFold CV.

    This class implements a cross-validated isotonic regression calibrator
    specifically designed for financial time series data. It prevents temporal
    leakage by using PurgedKFold and supports sample weights (e.g., uniqueness
    or time-decay weights as described in *Advances in Financial Machine Learning*).

    The calibration is performed on out-of-fold predictions to provide
    an unbiased estimate of the base estimator's probability outputs before
    fitting the isotonic regressor. After calibration, the base estimator is
    refitted on the full training set.

    The default ``score`` method returns **Probability-Weighted Accuracy (PWA)**.
    The ``brier_score`` method returns the classic Brier score (lower = better).
    """

    def __init__(self, estimator, cv=None):
        self.estimator = estimator
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """
        Fit the CV isotonic calibrator (exactly as before, with minor robustness improvements).
        """
        X = check_array(X, ensure_min_samples=2)
        y = check_array(y, ensure_min_samples=2, dtype="int").ravel()

        n_samples = X.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        # Default CV handling (you must pass a PurgedKFold with t1)
        if self.cv is None:
            raise NotImplementedError(
                "Please explicitly pass a PurgedKFold (or compatible) instance to cv."
            )
        else:
            self.cv_ = clone(self.cv) if hasattr(self.cv, "split") else self.cv

        # --- Phase 1: Collect OOF predictions ---
        oof_probs = np.full(n_samples, np.nan)

        for train_idx, test_idx in self.cv_.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            sw_train = sample_weight[train_idx]

            fold_clf = clone(self.estimator)

            try:
                fold_clf.fit(X_train, y_train, sample_weight=sw_train)
            except TypeError:
                fold_clf.fit(X_train, y_train)

            oof_probs[test_idx] = fold_clf.predict_proba(X_test)[:, 1]

        valid_mask = \~np.isnan(oof_probs)
        if valid_mask.sum() < 10:
            raise ValueError(
                "Too few valid OOF predictions generated. "
                "Consider reducing embargo size or increasing n_splits."
            )

        # --- Phase 2: Fit isotonic calibrator ---
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.calibrator_.fit(
            oof_probs[valid_mask],
            y[valid_mask],
            sample_weight=sample_weight[valid_mask],
        )

        # --- Phase 3: Refit base estimator on full data ---
        self.estimator_ = clone(self.estimator)
        try:
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.estimator_.fit(X, y)

        self.oof_probs_ = oof_probs
        self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["calibrator_", "estimator_"])
        X = check_array(X)
        raw_probs = self.estimator_.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator_.predict(raw_probs), 0.0, 1.0)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y, sample_weight=None):
        """
        Return the Probability-Weighted Accuracy (PWA) — higher is better.
        """
        check_is_fitted(self, ["calibrator_", "estimator_"])
        proba = self.predict_proba(X)
        return probability_weighted_accuracy(
            y_true=y,
            y_pred=proba,
            sample_weight=sample_weight,
            labels=self.classes_,
        )

    def brier_score(self, X, y, sample_weight=None):
        """
        Return the Brier score (lower is better).
        """
        check_is_fitted(self, ["calibrator_", "estimator_"])
        proba = self.predict_proba(X)[:, 1]
        y = check_array(y, ensure_2d=False, dtype="int").ravel()
        return brier_score_loss(y, proba, sample_weight=sample_weight)


# ===================================================================
# NEW COMPREHENSIVE FUNCTION (add this at the end of the file)
# ===================================================================
def analyze_calibrated_cross_val_scores(
    base_estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_score: Optional[pd.Series] = None,
    calibrator_cv: Optional[BaseCrossValidator] = None,
):
    """
    Comprehensive cross-validation analysis that compares:
      - the raw (unadjusted) base_estimator
      - the calibrated CVIsotonicCalibrator

    Returns PWA and Brier score for BOTH, plus the usual metrics from
    analyze_cross_val_scores. Uses the exact same style / return format.

    Parameters
    ----------
    base_estimator : unfitted sklearn-compatible classifier
    calibrator_cv : PurgedKFold (or compatible). If None, it is auto-created
                    from cv_gen (recommended to use the same PurgedKFold).

    Returns
    -------
    ret_scores : dict of np.ndarray (per-fold scores, keys like "raw_pwa", "cal_pwa", ...)
    scores_df : pd.DataFrame with mean / std
    confusion_matrix_breakdown : list of dicts (raw + calibrated per fold)
    """
    if calibrator_cv is None:
        if hasattr(cv_gen, "t1") and isinstance(cv_gen.t1, pd.Series):
            from .cross_validation import PurgedKFold

            t1_series = cv_gen.t1
            pct = getattr(cv_gen, "pct_embargo", 0.01)
            n_splits_cal = getattr(cv_gen, "n_splits", 5)
            calibrator_cv = PurgedKFold(
                n_splits=n_splits_cal, t1=t1_series, pct_embargo=pct
            )
        else:
            raise ValueError(
                "calibrator_cv must be provided (PurgedKFold with same t1/pct_embargo as cv_gen)"
            )

    # Score keys (raw + calibrated)
    score_keys = ["accuracy", "pwa", "neg_log_loss", "brier", "precision", "recall", "f1"]
    ret_scores = {f"raw_{k}": np.zeros(cv_gen.n_splits) for k in score_keys}
    ret_scores.update({f"cal_{k}": np.zeros(cv_gen.n_splits) for k in score_keys})

    cms_raw = []
    cms_cal = []

    # Default weights
    if sample_weight_train is None:
        sample_weight_train = pd.Series(np.ones((X.shape[0],)), index=y.index)
    if sample_weight_score is None:
        sample_weight_score = pd.Series(np.ones((X.shape[0],)), index=y.index)

    # Sequential bootstrap handling (same as in analyze_cross_val_scores)
    seq_bootstrap = isinstance(base_estimator, SequentiallyBootstrappedBaggingClassifier)
    if seq_bootstrap:
        t1 = base_estimator.samples_info_sets.copy()

    for i, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        # ====================== RAW ESTIMATOR ======================
        if seq_bootstrap:
            raw_clf = clone(base_estimator).set_params(
                samples_info_sets=t1.iloc[train], oob_score=False
            )
        else:
            raw_clf = clone(base_estimator)

        raw_fit = raw_clf.fit(
            X=X.iloc[train, :].to_numpy(),
            y=y.iloc[train].to_numpy(),
            sample_weight=sample_weight_train.iloc[train].to_numpy(),
        )

        prob_raw = raw_fit.predict_proba(X.iloc[test, :].to_numpy())
        pred_raw = (prob_raw[:, 1] > 0.5).astype(int)

        params_raw = dict(
            y_true=y.iloc[test],
            y_pred=pred_raw,
            labels=raw_clf.classes_,
            sample_weight=sample_weight_score.iloc[test].to_numpy(),
        )

        # Raw metrics (same logic as analyze_cross_val_scores)
        for k, scoring in zip(
            ["accuracy", "pwa", "neg_log_loss", "brier", "precision", "recall", "f1"],
            [
                accuracy_score,
                probability_weighted_accuracy,
                log_loss,
                brier_score_loss,
                precision_score,
                recall_score,
                f1_score,
            ],
        ):
            if scoring in (probability_weighted_accuracy, log_loss):
                params_raw["y_pred"] = prob_raw
                score = scoring(**params_raw)
                if k == "neg_log_loss":
                    score *= -1
            elif scoring == brier_score_loss:
                score = brier_score_loss(
                    y_true=params_raw["y_true"],
                    y_pred=prob_raw[:, 1],
                    sample_weight=params_raw["sample_weight"],
                )
            else:
                params_raw["y_pred"] = pred_raw
                try:
                    score = scoring(**params_raw)
                except Exception:
                    del params_raw["labels"]
                    score = scoring(**params_raw)
                    params_raw["labels"] = raw_clf.classes_
            ret_scores[f"raw_{k}"][i] = score

        cms_raw.append(confusion_matrix(**params_raw).round(2))

        # ====================== CALIBRATED (CVIsotonicCalibrator) ======================
        calibrator = CVIsotonicCalibrator(
            estimator=clone(base_estimator), cv=calibrator_cv
        )
        calibrator.fit(
            X=X.iloc[train, :].to_numpy(),
            y=y.iloc[train].to_numpy(),
            sample_weight=sample_weight_train.iloc[train].to_numpy(),
        )

        prob_cal = calibrator.predict_proba(X.iloc[test, :].to_numpy())
        pred_cal = calibrator.predict(X.iloc[test, :].to_numpy())

        # Use the calibrator's built-in methods for PWA and Brier
        ret_scores["cal_pwa"][i] = calibrator.score(
            X=X.iloc[test, :].to_numpy(),
            y=y.iloc[test].to_numpy(),
            sample_weight=sample_weight_score.iloc[test].to_numpy(),
        )
        ret_scores["cal_brier"][i] = calibrator.brier_score(
            X=X.iloc[test, :].to_numpy(),
            y=y.iloc[test].to_numpy(),
            sample_weight=sample_weight_score.iloc[test].to_numpy(),
        )

        # Other metrics for calibrated
        params_cal = dict(
            y_true=y.iloc[test],
            y_pred=pred_cal,
            labels=calibrator.classes_,
            sample_weight=sample_weight_score.iloc[test].to_numpy(),
        )

        for k, scoring in zip(
            ["accuracy", "neg_log_loss", "precision", "recall", "f1"],
            [accuracy_score, log_loss, precision_score, recall_score, f1_score],
        ):
            if scoring == log_loss:
                params_cal["y_pred"] = prob_cal
                score = scoring(**params_cal)
                score *= -1
            else:
                params_cal["y_pred"] = pred_cal
                try:
                    score = scoring(**params_cal)
                except Exception:
                    del params_cal["labels"]
                    score = scoring(**params_cal)
                    params_cal["labels"] = calibrator.classes_
            ret_scores[f"cal_{k}"][i] = score

        cms_cal.append(confusion_matrix(**params_cal).round(2))

    # Mean / std DataFrame (same format as analyze_cross_val_scores)
    scores_df = pd.DataFrame.from_dict(
        {
            scoring: {"mean": scores.mean(), "std": scores.std()}
            for scoring, scores in ret_scores.items()
        },
        orient="index",
    )

    # Confusion matrix breakdown (both raw and calibrated)
    confusion_matrix_breakdown = []
    for i, cm in enumerate(cms_raw, 1):
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "raw", "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            )
        else:
            confusion_matrix_breakdown.append({"fold": i, "type": "raw", "confusion_matrix": cm})

    for i, cm in enumerate(cms_cal, 1):
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "calibrated", "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            )
        else:
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "calibrated", "confusion_matrix": cm}
            )

    return ret_scores, scores_df, confusion_matrix_breakdown
