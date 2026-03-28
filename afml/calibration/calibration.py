"""
Probability Calibration Toolkit for Financial Machine Learning

This module provides comprehensive tools for calibrating classifier probabilities
and evaluating calibration quality, with special considerations for financial
time series data.

Key Features:
- Calibration metrics (Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE))
- Reliability diagrams with confidence intervals
- Multiple calibration methods (Platt scaling, isotonic regression)
- Integration with purged cross-validation
- Bootstrap confidence intervals for calibration curves

The module is designed to work seamlessly with the combinatorial purged
cross-validation framework from Chapter 12 of Advances in Financial Machine Learning.

Examples:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from cross_validation import PurgedKFold
    >>> from calibration import *

    >>> # Generate sample financial data
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> t1 = pd.Series(index=pd.date_range('2020-01-01', periods=1000, freq='D'),
    ...                data=pd.date_range('2020-01-15', periods=1000, freq='D'))

    >>> # Train a classifier with purged CV
    >>> clf = RandomForestClassifier()
    >>> cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

    >>> # Get out-of-fold probabilities for calibration
    >>> oof_probs = oof_predict_proba(clf, X, y, cv=cv)

    >>> # Evaluate calibration
    >>> ece = expected_calibration_error(y, oof_probs)
    >>> brier = brier_score(y, oof_probs)
    >>> print(f"ECE: {ece:.4f}, Brier: {brier:.4f}")

    >>> # Fit calibration mapping
    >>> calibrator, _ = calibrate_with_oof(clf, X, y, cv=cv, method='isotonic')

    >>> # Plot reliability diagram
    >>> plot_reliability_with_ci(y, oof_probs, n_bins=10, title="Pre-Calibration")

Reference:
    López de Prado, M. (2018) Advances in Financial Machine Learning, Chapter 7.
    Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.

Probability calibration utilities for financial time series.

Provides CVIsotonicCalibrator (OOF isotonic regression with PurgedKFold)
and a comprehensive cross-validation analysis that directly compares
raw vs calibrated performance on PWA and Brier score.
"""

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array, check_is_fitted

from ..cross_validation.scoring import probability_weighted_accuracy
from ..ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier


# ---- Calibration Methods ----

def fit_platt_scaling(
    y_calib: np.ndarray,
    scores_calib: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Fit Platt scaling calibration mapping.

    Platt scaling fits a logistic regression to map raw scores to calibrated probabilities.

    Args:
        y_calib: Calibration set true labels
        scores_calib: Calibration set predicted scores or probabilities
        C: Inverse of regularization strength
        solver: Optimization algorithm
        max_iter: Maximum number of iterations

    Returns:
        Fitted LogisticRegression calibrator
    """
    X_calib = np.asarray(scores_calib).reshape(-1, 1)
    platt = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    platt.fit(X_calib, y_calib, sample_weight=sample_weight)
    return platt


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


# ---- Core Calibration Metrics ----


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """
    Compute Brier score for probability predictions.

    The Brier score is the mean squared error between predicted probabilities
    and actual outcomes. Lower scores indicate better calibrated probabilities.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class

    Returns:
        Brier score (lower is better)

    Example:
        >>> y_true = np.array([0, 1, 0, 1])
        >>> p_pred = np.array([0.1, 0.9, 0.2, 0.8])
        >>> brier_score(y_true, p_pred)
        0.025
    """
    y_true = np.asarray(y_true).ravel()
    p_pred = np.asarray(p_pred).ravel()
    return float(brier_score_loss(y_true, p_pred))
    

def expected_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted probabilities
    and observed frequencies across probability bins.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        Expected Calibration Error (lower is better)

    Note:
        For financial applications, quantile bins are often preferred as they
        handle imbalanced probability distributions better.

    Example:
        >>> y_true = np.random.randint(0, 2, 1000)
        >>> p_pred = np.clip(y_true + np.random.normal(0, 0.2, 1000), 0, 1)
        >>> ece = expected_calibration_error(y_true, p_pred, n_bins=10)
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if y_true.shape != p_pred.shape:
        raise ValueError("y_true and p_pred must have the same shape")

    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bins[1:-1], right=True)
    ece_val = 0.0
    total_samples = len(p_pred)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = np.sum(mask)

        if bin_size == 0:
            continue

        # Average predicted probability in bin (confidence)
        mean_confidence = np.mean(p_pred[mask])
        # Observed frequency of positive class (accuracy)
        mean_accuracy = np.mean(y_true[mask])

        ece_val += (bin_size / total_samples) * abs(mean_accuracy - mean_confidence)

    return float(ece_val)


def maximum_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the worst-case absolute difference between predicted probabilities
    and observed frequencies across all probability bins.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        Maximum Calibration Error (lower is better)

    Example:
        >>> mce = maximum_calibration_error(y_true, p_pred, n_bins=10)
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if y_true.shape != p_pred.shape:
        raise ValueError("y_true and p_pred must have the same shape")

    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bins[1:-1], right=True)
    max_error = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx

        if not np.any(mask):
            continue

        mean_confidence = np.mean(p_pred[mask])
        mean_accuracy = np.mean(y_true[mask])
        bin_error = abs(mean_accuracy - mean_confidence)
        max_error = max(max_error, bin_error)

    return float(max_error)


# ---- Reliability Analysis and Visualization ----


def compute_reliability(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """
    Compute reliability curve data for calibration assessment.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        DataFrame with columns:
        - bin: Bin index
        - count: Number of samples in bin
        - pred_mean: Mean predicted probability in bin
        - true_frac: Observed frequency of positive class in bin
        - bin_center: Center point of the bin
        - bin_lower: Lower bound of bin
        - bin_upper: Upper bound of bin

    Example:
        >>> reliability_df = compute_reliability(y_true, p_pred, n_bins=10)
        >>> print(reliability_df[['bin_center', 'pred_mean', 'true_frac']])
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if strategy == "quantile":
        bin_edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
        bin_edges[0], bin_edges[-1] = 0.0, 1.0  # Ensure bounds
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bin_edges[1:-1], right=True)
    results = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask)
        bin_lower = bin_edges[bin_idx]
        bin_upper = bin_edges[bin_idx + 1]
        bin_center = (bin_lower + bin_upper) / 2

        if bin_count == 0:
            results.append(
                {
                    "bin": bin_idx,
                    "count": 0,
                    "pred_mean": np.nan,
                    "true_frac": np.nan,
                    "bin_center": bin_center,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                }
            )
        else:
            results.append(
                {
                    "bin": bin_idx,
                    "count": bin_count,
                    "pred_mean": float(np.mean(p_pred[mask])),
                    "true_frac": float(np.mean(y_true[mask])),
                    "bin_center": bin_center,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                }
            )

    return pd.DataFrame(results)


def plot_reliability(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Optional[plt.Axes] = None,
    show_perfect: bool = True,
    draw_hist: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Axes:
    """
    Plot reliability diagram (calibration curve).

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        strategy: "uniform" or "quantile" binning
        ax: Matplotlib axes to plot on (creates new if None)
        show_perfect: Whether to show perfect calibration line
        draw_hist: Whether to draw probability distribution histogram
        title: Plot title
        figsize: Figure size when creating new axes

    Returns:
        Matplotlib axes object

    Example:
        >>> ax = plot_reliability(y_true, p_pred, title="Model Calibration")
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    valid_mask = df["count"] > 0

    # Plot calibration curve
    ax.plot(
        df.loc[valid_mask, "pred_mean"],
        df.loc[valid_mask, "true_frac"],
        marker="o",
        markersize=6,
        linewidth=2,
        color="blue",
        label="Calibration Curve",
    )

    if show_perfect:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title, fontsize=14)

    # Add probability distribution histogram
    if draw_hist:
        hist_ax = ax.inset_axes([0.15, -0.25, 0.7, 0.15])
        hist_ax.hist(
            p_pred,
            bins=20,
            range=(0, 1),
            color="lightblue",
            edgecolor="black",
            alpha=0.7,
        )
        hist_ax.set_xlabel("Predicted Probability Distribution", fontsize=10)
        hist_ax.set_ylabel("Count", fontsize=10)
        hist_ax.set_xlim(0, 1)

    return ax


# ---- Bootstrap Confidence Intervals ----


def bootstrap_reliability_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    strategy: str = "uniform",
    random_state: Optional[int] = None,
    confidence_level: float = 0.95,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for reliability curve.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        n_bootstraps: Number of bootstrap samples
        strategy: "uniform" or "quantile" binning
        random_state: Random seed for reproducibility
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (base_dataframe, lower_bounds, upper_bounds)

    Example:
        >>> df, lower, upper = bootstrap_reliability_ci(y_true, p_pred)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    base_df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)

    # Bootstrap samples of true fractions
    boot_true_fracs = np.full((n_bootstraps, n_bins), np.nan)

    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, n_samples, size=n_samples)
        y_boot = y_true[indices]
        p_boot = p_pred[indices]

        df_boot = compute_reliability(y_boot, p_boot, n_bins=n_bins, strategy=strategy)
        boot_true_fracs[i, :] = df_boot["true_frac"].values

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower = np.nanpercentile(boot_true_fracs, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(boot_true_fracs, 100 * (1 - alpha / 2), axis=0)

    return base_df, lower, upper


def plot_reliability_with_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    strategy: str = "uniform",
    ax: Optional[plt.Axes] = None,
    random_state: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Axes:
    """
    Plot reliability diagram with bootstrap confidence intervals.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        n_bootstraps: Number of bootstrap samples
        strategy: "uniform" or "quantile" binning
        ax: Matplotlib axes to plot on
        random_state: Random seed for reproducibility
        title: Plot title
        figsize: Figure size when creating new axes

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    base_df, lower, upper = bootstrap_reliability_ci(
        y_true,
        p_pred,
        n_bins=n_bins,
        n_bootstraps=n_bootstraps,
        strategy=strategy,
        random_state=random_state,
    )

    valid_mask = base_df["count"] > 0

    # Plot calibration curve with confidence intervals
    ax.plot(
        base_df.loc[valid_mask, "pred_mean"],
        base_df.loc[valid_mask, "true_frac"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="Calibration Curve",
    )

    ax.fill_between(
        base_df.loc[valid_mask, "pred_mean"],
        lower[valid_mask],
        upper[valid_mask],
        color="blue",
        alpha=0.2,
        label="95% CI",
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title, fontsize=14)

    return ax


# ---- Comprehensive Calibration Report ----


def calibration_report(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    p_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generate comprehensive calibration assessment report.

    Args:
        y_true: True labels
        p_pred: Original predicted probabilities
        p_calibrated: Calibrated probabilities (optional)
        n_bins: Number of bins for analysis

    Returns:
        DataFrame with calibration metrics
    """
    metrics = {}

    # Original probabilities
    metrics["original_brier"] = brier_score(y_true, p_pred)
    metrics["original_ece"] = expected_calibration_error(y_true, p_pred, n_bins)
    metrics["original_mce"] = maximum_calibration_error(y_true, p_pred, n_bins)

    # Calibrated probabilities (if provided)
    if p_calibrated is not None:
        metrics["calibrated_brier"] = brier_score(y_true, p_calibrated)
        metrics["calibrated_ece"] = expected_calibration_error(
            y_true, p_calibrated, n_bins
        )
        metrics["calibrated_mce"] = maximum_calibration_error(
            y_true, p_calibrated, n_bins
        )
        metrics["brier_improvement"] = (
            metrics["original_brier"] - metrics["calibrated_brier"]
        )
        metrics["ece_improvement"] = metrics["original_ece"] - metrics["calibrated_ece"]

    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


# ===================================================================
# Public API definition
# ===================================================================
__all__ = [
    "CVIsotonicCalibrator",
    "analyze_calibrated_cross_val_scores",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "compute_reliability",
    "plot_reliability",
    "plot_reliability_with_ci",
    "calibration_report",
    "fit_platt_scaling",
    "apply_calibration",
]
