"""
Probability Calibration Toolkit for Financial Machine Learning

This module provides comprehensive tools for calibrating classifier probabilities
and evaluating calibration quality, with special considerations for financial
time series data.

Key Features:
- Calibration metrics (Brier score, ECE, MCE)
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
"""

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

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
        hist_ax.hist(p_pred, bins=20, range=(0, 1), color="lightblue", edgecolor="black", alpha=0.7)
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


# ---- Calibration Methods ----


def fit_platt_scaling(
    y_calib: np.ndarray,
    scores_calib: np.ndarray,
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
    platt.fit(X_calib, y_calib)
    return platt


def fit_isotonic_calibration(y_calib: np.ndarray, scores_calib: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression calibration mapping.

    Isotonic regression fits a non-decreasing step function to map scores to probabilities.

    Args:
        y_calib: Calibration set true labels
        scores_calib: Calibration set predicted scores or probabilities

    Returns:
        Fitted IsotonicRegression calibrator
    """
    isotonic = IsotonicRegression(out_of_bounds="clip", increasing=True)
    isotonic.fit(scores_calib, y_calib)
    return isotonic


def apply_calibration(
    calibrator: Union[LogisticRegression, IsotonicRegression], scores: np.ndarray
) -> np.ndarray:
    """
    Apply fitted calibrator to new scores.

    Args:
        calibrator: Fitted Platt or isotonic calibrator
        scores: Raw scores to calibrate

    Returns:
        Calibrated probabilities
    """
    scores_array = np.asarray(scores)

    if isinstance(calibrator, LogisticRegression):
        # Platt scaling
        X_calib = scores_array.reshape(-1, 1)
        return calibrator.predict_proba(X_calib)[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        # Isotonic regression
        return calibrator.predict(scores_array)
    else:
        raise ValueError("Unsupported calibrator type")


# ---- Cross-Validation Integration ----


def oof_predict_proba(
    estimator: object,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: object,
    n_jobs: int = 1,
    method: str = "predict_proba",
) -> np.ndarray:
    """
    Generate out-of-fold probabilities using cross-validation.

    This function is designed to work with purged cross-validation to avoid
    data leakage in financial time series.

    Args:
        estimator: Sklearn-compatible classifier
        X: Feature matrix
        y: Target labels
        cv: Cross-validation generator (e.g., PurgedKFold)
        n_jobs: Number of parallel jobs
        method: Prediction method ('predict_proba' or 'decision_function')

    Returns:
        Out-of-fold probabilities for positive class

    Example:
        >>> from cross_validation import PurgedKFold
        >>> oof_probs = oof_predict_proba(clf, X, y, cv=PurgedKFold(n_splits=5, t1=t1))
    """
    # Use sklearn's cross_val_predict with purged CV
    predictions = cross_val_predict(estimator, X, y, cv=cv, method=method, n_jobs=n_jobs)

    # Handle different prediction formats
    if predictions.ndim == 2 and predictions.shape[1] >= 2:
        return predictions[:, 1]  # Positive class probabilities
    else:
        return predictions  # Decision function or binary predictions


def calibrate_with_oof(
    estimator: object,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: object,
    method: str = "isotonic",
    n_jobs: int = 1,
) -> Tuple[object, np.ndarray]:
    """
    Fit calibration mapping using out-of-fold predictions.

    This is the recommended approach for financial applications as it properly
    handles the temporal dependencies in the data.

    Args:
        estimator: Sklearn-compatible classifier
        X: Feature matrix
        y: Target labels
        cv: Cross-validation generator
        method: Calibration method ('isotonic' or 'sigmoid')
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (fitted_calibrator, oof_probabilities)

    Example:
        >>> calibrator, oof_probs = calibrate_with_oof(
        ...     clf, X, y, cv=purged_cv, method='isotonic'
        ... )
    """
    # Get out-of-fold probabilities
    oof_probs = oof_predict_proba(estimator, X, y, cv=cv, n_jobs=n_jobs)

    # Fit calibrator on OOF probabilities
    if method == "sigmoid":
        calibrator = fit_platt_scaling(y, oof_probs)
    elif method == "isotonic":
        calibrator = fit_isotonic_calibration(y, oof_probs)
    else:
        raise ValueError("Method must be 'sigmoid' or 'isotonic'")

    return calibrator, oof_probs


def calibrated_cross_val_predict(
    estimator: object,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: object,
    calibration_method: str = "isotonic",
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Generate calibrated out-of-fold probabilities using nested cross-validation.

    This implements the full calibration workflow within cross-validation,
    preventing data leakage in both model training and calibration.

    Args:
        estimator: Sklearn-compatible classifier
        X: Feature matrix
        y: Target labels
        cv: Cross-validation generator
        calibration_method: Calibration method ('isotonic' or 'sigmoid')
        n_jobs: Number of parallel jobs

    Returns:
        Calibrated out-of-fold probabilities

    Example:
        >>> cal_probs = calibrated_cross_val_predict(
        ...     clf, X, y, cv=purged_cv, calibration_method='isotonic'
        ... )
    """
    calibrated_predictions = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Clone estimator for this fold
        fold_estimator = clone(estimator)

        # Fit on training data
        fold_estimator.fit(X_train, y_train)

        # Get probabilities on test set
        test_probs = fold_estimator.predict_proba(X_test)[:, 1]

        # For calibration, we need to split training data further
        # into model training and calibration sets
        from sklearn.model_selection import train_test_split

        X_model, X_calib, y_model, y_calib = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )

        # Refit on model set
        fold_estimator.fit(X_model, y_model)

        # Get probabilities on calibration set
        calib_probs = fold_estimator.predict_proba(X_calib)[:, 1]

        # Fit calibrator
        if calibration_method == "sigmoid":
            calibrator = fit_platt_scaling(y_calib, calib_probs)
        else:
            calibrator = fit_isotonic_calibration(y_calib, calib_probs)

        # Apply calibration to test set
        calibrated_test_probs = apply_calibration(calibrator, test_probs)
        calibrated_predictions[test_idx] = calibrated_test_probs

    return calibrated_predictions


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
        metrics["calibrated_ece"] = expected_calibration_error(y_true, p_calibrated, n_bins)
        metrics["calibrated_mce"] = maximum_calibration_error(y_true, p_calibrated, n_bins)
        metrics["brier_improvement"] = metrics["original_brier"] - metrics["calibrated_brier"]
        metrics["ece_improvement"] = metrics["original_ece"] - metrics["calibrated_ece"]

    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})
