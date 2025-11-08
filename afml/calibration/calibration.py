from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

# ---- Core metrics ----


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Return Brier score (mean squared error for probabilities)."""
    return float(brier_score_loss(y_true, p_pred))


def expected_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    strategy: "uniform" (equal-width bins) or "quantile" (equal-count bins).
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    assert y_true.shape == p_pred.shape
    n = len(p_pred)
    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p_pred, bins[1:-1], right=True)
    ece_val = 0.0
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        bin_size = mask.sum()
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece_val += (bin_size / n) * abs(acc - conf)
    return float(ece_val)


def maximum_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Maximum Calibration Error (largest absolute gap across bins)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    assert y_true.shape == p_pred.shape
    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p_pred, bins[1:-1], right=True)
    gaps = []
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        gaps.append(abs(acc - conf))
    return float(max(gaps) if gaps else 0.0)


# ---- Calibration curve and plotting ----


def compute_reliability(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """
    Return a DataFrame with bin centers, mean predicted prob (confidence),
    observed positive fraction (accuracy), and counts.
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    assert y_true.shape == p_pred.shape
    if strategy == "quantile":
        bin_edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
        # ensure monotonic edges
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p_pred, bin_edges[1:-1], right=True)
    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "count": 0,
                    "pred_mean": np.nan,
                    "true_frac": np.nan,
                    "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                }
            )
        else:
            rows.append(
                {
                    "bin": i,
                    "count": count,
                    "pred_mean": float(p_pred[mask].mean()),
                    "true_frac": float(y_true[mask].mean()),
                    "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                }
            )
    return pd.DataFrame(rows)


def plot_reliability(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Optional[plt.Axes] = None,
    show_perfect: bool = True,
    draw_hist: bool = True,
    title: Optional[str] = None,
):
    """
    Plot reliability diagram (calibration curve) with optional histogram of predictions.
    """
    df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    # plot model curve (line through bin centers where count>0)
    valid = df["count"] > 0
    ax.plot(
        df.loc[valid, "pred_mean"],
        df.loc[valid, "true_frac"],
        marker="o",
        color="cyan",
        label="Model",
    )
    if show_perfect:
        ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Frequency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    if title:
        ax.set_title(title)
    # optional histogram below
    if draw_hist:
        ax_hist = ax.inset_axes([0.15, -0.25, 0.7, 0.15])
        ax_hist.hist(p_pred, bins=20, range=(0, 1), color="lightgray", edgecolor="k")
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])
        ax_hist.set_xlabel("Predicted probability distribution")
    return ax


# ---- Bootstrap CI for reliability curve ----


def bootstrap_reliability_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 200,
    strategy: str = "uniform",
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Bootstrap to estimate confidence intervals for the true_frac by bin.
    Returns:
      - base_df: DataFrame from compute_reliability
      - lower: array of lower CI for each bin (nan where count==0)
      - upper: array of upper CI for each bin
    """
    rng = np.random.RandomState(random_state)
    n = len(p_pred)
    base_df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    boot_true_fracs = np.full((n_bootstraps, n_bins), np.nan)
    for b in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)  # bootstrap sample with replacement
        df_b = compute_reliability(y_true[idx], p_pred[idx], n_bins=n_bins, strategy=strategy)
        boot_true_fracs[b, :] = df_b["true_frac"].values
    lower = np.nanpercentile(boot_true_fracs, 2.5, axis=0)
    upper = np.nanpercentile(boot_true_fracs, 97.5, axis=0)
    return base_df, lower, upper


def plot_reliability_with_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 200,
    strategy: str = "uniform",
    ax: Optional[plt.Axes] = None,
    random_state: Optional[int] = None,
    title: Optional[str] = None,
):
    """Plot calibration curve with bootstrap 95% CIs around observed true fraction per bin."""
    base_df, lower, upper = bootstrap_reliability_ci(
        y_true,
        p_pred,
        n_bins=n_bins,
        n_bootstraps=n_bootstraps,
        strategy=strategy,
        random_state=random_state,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    valid = base_df["count"] > 0
    ax.plot(
        base_df.loc[valid, "pred_mean"],
        base_df.loc[valid, "true_frac"],
        marker="o",
        color="cyan",
        label="Model",
    )
    ax.fill_between(
        base_df.loc[valid, "pred_mean"], lower[valid], upper[valid], color="cyan", alpha=0.15
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Frequency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


# ---- Calibration fitting: Platt (sigmoid) and Isotonic ----


def fit_platt_sigmoid(
    y_cal: np.ndarray,
    scores_cal: np.ndarray,
    C: float = 1e6,
    solver: str = "lbfgs",
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Fit Platt scaling by training a logistic regression on raw scores.
    Returns fitted sklearn LogisticRegression model that maps score -> probability.
    Note: supply raw scores (decision function or predict_proba[:,1]) as 1D array.
    """
    X = np.asarray(scores_cal).reshape(-1, 1)
    lr = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    lr.fit(X, y_cal)
    return lr


def fit_isotonic(y_cal: np.ndarray, scores_cal: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression mapping scores -> probabilities (monotonic nonparametric).
    Returns fitted IsotonicRegression (use .predict(X) with X as 1D array).
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores_cal, y_cal)
    return ir


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    """
    Apply a fitted calibrator (LogisticRegression or IsotonicRegression) to scores.
    Returns probability array.
    """
    scores = (
        np.asarray(scores).reshape(-1, 1)
        if hasattr(calibrator, "predict_proba")
        else np.asarray(scores)
    )
    # logistic sklearn model:
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(scores)[:, 1]
    # isotonic
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(np.asarray(scores).ravel())
    # fallback: attempt predict_proba
    try:
        return calibrator.predict_proba(scores)[:, 1]
    except Exception:
        raise ValueError(
            "Calibrator not recognized. Provide fitted LogisticRegression or IsotonicRegression."
        )


# ---- Cross-validated out-of-fold calibration helpers ----


def oof_predict_proba(
    estimator, X, y, cv, n_jobs: int = 1, method: str = "predict_proba"
) -> np.ndarray:
    """
    Compute out-of-fold (OOF) predicted probabilities aligned with X/y order using cross_val_predict.
    Returns probability for positive class (1D array).
    """
    # cross_val_predict returns in-order predictions aligned to X
    probs = cross_val_predict(estimator, X, y, cv=cv, method=method, n_jobs=n_jobs)
    # if predict_proba returned shape (n,2), take positive column
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    return probs


def calibrate_with_oof(
    estimator, X, y, cv, method: str = "sigmoid", n_jobs: int = 1
) -> Tuple[object, np.ndarray]:
    """
    Fit calibration (Platt or isotonic) using out-of-fold predictions:
      - estimator: base classifier (unfitted or should support fit in cross_val_predict)
      - cv: cross-validation generator or int
      - method: "sigmoid" (Platt) or "isotonic"
    Returns (fitted_calibrator, oof_probs) where fitted_calibrator is a sklearn object that maps score->prob.
    """
    # 1) get OOF probabilities (raw)
    oof_probs = oof_predict_proba(estimator, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba")
    # Use positive-class probabilities as "scores" to calibrate (could use decision_function if available)
    scores = oof_probs
    # 2) fit calibrator on the OOF predictions vs true labels
    if method == "sigmoid":
        calibrator = fit_platt_sigmoid(y, scores)
    elif method == "isotonic":
        calibrator = fit_isotonic(y, scores)
    else:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    return calibrator, scores


def calibrated_predict_proba(
    estimator, X_train, y_train, X_test, cv, method="sigmoid", n_jobs: int = 1
) -> np.ndarray:
    """
    High-level convenience: compute a calibrated probability on X_test by:
      - getting OOF predictions for X_train,
      - fitting calibrator,
      - fitting base estimator on full X_train,
      - predicting raw scores on X_test and applying calibrator.
    Returns calibrated probabilities for X_test.
    """
    calibrator, _ = calibrate_with_oof(
        estimator, X_train, y_train, cv=cv, method=method, n_jobs=n_jobs
    )
    # fit base estimator on full training set
    estimator.fit(X_train, y_train)
    # obtain raw scores on X_test (prefer predict_proba)
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X_test)
    else:
        # fallback: use predict (not probabilistic)
        raw = estimator.predict(X_test)
    return apply_calibrator(calibrator, raw)


# ---- Sklearn's CalibratedClassifierCV wrapper usage ----


def calibrate_with_sklearn(
    estimator, X_train, y_train, cv=5, method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """
    Return a fitted CalibratedClassifierCV that wraps estimator and performs internal CV calibration.
    Call .predict_proba(X) on the returned object.
    """
    cal = CalibratedClassifierCV(base_estimator=estimator, cv=cv, method=method)
    cal.fit(X_train, y_train)
    return cal
