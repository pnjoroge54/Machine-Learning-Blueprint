import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..cross_validation.scoring import probability_weighted_accuracy
from ..util.misc import indices_to_mask


def compute_custom_oob_metrics(clf, X, y, sample_weight=None):
    """
    Compute custom OOB metrics for both RandomForestClassifier and BaggingClassifier.
    """

    # Input validation
    if not hasattr(clf, "estimators_") or len(clf.estimators_) == 0:
        raise ValueError("Classifier must be fitted before computing OOB metrics")

    # Convert to arrays if pandas objects
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values
    if sample_weight is not None and hasattr(sample_weight, "values"):
        sample_weight = sample_weight.values

    n_samples = y.shape[0]
    n_classes = clf.n_classes_

    # Handle different classifier types
    if isinstance(clf, RandomForestClassifier):
        # Use RandomForest's built-in OOB
        return _compute_rf_oob_metrics(clf, X, y, sample_weight, n_samples, n_classes)
    else:
        # Use BaggingClassifier approach
        return _compute_bagging_oob_metrics(clf, X, y, sample_weight, n_samples, n_classes)


def _compute_rf_oob_metrics(clf, X, y, sample_weight, n_samples, n_classes):
    """Compute OOB metrics for RandomForestClassifier using built-in method."""

    # Use RandomForest's built-in OOB predictions
    if not hasattr(clf, "oob_decision_function_"):
        raise ValueError("RandomForestClassifier must have oob_score=True for OOB metrics")

    oob_proba = clf.oob_decision_function_
    oob_pred = np.argmax(oob_proba, axis=1)

    # Create mask for samples that have OOB predictions
    oob_mask = ~np.any(np.isnan(oob_proba), axis=1)

    if not np.any(oob_mask):
        warnings.warn("No out-of-bag samples found in RandomForest.")
        return _get_default_metrics()

    # Prepare data for metrics
    y_oob = y[oob_mask]
    pred_oob = oob_pred[oob_mask]
    proba_oob = oob_proba[oob_mask]

    if sample_weight is not None:
        sample_weight_oob = sample_weight[oob_mask]
    else:
        sample_weight_oob = None

    return _compute_metrics(
        y_oob, pred_oob, proba_oob, sample_weight_oob, clf.classes_, oob_mask, n_samples
    )


def _compute_bagging_oob_metrics(clf, X, y, sample_weight, n_samples, n_classes):
    """Compute OOB metrics for BaggingClassifier with custom implementation."""

    # Check for required attributes
    if not hasattr(clf, "_estimators_samples") and not hasattr(clf, "estimators_samples_"):
        raise ValueError("BaggingClassifier must have sample indices available for OOB computation")

    # Get sample indices
    if hasattr(clf, "_estimators_samples") and clf._estimators_samples:
        estimators_samples = clf._estimators_samples
    else:
        estimators_samples = clf.estimators_samples_

    # Check for feature indices (may not exist in all BaggingClassifiers)
    if hasattr(clf, "estimators_features_"):
        estimators_features = clf.estimators_features_
    else:
        # If no feature indices, assume all features are used
        estimators_features = [None] * len(clf.estimators_)

    # Accumulate OOB predictions
    oob_proba = np.zeros((n_samples, n_classes))
    oob_count = np.zeros(n_samples)

    for i, (estimator, samples) in enumerate(zip(clf.estimators_, estimators_samples)):
        mask = ~indices_to_mask(samples, n_samples)

        if np.any(mask):
            X_oob = X[mask]
            features = estimators_features[i]

            # Handle feature subsetting if available
            if features is not None and len(features) > 0:
                X_oob = X_oob[:, features]

            try:
                proba = estimator.predict_proba(X_oob)
                oob_proba[mask] += proba
                oob_count[mask] += 1
            except (AttributeError, NotImplementedError):
                # Fallback: use hard voting if predict_proba not available
                pred = estimator.predict(X_oob)
                proba = np.eye(n_classes)[pred]
                oob_proba[mask] += proba
                oob_count[mask] += 1

    # Handle samples with no OOB predictions
    oob_mask = oob_count > 0
    if not np.any(oob_mask):
        warnings.warn("No out-of-bag samples found in BaggingClassifier.")
        return _get_default_metrics()

    # Average probabilities
    oob_proba[oob_mask] /= oob_count[oob_mask, np.newaxis]
    oob_pred = np.argmax(oob_proba, axis=1)

    # Prepare data for metrics
    y_oob = y[oob_mask]
    pred_oob = oob_pred[oob_mask]
    proba_oob = oob_proba[oob_mask]

    if sample_weight is not None:
        sample_weight_oob = sample_weight[oob_mask]
    else:
        sample_weight_oob = None

    return _compute_metrics(
        y_oob, pred_oob, proba_oob, sample_weight_oob, clf.classes_, oob_mask, n_samples
    )


def _compute_metrics(y_oob, pred_oob, proba_oob, sample_weight_oob, classes, oob_mask, n_samples):
    """Compute metrics from OOB predictions."""

    metrics = {
        "f1": f1_score(y_oob, pred_oob, average="weighted", sample_weight=sample_weight_oob),
        "precision": precision_score(
            y_oob, pred_oob, average="weighted", sample_weight=sample_weight_oob
        ),
        "recall": recall_score(
            y_oob, pred_oob, average="weighted", sample_weight=sample_weight_oob
        ),
        "pwa": probability_weighted_accuracy(y_oob, proba_oob, labels=classes),
        "neg_log_loss": -log_loss(
            y_oob, proba_oob, labels=classes, sample_weight=sample_weight_oob
        ),
        "accuracy": accuracy_score(y_oob, pred_oob, sample_weight=sample_weight_oob),
        "coverage": oob_mask.sum() / n_samples,
    }

    # Add AUC for binary classification
    if len(classes) == 2:
        try:
            metrics["auc"] = roc_auc_score(y_oob, proba_oob[:, 1], sample_weight=sample_weight_oob)
        except ValueError:
            metrics["auc"] = 0.5

    return metrics


def _get_default_metrics():
    """Return default metrics when no OOB samples are available."""
    return {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pwa": 0.0,
        "neg_log_loss": -np.inf,
        "accuracy": 0.0,
        "coverage": 0.0,
        "auc": 0.5,
    }


# Check ensemble memory footprint
def estimate_ensemble_size(clf):
    """Estimate memory usage of fitted ensemble."""
    total_bytes = 0

    # Estimators
    for est in clf.estimators_:
        total_bytes += sys.getsizeof(est)

    # Sample indices
    for samples in clf.estimators_samples_:
        total_bytes += sys.getsizeof(samples)

    # Feature indices
    if clf.estimators_features_ is not None:
        for features in clf.estimators_features_:
            total_bytes += sys.getsizeof(features)

    return total_bytes / (1024**2)  # Convert to MB
