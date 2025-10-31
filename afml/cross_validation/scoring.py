import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels


def probability_weighted_accuracy(y_true, y_pred, sample_weight=None, labels=None, eps=1e-15):
    """
    Calculates the Probability-Weighted Accuracy (PWA) score.

    PWA is a confidence-weighted accuracy that penalizes high-confidence
    mistakes more severely. This version is compatible with sklearn
    conventions: it accepts a `labels` argument to fix the class order,
    applies probability clipping, and supports sample weights.

    Args:
        y_true (array-like): True class labels, shape (n_samples,).
        y_pred (array-like or DataFrame): Predicted probabilities,
            shape (n_samples, n_classes). If DataFrame, columns must be
            class labels.
        sample_weight (array-like, optional): Per-sample weights.
        labels (array-like, optional): List of all expected class labels
            (in the order corresponding to columns of y_pred).
        eps (float): Small value to clip probabilities into [eps, 1 - eps].

    Returns:
        float: PWA score between 0 and 1.
    """
    # 1) Convert inputs to numpy arrays (or reorder DataFrame)
    y_true = np.asarray(y_true)
    if isinstance(y_pred, pd.DataFrame):
        # If labels given, reorder columns; otherwise infer column order
        cols = labels if labels is not None else y_pred.columns.tolist()
        y_pred = y_pred[cols].to_numpy()
    else:
        y_pred = np.asarray(y_pred)

    # 2) Clip probabilities to avoid zeros or ones
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3) Determine class list and validate
    if labels is not None:
        classes = np.asarray(labels)
    else:
        # Infer classes from y_true (sorted)
        classes = unique_labels(y_true)
    n_classes = classes.shape[0]

    # 4) Handle binary case where y_pred might be 1D
    if y_pred.ndim == 1:
        # Interpret as probability of class classes[1]
        y_pred = np.vstack([1 - y_pred, y_pred]).T
        n_classes = 2

    # 5) Shape checks
    if y_pred.ndim != 2 or y_pred.shape[1] != n_classes:
        raise ValueError(
            f"y_pred must be shape (n_samples, n_classes={n_classes}), " f"but got {y_pred.shape}"
        )

    if not np.all(np.isin(y_true, classes)):
        missing = set(y_true) - set(classes)
        raise ValueError(f"y_true contains labels not in `labels`: {missing}")

    # 6) Prepare sample weights
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError("sample_weight must have same length as y_true")

    # 7) Predicted class index and its probability
    pred_idx = np.argmax(y_pred, axis=1)
    p_n = y_pred[np.arange(len(y_true)), pred_idx]

    # 8) Correctness indicator y_n ∈ {0,1}
    #    Map y_true labels to indices in `classes`
    label_to_index = {c: i for i, c in enumerate(classes)}
    true_idx = np.vectorize(label_to_index.get)(y_true)
    y_n = (pred_idx == true_idx).astype(int)

    # 9) Confidence weights: p_n – (1/K)
    baseline = 1.0 / n_classes
    conf_w = p_n - baseline

    # 10) Compute numerator and denominator with sample weights
    numerator = np.sum(sample_weight * y_n * conf_w)
    denominator = np.sum(sample_weight * conf_w)

    # 11) Edge case: no confidence (all p_n == 1/K)
    if np.isclose(denominator, 0.0):
        return 0.5  # random-guess baseline

    # 12) Final PWA score
    return numerator / denominator
