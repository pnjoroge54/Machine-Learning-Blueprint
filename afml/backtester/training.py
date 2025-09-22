from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from feature_engine.selection import DropDuplicateFeatures
from loguru import logger
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, precision_recall_curve

from afml.cache.selective_cleaner import smart_cacheable

from ..cross_validation.cross_validation import PurgedSplit
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized


@dataclass
class ModelData:
    """
    Container for storing trained model, test data, predictions, probabilities, and event metadata

    :var fit: Trained scikit-learn estimator
    :vartype fit: BaseEstimator
    :var X_test: Test feature data as a DataFrame
    :var y_test: True labels for the test set
    :var sample_weight: Sample weights for test data
    :var pred: Model predictions for the test set
    :var prob: Predicted probabilities for the positive class
    :var events: Event metadata associated with test samples
    :var columns: Feature columns used in training
    """

    fit: BaseEstimator
    X_test: pd.DataFrame
    y_test: pd.Series
    sample_weight: pd.Series
    pred: pd.Series
    prob: pd.Series
    events: pd.DataFrame
    columns: pd.Index


def train_model(
    model: BaseEstimator,
    events: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series = None,
    test_size: float = 0.3,
    weighting: Union[str, None] = None,
    time_decay: float = 1,
    linear_decay: bool = False,
) -> ModelData:
    """
    Trains a meta-model using labeled financial events and returns structured model output.

    Args:
        model (BaggingClassifier): A scikit-learn compatible classifier.
        features (pd.DataFrame): Feature matrix indexed by timestamps.
        events (pd.DataFrame): Labeled events with required columns depending on labeling method.
        test_size (float, optional): Fraction of data to reserve for testing. Defaults to 0.3.
        weighting (str or None, optional): Sample weighting method. Options:
            - "return": Use return-based or t-value-based weights.
            - "time": Use time-decay weighting.
            - None: Equal weights. Defaults to None.
        time_decay (float, optional): Decay factor for time-based weighting. Defaults to 1.
        linear_decay (bool, optional): If True, applies linear decay instead of exponential. Defaults to False.

    Returns:
        ModelData: A dataclass containing model fit, test data, predictions, probabilities, and event metadata.
    """
    # Prepare features and target
    data_index = X.index
    X = X.reindex(events.index).replace([np.inf, -np.inf], np.nan).dropna()
    cont = events.loc[X.index]
    y = cont["bin"] if y is None else y
    w = pd.Series(1, index=y.index, name="w")  # Sample weights
    a, b = data_index.get_indexer([cont.index[0], cont["t1"].max()])
    data_index = data_index[a : b + 1]

    # Assign sample weights
    if weighting == "return" and "w" in cont:
        w = cont["w"]
        logger.info("Samples weighted by return attribution.")
    elif weighting == "t_value" and "t_value" in cont:
        w = cont["t_value"].abs()
        logger.info("Samples weighted by t_value.")
    elif weighting == "time":
        w = get_weights_by_time_decay_optimized(
            triple_barrier_events=cont,
            close_series_index=data_index,
            decay=time_decay,
            linear=linear_decay,
            av_uniqueness=(cont["tW"] if "tW" in cont else None),
        )
    else:
        logger.info("Samples are equally weighted.")

    # Split data
    train, test = PurgedSplit(cont["t1"], test_size).split(X)
    X_train, X_test, y_train, y_test, w_train, w_test = (
        X.iloc[train],
        X.iloc[test],
        y.iloc[train],
        y.iloc[test],
        w.iloc[train],
        w.iloc[test],
    )

    # Only clone when we actually need to modify parameters
    if isinstance(model, BaggingClassifier) and "tW" in cont:
        av_uniqueness = cont["tW"].iloc[train].mean()
        logger.info(f"Average uniqueness: {av_uniqueness:.4f}")
        # Clone before modifying to preserve original
        model = clone(model)
        model.set_params(max_samples=av_uniqueness)
    elif isinstance(model, BaggingClassifier):
        logger.warning("Warning: 'tW' column not found in labels. Using default max_samples.")

    logger.info(f"Training on {X_train.shape[0]:,} samples...")
    fit = model.fit(X_train, y_train, sample_weight=w_train)

    # Make predictions
    pred = fit.predict(X_test)
    pred = pd.Series(pred, index=X_test.index, name="pred")

    prob = fit.predict_proba(X_test)[:, 1]
    prob = pd.Series(prob, index=X_test.index, name="prob")

    # Validation events data
    events = cont.iloc[test]

    return ModelData(fit, X_test, y_test, w_test, pred, prob, events, X.columns)


def train_model_with_trend(
    model: BaseEstimator,
    ts_model_data: ModelData,
    X: pd.DataFrame,
    events: pd.DataFrame,
    test_size: float = 0.3,
    weighting: Union[str, None] = None,
    time_decay: float = 1,
    linear_decay: bool = False,
    agreement_only: bool = True,
) -> ModelData:
    """
    Trains a meta-model using BOTH trend-scanning and triple-barrier labels.

    Args:
        model (BaseEstimator): A scikit-learn compatible classifier.
        X (pd.DataFrame): Feature matrix indexed by timestamps.
        events (pd.DataFrame): Must contain:
            - 'bin': triple-barrier label (+1/-1/0 or binary)
            - 'ts_bin': trend-scanning label (+1/-1/0)
            - 't1': event end times
            - Optional: 'w' (sample weights), 'tW' (average uniqueness)
        test_size (float): Fraction of data for testing.
        weighting (str or None): 'return', 'time', or None.
        time_decay (float): Decay factor for time-based weighting.
        linear_decay (bool): Use linear instead of exponential decay.
        agreement_only (bool): If True, only keep samples where bin == ts_bin != 0.

    Returns:
        ModelData: Dataclass with fit, test data, predictions, probabilities, and event` metadata.
    """
    # Get index from features before modification
    if weighting == "time":
        data_index = X.index

    # --- Prepare features and target ---
    ts_events = ts_model_data.events  # Events used to train trend-scanning model
    t_events = events.index.intersection(ts_events.index)
    X = X.reindex(t_events).replace([np.inf, -np.inf], np.nan).dropna()
    cont = events.loc[X.index]

    ts_fit = ts_model_data.fit  # Trained trend-scanning model
    X["ts_bin"] = ts_fit.predict(X[ts_model_data.columns])
    X["ts_prob"] = ts_fit.predict_proba(X[ts_model_data.columns])[:, 1]

    if agreement_only:
        mask = cont["side"] == X["ts_bin"]
        cont = cont.loc[mask]
        X = X.loc[mask]

    # Combined meta-label: could be agreement, or any custom logic
    y = cont["bin"]  # or pd.Series(np.sign(cont["tb_bin"] + cont["ts_bin"]), index=cont.index)

    # --- Sample weights ---
    w = pd.Series(1, index=y.index, name="w")
    if weighting == "return" and "w" in cont:
        w = cont["w"]
        logger.info("Samples weighted by return attribution.")
    elif weighting == "time":
        w = get_weights_by_time_decay_optimized(
            triple_barrier_events=cont,
            close_series_index=data_index,
            decay=time_decay,
            linear=linear_decay,
            av_uniqueness=(cont["tW"] if "tW" in cont else None),
        )
    else:
        logger.info("Samples are equally weighted.")

    # --- Purged split ---
    train, test = PurgedSplit(cont["t1"], test_size).split(X)
    X_train, X_test, y_train, y_test, w_train, w_test = (
        X.iloc[train],
        X.iloc[test],
        y.iloc[train],
        y.iloc[test],
        w.iloc[train],
        w.iloc[test],
    )

    # New instance of model to prevent errors from expected column names
    model = clone(model)

    # --- Bagging uniqueness adjustment ---
    if isinstance(model, BaggingClassifier) and "tW" in cont:
        av_uniqueness = cont["tW"].iloc[train].mean()
        model.set_params(max_samples=av_uniqueness)

    # --- Fit ---
    logger.info(f"Training on {X_train.shape[0]:,} samples...")
    fit = model.fit(X_train, y_train, sample_weight=w_train)

    # --- Predict ---
    pred = pd.Series(fit.predict(X_test), index=X_test.index, name="pred")
    prob = pd.Series(fit.predict_proba(X_test)[:, 1], index=X_test.index, name="prob")

    events = cont.iloc[test]

    return ModelData(fit, X_test, y_test, w_test, pred, prob, events, X.columns)


def get_optimal_threshold(model_data: ModelData) -> dict:
    """
    Computes the optimal classification threshold that maximizes F1-score.

    Args:
        model_data (ModelData): Output from `train_model`, containing:
            - y_test (pd.Series): True binary labels.
            - prob (pd.Series): Predicted probabilities.
            - sample_weight (pd.Series): Sample weights for test set.

    Returns:
        dict: Dictionary with optimal threshold and performance metrics:
            - threshold (float): Optimal probability threshold.
            - f1_score (float): Best F1-score.
            - precision (float): Precision at optimal threshold.
            - recall (float): Recall at optimal threshold.
            - retained_trades (int): Number of trades above threshold.
            - total_trades (int): Total number of test trades.
    """

    y_true = model_data.y_test
    prob = model_data.prob
    sample_weight = getattr(model_data, "sample_weight", None)

    precision, recall, thresholds = precision_recall_curve(
        y_true, prob, sample_weight=sample_weight
    )

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    retained = int(np.sum(prob >= best_threshold))
    total = len(prob)

    logger.info(
        f"Optimal threshold: {best_threshold:.4f} | F1: {best_f1:.4f} | "
        f"Precision: {best_precision:.4f} | Recall: {best_recall:.4f} | "
        f"Trades retained: {retained}/{total}"
    )

    return {
        "threshold": round(best_threshold, 4),
        "f1_score": round(best_f1, 4),
        "precision": round(best_precision, 4),
        "recall": round(best_recall, 4),
        "retained_trades": retained,
        "total_trades": total,
    }

    #     # --- Default threshold metrics ---
    # y_pred_default = (data.prob >= 0.5).astype(int)
    # default_precision = precision_score(data.y_test, y_pred_default, zero_division=0)
    # default_recall = recall_score(data.y_test, y_pred_default, zero_division=0)
    # default_f1 = f1_score(data.y_test, y_pred_default, zero_division=0)

    # # --- Tuned threshold metrics ---
    # y_pred_tuned = (data.prob >= best_threshold).astype(int)
    # tuned_precision = precision_score(data.y_test, y_pred_tuned, zero_division=0)
    # tuned_recall = recall_score(data.y_test, y_pred_tuned, zero_division=0)
    # tuned_f1 = f1_score(data.y_test, y_pred_tuned, zero_division=0)

    # summary_rows.append(
    #     {
    #         "Model": name,
    #         "ROC AUC": auc,
    #         "AP": ap,
    #         "Default Threshold": 0.5,
    #         "Default Precision": default_precision,
    #         "Default Recall": default_recall,
    #         "Default F1": default_f1,
    #         "Tuned Threshold": best_threshold,
    #         "Tuned Precision": tuned_precision,
    #         "Tuned Recall": tuned_recall,
    #         "Tuned F1": tuned_f1,
    #     }
    # )
