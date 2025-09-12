from collections import namedtuple

import numpy as np
import pandas as pd
from feature_engine.selection import DropDuplicateFeatures
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier

from ..cross_validation.cross_validation import PurgedSplit
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized


def train_meta_model(
    model,
    features,
    labels,
    test_size=0.3,
    weighting=None,
    time_decay=1,
    linear_decay=False,
) -> namedtuple:
    """
    Generate meta-labeling report for both tick and time bar features.

    Args:
        model_template: Classifier to be used for meta-labeling.
        feature: DataFrame containing features.
        labels: DataFrame containing label features.
        weighting: Method to use for sample weights, default=None.
            - **"return"** to use return-weighted attribution
            - **"time** to use time-decay weighting


    Returns: namedtuple that contains fit, X_train, X_test, y_train, y_test, w_train, w_test, pred, prob
    """
    # Prepare features and target
    ddf = DropDuplicateFeatures()
    X = features.join(labels["side"]).replace([np.inf, -np.inf], np.nan).dropna()
    X = ddf.fit_transform(X)
    cont = labels.loc[X.index]
    y = cont["bin"]
    t1 = cont["t1"]
    w = pd.Series(1, index=y.index, dtype="float16", name="w")

    # Assign sample weights
    if weighting == "return":
        if "w" in cont:
            w = cont["w"]
        elif "t_value" in cont:
            w = cont["t_value"].abs()
            logger.info("Samples weighted by t-value.")
        else:
            logger.info("Samples are equally weighted.")

    elif weighting == "time":
        w = get_weights_by_time_decay_optimized(
            triple_barrier_events=cont,
            close_series_index=features.index,
            decay=time_decay,
            linear=linear_decay,
            av_uniqueness=(cont["tW"] if "tW" in cont else None),
        )
    else:
        logger.info("Samples are equally weighted.")

    # Split data
    train, test = PurgedSplit(t1, test_size).split(X)
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
        logger.info("Warning: 'tW' column not found in labels. Using default max_samples.")

    logger.info(f"Training on {X_train.shape[0]:,} samples...")
    fit = model.fit(X_train, y_train, sample_weight=w_train)

    # Make predictions
    pred = fit.predict(X_test)
    pred = pd.Series(pred, index=X_test.index, name="pred")

    prob = fit.predict_proba(X_test)[:, 1]
    prob = pd.Series(prob, index=X_test.index, name="prob")

    events = cont.iloc[test]  # Validation events data

    ModelData = namedtuple(
        "ModelData",
        [
            "fit",
            "X_test",
            "y_test",
            "w_test",
            "pred",
            "prob",
            "events",
        ],
    )

    return ModelData(fit, X_test, y_test, w_test, pred, prob, events)


# from typing import List, Tuple, Union


# def get_trend_scanning_meta_labels(
#     close: pd.Series,
#     side_prediction: pd.Series,
#     t_events: pd.DatetimeIndex,
#     span: Union[List[int], Tuple[int, int]] = (5, 20),
#     volatility_threshold: float = 0.1,
#     use_log: bool = True,
#     verbose: bool = False,
# ) -> pd.DataFrame:
#     """
#     Generate meta-labels using trend-scanning labels and existing side predictions.

#     Parameters
#     ----------
#     close : pd.Series
#         Time-indexed price series.
#     side_prediction : pd.Series
#         Primary model's side predictions (1 for long, -1 for short).
#     t_events : pd.DatetimeIndex
#         Index of event start times to align with trend events.
#     span : Union[List[int], Tuple[int, int]], default=(5, 20)
#         Window span for trend scanning.
#     volatility_threshold : float, default=0.1
#         Volatility threshold for trend scanning.
#     use_log : bool, default=True
#         Use log prices for trend scanning.
#     verbose : bool, default=False
#         Verbosity flag.

#     Returns
#     -------
#     pd.DataFrame
#         Meta-labeled events with columns:
#         - t1: End time of trend
#         - ret: Return adjusted for side prediction
#         - bin: Meta-label (1 if correct prediction, 0 otherwise)
#         - side: Original side prediction
#     """
#     # Generate trend-scanning labels
#     trend_events = trend_scanning_labels(
#         close, span, volatility_threshold, use_log=use_log, verbose=verbose
#     )
#     print(f"Trend-Scanning (σ = {volatility_threshold})")
#     value_counts_data(trend_events.bin, verbose=True)
#     trend_events["side"] = side_prediction.reindex(trend_events.index)

#     # Align with t_events
#     trend_events = trend_events.reindex(t_events.intersection(trend_events.index))

#     # Apply meta-labeling logic
#     # trend_events["ret"] *= trend_events["side"]  # Adjust returns for side
#     trend_events["bin"] = np.where((trend_events["side"] == trend_events["bin"]), 1, 0).astype(
#         "int8"
#     )

#     return trend_events
