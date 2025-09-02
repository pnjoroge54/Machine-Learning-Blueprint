from collections import namedtuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier

from afml.cross_validation.cross_validation import PurgedSplit
from afml.sample_weights.optimized_attribution import (
    get_weights_by_time_decay_optimized,
)


def train_meta_model(
    model,
    features,
    labels,
    test_size=0.3,
    sample_weights=False,
    weighting="return",
    time_decay=1,
    linear_decay=False,
) -> namedtuple:
    """
    Generate meta-labeling report for both tick and time bar features.

    Args:
        model_template: Classifier to be used for meta-labeling.
        feature: DataFrame containing features.
        labels: DataFrame containing label features.
        weighting: Method to use for sample weights

    Returns: namedtuple that contains fit, X_train, X_test, y_train, y_test, w_train, w_test, pred, prob
    """

    # Prepare features and target
    X = features.reindex(labels.index).replace([np.inf, -np.inf], np.nan).dropna()
    cont = labels.loc[X.index]
    X["side"] = cont["side"]
    y = cont["bin"]
    t1 = cont["t1"]

    # Assign sample weights
    if sample_weights:
        if "w" in cont:
            w = cont["w"]
        elif "t_value" in cont:
            w = cont["t_value"].abs()
            logger.info("Samples weighted by t-value.")
        else:
            w = pd.Series(1, index=y.index, dtype="float16", name="w")
            logger.info("Samples are equally weighted.")

        if weighting == "time":
            tw = get_weights_by_time_decay_optimized(
                triple_barrier_events=cont,
                close_series_index=features.index,
                decay=time_decay,
                linear=linear_decay,
                av_uniqueness=(cont[["tW"]] if "tW" in cont else None),
            )
            w *= tw  # Scale return weights by time
    else:
        logger.info("Samples are equally weighted.")
        w = pd.Series(1, index=y.index, dtype="float16", name="w")

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

    if isinstance(model, BaggingClassifier):
        # Set max_samples based on average uniqueness from the triple barrier
        try:
            av_uniqueness = cont["tW"].iloc[train].mean()
            logger.info(f"Average uniqueness: {av_uniqueness:.4f}")
            model.set_params(max_samples=av_uniqueness)
        except KeyError:
            model = clone(model)
            logger.info("Warning: 'tW' column not found in labels. Using default max_samples.")
    else:
        model = clone(model)  # Ensure independence of estimator instances

    logger.info(f"Training on {X_train.shape[0]:,} samples...")
    model.fit(X_train, y_train, sample_weight=w_train)

    # Make predictions
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    pred = pd.Series(pred, index=X_test.index, name="pred")
    prob = pd.Series(prob, index=X_test.index, name="prob")

    ModelData = namedtuple(
        "ModelData",
        [
            "fit",
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "w_train",
            "w_test",
            "pred",
            "prob",
        ],
    )
    return ModelData(model, X_train, X_test, y_train, y_test, w_train, w_test, pred, prob)
