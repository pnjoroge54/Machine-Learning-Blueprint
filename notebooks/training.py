from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from afml.backtest_statistics.reporting import create_classification_report_image
from afml.cross_validation.cross_validation import PurgedSplit


def train_meta_model(model, features, labels, test_size=0.3) -> namedtuple:
    """
    Generate meta-labeling report for both tick and time bar features.

    Args:
        model_template: Classifier to be used for meta-labeling.
        X: DataFrame containing features.
        cont: DataFrame containing label features.

    Returns: namedtuple that contains fit, X_train, X_test, y_train, y_test, w_train, w_test, pred, prob
    """

    # Prepare features and target
    X = features.reindex(labels.index).replace([np.inf, -np.inf], np.nan).dropna()
    cont = labels.loc[X.index]
    X["side"] = cont["side"]
    y = cont["bin"]
    w = cont["w"] if "w" in cont else np.ones_like(y)  # sample weights
    t1 = cont["t1"]

    # Split data
    train, test = PurgedSplit(t1, test_size).split(X)
    X_train, X_test, y_train, y_test, w_train, w_test = (
        X.iloc[train],
        X.iloc[test],
        y.iloc[train],
        y.iloc[test],
        w[train],
        w[test],
    )

    if isinstance(model, BaggingClassifier):
        # Set max_samples based on average uniqueness from the triple barrier
        try:
            av_uniqueness = cont["tW"].iloc[train].mean()
            print(f"Average uniqueness: {av_uniqueness:.4f}")
            model.set_params(max_samples=av_uniqueness)
        except KeyError:
            model = clone(model)
            print("Warning: 'tW' column not found in labels. Using default max_samples.\n")
    else:
        model = clone(model)  # Ensure independence of estimator instances

    model.fit(X_train, y_train, sample_weight=w_train)

    # Make predictions
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    pred = pd.Series(pred, index=X_test.index, name="pred")
    prob = pd.Series(prob, index=X_test.index, name="prob")

    model_data = namedtuple(
        "ModelData",
        ["fit", "X_train", "X_test", "y_train", "y_test", "w_train", "w_test", "pred", "prob"],
    )
    return model_data(model, X_train, X_test, y_train, y_test, w_train, w_test, pred, prob)


def compare_roc_curves(model_data: List[namedtuple], titles: List[str] = None):
    n = len(model_data)
    fig, axes = plt.subplots(
        nrows=round(n / 2), ncols=2, sharex=True, sharey=True, figsize=(7.5, 5), dpi=100
    )
    axes = axes.flatten()
    if not titles:
        titles = [""] * n

    # Plot ROC curve
    for data, ax, title in zip(model_data, axes, titles):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(data.y_test, data.prob, sample_weight=data.w_test)
        auc = roc_auc_score(data.y_test, data.prob, sample_weight=data.w_test)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}")
        ax.legend()

    plt.tight_layout()
    plt.style.use("dark_background")
    return fig


def meta_labelling_classification_reports(model_data, title, output_filename):
    create_classification_report_image(
        y_true=model_data.y_test,
        y_pred=np.ones_like(model_data.pred),
        title=f"{title} Primary Model",
        output_filename=f"{output_filename}_primary_clf_report.png",
        display=False,
    )
    create_classification_report_image(
        y_true=model_data.y_test,
        y_pred=model_data.pred,
        title=f"{title} Meta-Model",
        output_filename=f"{output_filename}_meta_clf_report.png",
        display=False,
    )
    print("Classification reports saved.\n")
