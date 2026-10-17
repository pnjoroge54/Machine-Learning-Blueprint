"""Synthetic datasets shared by the tests."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def get_classification_data(n_features=100, n_informative=25, n_redundant=25, n_samples=10000,
                            random_state=0, sigma=0.0):
    """
    Classification data with informative (``I_``), redundant (``R_``) and noise
    (``N_``) columns (AFML snippet 8.7).

    Redundant features are informative features plus Gaussian noise of scale
    ``sigma``, so feature-importance and clustering tests know which columns
    should rank where.

    Returns:
        (pd.DataFrame, pd.Series): features and labels, indexed by business days.
    """
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
    )
    cols = [f"I_{i}" for i in range(n_informative)]
    cols += [f"N_{i}" for i in range(n_features - n_informative - n_redundant)]
    index = pd.bdate_range(end=pd.Timestamp("2020-01-01"), periods=n_samples)
    X = pd.DataFrame(X, columns=cols, index=index)
    y = pd.Series(y, index=index)
    picks = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(picks):
        X[f"R_{k}"] = X[f"I_{j}"] + np.random.normal(size=X.shape[0]) * sigma
    return X, y
