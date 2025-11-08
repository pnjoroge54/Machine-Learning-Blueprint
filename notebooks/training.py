import time

import pandas as pd
from sklearn.base import clone

from afml.cache import cv_cacheable, mlflow_cached


@mlflow_cached(tags={"model": "random_forest"})
def train_random_forest(classifier, X, y, sample_weight=None):
    clf = clone(classifier).set_params(oob_score=True)
    return clf.fit(X, y, sample_weight)
