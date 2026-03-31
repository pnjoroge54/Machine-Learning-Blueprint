import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized


class _WeightedEstimator(BaseEstimator, ClassifierMixin):
    """
    A transparency wrapper for scikit-learn estimators to support AFML weights.

    Technical Constraints:
    - Required for seamless integration with Scikit-learn's Pipeline and
      GridSearchCV/Optuna, which do not always pass 'sample_weight' through
      standard 'fit' calls in complex nested structures.
    - Manages the internal application of 'Time Decay' and 'Uniqueness'
      weights at the moment of training.
    """

    def __init__(
        self,
        base_estimator,
        events,
        data_index,
        scheme="unweighted",
        decay=1.0,
        linear=True,
        **params,
    ):
        from sklearn.utils.validation import has_fit_parameter

        if not has_fit_parameter(base_estimator, "sample_weight"):
            raise TypeError("The base estimator must accept sample_weight.")

        self.base_estimator = base_estimator
        self.base_estimator.set_params(**params)
        self.scheme = scheme
        self.decay = decay
        self.linear = linear
        self.events = events
        self.data_index = data_index

    def fit(self, X, y):
        if self.scheme == "uniqueness":
            weights = self.events["tW"].copy()
        elif self.scheme == "return":
            weights = self.events["w"].copy()
        else:
            weights = pd.Series(np.ones(len(self.events)), index=self.events.index)
    
        if self.decay != 1.0:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=self.events,
                close_index=self.data_index,
                last_weight=self.decay,
                linear=self.linear,
                av_uniqueness=self.events["tW"],
            )
            weights *= decay_vec
    
        valid = X.index.intersection(y.index)
        weights = weights.loc[valid]
    
        # Normalize weights to sum to N (preserves relative structure,
        # maintains effective sample size for regularization and loss scaling)
        weights *= weights.shape[0] / weights.sum()
    
        self.sample_weight_ = weights
        self.base_estimator.fit(
            X.loc[valid].to_numpy(),
            y.loc[valid].to_numpy(),
            sample_weight=weights.to_numpy(),
        )
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = {
            "scheme": self.scheme,
            "decay": self.decay,
            "linear": self.linear,
            "base_estimator": self.base_estimator,
            "events": self.events,
            "data_index": self.data_index,
        }
        if deep:
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f"base_{k.split("__")[-1]}": v for k, v in base_params.items()})
        return params

    def set_params(self, **params):
        base_params = {}
        for key in list(params.keys()):
            if issubclass(self.base_estimator, Pipeline):
                key = key.split("__")[-1]
            if key.startswith("base_"):
                base_params[key[5:]] = params.pop(key)

        for key in ["scheme", "decay", "linear", "base_estimator", "events", "data_index"]:
            if key in params:
                setattr(self, key, params.pop(key))

        if base_params:
            self.base_estimator.set_params(**base_params)
        return self
