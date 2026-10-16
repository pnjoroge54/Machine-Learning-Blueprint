"""
Tests for afml.util.pipelines (weighted CV scorer, MyPipeline, pipeline helpers)
and its use by cross_validation.hyper_fit.clf_hyper_fit.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..cross_validation.hyper_fit import clf_hyper_fit
from ..util.pipelines import (
    MyPipeline,
    make_custom_pipeline,
    make_weighted_scorer,
    set_pipeline_params,
)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series((X["a"] + 0.5 * rng.normal(size=n) > 0).astype(int), index=idx)
    w = pd.Series(rng.uniform(0.1, 2.0, size=n), index=idx)
    t1 = pd.Series(idx + pd.Timedelta(hours=2), index=idx)
    return X, y, w, t1


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Predicts the majority class and records the sample weights it was fit with."""

    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = sample_weight
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


# --------------------------------------------------------------------------- weighted scorer


def test_f1_scorer_is_finite_and_uses_fold_weights():
    X, y, w, _ = _data()
    clf = LogisticRegression().fit(X, y)
    fold = X.index[100:200]
    score = make_weighted_scorer("f1", w)(clf, X.loc[fold], y.loc[fold])
    expected = f1_score(y.loc[fold], clf.predict(X.loc[fold]), sample_weight=w.loc[fold])
    assert np.isfinite(score)
    assert score == pytest.approx(expected)


def test_log_loss_scorer_is_negative_weighted_log_loss():
    X, y, w, _ = _data()
    clf = LogisticRegression().fit(X, y)
    fold = X.index[:150]
    score = make_weighted_scorer("neg_log_loss", w)(clf, X.loc[fold], y.loc[fold])
    expected = -log_loss(y.loc[fold], clf.predict_proba(X.loc[fold]), sample_weight=w.loc[fold])
    assert score == pytest.approx(expected)
    assert score < 0


def test_weights_change_the_score():
    X, y, w, _ = _data()
    clf = LogisticRegression().fit(X, y)
    wrong = clf.predict(X) != y
    heavy_on_errors = pd.Series(np.where(wrong, 10.0, 0.1), index=X.index)
    weighted = make_weighted_scorer("f1", heavy_on_errors)(clf, X, y)
    unweighted = make_weighted_scorer("f1", pd.Series(1.0, index=X.index))(clf, X, y)
    assert weighted < unweighted


# --------------------------------------------------------------------------- pipelines


def test_my_pipeline_routes_sample_weight_to_final_step():
    X, y, w, _ = _data()
    pipe = MyPipeline([("scale", StandardScaler()), ("clf", _RecordingClassifier())])
    pipe.fit(X, y, sample_weight=w.to_numpy())
    np.testing.assert_array_equal(pipe.named_steps["clf"].sample_weight_, w.to_numpy())


def test_make_custom_pipeline_wraps_estimators_and_pipelines():
    wrapped = make_custom_pipeline(LogisticRegression())
    assert isinstance(wrapped, MyPipeline) and wrapped.steps[0][0] == "clf"
    original = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression())])
    wrapped = make_custom_pipeline(original)
    assert isinstance(wrapped, MyPipeline)
    assert [name for name, _ in wrapped.steps] == ["scale", "lr"]


def test_set_pipeline_params_only_touches_steps_that_have_the_param():
    pipe = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(n_jobs=4))])
    set_pipeline_params(pipe, n_jobs=1, random_state=7)
    assert pipe.named_steps["rf"].n_jobs == 1
    assert pipe.named_steps["rf"].random_state == 7
    assert not hasattr(pipe.named_steps["scale"], "n_jobs")


# --------------------------------------------------------------------------- clf_hyper_fit


@pytest.mark.parametrize("score_weights", [True, False])
def test_clf_hyper_fit_returns_finite_weighted_scores(score_weights):
    """Regression: the weighted scorer raised NameError (np) and every fold scored NaN."""
    X, y, w, t1 = _data()
    pipe = MyPipeline([("clf", RandomForestClassifier(n_estimators=20, random_state=0))])
    model, res = clf_hyper_fit(
        X, y, t1, pipe, {"max_depth": [2, 4]},
        n_splits=3, n_jobs=1, random_state=0, pct_embargo=0.01,
        sample_weight_train=w, sample_weight_score=w if score_weights else None,
    )
    scores = res["cv_results"].filter(regex=r"split\d+_test_score").to_numpy()
    assert np.isfinite(scores).all()
    assert np.isfinite(res["best_score"])
    assert res["scoring"] == "f1"
    assert set(res["best_params"]) == {"clf__max_depth"}
    assert model.predict(X).shape == (len(X),)
