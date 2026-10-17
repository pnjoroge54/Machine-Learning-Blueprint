"""
Tests for cross_validation.optuna_hyper_fit.optimize_trading_model: purged folds with
the requested embargo, per-fold scores kept in cv_results, and scoring with events['w'].
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

optuna = pytest.importorskip("optuna")

from ..cross_validation import optuna_hyper_fit as ohf  # noqa: E402


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=1.0, size=n) > 0).astype(int), index=idx)
    events = pd.DataFrame({"t1": idx + pd.Timedelta(hours=3), "w": rng.uniform(0.5, 1.5, n),
                           "tW": rng.uniform(0.2, 1.0, n), "bin": y}, index=idx)
    return X, y, events


def _study(dataset, tmp_path, **kwargs):
    X, y, events = dataset
    return ohf.optimize_trading_model(
        RandomForestClassifier(n_estimators=10, random_state=0), X, y, events, X.index,
        {"max_depth": range(2, 5)}, n_trials=4, n_splits=3, study_name="t",
        db_path=f"sqlite:///{(tmp_path / 'o.db').as_posix()}", refit=False, **kwargs,
    )


def test_embargo_reaches_the_purged_folds(dataset, tmp_path, monkeypatch):
    seen = []
    real = ohf.PurgedKFold

    def recording(*args, **kwargs):
        seen.append(kwargs["pct_embargo"])
        return real(*args, **kwargs)

    monkeypatch.setattr(ohf, "PurgedKFold", recording)
    _study(dataset, tmp_path, pct_embargo=0.05)
    assert seen and set(seen) == {0.05}


def test_cv_results_keep_every_fold_score(dataset, tmp_path):
    study, cv_results = _study(dataset, tmp_path, pruner_type="median")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(cv_results) == len(complete) > 0
    splits = [c for c in cv_results.columns if c.startswith("split")]
    assert len(splits) == 3
    np.testing.assert_allclose(cv_results[splits].mean(axis=1), cv_results["mean_test_score"])


def test_oof_sink_gets_each_completed_trials_probabilities(dataset, tmp_path):
    got = {}
    study, cv_results = _study(dataset, tmp_path, oof_sink=lambda n, s: got.__setitem__(n, s))
    complete = {t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE}
    assert set(got) == complete
    X = dataset[0]
    for s in got.values():
        assert s.index.equals(X.index)
        covered = s.dropna()
        assert len(covered) > 0.5 * len(X) and covered.between(0, 1).all()
