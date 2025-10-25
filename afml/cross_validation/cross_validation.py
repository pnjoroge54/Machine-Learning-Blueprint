"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold

from ..cross_validation.scoring import probability_weighted_accuracy
from ..sampling import SequentiallyBootstrappedBaggingClassifier


def ml_get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Advances in Financial Machine Learning, Snippet 7.1, page 106.

    Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    :param t1: (pd.Series) The information range on which each record is constructed from
        *t1.index*: Time when the information extraction started.
        *t1.value*: Time when the information extraction ended.
    :param test_times: (pd.Series) Times for the test dataset.
    :return: (pd.Series) Training set
    """
    if test_times.empty:
        return t1.copy(deep=True)

    # Convert to numpy arrays
    train_start = t1.index.values[:, np.newaxis]  # Shape: (n_train, 1)
    train_end = t1.values[:, np.newaxis]  # Shape: (n_train, 1)
    test_start = test_times.index.values[np.newaxis, :]  # Shape: (1, n_test)
    test_end = test_times.values[np.newaxis, :]  # Shape: (1, n_test)

    # Vectorized conditions using broadcasting
    # Each condition results in shape (n_train, n_test)
    cond1 = (test_start <= train_start) & (train_start <= test_end)  # Train starts in test
    cond2 = (test_start <= train_end) & (train_end <= test_end)  # Train ends in test
    cond3 = (train_start <= test_start) & (test_end <= train_end)  # Train envelops test

    # Any overlap with any test period (reduce along test axis)
    has_overlap = np.any(cond1 | cond2 | cond3, axis=1)

    # Keep samples with no overlap
    return t1[~has_overlap]


class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals

    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: (int) The number of splits. Default to 3
    :param t1: (pd.Series) The information range on which each record is constructed from
        *t1.index*: Time when the information extraction started.
        *t1.value*: Time when the information extraction ended.
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")

        super().__init__(n_splits, shuffle=False, random_state=None)

        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """

        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if max_t1_idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg :]))
            yield train_indices, test_indices


class PurgedSplit:
    # No pct_embargo needed because just 1 train/test split (pct_embargo needed after test)
    def __init__(
        self,
        t1: pd.Series = None,
        test_size_pct=0.25,
    ):

        if not isinstance(t1, pd.Series):
            raise ValueError("The t1 param must be a pd.Series")

        self.t1 = t1
        self.test_size_pct = test_size_pct

    # noinspection PyPep8Naming
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and the 't1' series param must be the same length")

        indices = np.arange(X.shape[0])
        test_ranges = [(int(len(X) * (1 - self.test_size_pct)) + 1, len(X))]

        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            test_times = pd.Series(
                index=[self.t1.index[start_ix]],
                data=[self.t1[end_ix - 1]],
            )
            train_times = ml_get_train_times(self.t1, test_times)

            train_indices = []
            for train_ix in train_times.index.unique():
                loc = self.t1.index.get_loc(train_ix)
                if not isinstance(loc, int):
                    loc = np.arange(loc.start, loc.stop)
                    train_indices.extend(loc)
                else:
                    train_indices.append(loc)
            return np.array(train_indices), test_indices


# noinspection PyPep8Naming
def ml_cross_val_score(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
    scoring: Callable[[np.array, np.array], float] = log_loss,
):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.

    Using the PurgedKFold Class.

    Function to run a cross-validation evaluation of the classifier using sample weights and a custom CV generator.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
        scores_array = ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                          sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (BaseEstimator) A scikit-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :return: (np.array) The computed score.
    """

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))
        
    seq_bootstrap = isinstance(classifier, SequentiallyBootstrappedBaggingClassifier):
    if seq_bootstrap:
    	t1 = getattr(classifier, "sample_info_sets")

    # Score model on KFolds
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        if seq_bootstrap:
        	classifier = classifier.set_params(sample_info_sets=t1.iloc[train])
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        if scoring in (log_loss, probability_weighted_accuracy):
            prob = fit.predict_proba(X.iloc[test, :])
            score = scoring(
                y.iloc[test],
                prob,
                sample_weight=sample_weight_score[test],
                labels=classifier.classes_,
            )
            if scoring == log_loss:
                score *= -1
        else:
            pred = fit.predict(X.iloc[test, :])
            score = scoring(y.iloc[test], pred, sample_weight=sample_weight_score[test])
        ret_scores.append(score)

    return np.array(ret_scores)


def ml_cross_val_scores_all(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.

    Using the PurgedKFold Class.

    Function to run a cross-validation evaluation of the classifier using sample weights and a custom CV generator.
    Scores are computed using accuracy_score, probability_weighted_accuracy, log_loss and f1_score.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
        scores_array = ml_cross_val_scores_all(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                               sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (BaseEstimator) A scikit-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :return: (dict) The computed scores.
    """
    scoring_methods = [
        accuracy_score,
        probability_weighted_accuracy,
        log_loss,
        precision_score,
        recall_score,
        f1_score,
    ]
    ret_scores = {
        scoring.__name__.replace("_score", "") if scoring != log_loss else "neg_log_loss": []
        for scoring in scoring_methods
    }

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))

    # Score model on KFolds
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        prob = fit.predict_proba(X.iloc[test, :])
        pred = fit.predict(X.iloc[test, :])
        for method, scoring in zip(ret_scores.keys(), scoring_methods):
            if scoring in (probability_weighted_accuracy, log_loss):
                score = scoring(
                    y.iloc[test],
                    prob,
                    sample_weight=sample_weight_score[test],
                    labels=classifier.classes_,
                )
                if method == "neg_log_loss":
                    score *= -1
            else:
                score = scoring(y.iloc[test], pred, sample_weight=sample_weight_score[test])
            ret_scores[method].append(score)

    for k, v in ret_scores.items():
        ret_scores[k] = np.array(v)

    return ret_scores
    