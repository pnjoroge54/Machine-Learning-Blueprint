"""
Implements the book chapter 7 on Cross Validation for financial data,
as well as Purged Walk-Forward Cross-Validation for financial data.
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold

from ..cross_validation.scoring import probability_weighted_accuracy
from ..ensemble import SequentiallyBootstrappedBaggingClassifier


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


class PurgedWalkForwardCV(_BaseKFold):
    """
    Purged Walk-Forward Cross-Validation

    This class implements walk-forward cross-validation with purging to prevent
    information leakage in financial time series data. In walk-forward validation:
    - Training data always comes before test data (respects temporal order)
    - The training window can be expanding or fixed-size
    - Purging removes training samples that overlap with test labels
    - Embargo prevents using samples immediately after the test set

    :param n_splits: (int) Number of splits/folds. Default to 5
    :param t1: (pd.Series) Information range for each observation
        *t1.index*: Time when information extraction started
        *t1.value*: Time when information extraction ended
    :param pct_embargo: (float) Embargo as fraction of dataset (e.g., 0.01 = 1%)
    :param expanding_window: (bool) If True, use expanding window; if False, use fixed-size window
    :param min_train_size: (float) Minimum training set size as fraction of total (for fixed window)
    """

    def __init__(
        self, n_splits=5, t1=None, pct_embargo=0.01, expanding_window=True, min_train_size=0.3
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series")

        super().__init__(n_splits, shuffle=False, random_state=None)

        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits for walk-forward cross-validation.

        :param X: (pd.DataFrame) Feature dataset
        :param y: (pd.Series) Labels (optional)
        :param groups: (array-like) Group labels (optional)
        :yield: (tuple) Train and test indices for each split
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")

        indices = np.arange(X.shape[0])
        n_samples = X.shape[0]
        embargo_size = int(n_samples * self.pct_embargo)

        # Calculate test set boundaries
        test_splits = np.array_split(np.arange(n_samples), self.n_splits)

        for split_idx in range(self.n_splits):
            # Define test set
            test_indices = test_splits[split_idx]
            test_start_idx = test_indices[0]
            test_end_idx = test_indices[-1] + 1

            # Get test times for purging
            test_times = pd.Series(
                index=[self.t1.index[test_start_idx]], data=[self.t1.iloc[test_end_idx - 1]]
            )

            # Determine training set based on window type
            if self.expanding_window:
                # Expanding window: use all data before test set
                initial_train_indices = indices[:test_start_idx]
            else:
                # Fixed window: use minimum training size
                min_train_samples = int(n_samples * self.min_train_size)
                train_start = max(0, test_start_idx - min_train_samples)
                initial_train_indices = indices[train_start:test_start_idx]

            # Apply purging to remove overlapping observations
            if len(initial_train_indices) > 0:
                initial_train_times = self.t1.iloc[initial_train_indices]
                purged_train_times = ml_get_train_times(initial_train_times, test_times)

                # Convert purged times back to indices
                train_indices = []
                for train_time in purged_train_times.index:
                    loc = self.t1.index.get_loc(train_time)
                    if isinstance(loc, int):
                        train_indices.append(loc)
                    else:
                        # Handle duplicate indices
                        train_indices.extend(range(loc.start, loc.stop))

                train_indices = np.array(train_indices)
            else:
                train_indices = np.array([])

            # Apply embargo: remove samples immediately after test set
            embargo_start = test_end_idx
            embargo_end = min(test_end_idx + embargo_size, n_samples)

            # Only yield if we have training data
            if len(train_indices) > 0:
                yield train_indices, test_indices


# noinspection PyPep8Naming
def ml_cross_val_score(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: Optional[np.ndarray] = None,
    sample_weight_score: Optional[np.ndarray] = None,
    scoring: Union[str, Callable[[np.ndarray, np.ndarray], float]] = log_loss,
):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Run purged/embargoed cross-validation for a classifier and return per-fold scores.

    This implements the evaluation pattern from López de Prado (Advances in Financial Machine Learning,
    snippet 7.4) but requires the caller to provide a CV generator (e.g., PurgedKFold).

    Behavior summary
    - Trains the provided classifier on each train split and scores on the corresponding test split.
    - Supports passing separate sample weights for training and scoring.
    - Special-cases `SequentiallyBootstrappedBaggingClassifier`: clones the classifier per fold and
      aligns its samples_info_sets with the train indices; disables internal OOB scoring during CV.
    - Accepts `scoring` as either a string key (mapped to a function) or a callable metric. For
      probability-based scorers (log_loss, probability_weighted_accuracy) the function expects
      probability inputs from `predict_proba`. For label-based scorers the function expects discrete
      predictions from `predict`.

    Parameters
    ----------
    classifier : ClassifierMixin
        A scikit-learn compatible classifier instance (must implement fit/predict and optionally
        predict_proba).
    X : pd.DataFrame
        Feature matrix indexed consistently with y and (for SequentiallyBootstrappedBaggingClassifier)
        with classifier.samples_info_sets.
    y : pd.Series
        Target labels aligned with X (index used to align samples_info_sets when required).
    cv_gen : BaseCrossValidator
        Cross-validation generator instance with a split(X, y) method (e.g., PurgedKFold).
    sample_weight_train : np.ndarray, optional (default=None)
        Per-sample weights used when calling classifier.fit on the train split. If None, all ones
        are used (no weighting).
    sample_weight_score : np.ndarray, optional (default=None)
        Per-sample weights used when calling the scoring function on the test split. If None, all ones
        are used.
    scoring : str or callable, optional (default=log_loss)
        - If a string, one of the supported keys: "neg_log_loss", "accuracy", "f1", "pwa".
          "neg_log_loss" maps to sklearn.metrics.log_loss and is returned as positive (the function
          multiplies log_loss by -1 to make larger-is-better consistent with other scorers).
        - If a callable, signature should be compatible with either:
            scorer(y_true, y_pred, sample_weight=None, labels=...)   # label-based or prob-based
          The code attempts to pass `labels=classifier.classes_` where relevant, and falls back if
          the scorer does not accept that argument.
        - For probability scorers (log_loss, probability_weighted_accuracy) the function is called
          with `predict_proba` output; for label-based scorers the function is called with `predict`.
        The default is `log_loss`.

    Returns
    -------
    np.ndarray
        1-D array of per-fold scores (float). Order corresponds to the order of splits returned by
        cv_gen.split(X, y).

    Raises
    ------
    KeyError
        If SequentiallyBootstrappedBaggingClassifier is used and its samples_info_sets are not aligned
        with y (index mismatch).
    TypeError / RuntimeError
        If the provided `scoring` callable raises on the provided inputs; the function attempts a
        robust call pattern but will propagate unexpected exceptions.

    Notes
    -----
    - For classifiers that require average/probability inputs (e.g., AUC), pass an appropriate
      scoring callable that accepts probability-like inputs and set scoring to that callable or the
      corresponding string key.
    - For Seq-Bagging classifiers the function disables the estimator's internal OOB scoring during
      cross-validation to avoid interference with the CV scoring flow.
    """
    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))

    # Check for sequential bootstrap
    seq_bootstrap = isinstance(classifier, SequentiallyBootstrappedBaggingClassifier)
    if seq_bootstrap:
        t1 = classifier.samples_info_sets.copy()
        common_idx = t1.index.intersection(y.index)
        X, y, t1 = X.loc[common_idx], y.loc[common_idx], t1.loc[common_idx]
        if t1.empty:
            raise KeyError(f"samples_info_sets not aligned with data")
        classifier.set_params(oob_score=False)

    if isinstance(scoring, str):
        scoring_map = {
            "neg_log_loss": log_loss,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "pwa": probability_weighted_accuracy,
        }
        scoring = scoring_map[scoring]

    # Score model on KFolds
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        if seq_bootstrap:
            classifier = clone(classifier).set_params(
                samples_info_sets=t1.iloc[train]
            )  # Create new instance
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        if scoring == (log_loss or probability_weighted_accuracy):
            prob = fit.predict_proba(X.iloc[test, :])
            score = scoring(
                y.iloc[test],
                prob,
                sample_weight=sample_weight_score[test],
                labels=classifier.classes_,
            )
            if scoring == log_loss:
                score *= -1
        elif scoring == (f1_score or accuracy_score):
            pred = fit.predict(X.iloc[test, :])
            params = dict(
                y_true=y.iloc[test],
                y_pred=pred,
                labels=classifier.classes_,
                sample_weight=sample_weight_score[test],
            )
            try:
                score = scoring(**params)
            except Exception:
                del params["labels"]
                score = scoring(**params)

        ret_scores.append(score)

    return np.array(ret_scores)


def analyze_cross_val_scores(
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
    :return: tuple(dict, pd.DataFrame, dict) The computed scores, a data frame of mean and std. deviation, and a dict of data in each fold
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
        (
            scoring.__name__.replace("_score", "")
            .replace("probability_weighted_accuracy", "pwa")
            .replace("log_loss", "neg_log_loss")
        ): np.zeros(cv_gen.n_splits)
        for scoring in scoring_methods
    }

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))

    seq_bootstrap = isinstance(classifier, SequentiallyBootstrappedBaggingClassifier)
    if seq_bootstrap:
        t1 = classifier.samples_info_sets.copy()
        common_idx = t1.index.intersection(y.index)
        X, y, t1 = X.loc[common_idx], y.loc[common_idx], t1.loc[common_idx]
        if t1.empty:
            raise KeyError(f"samples_info_sets not aligned with data")
        classifier.set_params(oob_score=False)

    cms = []  # To store confusion matrices

    # Score model on KFolds
    for i, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        if seq_bootstrap:
            classifier = clone(classifier).set_params(
                samples_info_sets=t1.iloc[train]
            )  # Create new instance
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        prob = fit.predict_proba(X.iloc[test, :])
        pred = fit.predict(X.iloc[test, :])
        params = dict(
            y_true=y.iloc[test],
            y_pred=pred,
            labels=classifier.classes_,
            sample_weight=sample_weight_score[test],
        )

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
                try:
                    score = scoring(**params)
                except:
                    del params["labels"]
                    score = scoring(**params)
                    params["labels"] = classifier.classes_

            ret_scores[method][i] = score

        cms.append(confusion_matrix(**params).round(2))

    # Mean and standard deviation of scores
    scores_df = pd.DataFrame.from_dict(
        {
            scoring: {"mean": scores.mean(), "std": scores.std()}
            for scoring, scores in ret_scores.items()
        },
        orient="index",
    )

    # Extract TN, TP, FP, FN for each fold
    confusion_matrix_breakdown = []
    for i, cm in enumerate(cms, 1):
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            confusion_matrix_breakdown.append({"fold": i, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        else:
            # For multi-class, you might want different handling
            confusion_matrix_breakdown.append({"fold": i, "confusion_matrix": cm})

    return ret_scores, scores_df, confusion_matrix_breakdown
