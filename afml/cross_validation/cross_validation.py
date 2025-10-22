"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

from typing import Callable, Union

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

    # Score model on KFolds
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1 * scoring(
                y.iloc[test],
                prob,
                sample_weight=sample_weight_score[test],
                labels=classifier.classes_,
            )
        elif scoring == probability_weighted_accuracy:
            prob = fit.predict_proba(X.iloc[test, :])
            score = scoring(
                y.iloc[test],
                prob,
                sample_weight=sample_weight_score[test],
                labels=classifier.classes_,
            )
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


def best_weighting_by_mean_score(results: dict, labels: Union[np.ndarray, pd.Series]):
    """
    Find the weighting method whose array of values for a given metric has the highest mean.

    Parameters
    ----------
    results : dict
        Mapping from keys to metric dictionaries. Each value should be a mapping
        (for example another dict) that contains arrays or array-like numeric
        values for metrics, e.g. {'modelA': {'f1': [0.2, 0.3], 'precision': [...]}, ...}.
    labels : str
        Labels to evaluate determine if 'f1' or 'log-loss' is used, i.e., if labels
        contains only two unique values (0 and 1), 'f1' is used; otherwise, 'log-loss' is used.
    Returns
    -------
    tuple
        A 2-tuple (best_key, best_mean) where best_key is the dictionary key
        from `results` with the largest mean for the requested metric, and
        best_mean is the computed mean as a float. If no entries contain the
        requested metric, returns (None, -inf).

    Notes
    -----
    - This function uses numpy.mean to aggregate the metric values, so numpy
      must be available and the metric values must be numeric and iterable.
    - Ties are resolved by keeping the first encountered key with the maximum mean.

    Examples
    --------
    >>> results = {'a': {'f1': [0.2, 0.4]}, 'b': {'f1': [0.5, 0.1]}}
    >>> best_weighting_by_mean_score(results, 'f1')
    ('b', 0.3)
    """

    scoring = "f1" if set(np.unique(labels)) == {0, 1} else "neg_log_loss"
    best_key = None
    best_mean = -float("inf")
    for k, v in results.items():
        score = v.get(scoring)
        if score is None:
            continue
        mean_score = float(np.mean(score))
        if mean_score > best_mean:
            best_mean = mean_score
            best_key = k
    print(f"\nBest Scheme so far: {best_key} with mean {scoring} score {best_mean:.4f}")
    return best_key, best_mean


from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.utils.multiclass import unique_labels


def _confusion_counts_from_cm(cm: np.ndarray, classes: np.ndarray) -> Dict[str, int]:
    """
    Convert a 2x2 confusion matrix to named counts TP, FP, TN, FN for binary labels.
    Assumes classes are ordered [neg, pos]. If labels differ, caller must align.
    """
    if cm.shape != (2, 2):
        # For multi-class fallback, return flattened mapping with class pairs
        pairs = {}
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                pairs[f"y={ci}_pred={cj}"] = int(cm[i, j])
        return pairs

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def ml_cross_val_scores_all(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Extended cross-validation runner that returns:
    - ret_scores: dict of metric name -> np.array (per-fold)
    - fold_results: list of dicts (per-fold) containing raw/weighted confusion matrices,
      fold indices, y_true, y_pred, probs, sample weights used in fold
    - weight_stats: pandas.DataFrame summarising the distribution of sample_weight_train
    - degenerate_folds: list of fold indices with degenerate behavior detected
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

    # Weight distribution summary (global, before CV)
    sw = np.asarray(sample_weight_train, dtype=float)
    weight_stats = (
        pd.Series(sw)
        .describe(percentiles=[0.25, 0.5, 0.75])
        .loc[["min", "25%", "50%", "mean", "75%", "max"]]
    )
    weight_stats = weight_stats.rename({"50%": "median"})

    fold_results = []
    degenerate_folds = []

    classes = None  # will set on first fit

    # Score model on KFolds
    for fold_idx, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        # Basic fold-level checks
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        if y_train.nunique() == 1:
            # warns: single-class in training fold (can happen with purge/embargo)
            if verbose:
                print(f"[fold {fold_idx}] single-class in training: {y_train.unique()}")
        if y_test.nunique() == 1:
            if verbose:
                print(f"[fold {fold_idx}] single-class in test: {y_test.unique()}")

        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y_train,
            sample_weight=sample_weight_train[train],
        )

        # save classes ordering
        if classes is None:
            classes = np.array(fit.classes_, dtype=object)

        prob = fit.predict_proba(X.iloc[test, :])
        pred = fit.predict(X.iloc[test, :])

        # raw confusion matrix (counts)
        cm_raw = confusion_matrix(y_test, pred, labels=classes)
        cm_raw_named = _confusion_counts_from_cm(cm_raw, classes)

        # sample-weighted confusion matrix (sums of sample weights per cell)
        df_fold = pd.DataFrame(
            {"y_true": y_test.to_numpy(), "y_pred": pred, "sw": sample_weight_score[test]}
        )
        # ensure all class columns present and in same order
        cm_w_df = (
            df_fold.groupby(["y_true", "y_pred"])["sw"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=classes, columns=classes, fill_value=0)
        )
        cm_weighted = cm_w_df.to_numpy()
        cm_weighted_named = _confusion_counts_from_cm(cm_weighted, classes)

        # detect degenerate conditions
        # (a) no observed positives in test; (b) no predicted positives in this fold
        observed_pos = (
            int((y_test == classes[-1]).sum()) if len(classes) == 2 else int((y_test == 1).sum())
        )
        predicted_pos = (
            int((pred == classes[-1]).sum()) if len(classes) == 2 else int((pred == 1).sum())
        )

        if observed_pos == 0 or predicted_pos == 0:
            degenerate_folds.append(
                {
                    "fold": fold_idx,
                    "observed_pos": observed_pos,
                    "predicted_pos": predicted_pos,
                    "cm_raw": cm_raw_named,
                    "cm_weighted": cm_weighted_named,
                }
            )
            if verbose:
                print(
                    f"[fold {fold_idx}] degenerate: observed_pos={observed_pos}, predicted_pos={predicted_pos}"
                )

        # store fold-level payload for audit
        fold_results.append(
            {
                "fold": fold_idx,
                "train_idx": train,
                "test_idx": test,
                "y_true": y_test.to_numpy(),
                "y_pred": pred,
                "y_prob": prob,
                "sample_weight_train_fold": sample_weight_train[train],
                "sample_weight_score_fold": sample_weight_score[test],
                "cm_raw": cm_raw_named,
                "cm_weighted": cm_weighted_named,
            }
        )

        # compute and append scoring metrics (preserve previous behavior)
        for method, scoring in zip(ret_scores.keys(), scoring_methods):
            if scoring in (probability_weighted_accuracy, log_loss):
                score = scoring(
                    y_test,
                    prob,
                    sample_weight=sample_weight_score[test],
                    labels=classes,
                )
                if method == "neg_log_loss":
                    score *= -1
            else:
                # for precision/recall/f1 provide zero_division=0 to align with previous undefined->0 behaviour
                try:
                    score = scoring(
                        y_test, pred, sample_weight=sample_weight_score[test], zero_division=0
                    )
                except TypeError:
                    score = scoring(y_test, pred, sample_weight=sample_weight_score[test])
            ret_scores[method].append(score)

        if verbose:
            print(
                f"[fold {fold_idx}] metrics snapshot: "
                + ", ".join(f"{k}={ret_scores[k][-1]:.4f}" for k in ret_scores)
            )

    # convert lists -> arrays (per-fold)
    for k, v in ret_scores.items():
        ret_scores[k] = np.array(v)

    results = {
        "ret_scores": ret_scores,
        "fold_results": fold_results,
        "weight_stats": weight_stats,
        "degenerate_folds": degenerate_folds,
        "classes": classes,
    }

    return results


from typing import Dict, List

import numpy as np
import pandas as pd


def aggregate_named_confusions(fold_results: List[Dict], classes: np.ndarray):
    """
    Aggregate per-fold named confusion dictionaries (cm_raw / cm_weighted) into
    summary tables.

    Args:
        fold_results: list where each element is a dict returned by fold_results entry:
                      must contain 'cm_raw' and 'cm_weighted' where each is either:
                        - dict with keys {"TP","FP","TN","FN"} for binary
                        - or a 2D ndarray for multi-class
        classes: array-like order of classes used when building confusion matrices

    Returns:
        {
          "raw_sum": pd.DataFrame,    # summed raw counts (classes x classes)
          "weighted_sum": pd.DataFrame,# summed weighted counts (classes x classes)
          "raw_median": pd.DataFrame, # per-cell median across folds
          "weighted_median": pd.DataFrame,
          "per_fold": pd.DataFrame     # fold-by-fold flattened TP/FP/TN/FN summary
        }
    """
    # build list of DataFrames (aligned to classes) for raw and weighted
    raws = []
    wgs = []
    per_fold_rows = []

    for fr in fold_results:
        # handle named-dict form for binary case
        cmr = fr.get("cm_raw")
        cmw = fr.get("cm_weighted")

        if isinstance(cmr, dict) and {"TP", "FP", "TN", "FN"} <= set(cmr.keys()):
            # create 2x2 DF ordered [neg, pos]
            # order classes such that pos is last
            neg, pos = classes[0], classes[-1]
            cmr_df = pd.DataFrame(
                [[cmr.get("TN", 0), cmr.get("FP", 0)], [cmr.get("FN", 0), cmr.get("TP", 0)]],
                index=[neg, pos],
                columns=[neg, pos],
            )
        else:
            # assume ndarray-like
            cmr_arr = np.asarray(cmr, dtype=float)
            cmr_df = pd.DataFrame(cmr_arr, index=classes, columns=classes)

        if isinstance(cmw, dict) and {"TP", "FP", "TN", "FN"} <= set(cmw.keys()):
            neg, pos = classes[0], classes[-1]
            cmw_df = pd.DataFrame(
                [[cmw.get("TN", 0), cmw.get("FP", 0)], [cmw.get("FN", 0), cmw.get("TP", 0)]],
                index=[neg, pos],
                columns=[neg, pos],
            )
        else:
            cmw_arr = np.asarray(cmw, dtype=float)
            cmw_df = pd.DataFrame(cmw_arr, index=classes, columns=classes)

        raws.append(cmr_df)
        wgs.append(cmw_df)

        # compute flattened TP/FP/TN/FN for this fold
        tp = int(cmr_df.iloc[1, 1])
        fp = int(cmr_df.iloc[0, 1])
        tn = int(cmr_df.iloc[0, 0])
        fn = int(cmr_df.iloc[1, 0])
        tp_w = float(cmw_df.iloc[1, 1])
        fp_w = float(cmw_df.iloc[0, 1])
        tn_w = float(cmw_df.iloc[0, 0])
        fn_w = float(cmw_df.iloc[1, 0])

        per_fold_rows.append(
            {
                "fold": fr.get("fold"),
                "TP_raw": tp,
                "FP_raw": fp,
                "TN_raw": tn,
                "FN_raw": fn,
                "TP_w": tp_w,
                "FP_w": fp_w,
                "TN_w": tn_w,
                "FN_w": fn_w,
            }
        )

    # stack to compute sums and medians
    raw_stack = pd.concat([df.stack() for df in raws], axis=1)
    raw_stack.columns = [f"fold_{i}" for i in range(len(raws))]
    weighted_stack = pd.concat([df.stack() for df in wgs], axis=1)
    weighted_stack.columns = [f"fold_{i}" for i in range(len(wgs))]

    raw_sum = raw_stack.sum(axis=1).unstack()
    weighted_sum = weighted_stack.sum(axis=1).unstack()
    raw_median = raw_stack.median(axis=1).unstack()
    weighted_median = weighted_stack.median(axis=1).unstack()

    per_fold = pd.DataFrame(per_fold_rows).set_index("fold").sort_index()

    return {
        "raw_sum": raw_sum.astype(float),
        "weighted_sum": weighted_sum.astype(float),
        "raw_median": raw_median.astype(float),
        "weighted_median": weighted_median.astype(float),
        "per_fold": per_fold,
    }
