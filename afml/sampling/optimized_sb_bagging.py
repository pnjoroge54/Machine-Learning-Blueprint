"""
Implementation of Sequentially Bootstrapped Bagging Classifier using sklearn's library as base class
"""

import numbers
from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.ensemble._base import _partition_estimators

# from sklearn.utils import indices_to_mask
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    check_X_y,
)
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter

from ..util.misc import indices_to_mask
from .optimized_bootstrapping import get_active_indices, seq_bootstrap_optimized

MAX_INT = np.iinfo(np.int32).max


# pylint: disable=too-many-ancestors
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=len-as-condition
# pylint: disable=attribute-defined-outside-init
# pylint: disable=bad-super-call
# pylint: disable=no-else-raise


def _generate_random_features(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw random indices for features
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples, random_state=random_state)
    return indices


def _parallel_build_estimators_with_precomputed_indices(
    n_estimators,
    ensemble,
    X,
    y,
    sample_indices_list,  # PRE-COMPUTED indices
    feature_indices_list,  # PRE-COMPUTED indices
    sample_weight,
    seeds,
    total_n_estimators,
    verbose,
):
    """
    Build estimators using pre-computed indices.
    This is much faster since we don't recompute bootstrap indices.
    """
    from sklearn.utils.validation import has_fit_parameter

    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")

    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Use pre-computed indices (NO index generation here!)
        sample_indices = sample_indices_list[i]
        feature_indices = feature_indices_list[i]

        # Prepare sample weights
        if support_sample_weight and sample_weight is not None:
            curr_sample_weight = sample_weight[sample_indices]
        else:
            curr_sample_weight = None

        estimators_features.append(feature_indices)
        estimators_samples.append(sample_indices)

        # Train estimator
        X_ = X[sample_indices][:, feature_indices]
        y_ = y[sample_indices]

        estimator.fit(X_, y_, sample_weight=curr_sample_weight)
        estimators.append(estimator)

    return estimators, estimators_features, estimators_samples


def _generate_all_bootstrap_indices(
    active_indices, n_features, max_samples, max_features, bootstrap_features, seeds
):
    """
    Pre-generate all bootstrap indices for all estimators.

    This separates the expensive sequential bootstrap computation from
    the estimator building process.
    """
    sample_indices_list = []
    feature_indices_list = []

    # Determine sample size
    if isinstance(max_samples, numbers.Integral):
        n_samples = max_samples
    elif isinstance(max_samples, numbers.Real):
        n_samples = int(round(max_samples * len(active_indices)))
    else:
        n_samples = len(active_indices)

    # Determine feature size
    if isinstance(max_features, numbers.Integral):
        n_feat = max_features
    elif isinstance(max_features, numbers.Real):
        n_feat = int(round(max_features * n_features))
    else:
        n_feat = n_features

    # Generate indices for each estimator
    for seed in seeds:
        random_state = check_random_state(seed)

        # Generate sample indices using sequential bootstrap
        sample_indices = seq_bootstrap_optimized(
            active_indices, sample_length=n_samples, random_seed=random_state
        )

        # Generate feature indices
        feature_indices = _generate_random_features(
            random_state, bootstrap_features, n_features, n_feat
        )

        sample_indices_list.append(sample_indices)
        feature_indices_list.append(feature_indices)

    return sample_indices_list, feature_indices_list


class SequentiallyBootstrappedBaseBagging(BaseBagging, metaclass=ABCMeta):
    """
    Base class for Sequentially Bootstrapped Classifier and Regressor, extension of sklearn's BaseBagging
    """

    @abstractmethod
    def __init__(
        self,
        samples_info_sets,
        price_bars_index,
        estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            bootstrap=True,  # Always use bootstrap for sequential bootstrap
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.samples_info_sets = samples_info_sets
        self.price_bars_index = price_bars_index
        self.active_indices_ = None
        self._estimators_samples = []  # Initialize private variable
        self._seeds = None  # Initialize the attribute

    @property
    def estimators_samples_(self):
        """Return the stored sample indices for each estimator."""
        return self._estimators_samples

    def fit(self, X, y, sample_weight=None):
        """
        Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
        set (X, y).

        Parameters
        ----------
        X : (array-like, sparse matrix) of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : (array-like), shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : (array-like), shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : (object)
        """
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """
        Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
        set (X, y).

        Parameters
        ----------
        X : (array-like, sparse matrix) of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : (array-like), shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : (int or float), optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : (int), optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : (array-like), shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : (object)
        """
        # Set classes_ and n_classes_ for classifier compatibility
        if hasattr(self, "classes_") and not hasattr(self, "n_classes_"):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        # Validate parameters
        random_state = check_random_state(self.random_state)

        # Convert data and validate
        X, y = check_X_y(X, y, ["csr", "csc"])
        n_samples, n_features = X.shape

        # Check sample weight
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Validate estimator
        self._validate_estimator()

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples

        if not isinstance(max_samples, (numbers.Integral, numbers.Real)):
            raise ValueError("max_samples must be int or float, got %s" % type(max_samples))

        if isinstance(max_samples, numbers.Integral):
            max_samples = min(max_samples, n_samples)
        else:
            if not (0.0 < max_samples <= 1.0):
                raise ValueError("max_samples must be in (0, 1], got %r" % max_samples)
            max_samples = int(round(max_samples * n_samples))

        self._max_samples = max_samples

        # Compute indicator matrix for sequential bootstrap
        if self.active_indices_ is None:
            self.active_indices_ = get_active_indices(self.samples_info_sets, self.price_bars_index)

        # Check if indicator matrix matches data shape
        if len(self.active_indices_) != n_samples:
            raise ValueError(
                f"Indicator matrix shape {len(self.active_indices_)} "
                f"does not match number of samples {n_samples}"
            )

        # Warm start handling
        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []
            self.estimators_features_ = []
            self._estimators_samples = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )
        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not fit new trees.")
            return self

        # Generate random seeds for each estimator
        seeds = random_state.randint(MAX_INT, size=n_more_estimators)

        # Store seeds
        if not hasattr(self, "_seeds"):
            self._seeds = seeds
        else:
            self._seeds = np.concatenate([self._seeds, seeds])

        # ========== KEY CHANGE: Pre-generate all indices ==========
        if self.verbose:
            print("Generating bootstrap indices for all estimators...")

        # Call the helper function (defined in the SAME file)
        sample_indices_list, feature_indices_list = _generate_all_bootstrap_indices(
            self.active_indices_,
            n_features,
            max_samples,
            self.max_features,
            self.bootstrap_features,
            seeds,
        )

        if self.verbose:
            print("Bootstrap indices generated. Building estimators in parallel...")

        # ========== Build estimators in parallel with pre-computed indices ==========
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators, self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Use the NEW function (also defined in the SAME file)
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators_with_precomputed_indices)(
                n_estimators[i],
                self,
                X,
                y,
                sample_indices_list[starts[i] : starts[i + 1]],
                feature_indices_list[starts[i] : starts[i + 1]],
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
            )
            for i in range(n_jobs)
        )

        # Unpack results
        for result in all_results:
            self.estimators_ += result[0]
            self.estimators_features_ += result[1]
            self._estimators_samples += result[2]

        # Compute OOB score if requested
        if self.oob_score:
            self._set_oob_score(X, y)

        return self


class SequentiallyBootstrappedBaggingClassifier(
    SequentiallyBootstrappedBaseBagging, BaggingClassifier, ClassifierMixin
):
    """
    A Sequentially Bootstrapped Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset generated using
    Sequential Bootstrapping sampling procedure and then aggregate their individual predictions (
    either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    :param samples_info_sets: (pd.Series), The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param price_bars_index: (pd.DataFrame)
        Price bars index used in samples_info_sets generation
    :param estimator: (object or None), optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    :param n_estimators: (int), optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: (int or float), optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: (int or float), optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: (bool), optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: (bool), optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: (bool), optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: (int or None), optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: (int), RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: (int), optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar estimator_: (estimator)
        The base estimator from which the ensemble is grown.
    :ivar estimators_: (list of estimators)
        The collection of fitted base estimators.
    :ivar estimators_samples_: (list of arrays)
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: (list of arrays)
        The subset of drawn features for each base estimator.
    :ivar classes_: (array) of shape = [n_classes]
        The classes labels.
    :ivar n_classes_: (int or list)
        The number of classes.
    :ivar oob_score_: (float)
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_decision_function_: (array) of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def __init__(
        self,
        samples_info_sets,
        price_bars_index,
        estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            samples_info_sets=samples_info_sets,
            price_bars_index=price_bars_index,
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier())

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """
        Override _fit to set classes_ and n_classes_ for classifier compatibility.
        """
        # Set classes_ and n_classes_ before calling parent _fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Call parent _fit method
        return super()._fit(X, y, max_samples, max_depth, sample_weight)

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""

        # Safeguard: Ensure n_classes_ is set
        if not hasattr(self, "n_classes_"):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        n_samples = y.shape[0]
        n_classes = self.n_classes_

        predictions = np.zeros((n_samples, n_classes))

        for estimator, samples, features in zip(
            self.estimators_, self._estimators_samples, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if np.any(mask):
                # Get predictions for OOB samples
                X_oob = X[mask][:, features]
                predictions[mask] += estimator.predict_proba(X_oob)

        # Average predictions
        denominator = np.sum(predictions != 0, axis=1)
        denominator[denominator == 0] = 1  # avoid division by zero
        predictions /= denominator[:, np.newaxis]

        # Compute OOB score
        oob_decision_function = predictions
        oob_prediction = np.argmax(predictions, axis=1)

        if n_classes == 2:
            oob_prediction = oob_prediction.astype(np.int64)

        self.oob_decision_function_ = oob_decision_function
        self.oob_prediction_ = oob_prediction
        self.oob_score_ = accuracy_score(y, oob_prediction)


class SequentiallyBootstrappedBaggingRegressor(
    SequentiallyBootstrappedBaseBagging, BaggingRegressor, RegressorMixin
):
    """
    A Sequentially Bootstrapped Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset using Sequential Bootstrapping and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    :param samples_info_sets: (pd.Series), The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.

    :param price_bars_index: (pd.DatetimeIndex)
        Index of price bars used in samples_info_sets generation
    :param estimator: (object or None), optional (default=None)
        The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
    :param n_estimators: (int), optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: (int or float), optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: (int or float), optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: (bool), optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: (bool)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: (bool), optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: (int or None), optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: (int, RandomState instance or None), optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: (int), optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar estimators_: (list) of estimators
        The collection of fitted sub-estimators.
    :ivar estimators_samples_: (list) of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: (list) of arrays
        The subset of drawn features for each base estimator.
    :ivar oob_score_: (float)
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_prediction_: (array) of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.
    """

    def __init__(
        self,
        samples_info_sets,
        price_bars_index,
        estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            samples_info_sets=samples_info_sets,
            price_bars_index=price_bars_index,
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        n_samples = y.shape[0]
        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)

        for estimator, samples, features in zip(
            self.estimators_, self._estimators_samples, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if np.any(mask):
                # Get predictions for OOB samples
                X_oob = X[mask][:, features]
                predictions[mask] += estimator.predict(X_oob)
                n_predictions[mask] += 1

        # Avoid division by zero
        mask = n_predictions > 0
        if np.any(mask):
            predictions[mask] /= n_predictions[mask]

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y[mask], predictions[mask])
