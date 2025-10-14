"""
Production-ready implementation of Sequential Bootstrapping for ensemble methods.

Combines the best features from mlfinlab/MizarLabs implementations with Numba optimization
for superior performance while maintaining full scikit-learn compatibility.

References:
    - Advances in Financial Machine Learning, Chapter 4 (López de Prado, 2018)
    - Machine Learning for Asset Managers (López de Prado, 2020)
"""

import numbers
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import jit, njit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble._base import _partition_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    check_X_y,
)
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

MAX_INT = np.iinfo(np.int32).max


# ============================================================================
# Core Sequential Bootstrap Logic (Numba-optimized)
# ============================================================================


@jit(forceobj=True, cache=True)
def precompute_active_indices(samples_info_sets, price_bars_index):
    """
    Map each sample to the bar indices it influences.

    This precomputation is critical for efficient sequential bootstrap sampling,
    as it allows fast lookup of which bars are affected by each sample.

    Args:
        samples_info_sets (pd.Series): Triple barrier event end times (t1) from labeling.
            Index represents start times (t0), values represent end times (t1).
        price_bars_index (array-like): Array of bar timestamps/indices used to form
            triple barrier events.

    Returns:
        dict: Mapping from sample_id (int) to numpy array of bar indices that the
              sample influences. Used to compute uniqueness in sequential bootstrap.

    Example:
        >>> samples_info_sets = pd.Series(
        ...     index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']),
        ...     data=pd.DatetimeIndex(['2020-01-05', '2020-01-04'])
        ... )
        >>> price_bars = pd.DatetimeIndex(['2020-01-01', '2020-01-02', ...])
        >>> active_indices = precompute_active_indices(samples_info_sets, price_bars)
    """
    price_bars_index = np.asarray(price_bars_index)
    active_indices = {}

    for sample_id, (t0, t1) in enumerate(samples_info_sets.items()):
        # Find all bars between start and end time (inclusive)
        mask = (price_bars_index >= t0) & (price_bars_index <= t1)
        active_indices[sample_id] = np.where(mask)[0]

    return active_indices


def seq_bootstrap(active_indices, s_length=None, random_state=None):
    """
    Generate sample indices using sequential bootstrap sampling.

    Sequential bootstrap accounts for sample overlap by computing uniqueness
    (average information content) for each sample based on concurrent labels,
    then sampling proportional to uniqueness.

    Args:
        active_indices (dict): Dictionary mapping sample IDs to arrays of bar indices
            they influence. From precompute_active_indices().
        s_length (int, optional): Number of samples to generate. If None, uses
            len(active_indices) (bootstrap sample same size as original dataset).
        random_state (int, RandomState, or None): Random seed or state for reproducibility.

    Returns:
        list: List of sampled indices (with replacement) selected via sequential bootstrap.

    Notes:
        - Uniqueness = 1 / (concurrency + 1) for each bar
        - Samples with higher average uniqueness have higher selection probability
        - As sampling progresses, concurrency increases, affecting future probabilities

    References:
        AFML Chapter 4: Sample Weights
    """
    # Handle random state
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        try:
            rng = np.random.RandomState(int(random_state))
        except (ValueError, TypeError):
            rng = np.random.RandomState()

    if not active_indices:
        return []

    sample_ids = np.array(list(active_indices.keys()))
    phi = []  # Bootstrap sample indices

    # Determine maximum bar index
    active_indices_values = list(active_indices.values())
    T = max(max(indices) if len(indices) > 0 else 0 for indices in active_indices_values) + 1
    concurrency = np.zeros(T, dtype=np.int32)

    s_length = len(active_indices) if s_length is None else s_length

    # Sequential bootstrap sampling loop
    for _ in range(s_length):
        prob = _compute_selection_probabilities(active_indices_values, concurrency)
        chosen = rng.choice(sample_ids, p=prob)
        phi.append(chosen)
        # Update concurrency for bars influenced by chosen sample
        concurrency[active_indices[chosen]] += 1

    return phi


@njit(cache=True)
def _compute_selection_probabilities(active_indices_values, concurrency):
    """
    Compute selection probabilities based on average uniqueness (Numba-optimized).

    This is the computational bottleneck of sequential bootstrap, so it's optimized
    with Numba for ~10-100x speedup.

    Args:
        active_indices_values (list of arrays): Bar indices influenced by each sample.
        concurrency (array): Current concurrency count for each bar.

    Returns:
        array: Probability distribution over samples for np.random.choice.
    """
    N = len(active_indices_values)
    av_uniqueness = np.zeros(N, dtype=np.float64)

    for i in range(N):
        indices = active_indices_values[i]
        if len(indices) == 0:
            av_uniqueness[i] = 0.0
            continue

        c = concurrency[indices]  # Concurrency values for influenced bars
        uniqueness = 1.0 / (c + 1.0)  # Uniqueness = 1 / (concurrency + 1)
        av_uniqueness[i] = np.mean(uniqueness)

    total = av_uniqueness.sum()
    if total > 0:
        prob = av_uniqueness / total
    else:
        # Uniform distribution if all uniqueness values are zero
        prob = np.ones(N, dtype=np.float64) / N

    return prob


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_samples_info_sets(samples_info_sets, X, price_bars_index):
    """
    Validate that samples_info_sets is properly formatted and aligned with X.

    Args:
        samples_info_sets (pd.Series): Triple barrier event times.
        X (array): Feature matrix.
        price_bars_index (array-like): Bar timestamps.

    Raises:
        ValueError: If validation fails.
    """
    if samples_info_sets is None:
        raise ValueError(
            "samples_info_sets cannot be None. Must provide triple barrier event times."
        )

    if not isinstance(samples_info_sets, pd.Series):
        raise TypeError(
            f"samples_info_sets must be a pd.Series, got {type(samples_info_sets)}"
        )

    if len(samples_info_sets) != X.shape[0]:
        raise ValueError(
            f"samples_info_sets length ({len(samples_info_sets)}) must match "
            f"number of samples in X ({X.shape[0]})"
        )

    if price_bars_index is None:
        raise ValueError("price_bars_index cannot be None")

    # Check that all times in samples_info_sets are in price_bars_index
    price_bars_set = set(np.asarray(price_bars_index))
    for t0, t1 in samples_info_sets.items():
        if t0 not in price_bars_set:
            warnings.warn(
                f"Start time {t0} not found in price_bars_index. "
                "This may lead to incorrect concurrency calculations.",
                UserWarning
            )


def indices_to_mask(indices, n_samples):
    """
    Convert sample indices to a boolean mask.

    Args:
        indices (array-like): Sample indices.
        n_samples (int): Total number of samples.

    Returns:
        array: Boolean mask of length n_samples.
    """
    mask = np.zeros(n_samples, dtype=bool)
    mask[indices] = True
    return mask


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    n_features,
    max_features,
    max_samples,
    active_indices,
):
    """
    Generate sample and feature indices for one bootstrap iteration.

    Args:
        random_state (int or RandomState): Random state for reproducibility.
        bootstrap_features (bool): Whether to bootstrap features.
        n_features (int): Total number of features.
        max_features (int or float): Number or fraction of features to select.
        max_samples (int or float): Number or fraction of samples to select.
        active_indices (dict): Precomputed active indices for sequential bootstrap.

    Returns:
        tuple: (sample_indices, feature_indices)
    """
    rng = check_random_state(random_state)

    # Generate sample indices via sequential bootstrap
    if isinstance(max_samples, numbers.Integral):
        sample_indices = seq_bootstrap(active_indices, s_length=max_samples, random_state=rng)
    elif isinstance(max_samples, numbers.Real):
        n_samples = int(round(max_samples * len(active_indices)))
        sample_indices = seq_bootstrap(active_indices, s_length=n_samples, random_state=rng)
    else:
        sample_indices = seq_bootstrap(active_indices, s_length=None, random_state=rng)

    # Generate feature indices
    if isinstance(max_features, numbers.Integral):
        n_feat = max_features
    elif isinstance(max_features, numbers.Real):
        n_feat = max(1, int(round(max_features * n_features)))
    else:
        raise ValueError("max_features must be int or float")

    if bootstrap_features:
        feature_indices = rng.randint(0, n_features, n_feat)
    else:
        feature_indices = sample_without_replacement(n_features, n_feat, random_state=rng)

    return sample_indices, feature_indices


def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    active_indices,
    sample_weight,
    seeds,
    total_n_estimators,
    verbose,
):
    """
    Build a batch of estimators in parallel (called by joblib.Parallel).

    Args:
        n_estimators (int): Number of estimators to build in this job.
        ensemble (BaseEstimator): The ensemble object.
        X (array): Feature matrix.
        y (array): Target labels.
        active_indices (dict): Precomputed active indices.
        sample_weight (array or None): Sample weights.
        seeds (array): Random seeds for each estimator.
        total_n_estimators (int): Total number of estimators across all jobs.
        verbose (int): Verbosity level.

    Returns:
        tuple: (estimators, estimators_features, estimators_samples)
    """
    n_samples, n_features = X.shape
    max_samples = ensemble._max_samples
    max_features = ensemble._max_features
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")

    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                f"Building estimator {i + 1}/{n_estimators} for this parallel run "
                f"(total {total_n_estimators})..."
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Generate bootstrap indices
        sample_indices, feature_indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            n_features,
            max_features,
            max_samples,
            active_indices,
        )

        # Extract bootstrap sample
        X_boot = X[sample_indices][:, feature_indices]
        y_boot = y[sample_indices]

        # Fit estimator
        if support_sample_weight and sample_weight is not None:
            estimator.fit(X_boot, y_boot, sample_weight=sample_weight[sample_indices])
        else:
            estimator.fit(X_boot, y_boot)

        estimators.append(estimator)
        estimators_features.append(feature_indices)
        estimators_samples.append(sample_indices)

    return estimators, estimators_features, estimators_samples


# ============================================================================
# Base Sequential Bootstrap Bagging Class
# ============================================================================


class SequentiallyBootstrappedBaseBagging(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Sequential Bootstrap Bagging ensembles.

    Extends standard bagging with sequential bootstrap sampling that accounts for
    temporal label overlap in financial time series data.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on bootstrap samples. If None, uses DecisionTree.

    n_estimators : int, default=10
        Number of estimators in the ensemble.

    max_samples : int or float, default=1.0
        Number of samples to draw for each estimator.
        - If int: draw exactly max_samples samples
        - If float: draw max_samples * n_samples samples

    max_features : int or float, default=1.0
        Number of features to draw for each estimator.
        - If int: draw exactly max_features features
        - If float: draw max_features * n_features features

    bootstrap_features : bool, default=False
        Whether to sample features with replacement.

    oob_score : bool, default=False
        Whether to compute out-of-bag score.

    warm_start : bool, default=False
        Whether to reuse previous fit and add more estimators.

    n_jobs : int, default=None
        Number of parallel jobs. None means 1, -1 means use all processors.

    random_state : int, RandomState or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    samples_info_sets : pd.Series, default=None
        Triple barrier event times. Can be set at init or in fit().
        Index = start times (t0), values = end times (t1).

    price_bars_index : array-like, default=None
        Bar timestamps used in samples_info_sets generation.
        Can be set at init or in fit().

    Attributes
    ----------
    estimators_ : list
        Fitted sub-estimators.

    estimators_features_ : list
        Feature indices for each estimator.

    estimators_samples_ : list
        Sample indices for each estimator.

    active_indices_ : dict
        Precomputed mapping of samples to influenced bars.

    oob_score_ : float
        Out-of-bag score (if oob_score=True).

    oob_decision_function_ : array
        OOB decision function (if oob_score=True, classifier only).

    Notes
    -----
    Sequential bootstrap respects temporal overlap in labels by computing uniqueness
    based on concurrent samples and sampling proportional to average uniqueness.

    References
    ----------
    López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    """

    @abstractmethod
    def __init__(
        self,
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
        samples_info_sets=None,
        price_bars_index=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.samples_info_sets = samples_info_sets
        self.price_bars_index = price_bars_index

    @abstractmethod
    def _validate_estimator(self):
        """Set estimator_ attribute with default if necessary."""
        pass

    def _validate_data(self, X, y, sample_weight=None):
        """Validate input data and parameters."""
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        return X, y, sample_weight

    def _validate_max_samples(self, n_samples):
        """Validate and convert max_samples to absolute number."""
        max_samples = self.max_samples

        if not isinstance(max_samples, (numbers.Integral, numbers.Real)):
            raise ValueError(
                f"max_samples must be int or float, got {type(max_samples)}"
            )

        if isinstance(max_samples, numbers.Integral):
            if max_samples > n_samples:
                raise ValueError(
                    f"max_samples ({max_samples}) cannot be greater than "
                    f"n_samples ({n_samples})"
                )
            return max_samples
        else:  # float
            if not (0.0 < max_samples <= 1.0):
                raise ValueError(
                    f"max_samples must be in (0, 1], got {max_samples}"
                )
            return max(1, int(round(max_samples * n_samples)))

    def _validate_max_features(self, n_features):
        """Validate and convert max_features to absolute number."""
        max_features = self.max_features

        if not isinstance(max_features, (numbers.Integral, numbers.Real)):
            raise ValueError(
                f"max_features must be int or float, got {type(max_features)}"
            )

        if isinstance(max_features, numbers.Integral):
            if max_features > n_features:
                raise ValueError(
                    f"max_features ({max_features}) cannot be greater than "
                    f"n_features ({n_features})"
                )
            return max_features
        else:  # float
            if not (0.0 < max_features <= 1.0):
                raise ValueError(
                    f"max_features must be in (0, 1], got {max_features}"
                )
            return max(1, int(round(max_features * n_features)))

    def fit(self, X, y, sample_weight=None, samples_info_sets=None, price_bars_index=None):
        """
        Build ensemble of estimators using sequential bootstrap sampling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, samples are equally weighted.

        samples_info_sets : pd.Series, default=None
            Triple barrier event times. If None, uses value from __init__.
            Index = start times (t0), values = end times (t1).

        price_bars_index : array-like, default=None
            Bar timestamps. If None, uses value from __init__.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Override samples_info_sets if provided
        if samples_info_sets is not None:
            self.samples_info_sets = samples_info_sets
        if price_bars_index is not None:
            self.price_bars_index = price_bars_index

        # Validate data
        X, y, sample_weight = self._validate_data(X, y, sample_weight)
        n_samples, n_features = X.shape

        # Validate samples_info_sets
        _validate_samples_info_sets(self.samples_info_sets, X, self.price_bars_index)

        # Validate estimator
        self._validate_estimator()

        # Validate max_samples and max_features
        self._max_samples = self._validate_max_samples(n_samples)
        self._max_features = self._validate_max_features(n_features)

        # Precompute active indices for sequential bootstrap
        if self.verbose > 0:
            print("Precomputing active indices for sequential bootstrap...")

        self.active_indices_ = precompute_active_indices(
            self.samples_info_sets, self.price_bars_index
        )

        if len(self.active_indices_) != n_samples:
            raise ValueError(
                f"Active indices computation resulted in {len(self.active_indices_)} "
                f"samples, but X has {n_samples} samples"
            )

        # Setup random state
        random_state = check_random_state(self.random_state)

        # Handle warm start
        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be >= "
                f"len(estimators_)={len(self.estimators_)} when warm_start=True"
            )
        elif n_more_estimators == 0:
            warnings.warn(
                "Warm-start fitting without increasing n_estimators does not fit new trees.",
                UserWarning
            )
            return self

        # Generate random seeds for each estimator
        seeds = random_state.randint(MAX_INT, size=n_more_estimators)

        # Partition estimators across jobs
        n_jobs, n_estimators_per_job, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )

        if self.verbose > 0:
            print(f"Fitting {n_more_estimators} estimators across {n_jobs} parallel jobs...")

        # Build estimators in parallel
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators_per_job[i],
                self,
                X,
                y,
                self.active_indices_,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                n_more_estimators,
                self.verbose,
            )
            for i in range(n_jobs)
        )

        # Collect results
        for estimators, features, samples in all_results:
            self.estimators_.extend(estimators)
            self.estimators_features_.extend(features)
            self.estimators_samples_.extend(samples)

        # Compute OOB score if requested
        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a new estimator."""
        estimator = self.estimator

        if estimator is None:
            estimator = self._validate_estimator()

        if random_state is not None:
            estimator = estimator.__class__(**estimator.get_params())
            estimator.set_params(random_state=random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Compute out-of-bag score."""
        pass

    def predict(self, X):
        """
        Predict labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=["csr", "csc"])

        # Aggregate predictions from all estimators
        predictions = np.array(
            [
                estimator.predict(X[:, features])
                for estimator, features in zip(self.estimators_, self.estimators_features_)
            ]
        )

        # For classifier: majority vote; for regressor: mean
        if hasattr(self, "classes_"):
            # Classifier: mode along axis 0
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
            )
        else:
            # Regressor: mean
            return np.mean(predictions, axis=0)


# ============================================================================
# Sequential Bootstrap Bagging Classifier
# ============================================================================


class SequentiallyBootstrappedBaggingClassifier(
    SequentiallyBootstrappedBaseBagging, ClassifierMixin
):
    """
    Sequential Bootstrap Bagging Classifier for financial time series.

    An ensemble classifier that uses sequential bootstrap sampling to account for
    temporal label overlap in financial data. Each base estimator is trained on a
    bootstrap sample selected proportional to sample uniqueness (information content).

    Parameters
    ----------
    estimator : object, default=None
        Base classifier. If None, uses DecisionTreeClassifier().

    n_estimators : int, default=10
        Number of estimators in the ensemble.

    max_samples : int or float, default=1.0
        Number of samples for each estimator.

    max_features : int or float, default=1.0
        Number of features for each estimator.

    bootstrap_features : bool, default=False
        Whether to sample features with replacement.

    oob_score : bool, default=False
        Whether to compute out-of-bag score.

    warm_start : bool, default=False
        Reuse previous fit and add estimators.

    n_jobs : int, default=None
        Number of parallel jobs.

    random_state : int, RandomState or None, default=None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    samples_info_sets : pd.Series, default=None
        Triple barrier event times (can be set in fit()).

    price_bars_index : array-like, default=None
        Bar timestamps (can be set in fit()).

    Attributes
    ----------
    classes_ : array
        Class labels.

    n_classes_ : int
        Number of classes.

    estimators_ : list
        Fitted estimators.

    oob_score_ : float
        Out-of-bag accuracy (if oob_score=True).

    oob_decision_function_ : array
        OOB class probabilities (if oob_score=True).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> # Create mock samples_info_sets
    >>> import pandas as pd
    >>> idx = pd.date_range('2020-01-01', periods=1000, freq='H')
    >>> samples_info_sets = pd.Series(
    ...     index=idx,
    ...     data=idx + pd.Timedelta(hours=5)  # 5-hour labels
    ... )
    >>> clf = SequentiallyBootstrappedBaggingClassifier(
    ...     n_estimators=10,
    ...     random_state=42,
    ...     samples_info_sets=samples_info_sets,
    ...     price_bars_index=idx
    ... )
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """

    def __init__(
        self,
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
        samples_info_sets=None,
        price_bars_index=None,
    ):
        super().__init__(
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
            samples_info_sets=samples_info_sets,
            price_bars_index=price_bars_index,
        )

    def _validate_estimator(self):
        """Validate and set the base estimator."""
        if self.estimator is None:
            self.estimator_ = DecisionTreeClassifier()
        else:
            self.estimator_ = self.estimator

    def fit(self, X, y, sample_weight=None, samples_info_sets=None, price_bars_index=None):
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        samples_info_sets : pd.Series, default=None
            Triple barrier event times (overrides init value if provided).

        price_bars_index : array-like, default=None
            Bar timestamps (overrides init value if provided).

        Returns
        -------
        self : object
        """
        # Set classes before calling parent fit
        _, y = check_X_y(X, y, accept_sparse=["csr", "csc"], multi_output=False)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        return super().fit(X, y, sample_weight, samples_info_sets, price_bars_index)

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score for classifier."""
        n_samples = X.shape[0]
        n_classes = self.n_classes_

        oob_decision_function = np.zeros((n_samples, n_classes))
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # Find OOB samples (not in training set)
            mask = ~indices_to_mask(samples, n_samples)

            if not np.any(mask):
                continue

            # Predict probabilities for OOB samples
            X_oob = X[mask][:, features]
            proba = estimator.predict_proba(X_oob)

            oob_decision_function[mask] += proba
            oob_counts[mask] += 1

        # Handle samples that were never OOB
        if np.any(oob_counts == 0):
            warnings.warn(
                "Some samples were never out-of-bag. "
                "Consider using more estimators or smaller max_samples.",
                UserWarning
            )
            # Set to uniform distribution for samples never OOB
            oob_decision_function[oob_counts == 0] = 1.0 / n_classes

        # Average predictions
        mask_positive = oob_counts > 0
        oob_decision_function[mask_positive] /= oob_counts[mask_positive, np.newaxis]

        # Store results
        self.oob_decision_function_ = oob_decision_function
        self.oob_prediction_ = np.argmax(oob_decision_function, axis=1)

        # Compute OOB accuracy only on samples that were OOB at least once
        if np.any(mask_positive):
            self.oob_score_ = accuracy_score(
                y[mask_positive], self.oob_prediction_[mask_positive]
            )
        else:
            self.oob_score_ = np.nan

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=["csr", "csc"])

        # Aggregate probability predictions
        all_proba = np.zeros((X.shape[0], self.n_classes_))

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            all_proba += estimator.predict_proba(X[:, features])

        return all_proba / len(self.estimators_)

    def predict_log_proba(self, X):
        """
        Predict log probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        log_proba : array of shape (n_samples, n_classes)
            Log of class probabilities.
        """
        return np.log(self.predict_proba(X))


# ============================================================================
# Sequential Bootstrap Bagging Regressor
# ============================================================================


class SequentiallyBootstrappedBaggingRegressor(
    SequentiallyBootstrappedBaseBagging, RegressorMixin
):
    """
    Sequential Bootstrap Bagging Regressor for financial time series.

    An ensemble regressor that uses sequential bootstrap sampling to account for
    temporal label overlap in financial data. Each base estimator is trained on a
    bootstrap sample selected proportional to sample uniqueness (information content).

    Parameters
    ----------
    estimator : object, default=None
        Base regressor. If None, uses DecisionTreeRegressor().

    n_estimators : int, default=10
        Number of estimators in the ensemble.

    max_samples : int or float, default=1.0
        Number of samples for each estimator.

    max_features : int or float, default=1.0
        Number of features for each estimator.

    bootstrap_features : bool, default=False
        Whether to sample features with replacement.

    oob_score : bool, default=False
        Whether to compute out-of-bag score.

    warm_start : bool, default=False
        Reuse previous fit and add estimators.

    n_jobs : int, default=None
        Number of parallel jobs.

    random_state : int, RandomState or None, default=None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    samples_info_sets : pd.Series, default=None
        Triple barrier event times (can be set in fit()).

    price_bars_index : array-like, default=None
        Bar timestamps (can be set in fit()).

    Attributes
    ----------
    estimators_ : list
        Fitted estimators.

    oob_score_ : float
        Out-of-bag R² score (if oob_score=True).

    oob_prediction_ : array
        OOB predictions (if oob_score=True).

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, random_state=42)
    >>> # Create mock samples_info_sets
    >>> import pandas as pd
    >>> idx = pd.date_range('2020-01-01', periods=1000, freq='H')
    >>> samples_info_sets = pd.Series(
    ...     index=idx,
    ...     data=idx + pd.Timedelta(hours=5)
    ... )
    >>> reg = SequentiallyBootstrappedBaggingRegressor(
    ...     n_estimators=10,
    ...     random_state=42,
    ...     samples_info_sets=samples_info_sets,
    ...     price_bars_index=idx
    ... )
    >>> reg.fit(X, y)
    >>> y_pred = reg.predict(X)
    """

    def __init__(
        self,
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
        samples_info_sets=None,
        price_bars_index=None,
    ):
        super().__init__(
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
            samples_info_sets=samples_info_sets,
            price_bars_index=price_bars_index,
        )

    def _validate_estimator(self):
        """Validate and set the base estimator."""
        if self.estimator is None:
            self.estimator_ = DecisionTreeRegressor()
        else:
            self.estimator_ = self.estimator

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score for regressor."""
        n_samples = X.shape[0]

        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # Find OOB samples (not in training set)
            mask = ~indices_to_mask(samples, n_samples)

            if not np.any(mask):
                continue

            # Predict for OOB samples
            X_oob = X[mask][:, features]
            pred = estimator.predict(X_oob)

            oob_predictions[mask] += pred
            oob_counts[mask] += 1

        # Handle samples that were never OOB
        if np.any(oob_counts == 0):
            warnings.warn(
                "Some samples were never out-of-bag. "
                "Consider using more estimators or smaller max_samples.",
                UserWarning
            )

        # Average predictions
        mask_positive = oob_counts > 0
        if np.any(mask_positive):
            oob_predictions[mask_positive] /= oob_counts[mask_positive]
            self.oob_prediction_ = oob_predictions
            self.oob_score_ = r2_score(y[mask_positive], oob_predictions[mask_positive])
        else:
            self.oob_prediction_ = oob_predictions
            self.oob_score_ = np.nan


# ============================================================================
# Example Usage and Tests
# ============================================================================


if __name__ == "__main__":
    """
    Example usage and basic tests for Sequential Bootstrap Bagging.
    """
    print("=" * 80)
    print("Sequential Bootstrap Bagging - Example Usage")
    print("=" * 80)

    # Generate synthetic data
    from sklearn.datasets import make_classification

    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    # Create mock samples_info_sets with overlapping labels
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=1000, freq="H")
    # Each label spans 5 hours (creates temporal overlap)
    samples_info_sets = pd.Series(index=idx, data=idx + pd.Timedelta(hours=5))

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels span: {(samples_info_sets - samples_info_sets.index).mean()}")

    # Example 1: Basic classifier
    print("\n" + "-" * 80)
    print("Example 1: Basic Sequential Bootstrap Classifier")
    print("-" * 80)

    clf = SequentiallyBootstrappedBaggingClassifier(
        n_estimators=10,
        max_samples=0.8,
        max_features=1.0,
        oob_score=True,
        random_state=42,
        verbose=1,
        samples_info_sets=samples_info_sets,
        price_bars_index=idx,
    )

    clf.fit(X, y)
    print(f"OOB Score: {clf.oob_score_:.4f}")
    print(f"Number of estimators: {len(clf.estimators_)}")

    # Predictions
    y_pred = clf.predict(X[:10])
    y_proba = clf.predict_proba(X[:10])
    print(f"\nPredictions (first 10): {y_pred}")
    print(f"Prediction probabilities shape: {y_proba.shape}")

    # Example 2: Regressor
    print("\n" + "-" * 80)
    print("Example 2: Sequential Bootstrap Regressor")
    print("-" * 80)

    from sklearn.datasets import make_regression

    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)

    reg = SequentiallyBootstrappedBaggingRegressor(
        n_estimators=10,
        max_samples=0.8,
        oob_score=True,
        random_state=42,
        verbose=1,
        samples_info_sets=samples_info_sets,
        price_bars_index=idx,
    )

    reg.fit(X_reg, y_reg)
    print(f"OOB R² Score: {reg.oob_score_:.4f}")

    # Example 3: With sample weights
    print("\n" + "-" * 80)
    print("Example 3: With Sample Weights")
    print("-" * 80)

    # Compute sample weights based on uniqueness (simplified)
    sample_weights = np.random.uniform(0.5, 1.5, size=X.shape[0])

    clf_weighted = SequentiallyBootstrappedBaggingClassifier(
        n_estimators=10,
        oob_score=True,
        random_state=42,
        samples_info_sets=samples_info_sets,
        price_bars_index=idx,
    )

    clf_weighted.fit(X, y, sample_weight=sample_weights)
    print(f"Weighted OOB Score: {clf_weighted.oob_score_:.4f}")

    # Example 4: Cross-validation compatibility
    print("\n" + "-" * 80)
    print("Example 4: Cross-Validation")
    print("-" * 80)

    from sklearn.model_selection import KFold, cross_val_score

    # Note: In production, use PurgedKFold instead of KFold
    cv = KFold(n_splits=3, shuffle=False)

    clf_cv = SequentiallyBootstrappedBaggingClassifier(
        n_estimators=5,  # Fewer estimators for faster CV
        random_state=42,
        samples_info_sets=samples_info_sets,
        price_bars_index=idx,
    )

    scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="accuracy")
    print(f"CV Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # Example 5: Warm start
    print("\n" + "-" * 80)
    print("Example 5: Warm Start")
    print("-" * 80)

    clf_warm = SequentiallyBootstrappedBaggingClassifier(
        n_estimators=5,
        warm_start=True,
        random_state=42,
        samples_info_sets=samples_info_sets,
        price_bars_index=idx,
    )

    clf_warm.fit(X, y)
    print(f"Initial estimators: {len(clf_warm.estimators_)}")

    clf_warm.n_estimators = 10
    clf_warm.fit(X, y)
    print(f"After warm start: {len(clf_warm.estimators_)}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)    print(f"After warm start: {len(clf_warm.estimators_)}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)