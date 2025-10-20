"""
Optimized implementation of sequential bootstrapping with full njit support and CV compatibility.
Integrates with financial ML cross-validation (PurgedKFold, embargo, sample weights).
"""

import numpy as np
import pandas as pd
import time
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@njit(cache=True)
def precompute_active_indices_njit(t0_array, t1_array, price_bars_index):
    """
    Map each sample to the bars it influences using pure njit for maximum performance.

    Args:
        t0_array (np.ndarray): Array of start times for each sample
        t1_array (np.ndarray): Array of end times for each sample
        price_bars_index (np.ndarray): Array of bar indices/timestamps

    Returns:
        tuple: (offsets array, flat_indices array) for efficient storage
    """
    n_samples = len(t0_array)
    n_bars = len(price_bars_index)

    # Pre-allocate storage for indices
    max_possible_indices = n_bars * n_samples  # Upper bound
    flat_indices = np.empty(max_possible_indices, dtype=np.int64)
    offsets = np.empty(n_samples + 1, dtype=np.int64)

    flat_idx = 0
    offsets[0] = 0

    for sample_id in range(n_samples):
        t0 = t0_array[sample_id]
        t1 = t1_array[sample_id]

        for j in range(n_bars):
            if price_bars_index[j] >= t0 and price_bars_index[j] <= t1:
                flat_indices[flat_idx] = j
                flat_idx += 1

        offsets[sample_id + 1] = flat_idx

    # Trim to actual size
    flat_indices = flat_indices[:flat_idx]

    return offsets, flat_indices


def precompute_active_indices(samples_info_sets, price_bars_index):
    """
    Wrapper function that converts pandas Series to arrays for njit processing.

    Args:
        samples_info_sets (pd.Series): Triple barrier events with (t0, t1) as index and values
        price_bars_index (list or array-like): Array of bar indices

    Returns:
        tuple: (offsets array, flat_indices array) for efficient storage
    """
    # Convert to numpy arrays for njit
    if isinstance(samples_info_sets, pd.Series):
        t0_array = np.array(samples_info_sets.index, dtype=np.int64)
        t1_array = np.array(samples_info_sets.values, dtype=np.int64)
    else:
        # Assume it's a dict-like structure
        t0_array = np.array(list(samples_info_sets.keys()), dtype=np.int64)
        t1_array = np.array(list(samples_info_sets.values()), dtype=np.int64)

    price_bars_index = np.asarray(price_bars_index, dtype=np.int64)

    return precompute_active_indices_njit(t0_array, t1_array, price_bars_index)


@njit(cache=True)
def seq_bootstrap_optimized(offsets, flat_indices, s_length, random_seed):
    """
    Generate sample indices using sequential bootstrap with pure njit.

    Args:
        offsets (np.ndarray): Offsets array for indexing flat_indices
        flat_indices (np.ndarray): Flattened array of all bar indices
        s_length (int): Number of samples to generate
        random_seed (int): Seed for random number generation

    Returns:
        np.ndarray: Array of generated sample indices
    """
    np.random.seed(random_seed)

    n_samples = len(offsets) - 1
    sample_ids = np.arange(n_samples, dtype=np.int64)

    # Determine maximum bar index
    T = np.max(flat_indices) + 1 if len(flat_indices) > 0 else 1
    concurrency = np.zeros(T, dtype=np.int64)

    phi = np.empty(s_length, dtype=np.int64)

    for iteration in range(s_length):
        # Compute probabilities
        av_uniqueness = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            start = offsets[i]
            end = offsets[i + 1]

            if end == start:
                av_uniqueness[i] = 0.0
                continue

            uniqueness_sum = 0.0
            for idx_pos in range(start, end):
                bar_idx = flat_indices[idx_pos]
                uniqueness_sum += 1.0 / (concurrency[bar_idx] + 1.0)

            av_uniqueness[i] = uniqueness_sum / (end - start)

        total = np.sum(av_uniqueness)
        if total > 0:
            prob = av_uniqueness / total
        else:
            prob = np.ones(n_samples) / n_samples

        # Sample using cumulative probabilities
        cumsum = np.cumsum(prob)
        r = np.random.random()

        chosen_idx = 0
        for i in range(len(cumsum)):
            if r <= cumsum[i]:
                chosen_idx = i
                break

        chosen = sample_ids[chosen_idx]
        phi[iteration] = chosen

        # Update concurrency
        start = offsets[chosen_idx]
        end = offsets[chosen_idx + 1]
        for idx_pos in range(start, end):
            bar_idx = flat_indices[idx_pos]
            concurrency[bar_idx] += 1

    return phi


class SeqBootstrapRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    A Random Forest classifier using sequential bootstrap sampling for training.
    Compatible with financial ML cross-validation (PurgedKFold, embargo, sample weights).

    This classifier requires additional metadata beyond X and y:
    - samples_info_sets: temporal span of each sample (for purging/sequential bootstrap)
    - price_bars_index: bar timestamps (for sequential bootstrap)
    - sample_weight: weights for both training and scoring

    These are passed via the fit() method and must be provided by a CV-compatible wrapper
    when using with cross-validation frameworks.

    Attributes:
        n_estimators (int): Number of decision trees in the random forest
        s_length (int): Length of the sequential bootstrap sample
        max_depth (int): Maximum depth of the decision trees
        random_seed (int): Random seed for reproducibility
        max_features (int or None): Number of features for splitting
        min_weight_fraction_leaf (float): Minimum weight fraction in leaf
        criterion (str): Split criterion ('gini' or 'entropy')
        class_weight (str): Class weight strategy
    """

    def __init__(
        self,
        n_estimators=10,
        s_length=None,
        max_depth=None,
        random_seed=None,
        max_features=None,
        min_weight_fraction_leaf=0.0,
        criterion="entropy",
        class_weight="balanced",
    ):
        self.n_estimators = n_estimators
        self.s_length = s_length
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.criterion = criterion
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None, samples_info_sets=None, price_bars_index=None):
        """
        Train the Random Forest using sequential bootstrap samples.

        Args:
            X (array-like): Input feature matrix
            y (array-like): Target labels
            sample_weight (array-like): Sample weights for training
            samples_info_sets (pd.Series): End times of sampled events (t1)
            price_bars_index (list or array-like): Array of bar indices

        Returns:
            self: The fitted classifier instance
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Handle default sample weights
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        # Precompute active indices using optimized njit function
        offsets, flat_indices = precompute_active_indices(samples_info_sets, price_bars_index)

        s_length = X.shape[0] if self.s_length is None else self.s_length

        # Initialize random seeds
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            seeds = np.random.randint(0, np.iinfo(np.int32).max, self.n_estimators)
        else:
            seeds = np.random.randint(0, np.iinfo(np.int32).max, self.n_estimators)

        self.estimators_ = []
        for seed in seeds:
            phi = seq_bootstrap_optimized(offsets, flat_indices, s_length, int(seed))

            X_boot = X[phi]
            y_boot = y[phi]
            w_boot = sample_weight[phi]

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                random_state=int(seed),
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
            )

            tree.fit(X=X_boot, y=y_boot, sample_weight=w_boot)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict labels for input samples.

        Args:
            X (array-like): Input feature matrix

        Returns:
            array: Predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Args:
            X (array-like): Input feature matrix

        Returns:
            array: Predicted class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return np.mean(proba, axis=0)


class FinancialMLWrapper(BaseEstimator, ClassifierMixin):
    """
    Universal wrapper for using any classifier with financial ML cross-validation.

    This wrapper makes ANY sklearn-compatible classifier work with:
    - PurgedKFold / PurgedSplit (temporal purging with t1)
    - Embargo periods
    - Sequential bootstrapping (if base estimator supports it)
    - Separate sample weights for training and scoring
    - ml_cross_val_score / ml_cross_val_scores_all

    The wrapper stores metadata globally and extracts appropriate subsets during CV splits.

    Parameters
    ----------
    base_estimator : sklearn estimator
        The underlying classifier to wrap (e.g., SeqBootstrapRandomForestClassifier)
    samples_info_sets : pd.Series
        Triple barrier events with index=t0, values=t1 (end times)
    price_bars_index : array-like
        Bar timestamps/indices used for sequential bootstrap
    sample_weight_train : array-like, optional
        Sample weights for training the model
    sample_weight_score : array-like, optional
        Sample weights for scoring/evaluation (can differ from training weights)

    Usage with ml_cross_val_score
    ------------------------------
    >>> from your_module.cross_validation import ml_cross_val_score, PurgedKFold
    >>>
    >>> # Create base estimator
    >>> base = SeqBootstrapRandomForestClassifier(n_estimators=100, random_seed=42)
    >>>
    >>> # Wrap it with metadata
    >>> wrapper = FinancialMLWrapper(
    ...     base_estimator=base,
    ...     samples_info_sets=samples_info_sets,  # pd.Series with t1 times
    ...     price_bars_index=price_bars_index,     # bar indices
    ...     sample_weight_train=train_weights,     # for model training
    ...     sample_weight_score=score_weights      # for evaluation
    ... )
    >>>
    >>> # Use with PurgedKFold
    >>> cv_gen = PurgedKFold(n_splits=5, t1=samples_info_sets, pct_embargo=0.01)
    >>> scores = ml_cross_val_score(
    ...     classifier=wrapper,
    ...     X=X, y=y,
    ...     cv_gen=cv_gen,
    ...     sample_weight_train=train_weights,  # wrapper will handle these
    ...     sample_weight_score=score_weights,
    ...     scoring=log_loss
    ... )

    Usage with ml_cross_val_scores_all
    -----------------------------------
    >>> results = ml_cross_val_scores_all(
    ...     classifier=wrapper,
    ...     X=X, y=y,
    ...     cv_gen=cv_gen,
    ...     sample_weight_train=train_weights,
    ...     sample_weight_score=score_weights,
    ...     verbose=True
    ... )
    >>>
    >>> # Access detailed results
    >>> print(results['ret_scores'])  # All metrics per fold
    >>> print(results['weight_stats'])  # Weight distribution
    >>> print(results['degenerate_folds'])  # Problem folds

    Notes
    -----
    - The wrapper handles index alignment automatically during CV splits
    - Works with both pandas DataFrames and numpy arrays
    - Compatible with all sklearn CV splitters (KFold, StratifiedKFold, etc.)
    - Especially designed for PurgedKFold and financial time series
    """

    def __init__(
        self,
        base_estimator,
        samples_info_sets,
        price_bars_index,
        sample_weight_train=None,
        sample_weight_score=None,
    ):
        self.base_estimator = base_estimator
        self.samples_info_sets = samples_info_sets
        self.price_bars_index = price_bars_index
        self.sample_weight_train = sample_weight_train
        self.sample_weight_score = sample_weight_score

    def _extract_subset(self, indices, X):
        """
        Extract metadata subset corresponding to the given indices.

        Args:
            indices: Array of indices (from CV split)
            X: Feature matrix (used to get original indices if pandas)

        Returns:
            tuple: (subset_info_sets, subset_train_weight)
        """
        # Get actual indices from X if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            actual_indices = X.index[indices]
        else:
            actual_indices = indices

        # Extract samples_info_sets subset
        if isinstance(self.samples_info_sets, pd.Series):
            subset_info = self.samples_info_sets.iloc[indices]
        else:
            # Handle dict-like or array-like
            subset_info = {
                k: self.samples_info_sets[k] for k in actual_indices if k in self.samples_info_sets
            }

        # Extract sample weight subset
        if self.sample_weight_train is not None:
            subset_weight = self.sample_weight_train[indices]
        else:
            subset_weight = None

        return subset_info, subset_weight

    def fit(self, X, y, sample_weight=None):
        """
        Fit the base estimator with appropriate metadata.

        During cross-validation, X and y will be subsets (train fold).
        We need to extract the corresponding metadata subsets.

        Args:
            X (array-like): Input feature matrix (train fold)
            y (array-like): Target labels (train fold)
            sample_weight (array-like): Sample weights (passed by ml_cross_val_score)

        Returns:
            self: The fitted wrapper instance
        """
        # Clone the base estimator for this fold
        self.estimator_ = clone(self.base_estimator)

        # Determine which samples we're working with
        if isinstance(X, pd.DataFrame):
            # Get position-based indices for subsetting
            train_indices = np.arange(len(X))
            # But we need the original indices for metadata lookup
            original_indices = X.index
        else:
            # For numpy arrays, assume indices are positions
            train_indices = np.arange(len(X))
            original_indices = train_indices

        # Extract metadata for this subset
        if isinstance(self.samples_info_sets, pd.Series):
            # Use the original indices to look up in samples_info_sets
            subset_info = self.samples_info_sets.loc[original_indices]
        else:
            subset_info = self.samples_info_sets

        # Use provided sample_weight (from ml_cross_val_score) or our stored one
        if sample_weight is None and self.sample_weight_train is not None:
            if isinstance(X, pd.DataFrame):
                # Need to map original indices to stored weights
                subset_weight = self.sample_weight_train[train_indices]
            else:
                subset_weight = self.sample_weight_train[train_indices]
        else:
            subset_weight = sample_weight

        # Check if base estimator expects these parameters
        fit_params = {}
        if hasattr(self.estimator_, "fit"):
            import inspect

            sig = inspect.signature(self.estimator_.fit)
            if "samples_info_sets" in sig.parameters:
                fit_params["samples_info_sets"] = subset_info
            if "price_bars_index" in sig.parameters:
                fit_params["price_bars_index"] = self.price_bars_index
            if "sample_weight" in sig.parameters and subset_weight is not None:
                fit_params["sample_weight"] = subset_weight

        # Fit the estimator
        if fit_params:
            self.estimator_.fit(X, y, **fit_params)
        else:
            # Fallback for standard sklearn estimators
            if subset_weight is not None:
                self.estimator_.fit(X, y, sample_weight=subset_weight)
            else:
                self.estimator_.fit(X, y)

        # Copy over required attributes
        self.classes_ = self.estimator_.classes_
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_

        return self

    def predict(self, X):
        """Predict labels for input samples."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for input samples."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def benchmark_implementations(samples_info_sets, price_bars_index, n_iterations=10):
    """
    Benchmark the old vs new implementation of precompute_active_indices.

    Args:
        samples_info_sets (pd.Series): Triple barrier events
        price_bars_index (array-like): Bar indices
        n_iterations (int): Number of iterations for timing

    Returns:
        dict: Benchmark results with timing information
    """
    from numba import jit

    # Old implementation
    @jit(forceobj=True, cache=False)
    def precompute_active_indices_old(samples_info_sets, price_bars_index):
        price_bars_index = np.asarray(price_bars_index)
        active_indices = {}
        for sample_id, (t0, t1) in enumerate(samples_info_sets.items()):
            mask = (price_bars_index >= t0) & (price_bars_index <= t1)
            active_indices[sample_id] = np.where(mask)[0]
        return active_indices

    # Warm up JIT
    _ = precompute_active_indices_old(samples_info_sets, price_bars_index)
    _ = precompute_active_indices(samples_info_sets, price_bars_index)

    # Benchmark old implementation
    times_old = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = precompute_active_indices_old(samples_info_sets, price_bars_index)
        times_old.append(time.perf_counter() - start)

    # Benchmark new implementation
    times_new = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = precompute_active_indices(samples_info_sets, price_bars_index)
        times_new.append(time.perf_counter() - start)

    avg_old = np.mean(times_old)
    avg_new = np.mean(times_new)
    speedup = avg_old / avg_new

    results = {
        "old_mean_time": avg_old,
        "old_std_time": np.std(times_old),
        "new_mean_time": avg_new,
        "new_std_time": np.std(times_new),
        "speedup": speedup,
        "n_samples": len(samples_info_sets),
        "n_bars": len(price_bars_index),
    }

    print("=" * 60)
    print("BENCHMARK RESULTS: precompute_active_indices")
    print("=" * 60)
    print(f"Dataset size: {results['n_samples']} samples, {results['n_bars']} bars")
    print(f"\nOld implementation (jit forceobj=True):")
    print(f"  Mean time: {results['old_mean_time']*1000:.2f} ms")
    print(f"  Std dev:   {results['old_std_time']*1000:.2f} ms")
    print(f"\nNew implementation (pure njit):")
    print(f"  Mean time: {results['new_mean_time']*1000:.2f} ms")
    print(f"  Std dev:   {results['new_std_time']*1000:.2f} ms")
    print(f"\nSpeedup: {results['speedup']:.2f}x faster")
    print("=" * 60)

    return results


def benchmark_full_training(
    X, y, sample_weight, samples_info_sets, price_bars_index, n_estimators=50
):
    """
    Benchmark full training time for the sequential bootstrap RF.

    Args:
        X (array-like): Feature matrix
        y (array-like): Target labels
        sample_weight (array-like): Sample weights
        samples_info_sets (pd.Series): Triple barrier events
        price_bars_index (array-like): Bar indices
        n_estimators (int): Number of trees to train

    Returns:
        dict: Benchmark results
    """
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Training Pipeline")
    print("=" * 60)

    # Initialize model
    model = SeqBootstrapRandomForestClassifier(
        n_estimators=n_estimators, random_seed=42, max_depth=5
    )

    # Benchmark training
    start = time.perf_counter()
    model.fit(X, y, sample_weight, samples_info_sets, price_bars_index)
    training_time = time.perf_counter() - start

    # Benchmark prediction
    start = time.perf_counter()
    predictions = model.predict(X)
    prediction_time = time.perf_counter() - start

    results = {
        "training_time": training_time,
        "prediction_time": prediction_time,
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "time_per_tree": training_time / n_estimators,
    }

    print(f"Dataset: {results['n_samples']} samples, {results['n_features']} features")
    print(f"Number of trees: {results['n_estimators']}")
    print(f"\nTraining time: {results['training_time']:.3f} seconds")
    print(f"Time per tree:  {results['time_per_tree']*1000:.2f} ms")
    print(f"Prediction time: {results['prediction_time']*1000:.2f} ms")
    print("=" * 60)

    return results


# # Standard approach: Direct usage
# model = SeqBootstrapRandomForestClassifier(n_estimators=100)
# model.fit(X, y, sample_weight, samples_info_sets, price_bars_index)

# # CV approach: Use wrapper with your financial ML CV
# wrapper = FinancialMLWrapper(
#     base_estimator=SeqBootstrapRandomForestClassifier(n_estimators=100),
#     samples_info_sets=samples_info_sets,
#     price_bars_index=price_bars_index,
#     sample_weight_train=train_weights,
#     sample_weight_score=score_weights,
# )

# # Now works with your ml_cross_val_score
# from ..cross_validation import ml_cross_val_score, PurgedKFold

# cv_gen = PurgedKFold(n_splits=5, t1=samples_info_sets, pct_embargo=0.01)
# scores = ml_cross_val_score(
#     classifier=wrapper,
#     X=X, y=y,
#     cv_gen=cv_gen,
#     sample_weight_train=train_weights,
#     sample_weight_score=score_weights,
#     scoring=log_loss
# )

# # Or with full diagnostics
# results = ml_cross_val_scores_all(
#     classifier=wrapper,
#     X=X, y=y,
#     cv_gen=cv_gen,
#     sample_weight_train=train_weights,
#     sample_weight_score=score_weights,
#     verbose=True
# )
