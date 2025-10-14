"""
Optimized implementation of logic regarding sequential bootstrapping from chapter 4.
"""

import numpy as np
import pandas as pd
from numba import jit, njit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@jit(forceobj=True, cache=True)
def precompute_active_indices(samples_info_sets, price_bars_index):
    """
    Map each sample to the bars it influences.

    Args:
        samples_info_sets (pd.Series): Triple barrier events(t1) from labeling.get_events.
        price_bars_index (list or array-like): Array of bar indices which were used to form triple barrier events.

    Returns:
        dict: A dictionary mapping each sample identifier to an array of indices
              of the bars it influences.
    """
    price_bars_index = np.asarray(
        price_bars_index
    )  # Ensure bar indices are a numpy array for efficient computation.
    active_indices = {}  # Initialize dictionary to store results.
    for sample_id, (t0, t1) in enumerate(samples_info_sets.items()):
        # Create a mask indicating the bars influenced by the sample.
        mask = (price_bars_index >= t0) & (price_bars_index <= t1)
        active_indices[sample_id] = np.where(mask)[0]  # Store the indices of the influenced bars.
    return active_indices


def seq_bootstrap_optimized(active_indices, s_length=None, random_seed=None):
    """
    Generate sample indices using sequential bootstrap.

    Args:
        active_indices (dict): Dictionary mapping sample identifiers to arrays of bar indices.
        s_length (int): Desired number of samples to generate.
        random_seed (int, optional): Seed for random number generation.

    Returns:
        list: A list of generated sample indices.
    """
    np.random.seed(random_seed)  # Set random seed for reproducibility.
    sample_ids = np.array(list(active_indices.keys()))  # Get array of sample identifiers.
    phi = []  # Initialize list for sampled indices.

    # Determine the maximum bar index influenced by all samples for concurrency tracking.
    active_indices_values = list(active_indices.values())
    T = max(max(indices) for indices in active_indices_values) + 1 if active_indices else 0
    concurrency = np.zeros(
        T, dtype=int
    )  # Array to track the number of samples influencing each bar.

    s_length = len(active_indices) if s_length is None else s_length  # Determine sample length.

    # Sequential bootstrap sampling loop.
    for _ in range(s_length):
        prob = _seq_bootstrap_optimized_loop(active_indices_values, concurrency)
        chosen = np.random.choice(sample_ids, p=prob)  # Choose a sample based on probabilities.
        phi.append(chosen)  # Append the selected sample.
        concurrency[
            active_indices[chosen]
        ] += 1  # Update concurrency for the chosen sample's indices.

    return phi


@njit(cache=True)
def _seq_bootstrap_optimized_loop(active_indices_values, concurrency):
    N = len(active_indices_values)
    av_uniqueness = np.zeros(N)  # Array to store average uniqueness of each sample.

    for i in range(N):
        indices = active_indices_values[i]  # Get influenced bar indices for the sample.
        c = concurrency[indices]  # Retrieve concurrency values for these indices.
        uniqueness = 1 / (c + 1)  # Calculate uniqueness as the inverse of concurrency.
        av_uniqueness[i] = (
            np.mean(uniqueness) if len(uniqueness) > 0 else 0.0
        )  # Compute average uniqueness.
    total = av_uniqueness.sum()  # Sum of uniqueness values across all samples.
    prob = (
        av_uniqueness / total if total > 0 else np.ones(N) / N
    )  # Compute probabilities for sampling.

    return prob


class SeqBootstrapRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    A Random Forest classifier using sequential bootstrap sampling for training.

    Attributes:
        n_estimators (int): Number of decision trees in the random forest.
        s_length (int): Length of the sequential bootstrap sample.
        max_depth (int): Maximum depth of the decision trees.
        random_seed (int): Random seed for reproducibility.
        max_features (int or None): Number of features considered for splitting nodes.
        min_weight_fraction_leaf (float): Minimum fraction of weight required in a leaf node.
        criterion (str): Criterion for splitting nodes ('gini' or 'entropy').
        estimators_ (list): List of trained decision tree estimators.
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
        self.estimators_ = []  # List to store trained decision tree estimators.

    def fit(self, X, y, sample_weight, samples_info_sets, price_bars_index):
        """
        Train the Random Forest using sequential bootstrap samples.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target labels.
            sample_weight (array-like): Sample weights.
            samples_info_sets (pd.Series): End times of sampled events.
            price_bars_index (list or array-like): Array of bar indices which were used to form triple barrier events

        Returns:
            self: The fitted classifier instance.
        """
        X, y = check_X_y(X, y)  # Validate and format input arrays.
        self.classes_ = np.unique(y)  # Determine unique classes in the target labels.
        self.n_features_ = X.shape[1]  # Number of features in the input matrix.

        self.active_indices_ = precompute_active_indices(
            samples_info_sets, price_bars_index
        )  # Compute active indices.
        self.s_length_ = (
            X.shape[0] if self.s_length is None else self.s_length
        )  # Determine sample length.

        # Initialize random seeds for each estimator if specified.
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            seeds = np.random.randint(0, np.iinfo(np.int32).max, self.n_estimators)
        else:
            seeds = [None] * self.n_estimators

        self.estimators_ = []  # Clear estimators before training.
        for seed in seeds:
            phi = seq_bootstrap_optimized(
                self.active_indices_, self.s_length_, random_seed=seed
            )  # Bootstrap samples.
            X_boot = X[phi]  # Bootstrap feature matrix.
            y_boot = y[phi]  # Bootstrap target labels.
            w_boot = sample_weight[phi]  # Bootstrap sample weight.
            # Initialize and train a decision tree classifier.
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                random_state=seed,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
            ).set_fit_request(sample_weight=True)
            tree.fit(X=X_boot, y=y_boot, sample_weight=w_boot)  # Fit the tree.
            self.estimators_.append(tree)  # Append the trained tree to the list.
        return self

    def predict(self, X):
        """
        Predict labels for input samples.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            array: Predicted labels for the input samples.
        """
        check_is_fitted(self)  # Ensure the classifier has been fitted.
        X = check_array(X)  # Validate input feature matrix.
        preds = np.array(
            [tree.predict(X) for tree in self.estimators_]
        )  # Predict using all estimators.
        # Aggregate predictions by selecting the most frequent label for each sample.
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
