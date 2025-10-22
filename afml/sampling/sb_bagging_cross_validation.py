"""
Cross-validation functions with Sequential Bootstrapping support.
Based on Chapter 4 and Chapter 7 of Advances in Financial Machine Learning.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator


def ml_cross_val_score_seq_bootstrap(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    samples_info_sets: pd.Series,
    price_bars_index: pd.DatetimeIndex,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
    scoring: Callable[[np.array, np.array], float] = log_loss,
    use_seq_bootstrap: bool = True,
):
    """
    Cross-validation with Sequential Bootstrapping for sample weights.
    
    This function performs cross-validation while computing sample weights
    using sequential bootstrapping within each fold. This addresses the issue
    of overlapping labels in financial data.
    
    Args:
        classifier: A scikit-learn Classifier object instance.
        X: The dataset of records to evaluate.
        y: The labels corresponding to the X dataset.
        cv_gen: Cross Validation generator object instance (e.g., PurgedKFold).
        samples_info_sets: Triple barrier events (t1) from labeling.
            Index: start times (t0), Values: end times (t1)
        price_bars_index: Index of price bars used in samples_info_sets generation.
        sample_weight_train: Base sample weights for training (before sequential bootstrap).
        sample_weight_score: Sample weights used to evaluate the model quality.
        scoring: A metric scoring function.
        use_seq_bootstrap: If True, compute sequential bootstrap weights per fold.
    
    Returns:
        np.array: The computed scores for each fold.
    
    Example:
        >>> from cross_validation import PurgedKFold
        >>> cv_gen = PurgedKFold(n_splits=5, t1=samples_info_sets, pct_embargo=0.01)
        >>> scores = ml_cross_val_score_seq_bootstrap(
        ...     classifier=RandomForestClassifier(),
        ...     X=X_train,
        ...     y=y_train,
        ...     cv_gen=cv_gen,
        ...     samples_info_sets=samples_info_sets,
        ...     price_bars_index=price_bars.index,
        ...     scoring=log_loss
        ... )
    """
    from ..cross_validation.scoring import probability_weighted_accuracy
    from .optimized_bootstrapping import (
        precompute_active_indices,
        seq_bootstrap_optimized,
    )

    # If no sample_weight then broadcast a value of 1 to all samples
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))
    
    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))
    
    # Precompute active indices once for the entire dataset
    if use_seq_bootstrap:
        active_indices_full = precompute_active_indices(samples_info_sets, price_bars_index)
    
    # Score model on KFolds
    ret_scores = []
    for fold_idx, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        # Compute sequential bootstrap weights for this fold's training set
        if use_seq_bootstrap:
            # Create fold-specific active indices (subset to training samples)
            active_indices_fold = {
                new_idx: active_indices_full[orig_idx] 
                for new_idx, orig_idx in enumerate(train)
            }
            
            # Generate sequential bootstrap sample indices
            # This returns indices relative to the fold (0 to len(train)-1)
            bootstrap_indices = seq_bootstrap_optimized(
                active_indices_fold, 
                s_length=len(train)
            )
            
            # Count occurrences to create sample weights
            fold_sample_weights = np.zeros(len(train))
            unique_indices, counts = np.unique(bootstrap_indices, return_counts=True)
            fold_sample_weights[unique_indices] = counts
            
            # Normalize weights and apply base weights
            fold_sample_weights = fold_sample_weights / fold_sample_weights.sum() * len(train)
            fold_sample_weights = fold_sample_weights * sample_weight_train[train]
        else:
            fold_sample_weights = sample_weight_train[train]
        
        # Fit the model with sequential bootstrap weights
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=fold_sample_weights,
        )
        
        # Score the model
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


def get_seq_bootstrap_sample_weights(
    samples_info_sets: pd.Series,
    price_bars_index: pd.DatetimeIndex,
    sample_indices: np.ndarray = None,
) -> np.ndarray:
    """
    Compute sample weights using sequential bootstrapping.
    
    This is a utility function to get sequential bootstrap weights for a given
    set of samples. Useful when you want to pre-compute weights before training.
    
    Args:
        samples_info_sets: Triple barrier events (t1) from labeling.
            Index: start times (t0), Values: end times (t1)
        price_bars_index: Index of price bars.
        sample_indices: Indices of samples to include. If None, uses all samples.
    
    Returns:
        np.ndarray: Sample weights based on sequential bootstrap frequency.
    
    Example:
        >>> weights = get_seq_bootstrap_sample_weights(
        ...     samples_info_sets=t1_series,
        ...     price_bars_index=price_bars.index
        ... )
        >>> classifier.fit(X, y, sample_weight=weights)
    """
    from .optimized_bootstrapping import (
        precompute_active_indices,
        seq_bootstrap_optimized,
    )

    # Subset if indices provided
    if sample_indices is not None:
        samples_info_sets = samples_info_sets.iloc[sample_indices]
    
    n_samples = len(samples_info_sets)
    
    # Precompute active indices
    active_indices = precompute_active_indices(samples_info_sets, price_bars_index)
    
    # Generate sequential bootstrap samples
    bootstrap_indices = seq_bootstrap_optimized(active_indices, s_length=n_samples)
    
    # Count occurrences to create sample weights
    sample_weights = np.zeros(n_samples)
    unique_indices, counts = np.unique(bootstrap_indices, return_counts=True)
    sample_weights[unique_indices] = counts
    
    # Normalize so mean weight is 1.0 (compatible with sklearn conventions)
    sample_weights = sample_weights / sample_weights.mean()
    
    return sample_weights


class SequentialBootstrapCV:
    """
    Wrapper class for cross-validation with sequential bootstrapping.
    
    This class provides a convenient interface for performing cross-validation
    with sequential bootstrap sample weights, following the methodology from
    Chapter 4 of Advances in Financial Machine Learning.
    
    Args:
        cv_gen: Cross-validation generator (e.g., PurgedKFold).
        samples_info_sets: Triple barrier events (t1) series.
        price_bars_index: Price bar timestamps.
        use_seq_bootstrap: Whether to use sequential bootstrap weights.
    
    Example:
        >>> from cross_validation import PurgedKFold
        >>> cv_gen = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        >>> seq_cv = SequentialBootstrapCV(
        ...     cv_gen=cv_gen,
        ...     samples_info_sets=t1,
        ...     price_bars_index=price_bars.index
        ... )
        >>> scores = seq_cv.cross_val_score(classifier, X, y, scoring=log_loss)
    """
    
    def __init__(
        self,
        cv_gen: BaseCrossValidator,
        samples_info_sets: pd.Series,
        price_bars_index: pd.DatetimeIndex,
        use_seq_bootstrap: bool = True,
    ):
        self.cv_gen = cv_gen
        self.samples_info_sets = samples_info_sets
        self.price_bars_index = price_bars_index
        self.use_seq_bootstrap = use_seq_bootstrap
    
    def cross_val_score(
        self,
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight_train: np.ndarray = None,
        sample_weight_score: np.ndarray = None,
        scoring: Callable = log_loss,
    ) -> np.ndarray:
        """
        Perform cross-validation with sequential bootstrapping.
        
        Returns:
            np.ndarray: Scores for each fold.
        """
        return ml_cross_val_score_seq_bootstrap(
            classifier=classifier,
            X=X,
            y=y,
            cv_gen=self.cv_gen,
            samples_info_sets=self.samples_info_sets,
            price_bars_index=self.price_bars_index,
            sample_weight_train=sample_weight_train,
            sample_weight_score=sample_weight_score,
            scoring=scoring,
            use_seq_bootstrap=self.use_seq_bootstrap,
        )            samples_info_sets=self.samples_info_sets,
            price_bars_index=self.price_bars_index,
            sample_weight_train=sample_weight_train,
            sample_weight_score=sample_weight_score,
            scoring=scoring,
            use_seq_bootstrap=self.use_seq_bootstrap,
        )