import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..cache.unified_cache_system import (
    cacheable,
    create_cacheable_param_grid,
    reconstruct_param_grid,
)
from .cross_validation import PurgedKFold


class MyPipeline(Pipeline):
    """Allows for a sample_weight in fit method"""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def clf_hyper_fit(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid,
    cv=5,
    bagging_n_estimators=0,
    bagging_max_samples=1.0,
    bagging_max_features=1.0,
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0,
    random_state=None,
    verbose=0,
    **fit_params,
):
    """
    Hyper-Parameter Fitting with Purged K-Fold Cross-Validation

    Performs hyperparameter optimization using purged k-fold cross-validation
    to prevent leakage in time-series data, then optionally fits a bagged
    ensemble on the full dataset using the best parameters found.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix for training.
    labels : pd.Series
        Target labels for classification.
    t1 : pd.Series
        Information range for each record, used for purged cross-validation.
        Index: Time when information extraction started.
        Values: Time when information extraction ended.
    pipe_clf : sklearn.pipeline.Pipeline or MyPipeline
        Pipeline containing preprocessing and classification steps.
    param_grid : dict or list of dicts
        Hyperparameter grid for search. Keys should include pipeline step
        names as prefixes (e.g., 'classifier__max_depth').
    cv : int, default=5
        Number of folds for purged k-fold cross-validation.
    bagging_n_estimators : int, default=0
        Number of base estimators in bagging ensemble. If 0, no bagging
        is applied and the best single estimator is returned. If > 0,
        returns a BaggingClassifier fitted on the full dataset.
    bagging_max_samples : float or int, default=1.0
        For bagging: fraction (if float in (0, 1]) or number (if int) of
        samples to draw for each base estimator.
    bagging_max_features : float or int, default=1.0
        For bagging: fraction (if float in (0, 1]) or number (if int) of
        features to draw for each base estimator.
    rnd_search_iter : int, default=0
        If 0, uses GridSearchCV (exhaustive search). If > 0, uses
        RandomizedSearchCV with this many iterations.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    pct_embargo : float, default=0
        Percentage of samples to embargo in test folds to prevent leakage
        from serially correlated labels. Range: [0, 1).
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Controls verbosity of output.
    **fit_params : dict
        Additional parameters passed to the fit method.

    Returns
    -------
    estimator : Pipeline or BaggingClassifier
        The trained model.
    cv_results : Dict
        Cross-validation results including best parameters and scores.
    """

    # Clone the pipeline to avoid modifying the original
    pipe_clf = clone(pipe_clf)

    # Determine scoring metric
    if set(labels.unique()) == {0, 1}:
        scoring = "f1"  # f1 for meta-labeling
    else:
        scoring = "neg_log_loss"

    # Create purged K-Fold
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    # Perform hyperparameter search
    if rnd_search_iter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            n_iter=rnd_search_iter,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )

    # Fit the grid search
    gs.fit(features, labels, **fit_params)

    # Extract results
    cv_results = {
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
        "cv_results": pd.DataFrame(gs.cv_results_),
        "scoring": scoring,
    }

    best_estimator = gs.best_estimator_

    # Handle bagging if requested
    if bagging_n_estimators > 0:
        # For bagging, set n_jobs=1 for base estimator to avoid nested parallelism
        base_estimator = clone(best_estimator).set_params(njobs=1)

        # Create and fit bagging classifier
        bag = BaggingClassifier(
            estimator=MyPipeline(base_estimator.steps),
            n_estimators=int(bagging_n_estimators),
            max_samples=bagging_max_samples,
            max_features=bagging_max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Fit bagging classifier with sample_weight if provided
        if "sample_weight" in fit_params:
            bag.fit(features, labels, sample_weight=fit_params["sample_weight"])
        else:
            bag.fit(features, labels)

        bag = Pipeline([("bag", bag)])
        return bag, cv_results
    else:
        return best_estimator, cv_results


# Helper function to safely handle n_jobs in pipelines
def set_pipeline_n_jobs(pipeline, n_jobs_value=1):
    """
    Safely set n_jobs for all estimators in a pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to modify.
    n_jobs_value : int
        Value to set for n_jobs.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Modified pipeline.
    """
    for step_name, estimator in pipeline.named_steps.items():
        param_key = f"{step_name}__n_jobs"
        try:
            # Check if this parameter exists in the pipeline
            current_params = pipeline.get_params()
            if param_key in current_params:
                pipeline.set_params(**{param_key: n_jobs_value})
        except:
            pass
    return pipeline


@cacheable()
def clf_hyper_fit_internal(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid_cacheable,  # ← Now accepts cacheable version
    cv=5,
    bagging_n_estimators=0,
    bagging_max_samples=1.0,
    bagging_max_features=1.0,
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0,
    random_state=None,
    verbose=0,
    **fit_params,
):
    """
    Cached version of clf_hyper_fit that properly handles scipy distributions.

    This is the FIXED version that will cache properly.
    """
    # Reconstruct param_grid from cacheable version
    param_grid = reconstruct_param_grid(param_grid_cacheable)

    return clf_hyper_fit(
        features=features,
        labels=labels,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid=param_grid,
        cv=cv,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_max_features=bagging_max_features,
        rnd_search_iter=rnd_search_iter,
        n_jobs=n_jobs,
        pct_embargo=pct_embargo,
        random_state=random_state,
        verbose=verbose,
        **fit_params,
    )


# ============================================================================
# Convenience wrapper that handles conversion automatically
# ============================================================================


def clf_hyper_fit_cached(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid,  # ← Accepts scipy distributions directly
    cv=5,
    bagging_n_estimators=0,
    bagging_max_samples=1.0,
    bagging_max_features=1.0,
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0,
    random_state=None,
    verbose=0,
    **fit_params,
):
    """
    Wrapper that automatically converts param_grid for caching.

    Usage:
        from scipy.stats import randint, uniform

        param_grid = {
            'clf__n_estimators': randint(100, 500),
            'clf__max_depth': randint(3, 20),
        }

        # Just call this instead of clf_hyper_fit
        model, results = clf_hyper_fit_auto_cache(
            features, labels, t1, pipe_clf, param_grid
        )
    """
    # Convert to cacheable format
    param_grid_cacheable = create_cacheable_param_grid(param_grid)

    # Call cached version
    return clf_hyper_fit_internal(
        features=features,
        labels=labels,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid_cacheable=param_grid_cacheable,
        cv=cv,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_max_features=bagging_max_features,
        rnd_search_iter=rnd_search_iter,
        n_jobs=n_jobs,
        pct_embargo=pct_embargo,
        random_state=random_state,
        verbose=verbose,
        **fit_params,
    )


def print_result(clf, n_splits):
    num_nodes = len(clf.cv_results_["rank_test_score"])
    total_time = clf.refit_time_ * num_nodes * n_splits

    best_scores, best_scores_idx, mean_score_time, mean_fit_time = [], 0, 0.0, 0.0

    for i in np.arange(n_splits):
        best_scores.append(clf.cv_results_["split" + str(i) + "_test_score"][clf.best_index_])
        idx = np.where(max(best_scores) == best_scores)[0][
            0
        ]  # always + 1 because index starts from 0 as default

    best_scores_idx = (clf.best_index_ + 1) + len(clf.cv_results_["mean_score_time"]) * idx
    print(
        f"Best params for estimator: {clf.best_estimator_} \nBest CV Score: {max(best_scores):.6f}\n"
    )
    print(
        f"A total of {best_scores_idx} was performed before optimal CV score found! (Under split{idx}_test_score / {idx + 1}th split)\n"
    )

    print(
        f"Estimated time taken for entire process: {total_time:.6f} seconds \nTotal number of candidates / nodes: {num_nodes}\n"
    )

    i = 0
    while i < (clf.best_index_ + 1):
        mean_score_time += clf.cv_results_["mean_score_time"][i]
        mean_fit_time += clf.cv_results_["mean_fit_time"][i]
        i += 1
    print("=" * 55)
    print(
        f"Estimated total mean time required for optimal solution: \n\nScore Time: {mean_score_time:.6f}s \nFit Time:  {mean_fit_time:.6f}s"
    )


def param_grid_size(param_grid: dict):
    # Compute total combinations
    from functools import reduce
    from operator import mul

    return reduce(mul, [len(v) for v in param_grid.values()])
