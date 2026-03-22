import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..cache.cv_cache import cv_cacheable
from ..cache.unified_cache_system import (
    create_cacheable_param_grid,
    reconstruct_param_grid,
)
from ..util.pipelines import MyPipeline, make_custom_pipeline, set_pipeline_params
from .cross_validation import PurgedKFold


def clf_hyper_fit(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid,
    n_splits=5,
    bagging_n_estimators=0,
    bagging_max_samples=1.0,
    bagging_max_features=1.0,
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0.02,
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
    pipe_clf : BaseEstimator or sklearn.pipeline.Pipeline or MyPipeline
        A BaseEstimator or Pipeline containing preprocessing and classification steps.
    param_grid : dict or list of dicts
        Hyperparameter grid for search. Keys should include pipeline step
        names as prefixes (e.g., 'classifier__max_depth').
    n_splits : int, default=5
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
    pct_embargo : float, default=0.02
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
    estimator : Pipeline
        The trained model.
    cv_results : Dict
        Cross-validation results including best parameters and scores.
    """

    # Clone the pipeline to avoid modifying the original
    pipe_clf = make_custom_pipeline(clone(pipe_clf))
    name_of_clf, estimator = pipe_clf.steps[-1]

    # Ensures no issues with oversubscription during parallelization
    pipe_clf = set_pipeline_params(pipe_clf, n_jobs=1)

    # Clean param_grid to only include valid parameters
    for k in reversed(list(param_grid.keys())):
        if not hasattr(estimator, k.split(f"{name_of_clf}__")[-1]):
            param_grid.pop(k)
        elif not k.startswith(f"{name_of_clf}__"):
            param_grid[f"{name_of_clf}__{k}"] = param_grid.pop(k)

    # Determine scoring metric
    if set(labels.unique()) == {0, 1}:
        scoring = "f1"  # for meta-labeling
    else:
        scoring = "neg_log_loss"

    # Create purged K-Fold
    inner_cv = PurgedKFold(n_splits, t1, pct_embargo)

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
        base_estimator = set_pipeline_params(best_estimator, n_jobs=1)

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
        
        bag.estimator = Pipeline(bag.estimator.steps)
        bag = Pipeline([("bag", bag)])
        return bag, cv_results
    else:
        return Pipeline(best_estimator.steps), cv_results


@cv_cacheable
def clf_hyper_fit_internal(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid_cacheable,
    n_splits,
    bagging_n_estimators,
    bagging_max_samples,
    bagging_max_features,
    rnd_search_iter,
    n_jobs,
    pct_embargo,
    random_state,
    verbose,
    **fit_params,
):
    """
    Cached version of clf_hyper_fit that properly handles scipy distributions.
    """
    # Reconstruct param_grid from cacheable version
    param_grid = reconstruct_param_grid(param_grid_cacheable)

    return clf_hyper_fit(
        features=features,
        labels=labels,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid=param_grid,
        n_splits=n_splits,
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
    param_grid,
    n_splits,
    bagging_n_estimators,
    bagging_max_samples,
    bagging_max_features,
    rnd_search_iter,
    n_jobs,
    pct_embargo,
    random_state,
    verbose,
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
        n_splits=n_splits,
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
