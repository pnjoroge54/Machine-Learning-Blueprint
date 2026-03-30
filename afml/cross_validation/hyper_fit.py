import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..cache import cacheable
from ..util.pipelines import MyPipeline, make_custom_pipeline, make_weighted_scorer, set_pipeline_params
from .cross_validation import PurgedKFold


@cacheable(time_aware=True, auto_versioning=False)
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
    sample_weight_train=None,
    sample_weight_score=None,
):
    """
    Hyper-Parameter Fitting with Purged K-Fold Cross-Validation.

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
        A BaseEstimator or Pipeline containing preprocessing and
        classification steps.
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
    sample_weight_train : pd.Series or None, default=None
        Per-sample weights used during fitting on each CV fold and on the
        final refit. Recommended: uniqueness weights (tW) or a time-decay
        variant thereof, reflecting how much information each label
        contributes independently of overlapping labels.
        When None, uniform weights are used for fitting.
    sample_weight_score : pd.Series or None, default=None
        Per-sample weights used when evaluating the scorer on each CV test
        fold. Recommended: return-attribution weights (w), so that the
        hyperparameter selection criterion favours parameter combinations
        that perform well on high-magnitude price moves rather than
        treating all outcomes equally.
        When None, falls back to sample_weight_train if provided, otherwise
        uniform weights are used for scoring.

    Returns
    -------
    estimator : Pipeline
        The trained model.
    cv_results : dict
        Cross-validation results with keys:
        - best_params  : dict of best hyperparameters found.
        - best_score   : float, best CV score achieved.
        - cv_results   : pd.DataFrame of full grid/random search results.
        - scoring      : str, scoring metric used ('f1' or 'neg_log_loss').

    Notes
    -----
    Separating sample_weight_train from sample_weight_score mirrors the
    approach in ml_cross_val_score and reflects the AFML rationale that
    the weighting scheme appropriate for reducing label redundancy during
    training (uniqueness-based) is not necessarily the same as the scheme
    appropriate for evaluating predictive quality (return-based).
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

    # Determine scoring metric.
    # Scoring weights fall back to training weights if score weights not provided.
    scoring_name = "f1" if set(labels.unique()) == {0, 1} else "neg_log_loss"
    effective_score_weight = (
        sample_weight_score
        if sample_weight_score is not None
        else sample_weight_train
    )
    scoring = (
        make_weighted_scorer(scoring_name, effective_score_weight)
        if effective_score_weight is not None
        else scoring_name
    )

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

    # Training weights are passed to fit(). Scoring weights are handled
    # inside the closure created by make_weighted_scorer() via index
    # alignment and do not need to be passed here.
    gs.fit(features, labels, sample_weight=sample_weight_train)

    cv_results = {
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
        "cv_results": pd.DataFrame(gs.cv_results_),
        "scoring": scoring_name,
    }

    best_estimator = gs.best_estimator_

    # Handle bagging if requested
    if bagging_n_estimators > 0:
        # Set n_jobs=1 on the base estimator to avoid nested parallelism
        base_estimator = set_pipeline_params(best_estimator, n_jobs=1)

        bag = BaggingClassifier(
            estimator=MyPipeline(base_estimator.steps),
            n_estimators=int(bagging_n_estimators),
            max_samples=bagging_max_samples,
            max_features=bagging_max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Final bagging fit uses training weights only — there is no scoring
        # step here so sample_weight_score is not relevant.
        bag.fit(features, labels, sample_weight=sample_weight_train)        
        return Pipeline([("bag", bag)]), cv_results

    return Pipeline(best_estimator.steps), cv_results
