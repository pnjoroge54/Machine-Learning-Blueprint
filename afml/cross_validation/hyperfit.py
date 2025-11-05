from copy import deepcopy

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .cross_validation import PurgedKFold


class MyPipeline(Pipeline):
    """Allows for a sample_weight in fit method"""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clf_hyper_fit(
    feat,
    labels,
    t1,
    pipe_clf,
    param_grid,
    cv=3,
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
    Hyper-Parameter Search with Purged K-Fold Cross-Validation

    Performs hyperparameter optimization using purged k-fold cross-validation
    to prevent leakage in time-series data, then optionally fits a bagged
    ensemble on the full dataset using the best parameters found.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature matrix for training.
    labels : pd.Series
        Target labels for classification.
    t1 : pd.Series
        Information range for each record, used for purged cross-validation.
        Index: Time when information extraction started.
        Values: Time when information extraction ended.
    pipe_clf : sklearn.pipeline.Pipeline
        Pipeline containing preprocessing and classification steps.
    param_grid : dict or list of dicts
        Hyperparameter grid for search. Keys should include pipeline step
        names as prefixes (e.g., 'classifier__max_depth').
    cv : int, default=3
        Number of folds for purged k-fold cross-validation.
    bagging_n_estimators : int, default=0
        Number of base estimators in bagging ensemble. If 0, no bagging
        is applied and the best single estimator is returned. If > 0,
        returns a BaggingClassifier fitted on the full dataset.
    bagging_max_samples : float or int, default=1.0
        For bagging: fraction (if float in (0, 1]) or number (if int) of
        samples to draw for each base estimator. Set to average uniqueness
        from sequential bootstrapping to account for sample overlap.
    bagging_max_features : float or int, default=1.0
        For bagging: fraction (if float in (0, 1]) or number (if int) of
        features to draw for each base estimator.
    rnd_search_iter : int, default=0
        If 0, uses GridSearchCV (exhaustive search). If > 0, uses
        RandomizedSearchCV with this many iterations.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores. The function
        handles nested parallelism automatically to prevent oversubscription.
    pct_embargo : float, default=0
        Percentage of samples to embargo in test folds to prevent leakage
        from serially correlated labels. Range: [0, 1).
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility in RandomizedSearchCV and
        BaggingClassifier.
    verbose : int, default=0
        Controls verbosity of GridSearchCV/RandomizedSearchCV output.
        0: silent, 1: print fold scores, 2: print fold scores and timing,
        3+: print more detailed information.
    **fit_params : dict
        Additional parameters passed to the fit method. Use format
        'step_name__parameter' for pipeline parameters (e.g.,
        'classifier__sample_weight' for sample weights).

    Returns
    -------
    estimator : Pipeline or BaggingClassifier
        If bagging_n_estimators == 0: Returns the best_estimator_ Pipeline
        from hyperparameter search, fitted on full data.
        If bagging_n_estimators > 0: Returns a BaggingClassifier with the
        best estimator as base, fitted on full data.

    Notes
    -----
    - Scoring metric is automatically selected: 'f1' for binary {0, 1}
      labels (meta-labeling), 'neg_log_loss' otherwise.
    - When bagging is enabled, it's recommended to set bagging_max_samples
      to the average uniqueness from sequential bootstrapping to account
      for overlapping samples in financial time series.
    - The function automatically prevents nested parallelism by setting
      inner estimator n_jobs=1 when outer search uses parallel jobs.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', RandomForestClassifier(random_state=42))
    ... ])
    >>> param_grid = {
    ...     'clf__n_estimators': [100, 200],
    ...     'clf__max_depth': [5, 10, None]
    ... }
    >>> # Without bagging
    >>> model = clf_hyper_fit(
    ...     feat=X, labels=y, t1=t1,
    ...     pipe_clf=pipe, param_grid=param_grid,
    ...     cv=5, n_jobs=-1
    ... )
    >>> # With bagging (using average uniqueness from sequential bootstrap)
    >>> avg_uniqueness = 0.65  # calculated from your data
    >>> model = clf_hyper_fit(
    ...     feat=X, labels=y, t1=t1,
    ...     pipe_clf=pipe, param_grid=param_grid,
    ...     cv=5, bagging_n_estimators=100,
    ...     bagging_max_samples=avg_uniqueness,
    ...     n_jobs=-1
    ... )
    """
    if set(labels.values) == {0, 1}:
        scoring = "f1"  # f1 for meta-labeling
    else:
        scoring = "neg_log_loss"  # symmetric towards all cases

    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    # Normalize outer n_jobs
    outer_n_jobs = 1 if n_jobs is None else n_jobs

    # Save original n_jobs from pipeline estimators
    orig_n_jobs = {}
    n_jobs_steps = []
    for name, est in pipe_clf.named_steps.items():
        if hasattr(est, "n_jobs"):
            n_jobs_steps.append(name)
            orig_n_jobs[name] = deepcopy(getattr(est, "n_jobs"))

    # Avoid nested parallelism: if outer search is parallel, make inner estimators sequential
    try:
        if outer_n_jobs != 1 and n_jobs_steps:
            set_params = {f"{name}__n_jobs": 1 for name in n_jobs_steps}
            pipe_clf.set_params(**set_params)

        if rnd_search_iter == 0:
            gs = GridSearchCV(
                estimator=pipe_clf,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=outer_n_jobs,
                verbose=verbose,
                refit=True,
            )
        else:
            gs = RandomizedSearchCV(
                estimator=pipe_clf,
                param_distributions=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=outer_n_jobs,
                n_iter=rnd_search_iter,
                random_state=random_state,
                verbose=verbose,
                refit=True,
            )

        gs = gs.fit(feat, labels, **fit_params).best_estimator_

        # 2) fit validated model on the entirety of the data
        if bagging_n_estimators > 0:
            # Create base pipeline with single-threaded estimators to avoid nested parallelism
            base_pipe = MyPipeline(gs.steps)
            for name in n_jobs_steps:
                if hasattr(base_pipe.named_steps.get(name), "n_jobs"):
                    base_pipe.set_params(**{f"{name}__n_jobs": 1})

            bag = BaggingClassifier(
                estimator=base_pipe,
                n_estimators=int(bagging_n_estimators),
                max_samples=bagging_max_samples,
                max_features=bagging_max_features,
                n_jobs=outer_n_jobs,
                random_state=random_state,
            )

            # Safely extract sample_weight if it exists
            sample_weight_key = base_pipe.steps[-1][0] + "__sample_weight"
            bag_fit_params = {}
            if sample_weight_key in fit_params:
                bag_fit_params["sample_weight"] = fit_params[sample_weight_key]

            bag = bag.fit(feat, labels, **bag_fit_params)
            return bag  # Return BaggingClassifier directly

        return gs

    finally:
        # Restore original n_jobs to avoid side effects
        if orig_n_jobs:
            restore_params = {f"{name}__n_jobs": orig_n_jobs[name] for name in orig_n_jobs}
            try:
                pipe_clf.set_params(**restore_params)
            except Exception:
                pass  # Best effort restore


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
