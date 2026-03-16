"""
clf_hyper_fit.py
----------------
Hyperparameter optimisation with Purged K-Fold Cross-Validation for
financial machine learning, following López de Prado (AFML, Ch. 4 & 7).

Search priority when rnd_search_iter > 0:
    1. Optuna with HyperbandPruner   (preferred)
    2. RandomizedSearchCV            (fallback if optuna not installed)

When rnd_search_iter == 0, exhaustive GridSearchCV is used.

Key design decisions
--------------------
1.  Sample weights are normalized per fold so they sum to N_fold.
    Prado's snippet 4.10 normalizes weights to sum to N on the full sample,
    but never re-normalizes inside the fold loop. After PurgedKFold slices
    the full-sample weights, they no longer sum to N_fold because purging
    and embargo remove boundary observations, and overlap density varies
    across time. Per-fold normalization restores comparability.

    Formula:  w_normalized = w_raw * (N_fold / sum(w_raw))
    Effect:   mean(w_normalized) == 1  (each obs = "one average obs worth")

2.  log_loss is called with normalize=True even after per-fold weight
    normalization. These are not redundant:
        - Weight normalization  → correct relative importance *within* a fold
        - normalize=True        → comparable loss scale *across* folds of
                                  unequal size (PurgedKFold folds differ in
                                  size due to purging and embargo)

3.  HyperbandPruner is used unconditionally. With 5-10 folds per trial,
    Hyperband is the only pruner that works across trials rather than
    within them. MedianPruner and SuccessiveHalving are special cases of
    Hyperband; letting it manage brackets automatically is strictly more
    general.

4.  CPCV is intentionally absent from this function. CPCV is a backtesting
    tool that produces a distribution of equity curves for a fixed, already-
    tuned model. Using it inside GridSearchCV is a category error: it has no
    scalar output, is combinatorially expensive, and its paths are not
    independent. PurgedKFold is the correct inner CV for hyperparameter
    selection.

Dependencies
------------
    pip install scikit-learn pandas numpy optuna
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..cache.unified_cache_system import (
    cacheable,
    create_cacheable_param_grid,
    reconstruct_param_grid,
)
from ..util.pipelines import MyPipeline, make_custom_pipeline, set_pipeline_params


# ============================================================================
# Weighted scorer — sklearn path (GridSearchCV / RandomizedSearchCV)
# ============================================================================

def _build_weighted_scorer(scoring: str, sample_weight=None) -> object:
    """
    Build a make_scorer instance that forwards sample_weight to the metric.

    When sample_weight is provided, each observation's contribution to the
    CV score is proportional to its uniqueness weight, preventing the scorer
    from treating informationally redundant (overlapping) bars as equal to
    independently drawn ones (Prado, AFML Ch. 4).

    Requires sklearn >= 1.3 for metadata routing. Falls back to an
    unweighted scorer on older versions with a UserWarning.

    Parameters
    ----------
    scoring : str
        "f1" for meta-labeling (binary) or "neg_log_loss" for multi-class.
    sample_weight : array-like or None
        Full-sample weights. When None, returns the plain string scorer —
        behaviour is identical to the original code.

    Returns
    -------
    scorer : str or sklearn scorer
    """
    if sample_weight is None:
        return scoring

    if scoring == "f1":
        scorer = make_scorer(
            f1_score,
            response_method="predict",
        )
    else:  # "neg_log_loss"
        scorer = make_scorer(
            log_loss,
            greater_is_better=False,
            response_method="predict_proba",
            # normalize=True so that folds of unequal size (due to purging
            # and embargo) produce comparable loss values before averaging.
            normalize=True,
        )

    try:
        # sklearn >= 1.3: tell the CV machinery to slice and forward
        # sample_weight to this scorer on every fold automatically.
        scorer.set_score_request(sample_weight=True)
    except AttributeError:
        warnings.warn(
            "sklearn >= 1.3 is required for weighted scoring via metadata "
            "routing. Falling back to unweighted scorer. Upgrade sklearn to "
            "incorporate sample_weight in CV score computation as recommended "
            "by Prado (AFML, Ch. 4).",
            UserWarning,
            stacklevel=3,
        )
        return scoring

    return scorer


def _enable_metadata_routing() -> None:
    """
    Activate sklearn's global metadata routing config.

    Required for sample_weight to be forwarded to scorers inside
    GridSearchCV / RandomizedSearchCV. No-op on sklearn < 1.3.
    """
    try:
        import sklearn
        sklearn.set_config(enable_metadata_routing=True)
    except (AttributeError, TypeError):
        pass


# ============================================================================
# Per-fold scoring — Optuna path
# ============================================================================

def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize sample weights so they sum to the fold size (mean == 1).

    This is the per-fold complement to Prado's full-sample normalization
    in snippet 4.10. After PurgedKFold slices the full-sample weights,
    they no longer sum to N_fold because:

        1. Purging and embargo remove boundary observations.
        2. Label overlap density — and therefore total raw weight — varies
           across time, so folds covering different regimes have different
           weight sums even at equal size.

    Re-normalizing here restores comparability between folds while
    preserving the relative down-weighting of overlapping observations.

    Formula:  w_normalized = w_raw * (N_fold / sum(w_raw))

    Parameters
    ----------
    w : np.ndarray
        Raw sample weights for one fold (train or test slice).

    Returns
    -------
    np.ndarray
        Normalized weights summing to len(w).
    """
    return w * (len(w) / w.sum())


def _compute_fold_score(
    estimator,
    features: pd.DataFrame,
    labels: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scoring: str,
    fit_params: dict,
) -> float:
    """
    Fit an estimator on one purged fold and return a normalized, weighted score.

    Score is always returned in higher-is-better convention so that Optuna
    can maximise uniformly regardless of the metric used.

    Parameters
    ----------
    estimator : sklearn estimator
        A cloned, unfitted estimator for this fold.
    features : pd.DataFrame
        Full feature matrix.
    labels : pd.Series
        Full label series.
    train_idx : np.ndarray
        Integer row indices for the training fold.
    test_idx : np.ndarray
        Integer row indices for the test fold.
    scoring : str
        "f1" or "neg_log_loss".
    fit_params : dict
        Full-sample fit params. sample_weight is sliced and normalized
        per fold automatically when present.

    Returns
    -------
    float
        Fold score in higher-is-better convention.
    """
    X_train = features.iloc[train_idx]
    y_train = labels.iloc[train_idx]
    X_test  = features.iloc[test_idx]
    y_test  = labels.iloc[test_idx]

    train_weight = None
    test_weight  = None

    if "sample_weight" in fit_params:
        raw_train = fit_params["sample_weight"][train_idx]
        raw_test  = fit_params["sample_weight"][test_idx]

        # Re-normalize within each fold so weights sum to fold size.
        # See _normalize_weights docstring for full rationale.
        train_weight = _normalize_weights(raw_train)
        test_weight  = _normalize_weights(raw_test)

    fold_fit_params = (
        {"sample_weight": train_weight} if train_weight is not None else {}
    )
    estimator.fit(X_train, y_train, **fold_fit_params)

    if scoring == "f1":
        y_pred = estimator.predict(X_test)
        score  = f1_score(y_test, y_pred, sample_weight=test_weight)

    else:  # "neg_log_loss"
        y_prob = estimator.predict_proba(X_test)
        # normalize=True: divide by N_fold so folds of unequal size are
        # comparable. Per-fold weight normalization and normalize=True
        # solve different problems and are both required — see module
        # docstring for the full explanation.
        score = -log_loss(
            y_test,
            y_prob,
            sample_weight=test_weight,
            normalize=True,
        )

    return score


# ============================================================================
# Optuna pruner
# ============================================================================

def _build_optuna_pruner(cv: int) -> "optuna.pruners.BasePruner":
    """
    Return a HyperbandPruner calibrated to the number of CV folds.

    HyperbandPruner is the unconditional choice here because:

    - It operates *across* trials (promoting the top 1/3 forward after
      fold 1) rather than *within* them. With only 5-10 intermediate
      values per trial this is the only pruner that produces meaningful
      savings.
    - MedianPruner and SuccessiveHalvingPruner are special cases of
      Hyperband. Letting Hyperband manage brackets automatically is
      strictly more general than either.
    - reduction_factor=3: roughly 2/3 of trials are pruned after fold 1,
      which is the right aggression level when each fold is an expensive
      PurgedKFold evaluation.

    Parameters
    ----------
    cv : int
        Number of CV folds — sets max_resource so the pruner is
        calibrated to the actual fold count.

    Returns
    -------
    optuna.pruners.HyperbandPruner
    """
    import optuna

    return optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=cv,
        reduction_factor=3,
    )


# ============================================================================
# Optuna search
# ============================================================================

def _optuna_search(
    pipe_clf,
    param_grid: dict,
    features: pd.DataFrame,
    labels: pd.Series,
    inner_cv: PurgedKFold,
    scoring: str,
    n_trials: int,
    n_jobs: int,
    random_state: Optional[int],
    verbose: int,
    fit_params: dict,
):
    """
    Run Optuna hyperparameter search with HyperbandPruning over PurgedKFold.

    Each trial:
        1. Samples a parameter combination via TPESampler.
        2. Runs the purged CV fold by fold.
        3. Reports the running mean fold score to HyperbandPruner.
        4. Raises TrialPruned if the pruner decides to stop early.

    After the study completes, the best parameters are refitted on the
    full dataset.

    Parameters
    ----------
    pipe_clf : Pipeline
        Cloned, ready-to-fit pipeline with n_jobs=1 already set.
    param_grid : dict
        Hyperparameter grid in sklearn style, e.g.:
            {'clf__max_depth': [3, 5, 7], 'clf__n_estimators': [100, 200]}
    features : pd.DataFrame
        Full feature matrix.
    labels : pd.Series
        Full label series.
    inner_cv : PurgedKFold
        Purged CV splitter pre-configured with t1 and pct_embargo.
    scoring : str
        "f1" or "neg_log_loss".
    n_trials : int
        Number of Optuna trials (equivalent to rnd_search_iter).
    n_jobs : int
        Parallel jobs passed to study.optimize.
    random_state : int or None
        Seed for TPESampler reproducibility.
    verbose : int
        0 = silent, >0 = progress bar.
    fit_params : dict
        Full-sample fit params. sample_weight is sliced and normalized
        per fold inside _compute_fold_score.

    Returns
    -------
    best_params : dict
        Best hyperparameters found by Optuna.
    best_score : float
        Mean CV score for the best trial (higher-is-better).
    best_estimator : Pipeline
        Best pipeline refitted on the full dataset.
    """
    import optuna
    from .optuna_hyper_fit import FinancialModelSuggester as suggester


    if verbose == 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner  = _build_optuna_pruner(cv=inner_cv.n_splits)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: "optuna.Trial") -> float:
        # Apply both weighting params and base model params
        trial_pipe = suggester.suggest_and_apply(
        trial, pipe_clf, param_grid, events, data_index
        )      

        fold_scores = []
        splits = list(inner_cv.split(features, labels))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            score = _compute_fold_score(
                estimator=clone(trial_pipe),
                features=features,
                labels=labels,
                train_idx=train_idx,
                test_idx=test_idx,
                scoring=scoring,
                fit_params=fit_params,
            )
            fold_scores.append(score)

            # Report running mean to the pruner after each fold.
            # HyperbandPruner uses these intermediate values to decide
            # whether to promote this trial to the next resource level.
            trial.report(float(np.mean(fold_scores)), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=verbose > 0,
    )

    # Refit the best parameter combination on the full dataset.
    best_pipe = clone(pipe_clf)
    best_pipe.set_params(**study.best_params)
    best_pipe.fit(features, labels, **fit_params)

    return study.best_params, study.best_value, best_pipe


# ============================================================================
# Main entry point
# ============================================================================

def clf_hyper_fit(
    features: pd.DataFrame,
    labels: pd.Series,
    t1: pd.Series,
    pipe_clf,
    param_grid: dict,
    cv: int = 5,
    bagging_n_estimators: int = 0,
    bagging_max_samples: float = 1.0,
    bagging_max_features: float = 1.0,
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    pct_embargo: float = 0.02,
    random_state: Optional[int] = None,
    verbose: int = 0,
    use_optuna: bool = True,
    **fit_params,
):
    """
    Hyperparameter fitting with Purged K-Fold Cross-Validation.

    Search priority
    ---------------
    rnd_search_iter == 0              →  GridSearchCV (exhaustive)
    rnd_search_iter  > 0, use_optuna  →  Optuna + HyperbandPruner (preferred)
    rnd_search_iter  > 0, not use_optuna → RandomizedSearchCV (forced fallback)

    Sample weight handling
    ----------------------
    When sample_weight is present in fit_params it is:
        - Sliced to each fold's train and test indices
        - Normalized per fold so weights sum to the fold size
        - Passed to both estimator.fit and the scoring function

    This implements Prado's recommendation (AFML, Ch. 4) while completing
    the per-fold normalization gap he leaves unaddressed in snippet 7.4.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    labels : pd.Series
        Target labels.
    t1 : pd.Series
        Information end times for each observation, used by PurgedKFold
        to prevent lookahead leakage. Index = label start, value = label end.
    pipe_clf : estimator or Pipeline
        Preprocessing and classification pipeline.
    param_grid : dict or list of dicts
        Hyperparameter search space, e.g.:
            {'clf__max_depth': [3, 5], 'clf__n_estimators': [100, 200]}
    cv : int, default=5
        Number of purged K-Fold splits.
    bagging_n_estimators : int, default=0
        If > 0, wraps the best estimator in a BaggingClassifier fitted
        on the full dataset. If 0, returns the best single estimator.
    bagging_max_samples : float or int, default=1.0
        Samples per bagging base estimator.
    bagging_max_features : float or int, default=1.0
        Features per bagging base estimator.
    rnd_search_iter : int, default=0
        Optuna trials (or RandomizedSearchCV iterations if Optuna is
        unavailable). 0 = exhaustive GridSearchCV.
    n_jobs : int, default=-1
        Parallel jobs. -1 uses all cores.
    pct_embargo : float, default=0.02
        Fraction of samples to embargo after each test fold.
    random_state : int or None, default=None
        Reproducibility seed.
    verbose : int, default=0
        Verbosity level.
    use_optuna : bool, default=True
        When True and rnd_search_iter > 0, uses Optuna with HyperbandPruner
        if optuna is installed, falling back to RandomizedSearchCV if not.
        When False and rnd_search_iter > 0, forces RandomizedSearchCV
        unconditionally — useful for benchmarking Optuna's improvement over
        a standard randomized search on the same number of iterations.
    **fit_params
        Additional params passed to estimator.fit, e.g. sample_weight.

    Returns
    -------
    estimator : Pipeline
        Trained model (possibly wrapped in BaggingClassifier).
    cv_results : dict
        Keys:
            best_params     : dict   — best hyperparameters
            best_score      : float  — best CV score (higher-is-better)
            cv_results      : DataFrame or None
            scoring         : str    — metric used
            weighted_scoring: bool   — whether sample_weight was active
            search_method   : str    — "optuna", "randomized", or "grid"
    """
    from .cross_validation import PurgedKFold

    # ------------------------------------------------------------------
    # Pipeline setup
    # ------------------------------------------------------------------
    pipe_clf = make_custom_pipeline(clone(pipe_clf))
    name_of_clf, estimator = pipe_clf.steps[-1]

    # Set n_jobs=1 on the inner estimator to avoid nested parallelism
    # when the outer search already uses n_jobs > 1.
    pipe_clf = set_pipeline_params(pipe_clf, n_jobs=1)

    # Validate and prefix param_grid keys
    for k in reversed(list(param_grid.keys())):
        if not hasattr(estimator, k.split(f"{name_of_clf}__")[-1]):
            param_grid.pop(k)
        elif not k.startswith(f"{name_of_clf}__"):
            param_grid[f"{name_of_clf}__{k}"] = param_grid.pop(k)

    # ------------------------------------------------------------------
    # Scoring metric
    # ------------------------------------------------------------------
    sample_weight = fit_params.get("sample_weight", None)

    # Binary {0,1} labels indicate meta-labeling; use F1.
    # All other cases use negative log loss (probabilistic output).
    if set(labels.unique()) == {0, 1}:
        base_scoring = "f1"
    else:
        base_scoring = "neg_log_loss"

    # Enable sklearn metadata routing before constructing search objects
    # so sample_weight is forwarded correctly on every fold.
    if sample_weight is not None:
        _enable_metadata_routing()

    # ------------------------------------------------------------------
    # Cross-validation splitter
    # ------------------------------------------------------------------
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    if rnd_search_iter > 0:
        if use_optuna:
            try:
                import optuna  # noqa: F401

                best_params, best_score, best_estimator = _optuna_search(
                    pipe_clf=pipe_clf,
                    param_grid=param_grid,
                    features=features,
                    labels=labels,
                    inner_cv=inner_cv,
                    scoring=base_scoring,
                    n_trials=rnd_search_iter,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=verbose,
                    fit_params=fit_params,
                )
                cv_results = {
                    "best_params":      best_params,
                    "best_score":       best_score,
                    "cv_results":       None,   # Optuna has no cv_results_ DataFrame
                    "scoring":          base_scoring,
                    "weighted_scoring": sample_weight is not None,
                    "search_method":    "optuna",
                }

            except ImportError:
                warnings.warn(
                    "Optuna is not installed. Falling back to RandomizedSearchCV. "
                    "Install optuna for pruning support: pip install optuna",
                    UserWarning,
                    stacklevel=2,
                )
                scoring = _build_weighted_scorer(base_scoring, sample_weight)
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
                gs.fit(features, labels, **fit_params)
                best_estimator = gs.best_estimator_
                cv_results = {
                    "best_params":      gs.best_params_,
                    "best_score":       gs.best_score_,
                    "cv_results":       pd.DataFrame(gs.cv_results_),
                    "scoring":          base_scoring,
                    "weighted_scoring": sample_weight is not None,
                    "search_method":    "randomized",
                }
        else:
            # use_optuna=False: bypass Optuna unconditionally.
            # Useful for benchmarking Optuna's gains over a plain
            # randomized search on the same number of iterations.
            scoring = _build_weighted_scorer(base_scoring, sample_weight)
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
            gs.fit(features, labels, **fit_params)
            best_estimator = gs.best_estimator_
            cv_results = {
                "best_params":      gs.best_params_,
                "best_score":       gs.best_score_,
                "cv_results":       pd.DataFrame(gs.cv_results_),
                "scoring":          base_scoring,
                "weighted_scoring": sample_weight is not None,
                "search_method":    "randomized",
            }

    else:
        scoring = _build_weighted_scorer(base_scoring, sample_weight)
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
        gs.fit(features, labels, **fit_params)
        best_estimator = gs.best_estimator_
        cv_results = {
            "best_params":      gs.best_params_,
            "best_score":       gs.best_score_,
            "cv_results":       pd.DataFrame(gs.cv_results_),
            "scoring":          base_scoring,
            "weighted_scoring": sample_weight is not None,
            "search_method":    "grid",
        }

    # ------------------------------------------------------------------
    # Optional bagging
    # ------------------------------------------------------------------
    if bagging_n_estimators > 0:
        base_estimator = set_pipeline_params(best_estimator, n_jobs=1)
        bag = BaggingClassifier(
            estimator=MyPipeline(base_estimator.steps),
            n_estimators=int(bagging_n_estimators),
            max_samples=bagging_max_samples,
            max_features=bagging_max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if "sample_weight" in fit_params:
            bag.fit(features, labels, sample_weight=fit_params["sample_weight"])
        else:
            bag.fit(features, labels)

        return Pipeline([("bag", bag)]), cv_results
    else:
        return Pipeline(best_estimator.steps), cv_results


# ============================================================================
# Cached variants
# ============================================================================

@cacheable()
def clf_hyper_fit_internal(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid_cacheable,
    cv,
    bagging_n_estimators,
    bagging_max_samples,
    bagging_max_features,
    rnd_search_iter,
    n_jobs,
    pct_embargo,
    random_state,
    verbose,
    use_optuna=True,
    **fit_params,
):
    """
    Cached version of clf_hyper_fit.

    Reconstructs param_grid from its cacheable representation (which
    serializes scipy distributions and other non-picklable objects) before
    delegating to clf_hyper_fit. All weighted scoring and per-fold
    normalization behaviour is inherited unchanged.

    use_optuna is threaded through so the bypass flag works correctly
    from the cached entry point too.
    """
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
        use_optuna=use_optuna,
        **fit_params,
    )


def clf_hyper_fit_cached(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid,
    cv,
    bagging_n_estimators,
    bagging_max_samples,
    bagging_max_features,
    rnd_search_iter,
    n_jobs,
    pct_embargo,
    random_state,
    verbose,
    use_optuna=True,
    **fit_params,
):
    """
    Convenience wrapper that converts param_grid to a cacheable format
    before calling the cached implementation.

    Usage
    -----
        from scipy.stats import randint

        # Standard call — uses Optuna if installed
        model, results = clf_hyper_fit_cached(
            features, labels, t1, pipe_clf,
            param_grid={
                'clf__n_estimators': randint(100, 500),
                'clf__max_depth':    randint(3, 20),
            },
            cv=5,
            rnd_search_iter=50,
            sample_weight=uniqueness_weights,
        )
        print(results["search_method"])    # "optuna"

        # Benchmark call — forces RandomizedSearchCV even if Optuna is installed
        model_rs, results_rs = clf_hyper_fit_cached(
            ...,
            rnd_search_iter=50,
            use_optuna=False,
        )
        print(results_rs["search_method"]) # "randomized"

        # Compare best_score between the two to demonstrate Optuna's advantage.
    """
    param_grid_cacheable = create_cacheable_param_grid(param_grid)

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
        use_optuna=use_optuna,
        **fit_params,
    )
