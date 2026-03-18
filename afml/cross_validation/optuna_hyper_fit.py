import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path 

import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import sqlite3
from loguru import logger
from optuna import TrialPruned, create_study
from optuna.exceptions import StorageInternalError
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss

from afml.cross_validation.cross_validation import PurgedKFold
from afml.production.model_development import _WeightedEstimator


class FinancialModelSuggester:
    """
    Translates Scikit-Learn style distribution dictionaries into
    Optuna trial suggestions for rigorous statistical HPT.

    Two core methods form a dual pair:
        suggest_and_apply  — Trial → params → model  (stochastic, used in objective)
        apply_from_params  — params → model           (deterministic, used for refit)
    """

    # ----- Central registry of weighting hyperparameters -----
    # Both methods reference this to separate weight keys from model keys.
    WEIGHT_KEYS = frozenset({"weight_scheme", "weight_decay", "weight_linear"})

    # ==================== SEARCH-TIME ====================

    @classmethod
    def suggest_and_apply(
        cls,
        trial: optuna.Trial,
        base_model,
        param_distributions: dict,
        events: pd.DataFrame,
        data_index: pd.DatetimeIndex,
    ):
        """
        Samples hyperparameters from distributions via an Optuna Trial
        and returns a configured _WeightedEstimator.

        Used inside the objective function during optimization.
        """
        # 1. Suggest Weighting Parameters
        scheme = trial.suggest_categorical("weight_scheme", ["unweighted", "uniqueness", "return"])
        decay = trial.suggest_float("weight_decay", 0.1, 1.0)
        linear = trial.suggest_categorical("weight_linear", [True, False])

        # 2. Suggest Base Model Parameters
        sampled_params = {}
        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                sampled_params[name] = trial.suggest_categorical(name, dist)
            elif hasattr(dist, 'ppf'):  # scipy.stats distribution
                low, high = dist.support()
                if dist.dist.name == "randint":
                    sampled_params[name] = trial.suggest_int(name, int(low), int(high))    
                else:
                    is_log = dist.dist.name in ['reciprocal', 'loguniform']
                    sampled_params[name] = trial.suggest_float(name, low, high, log=is_log)               
            elif isinstance(dist, range):
                sampled_params[name] = trial.suggest_int(name, dist.start, dist.stop - 1)
            else:
                sampled_params[name] = dist

        # 3. Clone and configure
        new_base = clone(base_model)
        new_base.set_params(**sampled_params)

        return _WeightedEstimator(
            base_estimator=new_base,
            events=events,
            data_index=data_index,
            scheme=scheme,
            decay=decay,
            linear=linear,
        )

    # ==================== REFIT-TIME ====================

    @classmethod
    def apply_from_params(
        cls,
        params: dict,
        base_model,
        events: pd.DataFrame,
        data_index: pd.DatetimeIndex,
    ) -> "_WeightedEstimator":
        """
        Reconstruct a WeightedEstimator from a flat params dict.

        This is the deterministic counterpart to suggest_and_apply():
        it takes resolved values (e.g. study.best_params) instead of
        sampling from distributions via a Trial.

        Parameters
        ----------
        params : dict
            Flat dictionary — typically study.best_params.
        base_model : estimator
            Unfitted sklearn-compatible classifier template.
        events : pd.DataFrame
            Event metadata with 't1', 'w', 'tW' columns.
        data_index : pd.DatetimeIndex
            Timestamps of bars in the training period.

        Returns
        -------
        _WeightedEstimator
            Unfitted estimator configured with the given params.

        Raises
        ------
        ValueError
            If params contains keys invalid for the base model.
        """
        weight_params = {k: params[k] for k in cls.WEIGHT_KEYS if k in params}
        model_params = {k: v for k, v in params.items() if k not in cls.WEIGHT_KEYS}

        # Validate against base model's accepted parameters
        valid_keys = set(base_model.get_params().keys())
        invalid = set(model_params) - valid_keys
        if invalid:
            raise ValueError(
                f"Parameters {invalid} are not valid for "
                f"{type(base_model).__name__}. "
                f"Valid: {sorted(valid_keys)}"
            )

        new_base = clone(base_model)
        new_base.set_params(**model_params)

        return _WeightedEstimator(
            base_estimator=new_base,
            events=events,
            data_index=data_index,
            scheme=weight_params.get("weight_scheme", "unweighted"),
            decay=weight_params.get("weight_decay", 1.0),
            linear=weight_params.get("weight_linear", False),
        )

    # ==================== SEARCH SPACES ====================

    @classmethod
    def get_search_space(cls, model_name: str):
        """Returns curated parameter distributions for financial models."""
        spaces = {
            "random_forest": {
                "n_estimators": range(100, 1000),
                "max_depth": range(3, 7),
                "min_weight_fraction_leaf": stats.uniform(0.025, 0.1),
                "max_features": ["sqrt", "log2", 0.5, 1.0],
                "ccp_alpha": stats.loguniform(1e-5, 1e-2),
            },
            "xgboost": {
                "n_estimators": range(100, 1000),
                "learning_rate": stats.loguniform(1e-3, 0.1),
                "max_depth": range(2, 8),
                "subsample": stats.uniform(0.6, 0.4),
                "colsample_bytree": stats.uniform(0.6, 0.4),
                "gamma": stats.uniform(0, 5),
            },
        }
        return spaces.get(model_name.lower(), {})

        
def optimize_trading_model_with_pruning(
    trial: "optuna.Trial",
    X, y, events, data_index,
    classifier,
    param_distributions: dict,
    n_splits: int = 5,
    metric="neg_log_loss",
):
    """
    Objective function for tuning models using Purged K-Fold cross-validation.
    """

    suggester = FinancialModelSuggester()
    # Apply both weighting params and base model params
    model = suggester.suggest_and_apply(
        trial, classifier, param_distributions, events, data_index
    )
        
    # Setup Cross-Validation
    t1 = events['t1']
    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=0.01)
    # Extract return-attribution weights from to evaluate validation set with respect to PnL
    w_score = events['w'].to_numpy()
        
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fit = clone(model).fit(X_train, y_train) # No need for weights due to _WeightedEstimator       
        w_val = w_score[val_idx]
        
        if metric == "neg_log_loss":
            y_prob = fit.predict_proba(X_val)                      
            score = -log_loss(y_val, y_prob, sample_weight=w_val)
        else:
            y_pred = fit.predict(X_val)
            score = f1_score(y_val, y_pred, sample_weight=w_val)            
        fold_scores.append(score)
        trial.report(score, step=fold_idx)

        if trial.should_prune():
            avg_score_so_far = np.mean(fold_scores)
            trial.set_user_attr("pruned_at_fold", fold_idx)
            trial.set_user_attr("score_when_pruned", avg_score_so_far)
            trial.set_user_attr("total_folds_attempted", len(fold_scores))
            raise TrialPruned(f"Pruned at fold {fold_idx}. Avg score: {avg_score_so_far:.4f}")

    final_score = np.mean(fold_scores)
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("score_std", np.std(fold_scores))
    return final_score
    

class TradingModelPruner(MedianPruner):
    """
    Financial-aware pruner that adjusts thresholds based on label entropy 
    and return-attribution weighted volatility.
    """
    def __init__(
        self,
        y,
        sample_weight,  # This should be your return-based weights |r|
        n_startup_trials: int = 10,
        n_warmup_steps: int = 2,
        multiplier: float = 1.15, # How much worse than baseline is 'trash'?
    ):
        super().__init__(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
        
        # 1. Calculate Baseline Entropy (The "Naïve" score)
        # We use weights to see the economic baseline - i.e., return-attribution (events['w'])
        weighted_counts = pd.Series(sample_weight).groupby(y.values).sum()
        probs = weighted_counts / weighted_counts.sum()

        if set(y.unique()) != {0,1}:
            self.baseline_entropy = -np.sum(probs * np.log(probs))
            # Threshold: e.g., if baseline is -0.5, threshold is -0.5 * 1.15 = -0.575
            self.min_score_threshold = -self.baseline_entropy * multiplier
        else:
            # For F1, baseline = weighted class prior accuracy
            majority_ratio = probs.max()
            self.min_score_threshold = majority_ratio / multiplier # e.g., 0.52 / 1.15 = 0.45
        
        logger.info(f"Minimum Score Threshold: {self.min_score_threshold:.4f}")
        
        # 2. Dynamic Volatility Tolerance
        # Higher return volatility = higher tolerance for score swings between folds
        # We use the Coefficient of Variation of weights as a proxy for noise
        weight_cv = np.std(sample_weight) / np.mean(sample_weight)
        self.volatility_tolerance = 0.1 * (1 + weight_cv)

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        if step is None: return False
        
        if trial.number >= 5 and len(trial.intermediate_values) >= 3:
            # Rule 1: Static baseline check (Is it worse than a coin flip/baseline?)
            current_score = trial.intermediate_values.get(step)
            if trial.number > 1 and current_score < self.min_score_threshold:
                return True

            # Rule 2: High-Variance check (Is the model unstable?)
            recent_scores = list(trial.intermediate_values.values())[-3:]
            if np.std(recent_scores) > self.volatility_tolerance:
                return True

        # Rule 3: Median Pruning (Standard Optuna logic)
        return super().prune(study, trial)


def optimize_trading_model(
    classifier,
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    data_index: pd.DatetimeIndex,
    param_distributions: dict,
    n_trials: int = 100,
    timeout: int = 3600,
    n_splits: int = 5,
    pruner_type: str = "hyperband",
    metric: str = "neg_log_loss",
    study_name: str = None,
    db_path: str = None,
    random_state: int = 42,
    refit: bool = True,
):
    """
    Executes a high-performance hyperparameter optimization (HPT) study for trading models.

    This orchestrator integrates temporal purging, automated parameter sampling, 
    and multi-stage pruning to identify robust model configurations while 
    minimizing computational waste.

    Args:
        classifier (estimator): A Scikit-Learn compatible classifier template.
        X (pd.DataFrame): Feature matrix with index aligned to 'events'.
        y (pd.Series): Binary or multi-class labels for training.
        events (pd.DataFrame): Event metadata; must contain 't1' column (observation end times).
        data_index (pd.DatetimeIndex): Timestamps of bars in training period
        param_distributions (dict): Search space template.
        n_trials (int): Maximum number of unique hyperparameter combinations to evaluate.
        timeout (int): Total search time limit in seconds.
        n_splits (int): Number of folds for PurgedKFold.
        pruner_type (str): Pruning strategy ('median', 'hyperband', or 'successive_halving').
        metric (str): Optimization objective ('neg_log_loss' or 'f1').
        study_name (str): Name of Optuna study.
        db_path (str): Path to store trials.      
        refit (bool): Fit best model on full data.
        
    Returns:
        optuna.study.Study: The completed study object with history and best params.
    """
    if pruner_type == "median":
        pruner = TradingModelPruner(y, sample_weight=events.loc[X.index, 'w']) 
    elif pruner_type == "hyperband":
        pruner = HyperbandPruner(min_resource=1, max_resource=n_splits, reduction_factor=3)
    else:
        pruner = SuccessiveHalvingPruner()

    sampler = TPESampler(seed=random_state)

    # Add a 30-second timeout for parallel workers
    if str(db_path).startswith("sqlite") and str(db_path).endswith("db"):
        storage_url = f"{db_path}?timeout=30"
    else:
        storage_url = f"sqlite:///{db_path}.db?timeout=30"
    
    try:
        study = optuna.create_study(
            direction="maximize", 
            sampler=sampler, 
            pruner=pruner, 
            study_name=study_name, 
            storage=storage_url, 
            load_if_exists=True,
        )
        
        # Force single-threaded inside CV to prevent oversubscription
        if hasattr(classifier, 'n_jobs'):
            classifier.set_params(n_jobs=1)
    
        # Ensure reproducibility
        if hasattr(classifier, 'random_state'):
            classifier.set_params(random_state=random_state)
    
        def objective(trial):
            return optimize_trading_model_with_pruning(
                trial, X, y, events, data_index, classifier,
                param_distributions, n_splits, metric
            )
    
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout, 
            callbacks=[print_best_trial, save_intermediate_results, check_for_overfitting]
        )
        
        # Reconstruct best model from best params
        if refit:
            best_trial = study.best_trial
            best_model = FinancialModelSuggester.apply_from_params(
                best_trial.params, classifier, events, data_index
            )

            # Return to parallelized mode
            if hasattr(classifier, 'n_jobs'):
                best_model.set_params(n_jobs=-1)
            
            best_model.fit(X, y)
            study.best_estimator_ = best_model  # attach for convenience
            
        cv_results = optuna_to_cv_results(study)
        return study, cv_results
        
    except StorageInternalError as e:
        print(f"❌ Storage Error: The database at {db_path} is locked or unreachable.")
        print(f"Technical details: {e}")
        # In a production environment, you might trigger a retry logic here
    except Exception as e:
        print(f"❌ Unexpected Error during study: {e}")
        raise e


def print_best_trial(study, trial):
    if study.best_trial.number == trial.number:
        print(f"\n🎯 New best trial #{trial.number} | Score: {trial.value:.4f}")


def save_intermediate_results(study, trial):
    results_dir = Path("optuna_results")
    results_dir.mkdir(exist_ok=True)
    trial_data = {
        "trial": trial.number, "value": trial.value, "params": trial.params,
        "state": str(trial.state), "user_attrs": trial.user_attrs
    }
    with open(results_dir / f"trial_{trial.number:04d}.json", "w") as f:
        json.dump(trial_data, f, indent=2, default=str)


def check_for_overfitting(study, trial):
    scores = trial.user_attrs.get("fold_scores", [])
    score_std = trial.user_attrs.get("score_std", 0.0)
    if len(scores) >= 3 and score_std > 0.3:
        print(f"⚠️ High variance detected in Trial {trial.number}")
        

def plot_model_vs_baseline(study, y, events):
    """
    Visualizes the best model's performance relative to the entropy baseline
    and market volatility across CV folds.
    """
    # 1. Re-calculate baseline for the whole period
    weighted_counts = events['w'].groupby(y.values).sum()
    probs = weighted_counts / weighted_counts.sum()
    
    if set(np.unique(y)) == {0, 1}:
        baseline = -probs.max()
        scoring = "F1"
    else:
        baseline = -np.sum(probs * np.log(probs))
        scoring = "Weighted Log-Loss"
    
    # 2. Extract best trial data
    best_trial = study.best_trial
    fold_scores = best_trial.user_attrs.get("fold_scores", [])
    
    # 3. Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the scores
    folds = range(len(fold_scores))
    plt.plot(folds, fold_scores, marker='o', label=f'Best Model ({scoring})', color='#1f77b4', lw=2)
    
    # Plot the baseline
    plt.axhline(y=-baseline, color='red', linestyle='--', label='Naive Baseline (Entropy)', alpha=0.7)
    
    # Fill the 'Alpha' area
    plt.fill_between(folds, fold_scores, -baseline, where=(np.array(fold_scores) > -baseline), 
                     color='green', alpha=0.1, label='Economic Edge')

    plt.title(f"Best Trial #{best_trial.number}: Performance vs. Information Baseline", fontsize=14)
    plt.xlabel("Cross-Validation Fold", fontsize=12)
    plt.ylabel(f"{scoring} Score (Higher is Better)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def optuna_to_cv_results(study):
    """Converts an Optuna study into a Scikit-Learn style cv_results DataFrame."""
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        # Core metrics
        res = {
            "mean_test_score": trial.value,
            "std_test_score": trial.user_attrs.get("score_std", 0),
            "mean_fit_time": (trial.datetime_complete - trial.datetime_start).total_seconds(),
            "params": trial.params,
        }
        
        # Add individual parameters with 'param_' prefix
        for k, v in trial.params.items():
            res[f"param_{k}"] = v
            
        # Add fold-specific scores for '6. CROSS-VALIDATION CONSISTENCY'
        fold_scores = trial.user_attrs.get("fold_scores", [])
        for i, score in enumerate(fold_scores):
            res[f"split{i}_test_score"] = score
            
        rows.append(res)
    
    return pd.DataFrame(rows)
