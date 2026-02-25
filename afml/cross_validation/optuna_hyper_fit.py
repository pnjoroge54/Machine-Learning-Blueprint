from pathlib import Path
import json
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
from optuna import TrialPruned, create_study
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss

from ..cross_validation.cross_validation import PurgedKFold
from ..production.model_development import _WeightedEstimator


class FinancialModelSuggester:
    """
    Translates Scikit-Learn style distribution dictionaries into 
    Optuna trial suggestions for rigorous statistical HPT.
    """
    
    @staticmethod
    def suggest_and_apply(trial: optuna.Trial, base_model, param_distributions: dict, events: pd.DataFrame, data_index: pd.DatetimeIndex):
        """
        Samples hyperparameters from a dictionary and applies them to a cloned model instance.

        Args:
            trial: The Optuna trial object.
            base_model: A pre-instantiated sklearn-style model or pipeline.
            param_distributions: A dict of distributions (list, range, or scipy.stats).
            events: Training events for the model.
            data_index: Timestamps of each bar in the training period.

        Engineering Note:
            Continuous distributions from scipy.stats are mapped to trial.suggest_float. 
            The method automatically detects 'reciprocal' or 'loguniform' distributions 
            to enable log-scaling in Optuna, which is critical for parameters like 
            learning rates and regularization coefficients (C, gamma, lambda).
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
            elif hasattr(dist, 'ppf'): # scipy.stats
                low, high = dist.support()
                is_log = dist.dist.name in ['reciprocal', 'loguniform']
                sampled_params[name] = trial.suggest_float(name, low, high, log=is_log)
            elif isinstance(dist, range):
                sampled_params[name] = trial.suggest_int(name, dist.start, dist.stop - 1)
            else:
                sampled_params[name] = dist

        # 3. Create the Weighted Estimator
        # We clone the base_model to keep the template pristine
        new_base = clone(base_model)
        new_base.set_params(**sampled_params)
        
        weighted_model = _WeightedEstimator(
            base_estimator=new_base,
            events=events,
            data_index=data_index,
            scheme=scheme,
            decay=decay,
            linear=linear
        )
        return weighted_model
        
    @classmethod
    def get_search_space(cls, model_name: str):
        """
        Returns a dictionary of parameter distributions curated for financial noise.

        Engineering Note:
            The 'random_forest' space prioritizes 'ccp_alpha' (Cost Complexity Pruning) 
            over 'max_depth' to allow for organic tree simplification, and utilizes 
            'min_weight_fraction_leaf' to ensure terminal nodes represent significant 
            economic or statistical information.
        """
        spaces = {
            "random_forest": {
                "n_estimators": range(100, 1000),
                "max_depth": range(3, 12),
                "min_weight_fraction_leaf": stats.uniform(0.01, 0.1),
                "max_features": ["sqrt", "log2", 0.5],
                "ccp_alpha": stats.reciprocal(1e-5, 1e-2)
            },
            "xgboost": {
                "n_estimators": range(100, 1000),
                "learning_rate": stats.reciprocal(1e-3, 0.1),
                "max_depth": range(2, 8),
                "subsample": stats.uniform(0.6, 0.4),
                "colsample_bytree": stats.uniform(0.6, 0.4),
                "gamma": stats.uniform(0, 5)
            }
        }
        return spaces.get(model_name.lower(), {})

        
def optimize_trading_model_with_pruning(
    trial: optuna.Trial,
    X, y, events, data_index,
    base_model_instance,
    param_distributions: dict,
    n_splits: int = 5,
    metric="neg_log_loss",
    cpcv=False,
):
    """
    Objective function for tuning models using Purged K-Fold cross-validation.
    """

    suggester = FinancialModelSuggester()
    # Apply both weighting params and base model params
    model = suggester.suggest_and_apply(
        trial, base_model_instance, param_distributions, events, data_index
    )

    # Setup Cross-Validation
    if not cpcv:
        cv = CombinatorialPurgedKFold(n_splits=n_splits+1, n_test_splits=2, t1=events.t1, pct_embargo=0.01)
    else:
        cv = PurgedKFold(n_splits=n_splits, t1=events.t1, pct_embargo=0.01)
        
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        if metric == "neg_log_loss":
            y_prob = model.predict_proba(X_val)
            # Use uniqueness weights for validation scoring to ensure statistical relevance
            w_val = events.loc[X_val.index, "tW"]
            score = log_loss(y_val, y_prob, sample_weight=w_val) * -1
        else:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
            
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
    Custom pruner that handles volatility and threshold constraints for financial models.
    """
    def __init__(
        self,
        n_startup_trials: int = 10,
        n_warmup_steps: int = 2,
        interval_steps: int = 1,
        min_score_threshold: float = -0.69,
        volatility_tolerance: float = 0.2
    ):
        super().__init__(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )
        self.min_score_threshold = min_score_threshold
        self.volatility_tolerance = volatility_tolerance

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None: return False

        current_score = trial.intermediate_values.get(step)
        if current_score < self.min_score_threshold:
            return True

        if len(trial.intermediate_values) >= 3:
            recent_scores = [trial.intermediate_values[i] for i in range(max(0, step - 2), step + 1)]
            if np.std(recent_scores) > self.volatility_tolerance:
                return True

        return super().prune(study, trial)


def optimize_trading_model_with_advanced_pruning(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    data_index: pd.DateTimeIndex,
    base_model_instance,
    param_distributions: dict,
    n_trials: int = 100,
    timeout: int = 3600,
    n_splits: int = 5,
    pruner_type: str = "median",
    metric: str = "neg_log_loss"
):
    """
    Executes a high-performance hyperparameter optimization (HPT) study for trading models.

    This orchestrator integrates temporal purging, automated parameter sampling, 
    and multi-stage pruning to identify robust model configurations while 
    minimizing computational waste.

    Args:
        X (pd.DataFrame): Feature matrix with index aligned to 'events'.
        y (pd.Series): Binary or multi-class labels for training.
        events (pd.DataFrame): Event metadata; must contain 't1' column (observation end times).
        data_index (pd.DateTimeIndex): Timestamps of bars in training period
        base_model_instance (estimator): A Scikit-Learn compatible classifier template.
        param_distributions (dict): Search space template.
        n_trials (int): Maximum number of unique hyperparameter combinations to evaluate.
        timeout (int): Total search time limit in seconds.
        n_splits (int): Number of folds for PurgedKFold.
        pruner_type (str): Pruning strategy ('median', 'hyperband', or 'successive_halving').
        metric (str): Optimization objective ('neg_log_loss' or 'f1').

    Returns:
        optuna.study.Study: The completed study object with history and best params.
    """
    if pruner_type == "median":
        pruner = TradingModelPruner(n_startup_trials=10, n_warmup_steps=2)
    elif pruner_type == "hyperband":
        pruner = HyperbandPruner(min_resource=1, max_resource=n_splits, reduction_factor=3)
    else:
        pruner = SuccessiveHalvingPruner()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial):
        return optimize_trading_model_with_pruning(
            trial=trial, X=X, y=y, events=events, data_index=data_index,
            base_model_instance=base_model_instance,
            param_distributions=param_distributions,
            n_splits=n_splits, metric=metric
        )

    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout, 
        callbacks=[print_best_trial, save_intermediate_results, check_for_overfitting]
    )
    return study


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
    if len(scores) >= 3 and (max(scores) - min(scores)) > 0.3:
        print(f"⚠️ High variance detected in Trial {trial.number}")
