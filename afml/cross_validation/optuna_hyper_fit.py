from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna import TrialPruned, create_study
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from afml.cross_validation.cross_validation import PurgedKFold


def optimize_trading_model_with_pruning(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    events: pd.DataFrame,
    n_splits: int = 5,
):
    """
    Optimize trading model with intelligent pruning.

    Key Features:
    1. Early pruning of unpromising trials
    2. Intermediate score reporting
    3. Custom pruning logic for trading models
    """

    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 0.01, 0.5
        ),
    }

    # Create cross-validation splits
    cv = PurgedKFold(n_splits=n_splits, t1=events.t1, pct_embargo=0.01)

    # Track scores for each fold
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weight.iloc[train_idx]
        w_val = sample_weight.iloc[val_idx]

        # Create and train model
        model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)

        # Predict and score
        y_pred = model.predict(X_val)
        fold_score = f1_score(y_val, y_pred, sample_weight=w_val)
        fold_scores.append(fold_score)

        # Report intermediate score to Optuna for pruning
        trial.report(fold_score, step=fold_idx)

        # Check if trial should be pruned
        if trial.should_prune():
            # Calculate average score so far for analysis
            avg_score_so_far = np.mean(fold_scores)

            # Log pruning event
            trial.set_user_attr("pruned_at_fold", fold_idx)
            trial.set_user_attr("score_when_pruned", avg_score_so_far)
            trial.set_user_attr("total_folds_attempted", len(fold_scores))

            raise TrialPruned(
                f"Trial pruned at fold {fold_idx}. "
                f"Average score: {avg_score_so_far:.4f}"
            )

    # If we complete all folds without pruning
    final_score = np.mean(fold_scores)

    # Store additional metrics
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("score_std", np.std(fold_scores))
    trial.set_user_attr("min_score", np.min(fold_scores))
    trial.set_user_attr("max_score", np.max(fold_scores))

    return final_score


# Create a custom pruner for trading models
class TradingModelPruner(MedianPruner):
    """
    Custom pruner for trading model optimization.

    Features:
    1. Dynamic pruning based on validation consistency
    2. Market-regime aware pruning thresholds
    3. Adaptive aggressiveness based on search progress
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        n_warmup_steps: int = 2,  # Minimum folds before pruning
        interval_steps: int = 1,
        aggressive_pruning: bool = True,
        min_score_threshold: float = 0.5,  # Minimum acceptable F1 score
        volatility_tolerance: float = 0.2,  # Allowable score volatility
    ):
        super().__init__(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )
        self.aggressive_pruning = aggressive_pruning
        self.min_score_threshold = min_score_threshold
        self.volatility_tolerance = volatility_tolerance

    def prune(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> bool:
        """Custom pruning logic for trading models."""

        # Get all completed trials
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Don't prune if we don't have enough data
        if len(completed_trials) < self._n_startup_trials:
            return False

        # Get current trial's intermediate values
        step = trial.last_step
        if step is None:
            return False

        # Check minimum score threshold
        current_score = trial.intermediate_values.get(step)
        if current_score < self.min_score_threshold:
            return True  # Prune if below minimum acceptable score

        # Check for high volatility (unstable models)
        if len(trial.intermediate_values) >= 3:
            recent_scores = [
                trial.intermediate_values[i] for i in range(max(0, step - 2), step + 1)
            ]
            score_volatility = np.std(recent_scores)
            if score_volatility > self.volatility_tolerance:
                return True  # Prune unstable models

        # Use parent's pruning logic
        return super().prune(study, trial)


# Main optimization function with pruning
def optimize_trading_model_with_advanced_pruning(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    events: pd.DataFrame,
    n_trials: int = 100,
    timeout: int = 3600,  # 1 hour timeout
    n_splits: int = 5,
    pruner_type: str = "median",  # 'median', 'hyperband', 'successive_halving'
    callback_functions: list = None,
):
    """
    Main function to run optimization with pruning.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    sample_weight : pd.Series
        Sample weights
    events : pd.DataFrame
        Event data with t1 for purged CV
    n_trials : int
        Maximum number of trials
    timeout : int
        Maximum time in seconds
    n_splits : int
        Number of CV folds
    pruner_type : str
        Type of pruner to use
    callback_functions : list
        List of callback functions to run after each trial

    Returns
    -------
    study : optuna.Study
        Complete study object with all trial results
    """

    # Select pruner based on type
    if pruner_type == "median":
        pruner = TradingModelPruner(
            n_startup_trials=10,
            n_warmup_steps=2,  # Wait for 2 folds before pruning
            interval_steps=1,
            aggressive_pruning=True,
            min_score_threshold=0.55,  # Prune trials with F1 < 0.55
        )
    elif pruner_type == "hyperband":
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=n_splits,  # Maximum resource is number of folds
            reduction_factor=3,
        )
    elif pruner_type == "successive_halving":
        pruner = SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=3, min_early_stopping_rate=0
        )
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")

    # Create sampler
    sampler = TPESampler(
        seed=42,
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        n_startup_trials=20,  # More initial random searches
        n_ei_candidates=24,
    )

    # Create study
    study = create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"trading_model_{X.shape[1]}features",
        load_if_exists=False,  # Set to True to continue from previous study
    )

    # Define objective function with data
    def objective(trial):
        return optimize_trading_model_with_pruning(
            trial=trial,
            X=X,
            y=y,
            sample_weight=sample_weight,
            events=events,
            n_splits=n_splits,
        )

    # Set up callbacks
    if callback_functions is None:
        callback_functions = [
            print_best_trial,
            save_intermediate_results,
            check_for_overfitting,
        ]

    # Add study-specific callbacks
    study_callbacks = []
    for callback in callback_functions:
        study_callbacks.append(lambda study, trial: callback(study, trial))

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        catch=(ValueError,),  # Catch specific exceptions
        callbacks=study_callbacks,
        gc_after_trial=True,  # Clean up memory after each trial
        show_progress_bar=True,
    )

    return study


# Callback functions for monitoring
def print_best_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Print best trial information after each trial."""
    if study.best_trial.number == trial.number:
        print(f"\n🎯 New best trial #{trial.number}:")
        print(f"   Score: {trial.value:.4f}")
        print(f"   Params: {trial.params}")

        # Print pruning info if trial was pruned
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(
                f"   ⚠️  Trial was pruned at fold {trial.user_attrs.get('pruned_at_fold', 'N/A')}"
            )
            print(
                f"   Score when pruned: {trial.user_attrs.get('score_when_pruned', 0):.4f}"
            )


def save_intermediate_results(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Save intermediate results to disk."""
    import json
    from datetime import datetime

    # Create results directory if it doesn't exist
    results_dir = Path("optuna_results")
    results_dir.mkdir(exist_ok=True)

    # Save trial results
    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": str(trial.state),
        "datetime": datetime.now().isoformat(),
        "user_attrs": trial.user_attrs,
        "duration": trial.duration.total_seconds() if trial.duration else None,
    }

    filename = results_dir / f"trial_{trial.number:04d}.json"
    with open(filename, "w") as f:
        json.dump(trial_data, f, indent=2, default=str)

    # Update summary file
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
    else:
        summary = {"trials": [], "best_trial": None}

    summary["trials"].append(trial_data)
    if study.best_trial.number == trial.number:
        summary["best_trial"] = trial_data

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def check_for_overfitting(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Check for signs of overfitting."""
    if len(trial.user_attrs.get("fold_scores", [])) >= 3:
        fold_scores = trial.user_attrs["fold_scores"]
        score_range = max(fold_scores) - min(fold_scores)

        if score_range > 0.3:  # Large variation between folds
            print(f"⚠️  Trial {trial.number} shows high variance: {score_range:.3f}")
            trial.set_user_attr("high_variance", True)

        # Check for declining performance across folds
        if len(fold_scores) >= 4:
            first_half = np.mean(fold_scores[:2])
            second_half = np.mean(fold_scores[2:])
            if second_half < first_half * 0.8:  # 20% decline
                print(f"⚠️  Trial {trial.number} shows performance decline")
                trial.set_user_attr("performance_decline", True)


# Analysis and visualization functions
def analyze_pruning_effectiveness(study: optuna.Study):
    """Analyze how effective pruning was."""
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\n" + "=" * 60)
    print("PRUNING EFFECTIVENESS ANALYSIS")
    print("=" * 60)

    print("\n📊 Trial Statistics:")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Completed trials: {len(completed_trials)}")
    print(f"   Pruned trials: {len(pruned_trials)}")
    print(f"   Pruning rate: {len(pruned_trials) / len(study.trials) * 100:.1f}%")

    if pruned_trials:
        # Calculate average time saved
        pruned_folds = []
        for trial in pruned_trials:
            pruned_at = trial.user_attrs.get("pruned_at_fold", 0)
            total_folds = trial.user_attrs.get("total_folds_attempted", 5)
            if total_folds > 0:
                pruned_folds.append(pruned_at / total_folds)

        avg_folds_saved = 1 - np.mean(pruned_folds) if pruned_folds else 0
        print("\n⏱️  Time Saved by Pruning:")
        print(
            f"   Average folds completed before pruning: {np.mean([t.user_attrs.get('pruned_at_fold', 0) for t in pruned_trials]):.1f}"
        )
        print(f"   Estimated time saved: {avg_folds_saved * 100:.1f}%")

        # Analyze pruned trial quality
        pruned_scores = [
            t.user_attrs.get("score_when_pruned", 0) for t in pruned_trials
        ]
        completed_scores = [t.value for t in completed_trials]

        print("\n📈 Score Analysis:")
        print(f"   Average score of pruned trials: {np.mean(pruned_scores):.4f}")
        print(f"   Average score of completed trials: {np.mean(completed_scores):.4f}")
        print(f"   Best score among pruned trials: {np.max(pruned_scores):.4f}")

        # Check if any good trials were pruned
        good_pruned = [s for s in pruned_scores if s > np.mean(completed_scores)]
        if good_pruned:
            print(f"   ⚠️  {len(good_pruned)} potentially good trials were pruned")

    return {
        "pruning_rate": len(pruned_trials) / len(study.trials),
        "avg_folds_saved": avg_folds_saved if pruned_trials else 0,
        "pruned_scores_mean": np.mean(pruned_scores) if pruned_scores else 0,
        "completed_scores_mean": np.mean(completed_scores) if completed_scores else 0,
    }


def plot_pruning_analysis(study: optuna.Study, save_path: str = None):
    """Visualize pruning effectiveness."""
    import matplotlib.pyplot as plt

    # Prepare data
    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "trial": trial.number,
                "state": str(trial.state),
                "value": trial.value if trial.value is not None else 0,
                "pruned_at": trial.user_attrs.get("pruned_at_fold", None),
                "duration": trial.duration.total_seconds() if trial.duration else 0,
            }
        )

    df = pd.DataFrame(trials_data)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Trial states distribution
    state_counts = df["state"].value_counts()
    axes[0, 0].pie(state_counts.values, labels=state_counts.index, autopct="%1.1f%%")
    axes[0, 0].set_title("Trial States Distribution")

    # 2. Score progression
    completed_mask = df["state"] == "TrialState.COMPLETE"
    axes[0, 1].plot(
        df.loc[completed_mask, "trial"],
        df.loc[completed_mask, "value"],
        "bo-",
        alpha=0.5,
        label="Completed",
    )

    pruned_mask = df["state"] == "TrialState.PRUNED"
    axes[0, 1].scatter(
        df.loc[pruned_mask, "trial"],
        df.loc[pruned_mask, "value"],
        color="red",
        alpha=0.6,
        label="Pruned",
    )

    axes[0, 1].set_xlabel("Trial Number")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Score Progression")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. When trials were pruned (histogram)
    if pruned_mask.any():
        pruned_folds = df.loc[pruned_mask, "pruned_at"].dropna()
        axes[1, 0].hist(pruned_folds, bins=range(1, 7), edgecolor="black")
        axes[1, 0].set_xlabel("Fold Number When Pruned")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Pruning Timing Distribution")
        axes[1, 0].set_xticks(range(1, 7))

    # 4. Duration comparison
    box_data = []
    labels = []
    for state in ["TrialState.COMPLETE", "TrialState.PRUNED"]:
        if state in df["state"].values:
            box_data.append(
                df[df["state"] == state]["duration"].values / 60
            )  # Convert to minutes
            labels.append(state.replace("TrialState.", ""))

    if box_data:
        axes[1, 1].boxplot(box_data, labels=labels)
        axes[1, 1].set_ylabel("Duration (minutes)")
        axes[1, 1].set_title("Trial Duration Comparison")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# Simple wrapper for quick start
def quick_optimize_with_pruning(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    events: pd.DataFrame,
    n_trials: int = 50,
):
    """
    Quick start function for pruning optimization.
    """
    study = create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1),
    )

    def objective(trial):
        return optimize_trading_model_with_pruning(
            trial=trial,
            X=X,
            y=y,
            sample_weight=sample_weight,
            events=events,
            n_splits=5,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study


"""
# Usage example
def main():
    '''Example usage of the pruning optimization'''

    # Assuming you have your data prepared
    # X, y, sample_weight, events = prepare_data(...)

    # Best weighting scheme found previously
    # best_weight = ...

    # Run optimization with pruning
    print("🚀 Starting optimization with pruning...")

    study = optimize_trading_model_with_advanced_pruning(
        X=X.reindex(events.index),
        y=events.bin,
        sample_weight=best_weight,
        events=events,
        n_trials=100,
        timeout=7200,  # 2 hours max
        n_splits=5,
        pruner_type="median",  # Try 'hyperband' for larger searches
        enable_parallel=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\n🏆 Best trial:")
    print(f"   Trial #{study.best_trial.number}")
    print(f"   Score (F1): {study.best_trial.value:.4f}")
    print(f"   Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"     {key}: {value}")

    # Analyze pruning effectiveness
    pruning_stats = analyze_pruning_effectiveness(study)

    # Visualize results
    plot_pruning_analysis(study, save_path="pruning_analysis.png")

    # Save best model
    best_model = RandomForestClassifier(**study.best_trial.params, n_jobs=-1, random_state=42)

    # Train on full data
    best_model.fit(X.reindex(events.index), events.bin, sample_weight=best_weight)

    return study, best_model
"""
