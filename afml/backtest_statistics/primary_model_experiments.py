# run_my_experiment.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from ..cache import cacheable
from ..data_structures.bars import make_bars
from ..labeling.triple_barrier import add_vertical_barrier, triple_barrier_labels

from ..cross_validation import (
    PurgedKFold,
    PurgedSplit,
    ml_cross_val_score,
    probability_weighted_accuracy,
)
from .logging_hooks import LoggingHooks
from .performance_analysis import (
    print_meta_labeling_comparison,
    run_meta_labeling_analysis,
)
from .research_framework import (
    ExperimentHook,
    ExperimentRunner,
    ResearchExperiment,
    log_model_configuration,
    log_performance_results,
)

# --- Placeholder Functions (Implement with your logic) ---


def load_my_data(
    tick_df,
    bar_type,
    timeframe,
    bar_size,
    strategy_func,
    strategy_kwargs,
    label_func,
    label_kwargs,
    features_func,
    features_kwargs,
):
    # Load your historical data, features, and primary signals
    # Must return a DataFrame and a Series of primary model signals
    print("Hook: Loading data and primary signals...")

    df = make_bars(tick_df, bar_type, timeframe, price="midprice", bar_size=bar_size, verbose=False)
    df0 = make_bars(tick_df, bar_type, timeframe, price="bid_ask", bar_size=bar_size, verbose=False)
    df["spread"] = df0.ask_close - df0.bid_close
    strategy_kwargs["data"] = df
    label_kwargs["data"] = df

    # Get signals from primary model
    primary_signals = strategy_func(**strategy_kwargs)
    df["side"] = primary_signals

    # Run labeling method
    if label_func.__name__ == "triple_barrier_labels":
        t_events = primary_signals[primary_signals.notna() & primary_signals != 0].index
        label_kwargs["t_events"] = t_events
        label_kwargs["vertical_barrier_times"] = add_vertical_barrier(
            t_events, df.close, **label_kwargs["vertical_barrier_times"]
        )
    labels = label_func(**label_kwargs)

    features_kwargs["data"] = df
    features = features_func(**features_kwargs)

    return df, features, labels


def get_train_test_split(df, labels, test_size=0.3):
    # Create your features (X) and split data
    print("Hook: Splitting data into train/test sets...")
    X = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = labels["bin"].loc[X.index]
    w = labels["w"].loc[X.index]  # Sample weights
    t1 = labels["t1"].loc[X.index]  # barrier touch timestamps
    train_indices, test_indices = PurgedSplit(t1, test_size).split(X)
    X_train, X_test, y_train, y_test, w_train, w_test = (
        X.iloc[train_indices],
        X.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices],
        w.iloc[train_indices],
        w.iloc[test_indices],
    )
    return X_train, X_test, y_train, y_test, w_train, w_test


def train_my_model(
    model_template: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
):
    print("Hook: Training meta model...")

    params = model_template.get_params()
    model = model_template.__class__(**params)
    model.fit(X_train, y_train, sample_weight.loc[X_train.index])
    return model


# --- Custom Hook Implementations ---


class DataSetupHook(ExperimentHook):
    priority = 100

    def on_data_load(self, experiment: ResearchExperiment):
        df, features, labels = load_my_data(
            tick_df=experiment.data["tick_df"],
            bar_type=experiment.data["bar_type"],
            timeframe=experiment.data["timeframe"],
            strategy_func=experiment.data["strategy_func"],
            strategy_kwargs=experiment.data["strategy_kwargs"],
            label_func=experiment.data["label_func"],
            label_kwargs=experiment.data["label_kwargs"],
            features_func=experiment.data["features_func"],
            features_kwargs=experiment.data["features_kwargs"],
        )
        X_train, X_test, y_train, y_test, w_train, w_test = get_train_test_split(
            features, labels, experiment.test_size
        )

        experiment.data["full_df"] = df
        experiment.data["X_train"] = X_train
        experiment.data["X_test"] = X_test
        experiment.data["y_train"] = y_train
        experiment.data["y_test"] = y_test
        experiment.data["sample_weight_train"] = w_train
        experiment.data["sample_weight_test"] = w_test


class ModelTrainingHook(ExperimentHook):
    priority = 50

    def on_model_train(self, experiment: ResearchExperiment):
        model = train_my_model(
            experiment.model_template,
            experiment.data["X_train"],
            experiment.data["y_train"],
            experiment.data["sample_weight_train"],
        )
        experiment.artifacts["meta_model"] = model


class PredictionHook(ExperimentHook):
    priority = 40

    def on_predict(self, experiment: ResearchExperiment):
        model = experiment.artifacts["meta_model"]
        X_test = experiment.data["X_test"]
        y_test = experiment.data["y_test"]
        w_test = experiment.data["sample_weight_test"]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Store predictions in the experiment
        experiment.artifacts["y_pred_proba"] = pd.Series(y_pred_proba, index=X_test.index)
        experiment.artifacts["metrics"] = {
            "accuracy": accuracy_score(y_test, y_pred, sample_weight=w_test),
            "precision": precision_score(y_test, y_pred, sample_weight=w_test),
            "recall": recall_score(y_test, y_pred, sample_weight=w_test),
            "f1_score": f1_score(y_test, y_pred, sample_weight=w_test),
            "PWA": probability_weighted_accuracy(y_test, y_pred, sample_weight=w_test),
        }


class PerformanceLoggingHook(ExperimentHook):
    priority = 10

    def on_evaluation(self, experiment: ResearchExperiment):
        print("Hook: Running performance analysis and logging...")
        # Use the powerful, all-in-one function from your refactored module
        results = run_meta_labeling_analysis(
            df=experiment.data["full_df"],
            signals=experiment.data["meta_model"].side,
            y_pred_proba=experiment.artifacts["y_pred_proba"],
            confidence_threshold=0.5,
            timeframe=experiment.data["timeframe"],
            trading_days_per_year=experiment.data["trading_days_per_year"],
            trading_hours_per_day=experiment.data["trading_hours_per_day"],
            strategy_name=experiment.strategy_name,
        )

        experiment.results = results

        log_model_configuration(
            model=experiment.model_template,
            X_train=experiment.data["X_train"],
            X_test=experiment.data["X_test"],
            strategy_name=experiment.strategy_name,
        )
        log_performance_results(results=experiment.results, model=experiment.model_template)


# --- Main Execution ---
class PerformanceEvaluationHook(ExperimentHook):
    """This hook now ONLY handles calculation, not logging or printing."""

    priority = 20

    def on_evaluation(self, experiment: ResearchExperiment):
        # Call the pure analysis function.
        results = run_meta_labeling_analysis(
            df=experiment.data["full_df"],
            signals=experiment.data["signals_test"],
            y_pred_proba=experiment.artifacts["y_pred_proba"],
            strategy_name=experiment.strategy_name,
        )
        experiment.results = results


class ReportingHook(ExperimentHook):
    """A dedicated hook for printing the final report to the console."""

    priority = 5  # Runs after evaluation and logging.

    def on_end(self, experiment: ResearchExperiment):
        if experiment.results:
            print_meta_labeling_comparison(experiment.results)


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Define the experiment
    my_experiment = ResearchExperiment(strategy_name="Momentum Strategy with RF Meta-Labeling v2")

    # 2. Define the hooks (workflow steps)
    # The list now includes the dedicated logging and reporting hooks.
    my_hooks = [
        DataSetupHook(),
        ModelTrainingHook(),
        PredictionHook(),
        PerformanceEvaluationHook(),
        LoggingHooks(),  # <-- ADDED
        ReportingHook(),  # <-- ADDED
    ]

    # 3. Create a runner and execute
    runner = ExperimentRunner(hooks=my_hooks)
    final_experiment = runner.run(my_experiment)

    # You can now access all results from the final_experiment object
    print("\nAccess final results:")
    print(f"Sharpe (Primary): {final_experiment.results['primary_metrics']['sharpe_ratio']:.4f}")
    print(f"Sharpe (Meta): {final_experiment.results['meta_metrics']['sharpe_ratio']:.4f}")
