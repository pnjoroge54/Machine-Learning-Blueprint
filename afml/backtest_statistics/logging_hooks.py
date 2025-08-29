# logging_hooks.py

import pandas as pd

from .performance_analysis import (
    log_model_configuration,
    log_performance_results,
    log_trade_decisions,
)
from .research_framework import ExperimentHook


class LoggingHooks(ExperimentHook):
    """
    A collection of hooks to handle all logging aspects of an experiment
    at different lifecycle stages.
    """

    priority = 1  # Low priority to ensure it runs after calculations.

    def on_start(self, experiment):
        """Log the model and data configuration at the start of the run."""
        log_model_configuration(
            model=experiment.artifacts["meta_model"],
            X_train=experiment.data["X_train"],
            X_test=experiment.data["X_test"],
            strategy_name=experiment.strategy_name,
        )

    def on_evaluation(self, experiment):
        """Log performance and trade details after evaluation."""
        if not experiment.results:
            print("Warning: No results found in experiment to log.")
            return

        # Log the detailed performance metrics dictionary.
        log_performance_results(
            results=experiment.results, model=experiment.artifacts["meta_model"]
        )

        # Log the individual trade-by-trade decisions.
        if isinstance(experiment.artifacts["y_pred_proba"], pd.Series):
            meta_probs = experiment.artifacts["y_pred_proba"]
        else:
            meta_probs = pd.Series(
                experiment.artifacts["y_pred_proba"],
                index=experiment.data["signals_test"].index[
                    -len(experiment.artifacts["y_pred_proba"]) :
                ],
            )

        test_start = experiment.data["signals_test"].index[0]
        pre_test_point = experiment.data["full_df"].index[
            experiment.data["full_df"].index.get_loc(test_start) - 1
        ]
        return_index = experiment.data["signals_test"].index.union([pre_test_point])
        returns_data = (
            experiment.data["full_df"]
            .loc[return_index, "close"]
            .pct_change()
            .loc[experiment.data["signals_test"].index]
        )

        log_trade_decisions(
            signals=experiment.data["signals_test"],
            meta_probabilities=meta_probs,
            returns=returns_data,
            confidence_threshold=experiment.results["meta_metrics"].get(
                "confidence_threshold", 0.6
            ),
        )
