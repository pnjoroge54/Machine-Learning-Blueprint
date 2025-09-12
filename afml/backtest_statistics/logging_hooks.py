# logging_hooks.py

import sys
from datetime import datetime

import pandas as pd
from loguru import logger
from pytz import timezone

from .research_framework import ExperimentHook

# --- REFINED LOGGING CONFIGURATION ---

# Remove default handlers to ensure full control.
logger.remove()

# 1. File Logger: Captures detailed DEBUG messages and serializes them to JSON.
# This file will contain the full, structured data for analysis.
logger.add(
    "logs/model_logs_{time:YYYYMMDD}.jsonl",
    format="{message}",
    serialize=True,  # Crucial for writing JSON
    level="DEBUG",  # Captures DEBUG, INFO, WARNING, etc.
    rotation="500 MB",
    retention="30 days",
    compression="zip",
)

# 2. Console Logger: Captures only INFO messages for a clean, readable output.
# It will ignore the verbose dictionaries logged at the DEBUG level.
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",  # Ignores DEBUG messages
    colorize=True,
)


# --- REFINED LOGGING FUNCTIONS ---


def log_model_configuration(model, X_train, X_test, params=None, strategy_name="Strategy"):
    """Logs model configuration with dual-level output."""
    model_metadata = {
        "event": "model_configuration",
        "timestamp": f"{datetime.utcnow().isoformat()}Z",
        "strategy": strategy_name,
        "model_type": type(model).__name__,
        "feature_set": list(X_train.columns),
        "training_data": {
            "start": X_train.index.min().isoformat(),
            "end": X_train.index.max().isoformat(),
            "samples": len(X_train),
            "features": X_train.shape[1],
        },
        "testing_data": {
            "start": X_test.index.min().isoformat(),
            "end": X_test.index.max().isoformat(),
            "samples": len(X_test),
        },
        "hyperparameters": (params if params else getattr(model, "get_params", lambda: {})()),
        "environment": {"python_version": sys.version, "platform": sys.platform},
    }
    # User-friendly message for the console.
    logger.info(f"Model configured: {strategy_name} ({type(model).__name__})")
    # Detailed dictionary for the JSON log file.
    logger.debug(model_metadata)


def log_performance_results(results, model=None):
    """Logs performance results with dual-level output."""
    performance_log = {
        "event": "performance_results",
        "timestamp": datetime.now(timezone("UTC")).isoformat(),
        "strategy": results["strategy_name"],
        "confidence_threshold": results["meta_metrics"].get("confidence_threshold", 0.6),
        "primary_metrics": results["primary_metrics"],
        "meta_metrics": results["meta_metrics"],
        "signal_stats": {
            "total_signals": results["total_primary_signals"],
            "filtered_signals": results["filtered_signals"],
            "filter_rate": results["meta_metrics"].get("signal_filter_rate", 0),
        },
    }
    if model and hasattr(model, "feature_importances_"):
        features = results.get("features", list(range(len(model.feature_importances_))))
        performance_log["feature_importances"] = dict(
            zip(features, model.feature_importances_.tolist())
        )

    # User-friendly message for the console.
    sharpe = performance_log["meta_metrics"].get("sharpe_ratio", 0)
    logger.info(f"Performance logged for {results['strategy_name']}. Meta Sharpe: {sharpe:.2f}")
    # Detailed dictionary for the JSON log file.
    logger.debug(performance_log)


def log_trade_decisions(signals, meta_probabilities, returns, confidence_threshold=0.6):
    """Logs individual trade decisions with dual-level output."""
    trade_logs = []
    for idx, signal in signals.items():
        if signal != 0:
            trade = {
                "event": "trade_decision",
                "timestamp": idx.isoformat(),
                "signal": int(signal),
                "confidence": float(meta_probabilities.get(idx, 0)),
                "action": (
                    "take" if meta_probabilities.get(idx, 0) > confidence_threshold else "skip"
                ),
                "return": float(returns.get(idx, 0)),
            }
            trade_logs.append(trade)

    if trade_logs:
        # User-friendly message for the console.
        logger.info(f"Trade decisions logged: {len(trade_logs)} total signals.")
        # Detailed dictionary for the JSON log file.
        logger.debug({"batch_event": "trade_decisions", "trades": trade_logs})


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
