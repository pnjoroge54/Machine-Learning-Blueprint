import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from pytz import timezone

# --- Hook System ---


class ExperimentHook:
    """
    Base hook interface for the research lifecycle.
    Subclass and override any methods you need.
    Priority determines execution order (higher runs earlier).
    """

    priority: int = 0

    def on_start(self, experiment: "ResearchExperiment"):
        """Called at the very beginning of a run."""
        pass

    def on_data_load(self, experiment: "ResearchExperiment"):
        """Called after initial data is loaded."""
        pass

    def on_model_train(self, experiment: "ResearchExperiment"):
        """Called to perform model training."""
        pass

    def on_predict(self, experiment: "ResearchExperiment"):
        """Called to generate model predictions."""
        pass

    def on_evaluation(self, experiment: "ResearchExperiment"):
        """Called to perform performance analysis and logging."""
        pass

    def on_end(self, experiment: "ResearchExperiment"):
        """Called at the very end of a run."""
        pass


# --- Experiment State ---


class ResearchExperiment:
    """A data class to hold the state of a single research run."""

    def __init__(
        self,
        strategy_name: str,
        **kwargs,
    ):
        self.strategy_name = strategy_name
        self.data: Dict[str, Any] = {}
        self.artifacts: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.data.update(kwargs)

    def __repr__(self):
        return f"ResearchExperiment(strategy='{self.strategy_name}', data={list(self.data.keys())}, artifacts={list(self.artifacts.keys())})"


# --- Orchestrator ---


class ExperimentRunner:
    """
    Orchestrates the execution of an experiment by invoking hooks
    in a predefined lifecycle order.
    """

    def __init__(self, hooks: List[ExperimentHook]):
        # Sort hooks by priority, descending. Higher priority runs first.
        self.hooks = sorted(hooks, key=lambda h: h.priority, reverse=True)
        self.lifecycle_methods = [
            "on_start",
            "on_data_load",
            "on_model_train",
            "on_predict",
            "on_evaluation",
            "on_end",
        ]

    def run(self, experiment: ResearchExperiment):
        """
        Runs the experiment through its lifecycle.
        """
        print(f"--- Starting Experiment: {experiment.strategy_name} ---")
        for method_name in self.lifecycle_methods:
            print(f"Executing stage: {method_name}...")
            for hook in self.hooks:
                method = getattr(hook, method_name)
                method(experiment)
        print(f"--- Finished Experiment: {experiment.strategy_name} ---")
        return experiment


# --- Structured Logging Configuration and Functions ---


# Configure Loguru logger
logger.remove()  # Remove default handlers

# Add structured JSON file logging
logger.add(
    "logs/model_logs_{time:YYYYMMDD}.jsonl",
    format="{message}",
    serialize=True,  # Automatically serialize log records to JSON
    rotation="500 MB",  # Rotate logs every 500MB
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    level="INFO",
)

# Add human-readable console logging
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def log_model_configuration(
    model, X_train, X_test, params=None, strategy_name="Strategy"
):
    """Log comprehensive model metadata"""
    model_metadata = {
        "event": "model_configuration",
        "timestamp": datetime.now(timezone("UTC")).isoformat(),
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
        "hyperparameters": (
            params if params else getattr(model, "get_params", lambda: {})()
        ),
        "environment": {"python_version": sys.version, "platform": sys.platform},
    }
    logger.info(model_metadata)


def log_performance_results(results, model=None):
    """Log performance metrics in structured format"""
    performance_log = {
        "event": "performance_results",
        "timestamp": datetime.now(timezone("UTC")).isoformat() + "Z",
        "strategy": results["strategy_name"],
        "confidence_threshold": results["meta_metrics"].get(
            "confidence_threshold", 0.6
        ),
        "primary_metrics": results["primary_metrics"],
        "meta_metrics": results["meta_metrics"],
        "signal_stats": {
            "total_signals": results["total_primary_signals"],
            "filtered_signals": results["filtered_signals"],
            "filter_rate": results["meta_metrics"].get("signal_filter_rate", 0),
        },
    }

    # Add model-specific information if available
    if model:
        if hasattr(model, "feature_importances_"):
            features = results.get(
                "features", list(range(len(model.feature_importances_)))
            )
            performance_log["feature_importances"] = dict(
                zip(features, model.feature_importances_.tolist())
            )

        if hasattr(model, "coef_"):
            performance_log["model_coefficients"] = model.coef_.tolist()

    logger.info(performance_log)


def log_trade_decisions(signals, meta_probabilities, returns, confidence_threshold=0.6):
    """Log individual trade decisions with context"""
    trade_logs = []

    for idx, signal in signals.items():
        if signal != 0:  # Only log actual trade signals
            trade = {
                "event": "trade_decision",
                "timestamp": idx.isoformat(),
                "signal": int(signal),
                "confidence": float(meta_probabilities.get(idx, 0)),
                "action": (
                    "take"
                    if meta_probabilities.get(idx, 0) > confidence_threshold
                    else "skip"
                ),
                "return": float(returns.get(idx, 0)),
            }
            trade_logs.append(trade)

    # Batch log trades to improve performance
    if trade_logs:
        logger.info(
            json.dumps({"batch_event": "trade_decisions", "trades": trade_logs})
        )
