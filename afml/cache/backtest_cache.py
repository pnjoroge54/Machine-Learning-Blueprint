"""
Specialized caching system for backtesting workflows.
Handles walk-forward analysis, parameter optimization, and trade-level caching.
"""

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class BacktestMetadata:
    """Metadata for a cached backtest run."""

    strategy_name: str
    parameters: Dict[str, Any]
    data_hash: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    timestamp: float
    run_id: str
    splits: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None
    performance_summary: Optional[Dict[str, float]] = None


@dataclass
class BacktestResult:
    """Complete backtest result with metadata."""

    metadata: BacktestMetadata
    metrics: Dict[str, float]
    trades: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    diagnostics: Optional[Dict[str, Any]] = None


class BacktestCache:
    """
    Specialized cache for backtesting workflows.
    Handles parameter optimization, walk-forward analysis, and result comparison.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize backtest cache.

        Args:
            cache_dir: Directory for cache storage (None = use default)
        """
        # Import at runtime to avoid circular imports
        from . import CACHE_DIRS

        self.cache_dir = cache_dir or CACHE_DIRS["base"] / "backtests"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.runs_dir = self.cache_dir / "runs"
        self.splits_dir = self.cache_dir / "splits"
        self.trades_dir = self.cache_dir / "trades"
        self.metadata_dir = self.cache_dir / "metadata"

        for d in [self.runs_dir, self.splits_dir, self.trades_dir, self.metadata_dir]:
            d.mkdir(exist_ok=True)

        # Load index
        self.index_file = self.cache_dir / "backtest_index.json"
        self.index = self._load_index()

    def cache_backtest(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        metrics: Dict[str, float],
        trades: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.Series] = None,
        positions: Optional[pd.DataFrame] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Cache a complete backtest run.

        Args:
            strategy_name: Name of the trading strategy
            parameters: Strategy parameters used
            data: Market data used for backtest
            metrics: Performance metrics
            trades: DataFrame of individual trades
            equity_curve: Time series of equity
            positions: Position history
            diagnostics: Additional diagnostic information

        Returns:
            Run ID for the cached backtest
        """
        # Generate run ID
        run_id = self._generate_run_id(strategy_name, parameters, data)

        # Check if already cached
        if run_id in self.index:
            logger.info(f"Backtest already cached: {run_id}")
            return run_id

        # Create metadata
        data_hash = self._hash_dataframe(data)
        metadata = BacktestMetadata(
            strategy_name=strategy_name,
            parameters=parameters,
            data_hash=data_hash,
            start_date=data.index[0] if len(data) > 0 else pd.Timestamp.now(),
            end_date=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
            timestamp=time.time(),
            run_id=run_id,
            performance_summary=self._extract_key_metrics(metrics),
        )

        # Create result object
        result = BacktestResult(
            metadata=metadata,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            positions=positions,
            diagnostics=diagnostics,
        )

        # Save to disk
        self._save_backtest_result(result)

        # Update index
        self.index[run_id] = {
            "strategy": strategy_name,
            "parameters": parameters,
            "data_hash": data_hash,
            "timestamp": metadata.timestamp,
            "metrics": metadata.performance_summary,
        }
        self._save_index()

        logger.info(f"Cached backtest: {run_id} ({strategy_name})")
        return run_id

    def get_cached_backtest(
        self, strategy_name: str, parameters: Dict[str, Any], data: pd.DataFrame
    ) -> Optional[BacktestResult]:
        """
        Retrieve a cached backtest result if it exists.

        Args:
            strategy_name: Strategy name
            parameters: Strategy parameters
            data: Market data (used to verify cache validity)

        Returns:
            BacktestResult if cached, None otherwise
        """
        run_id = self._generate_run_id(strategy_name, parameters, data)

        if run_id not in self.index:
            return None

        # Verify data hasn't changed
        cached_info = self.index[run_id]
        current_data_hash = self._hash_dataframe(data)

        if cached_info["data_hash"] != current_data_hash:
            logger.warning(f"Data hash mismatch for {run_id} - cache invalid")
            return None

        # Load from disk
        return self._load_backtest_result(run_id)

    def cache_walk_forward_split(
        self,
        split_id: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold_number: int,
        total_folds: int,
    ) -> str:
        """
        Cache a walk-forward analysis split.

        Args:
            split_id: Unique identifier for the split set
            train_data: Training data for this fold
            test_data: Test data for this fold
            fold_number: Current fold number
            total_folds: Total number of folds

        Returns:
            Cache key for this split
        """
        split_key = f"{split_id}_fold_{fold_number}"
        split_path = self.splits_dir / f"{split_key}.pkl"

        split_data = {
            "split_id": split_id,
            "fold": fold_number,
            "total_folds": total_folds,
            "train_range": (train_data.index[0], train_data.index[-1]),
            "test_range": (test_data.index[0], test_data.index[-1]),
            "train_hash": self._hash_dataframe(train_data),
            "test_hash": self._hash_dataframe(test_data),
        }

        with open(split_path, "wb") as f:
            pickle.dump(split_data, f)

        logger.debug(f"Cached WF split: {split_key}")
        return split_key

    def get_walk_forward_split(self, split_id: str, fold_number: int) -> Optional[Dict[str, Any]]:
        """Get cached walk-forward split metadata."""
        split_key = f"{split_id}_fold_{fold_number}"
        split_path = self.splits_dir / f"{split_key}.pkl"

        if not split_path.exists():
            return None

        try:
            with open(split_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load split {split_key}: {e}")
            return None

    def cache_trades(self, run_id: str, trades: pd.DataFrame) -> Path:
        """
        Cache trade-level data separately for efficient access.

        Args:
            run_id: Backtest run ID
            trades: DataFrame of trades

        Returns:
            Path to cached trades file
        """
        trades_path = self.trades_dir / f"{run_id}_trades.parquet"

        try:
            trades.to_parquet(trades_path, compression="gzip")
            logger.debug(f"Cached {len(trades)} trades for {run_id}")
        except Exception as e:
            logger.warning(f"Failed to cache trades for {run_id}: {e}")
            # Fallback to CSV
            trades_path = self.trades_dir / f"{run_id}_trades.csv.gz"
            trades.to_csv(trades_path, compression="gzip")

        return trades_path

    def get_cached_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        """Load cached trades for a run."""
        # Try parquet first
        trades_path = self.trades_dir / f"{run_id}_trades.parquet"
        if trades_path.exists():
            try:
                return pd.read_parquet(trades_path)
            except Exception as e:
                logger.warning(f"Failed to load parquet trades: {e}")

        # Fallback to CSV
        trades_path = self.trades_dir / f"{run_id}_trades.csv.gz"
        if trades_path.exists():
            try:
                return pd.read_csv(trades_path, compression="gzip", index_col=0)
            except Exception as e:
                logger.warning(f"Failed to load CSV trades: {e}")

        return None

    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare metrics across multiple backtest runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            DataFrame with runs as rows and metrics as columns
        """
        comparison_data = []

        for run_id in run_ids:
            if run_id not in self.index:
                logger.warning(f"Run {run_id} not found in index")
                continue

            result = self._load_backtest_result(run_id)
            if result is None:
                continue

            row_data = {
                "run_id": run_id,
                "strategy": result.metadata.strategy_name,
                "timestamp": pd.Timestamp.fromtimestamp(result.metadata.timestamp),
                "start_date": result.metadata.start_date,
                "end_date": result.metadata.end_date,
            }

            # Add parameters as columns
            for param_name, param_value in result.metadata.parameters.items():
                row_data[f"param_{param_name}"] = param_value

            # Add metrics
            for metric_name, metric_value in result.metrics.items():
                if metrics is None or metric_name in metrics:
                    row_data[metric_name] = metric_value

            comparison_data.append(row_data)

        if not comparison_data:
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        return df.set_index("run_id")

    def find_best_parameters(
        self,
        strategy_name: str,
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Find best performing parameter combinations for a strategy.

        Args:
            strategy_name: Strategy to analyze
            metric: Metric to optimize
            maximize: True to maximize, False to minimize
            top_n: Number of top results to return

        Returns:
            DataFrame of top parameter combinations
        """
        matching_runs = [
            run_id for run_id, info in self.index.items() if info["strategy"] == strategy_name
        ]

        if not matching_runs:
            logger.warning(f"No cached runs found for strategy: {strategy_name}")
            return pd.DataFrame()

        comparison = self.compare_runs(matching_runs, metrics=[metric])

        if comparison.empty or metric not in comparison.columns:
            return pd.DataFrame()

        # Sort and get top N
        sorted_df = comparison.sort_values(metric, ascending=not maximize)
        return sorted_df.head(top_n)

    def get_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific run."""
        if run_id not in self.index:
            return None

        metadata_path = self.metadata_dir / f"{run_id}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {run_id}: {e}")

        return self.index[run_id]

    def clear_old_runs(self, days: int = 30) -> int:
        """
        Clear cached runs older than specified days.

        Args:
            days: Remove runs older than this many days

        Returns:
            Number of runs cleared
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        cleared_count = 0

        for run_id, info in list(self.index.items()):
            if info["timestamp"] < cutoff_time:
                self._delete_run(run_id)
                del self.index[run_id]
                cleared_count += 1

        if cleared_count > 0:
            self._save_index()
            logger.info(f"Cleared {cleared_count} old backtest runs (>{days} days)")

        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached backtests."""
        total_runs = len(self.index)
        strategies = {}
        total_size_mb = 0

        for run_id, info in self.index.items():
            strategy = info["strategy"]
            strategies[strategy] = strategies.get(strategy, 0) + 1

        # Calculate disk usage
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size_mb += file_path.stat().st_size / (1024 * 1024)

        return {
            "total_runs": total_runs,
            "strategies": strategies,
            "cache_size_mb": round(total_size_mb, 2),
            "runs_dir": str(self.runs_dir),
        }

    # Private methods

    def _generate_run_id(
        self, strategy_name: str, parameters: Dict[str, Any], data: pd.DataFrame
    ) -> str:
        """Generate unique run ID from strategy, parameters, and data."""
        param_str = json.dumps(parameters, sort_keys=True)
        data_hash = self._hash_dataframe(data)
        combined = f"{strategy_name}_{param_str}_{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Create hash of DataFrame content."""
        if len(df) == 0:
            return "empty"

        # Hash based on shape, columns, index range, and sample of data
        parts = [
            str(df.shape),
            str(tuple(df.columns)),
            str(df.index[0]),
            str(df.index[-1]),
        ]

        # Sample data for hashing (for performance)
        if len(df) > 100:
            sample = df.iloc[:: max(1, len(df) // 100)]
        else:
            sample = df

        parts.append(hashlib.md5(sample.values.tobytes()).hexdigest()[:8])

        return hashlib.md5("_".join(parts).encode()).hexdigest()

    def _extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract most important metrics for summary."""
        key_metrics = [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]

        return {k: v for k, v in metrics.items() if k in key_metrics}

    def _save_backtest_result(self, result: BacktestResult):
        """Save backtest result to disk."""
        run_id = result.metadata.run_id

        # Save main result
        result_path = self.runs_dir / f"{run_id}.pkl"
        with open(result_path, "wb") as f:
            pickle.dump(result, f)

        # Save metadata separately for quick access
        metadata_path = self.metadata_dir / f"{run_id}_metadata.json"
        metadata_dict = asdict(result.metadata)
        # Convert timestamps to strings for JSON
        metadata_dict["start_date"] = str(metadata_dict["start_date"])
        metadata_dict["end_date"] = str(metadata_dict["end_date"])

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Save trades separately if present
        if result.trades is not None and not result.trades.empty:
            self.cache_trades(run_id, result.trades)

    def _load_backtest_result(self, run_id: str) -> Optional[BacktestResult]:
        """Load backtest result from disk."""
        result_path = self.runs_dir / f"{run_id}.pkl"

        if not result_path.exists():
            logger.warning(f"Result file not found for {run_id}")
            return None

        try:
            with open(result_path, "rb") as f:
                result = pickle.load(f)

            # Load trades separately if not in main result
            if result.trades is None:
                result.trades = self.get_cached_trades(run_id)

            return result

        except Exception as e:
            logger.error(f"Failed to load backtest result {run_id}: {e}")
            return None

    def _delete_run(self, run_id: str):
        """Delete all files associated with a run."""
        # Delete main result
        result_path = self.runs_dir / f"{run_id}.pkl"
        if result_path.exists():
            result_path.unlink()

        # Delete metadata
        metadata_path = self.metadata_dir / f"{run_id}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Delete trades
        for trades_path in [
            self.trades_dir / f"{run_id}_trades.parquet",
            self.trades_dir / f"{run_id}_trades.csv.gz",
        ]:
            if trades_path.exists():
                trades_path.unlink()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load backtest index: {e}")

        return {}

    def _save_index(self):
        """Save index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save backtest index: {e}")


# =============================================================================
# Convenience decorator
# =============================================================================

# Global instance
_global_backtest_cache: Optional[BacktestCache] = None


def get_backtest_cache() -> BacktestCache:
    """Get global backtest cache instance."""
    global _global_backtest_cache
    if _global_backtest_cache is None:
        _global_backtest_cache = BacktestCache()
    return _global_backtest_cache


def cached_backtest(strategy_name: str, save_trades: bool = True):
    """
    Decorator for caching backtest functions.

    Usage:
        @cached_backtest("momentum_strategy", save_trades=True)
        def run_backtest(data, params):
            # Backtest logic
            return metrics, trades, equity_curve
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(data: pd.DataFrame, params: Dict[str, Any], *args, **kwargs):
            cache = get_backtest_cache()

            # Check cache first
            cached_result = cache.get_cached_backtest(strategy_name, params, data)
            if cached_result is not None:
                logger.info(f"Cache hit for {strategy_name} backtest")
                return (
                    cached_result.metrics,
                    cached_result.trades,
                    cached_result.equity_curve,
                )

            # Run backtest
            logger.info(f"Running backtest: {strategy_name}")
            result = func(data, params, *args, **kwargs)

            # Unpack result
            if isinstance(result, tuple):
                metrics = result[0] if len(result) > 0 else {}
                trades = result[1] if len(result) > 1 else None
                equity_curve = result[2] if len(result) > 2 else None
            else:
                metrics = result
                trades = None
                equity_curve = None

            # Cache result
            cache.cache_backtest(
                strategy_name=strategy_name,
                parameters=params,
                data=data,
                metrics=metrics,
                trades=trades if save_trades else None,
                equity_curve=equity_curve,
            )

            return result

        return wrapper

    return decorator


__all__ = [
    "BacktestCache",
    "BacktestMetadata",
    "BacktestResult",
    "get_backtest_cache",
    "cached_backtest",
]
