"""
MLflow integration for experiment tracking with local caching.
Combines the speed of local caching with the tracking power of MLflow.
"""

import hashlib
import json
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - install with: pip install mlflow")


class MLflowCacheIntegration:
    """
    Unified caching and experiment tracking system.
    Uses local cache for speed, MLflow for tracking and reproducibility.
    """

    def __init__(
        self,
        experiment_name: str = "afml_experiments",
        tracking_uri: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize MLflow + cache integration.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI (None = local ./mlruns)
            cache_dir: Directory for local cache (None = use default)
        """
        self.experiment_name = experiment_name
        self.enabled = MLFLOW_AVAILABLE

        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient()
            logger.info(f"MLflow tracking enabled: experiment={experiment_name}")
        else:
            logger.info("MLflow tracking disabled (not installed)")

        # Import cache at runtime to avoid circular imports
        from . import CACHE_DIRS, memory

        self.local_cache = memory
        self.cache_dir = cache_dir or CACHE_DIRS["base"] / "mlflow_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_experiment(
        self,
        tags: Optional[Dict[str, str]] = None,
        log_artifacts: bool = True,
        cache_metrics: bool = True,
    ):
        """
        Decorator that combines local caching with MLflow experiment tracking.

        Args:
            tags: Additional MLflow tags for the run
            log_artifacts: Whether to log results as MLflow artifacts
            cache_metrics: Whether to cache metric computations

        Usage:
            @mlflow_cache.cached_experiment(tags={"strategy": "momentum"})
            def train_model(data, params):
                # Training code
                return model, metrics
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract parameters for tracking
                params = self._extract_params(func, args, kwargs)
                run_name = self._generate_run_name(func.__name__, params)
                cache_key = self._generate_cache_key(func.__name__, params)

                # Check local cache first
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    logger.info(f"Cache hit for {func.__name__} (run: {run_name})")

                    # Log to MLflow even for cached results (for tracking)
                    if self.enabled:
                        with mlflow.start_run(run_name=run_name) as run:
                            mlflow.log_param("cache_hit", True)
                            mlflow.log_param("cache_key", cache_key)
                            self._log_params(params)
                            if tags:
                                mlflow.set_tags(tags)

                            # Log cached metrics if available
                            if cache_metrics and isinstance(cached_result, dict):
                                self._log_metrics(cached_result)

                    return cached_result

                # Cache miss - compute and track
                logger.info(f"Cache miss for {func.__name__} - computing and tracking")

                start_time = time.time()

                if self.enabled:
                    with mlflow.start_run(run_name=run_name) as run:
                        # Log parameters
                        mlflow.log_param("cache_hit", False)
                        mlflow.log_param("cache_key", cache_key)
                        self._log_params(params)

                        if tags:
                            mlflow.set_tags(tags)

                        # Execute function
                        result = func(*args, **kwargs)

                        # Log execution time
                        execution_time = time.time() - start_time
                        mlflow.log_metric("execution_time_seconds", execution_time)

                        # Log results
                        if isinstance(result, dict):
                            self._log_metrics(result)

                        # Log artifacts if requested
                        if log_artifacts:
                            self._log_result_artifacts(result, run.info.run_id)

                        # Cache the result
                        self._save_to_cache(cache_key, result)

                        logger.info(
                            f"Completed {func.__name__} in {execution_time:.2f}s (run_id: {run.info.run_id})"
                        )

                        return result
                else:
                    # MLflow disabled - just cache
                    result = func(*args, **kwargs)
                    self._save_to_cache(cache_key, result)
                    return result

            wrapper._mlflow_tracked = True
            return wrapper

        return decorator

    @contextmanager
    def experiment_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ):
        """
        Context manager for manual experiment tracking with optional caching.

        Usage:
            with mlflow_cache.experiment_run("my_experiment", use_cache=True) as ctx:
                result = expensive_computation()
                ctx.log_metric("accuracy", 0.95)
                ctx.cache_result("my_key", result)
        """

        class ExperimentContext:
            def __init__(self, integration, run_obj, use_cache):
                self.integration = integration
                self.run = run_obj
                self.use_cache = use_cache
                self.cached_data = {}

            def log_param(self, key: str, value: Any):
                if self.integration.enabled:
                    mlflow.log_param(key, value)

            def log_metric(self, key: str, value: float, step: Optional[int] = None):
                if self.integration.enabled:
                    mlflow.log_metric(key, value, step=step)

            def log_artifact(self, local_path: str):
                if self.integration.enabled:
                    mlflow.log_artifact(local_path)

            def cache_result(self, key: str, value: Any):
                if self.use_cache:
                    self.cached_data[key] = value
                    self.integration._save_to_cache(key, value)

            def get_cached(self, key: str) -> Optional[Any]:
                if self.use_cache:
                    if key in self.cached_data:
                        return self.cached_data[key]
                    return self.integration._get_from_cache(key)
                return None

        if self.enabled:
            with mlflow.start_run(run_name=run_name) as run:
                if tags:
                    mlflow.set_tags(tags)
                ctx = ExperimentContext(self, run, use_cache)
                yield ctx
        else:
            # MLflow disabled - provide mock context
            ctx = ExperimentContext(self, None, use_cache)
            yield ctx

    def compare_runs(
        self, run_ids: list[str], metrics: list[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across multiple MLflow runs.

        Args:
            run_ids: List of MLflow run IDs to compare
            metrics: List of metric names to compare (None = all metrics)

        Returns:
            Dict mapping run_id -> {metric_name: value}
        """
        if not self.enabled:
            logger.warning("MLflow not available - cannot compare runs")
            return {}

        comparison = {}

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_metrics = run.data.metrics

                if metrics:
                    # Filter to requested metrics
                    run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

                comparison[run_id] = run_metrics

            except Exception as e:
                logger.warning(f"Failed to get run {run_id}: {e}")

        return comparison

    def get_best_run(self, metric_name: str, maximize: bool = True) -> Optional[str]:
        """
        Find the best run based on a metric.

        Args:
            metric_name: Name of metric to optimize
            maximize: If True, find max; if False, find min

        Returns:
            Run ID of best run, or None if no runs found
        """
        if not self.enabled:
            return None

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.{metric_name} IS NOT NULL",
                order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
                max_results=1,
            )

            if len(runs) > 0:
                return runs.iloc[0]["run_id"]

        except Exception as e:
            logger.warning(f"Failed to find best run: {e}")

        return None

    def load_model_from_run(self, run_id: str, model_name: str = "model") -> Any:
        """
        Load a model artifact from an MLflow run.

        Args:
            run_id: MLflow run ID
            model_name: Name of the model artifact

        Returns:
            Loaded model object
        """
        if not self.enabled:
            raise RuntimeError("MLflow not available")

        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model from run {run_id}: {e}")
            raise

    def _extract_params(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract parameters from function call."""
        import inspect

        params = {}

        # Get function signature
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, param_value in bound.arguments.items():
                # Convert to JSON-serializable format
                if isinstance(param_value, (int, float, str, bool, type(None))):
                    params[param_name] = param_value
                elif isinstance(param_value, dict):
                    # Recursively handle dict parameters
                    params[param_name] = str(param_value)
                else:
                    # For complex types, use string representation
                    params[param_name] = str(type(param_value).__name__)

        except Exception as e:
            logger.debug(f"Failed to extract params for {func.__name__}: {e}")

        return params

    def _generate_run_name(self, func_name: str, params: Dict[str, Any]) -> str:
        """Generate unique run name."""
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        return f"{func_name}_{param_hash}"

    def _generate_cache_key(self, func_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key from function name and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{func_name}_{param_str}".encode()).hexdigest()

    def _log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        for key, value in params.items():
            try:
                # MLflow has limits on param string length
                str_value = str(value)
                if len(str_value) > 250:
                    str_value = str_value[:247] + "..."
                mlflow.log_param(key, str_value)
            except Exception as e:
                logger.debug(f"Failed to log param {key}: {e}")

    def _log_metrics(self, result: Dict[str, Any]):
        """Log numeric metrics from result dict."""
        if not isinstance(result, dict):
            return

        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    mlflow.log_metric(key, float(value))
                except Exception as e:
                    logger.debug(f"Failed to log metric {key}: {e}")

    def _log_result_artifacts(self, result: Any, run_id: str):
        """Save and log result as artifact."""
        try:
            import pickle

            artifact_path = self.cache_dir / f"result_{run_id}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(result, f)

            mlflow.log_artifact(str(artifact_path))

        except Exception as e:
            logger.debug(f"Failed to log result artifact: {e}")

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve from local cache."""
        import pickle

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache read failed for {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, result: Any):
        """Save to local cache."""
        import pickle

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.debug(f"Cache write failed for {cache_key}: {e}")


# =============================================================================
# Convenience functions
# =============================================================================


# Global instance for easy usage
_global_integration: Optional[MLflowCacheIntegration] = None


def setup_mlflow_cache(
    experiment_name: str = "afml_experiments",
    tracking_uri: Optional[str] = None,
) -> MLflowCacheIntegration:
    """
    Setup global MLflow + cache integration.

    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI

    Returns:
        MLflowCacheIntegration instance
    """
    global _global_integration
    _global_integration = MLflowCacheIntegration(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    return _global_integration


def get_mlflow_cache() -> MLflowCacheIntegration:
    """Get global MLflow cache integration instance."""
    global _global_integration
    if _global_integration is None:
        _global_integration = setup_mlflow_cache()
    return _global_integration


# Decorator shortcut
def mlflow_cached(
    tags: Optional[Dict[str, str]] = None,
    log_artifacts: bool = True,
):
    """
    Convenience decorator using global integration.

    Usage:
        @mlflow_cached(tags={"model": "random_forest"})
        def train_model(data, params):
            ...
    """
    integration = get_mlflow_cache()
    return integration.cached_experiment(tags=tags, log_artifacts=log_artifacts)


__all__ = [
    "MLflowCacheIntegration",
    "setup_mlflow_cache",
    "get_mlflow_cache",
    "mlflow_cached",
    "MLFLOW_AVAILABLE",
]
