"""Specialized caching for cross-validation functions."""

import hashlib
import inspect
import json
import pickle
import time
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, BaggingRegressor


def _hash_estimator(estimator: BaseEstimator) -> str:
    """
    Enhanced estimator hashing that handles nested estimators.
    """
    try:
        estimator_type = type(estimator).__name__
        params = estimator.get_params(deep=True)

        serializable_params = {}
        for k, v in params.items():
            try:
                # Handle nested estimators (like BaggingClassifier.base_estimator)
                if hasattr(v, "get_params") and callable(v.get_params):
                    nested_hash = _hash_estimator(v)
                    serializable_params[k] = f"nested_{nested_hash}"
                # Handle sklearn ensembles specifically
                elif isinstance(v, (BaggingClassifier, BaggingRegressor)):
                    nested_hash = _hash_estimator(v)
                    serializable_params[k] = f"ensemble_{nested_hash}"
                elif hasattr(v, "__name__"):
                    serializable_params[k] = f"func_{v.__name__}"
                elif callable(v):
                    serializable_params[k] = f"callable_{type(v).__name__}"
                elif isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
                    # Hash data structures efficiently
                    if hasattr(v, "shape"):
                        serializable_params[k] = (
                            f"data_{v.shape}_{hashlib.md5(v.tobytes()).hexdigest()[:8]}"
                        )
                    else:
                        serializable_params[k] = f"data_{len(v)}"
                else:
                    # Test JSON serialization
                    json.dumps(v)
                    serializable_params[k] = v
            except (TypeError, ValueError):
                serializable_params[k] = f"<{type(v).__name__}>"

        param_str = json.dumps(serializable_params, sort_keys=True)
        combined = f"{estimator_type}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    except Exception as e:
        logger.debug(f"Failed to hash estimator {type(estimator).__name__}: {e}")
        return f"est_{type(estimator).__name__}_{id(estimator)}"


def _hash_cv_generator(cv_gen) -> str:
    """Generate hash for CV generator (KFold, PurgedKFold, etc.)"""
    try:
        cv_type = type(cv_gen).__name__

        # Get CV parameters
        params = {}
        if hasattr(cv_gen, "n_splits"):
            params["n_splits"] = cv_gen.n_splits
        if hasattr(cv_gen, "pct_embargo"):
            params["pct_embargo"] = cv_gen.pct_embargo
        if hasattr(cv_gen, "t1"):
            # Hash the t1 series structure (not full content for speed)
            t1 = cv_gen.t1
            if isinstance(t1, pd.Series):
                params["t1_len"] = len(t1)
                params["t1_start"] = str(t1.index[0])
                params["t1_end"] = str(t1.index[-1])

        param_str = json.dumps(params, sort_keys=True)
        combined = f"{cv_type}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    except Exception as e:
        logger.debug(f"Failed to hash CV generator: {e}")
        return f"cv_{type(cv_gen).__name__}_{id(cv_gen)}"


def _hash_dataframe_fast(df: pd.DataFrame) -> str:
    """
    Fast DataFrame hashing for CV caching.
    Uses shape + columns + index range + sample of data.
    """
    parts = [
        f"shape_{df.shape}",
        f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}",
    ]

    # Hash index
    if isinstance(df.index, pd.DatetimeIndex):
        parts.append(f"idx_{df.index[0]}_{df.index[-1]}_{len(df)}")
    else:
        parts.append(f"idx_{df.index[0]}_{df.index[-1]}")

    # Sample data for hash (for speed)
    if len(df) > 100:
        sample = df.iloc[:: max(1, len(df) // 100)]
    else:
        sample = df

    data_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
    parts.append(f"data_{data_hash}")

    return "_".join(parts)


def _hash_series_fast(series: pd.Series) -> str:
    """Fast Series hashing."""
    parts = [f"len_{len(series)}", f"dtype_{series.dtype}"]

    if isinstance(series.index, pd.DatetimeIndex):
        parts.append(f"idx_{series.index[0]}_{series.index[-1]}")

    # Sample for hash
    if len(series) > 100:
        sample = series.iloc[:: max(1, len(series) // 100)]
    else:
        sample = series

    data_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
    parts.append(f"data_{data_hash}")

    return "_".join(parts)


def _generate_cv_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Generate specialized cache key for CV functions.
    Enhanced to handle complex sklearn estimators.
    """
    key_parts = [func.__module__, func.__qualname__]

    # Get function signature to map args to param names
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    for param_name, param_value in bound.arguments.items():
        try:
            # Handle different parameter types
            if param_value is None:
                key_parts.append(f"{param_name}_None")

            elif hasattr(param_value, "get_params") and callable(param_value.get_params):
                # Sklearn classifier/estimator (including nested ones)
                clf_hash = _hash_estimator(param_value)
                key_parts.append(f"{param_name}_est_{clf_hash}")

            elif hasattr(param_value, "split") and hasattr(param_value, "n_splits"):
                # CV generator
                cv_hash = _hash_cv_generator(param_value)
                key_parts.append(f"{param_name}_cv_{cv_hash}")

            elif isinstance(param_value, pd.DataFrame):
                df_hash = _hash_dataframe_fast(param_value)
                key_parts.append(f"{param_name}_df_{df_hash}")

            elif isinstance(param_value, pd.Series):
                series_hash = _hash_series_fast(param_value)
                key_parts.append(f"{param_name}_ser_{series_hash}")

            elif isinstance(param_value, np.ndarray):
                arr_hash = hashlib.md5(param_value.tobytes()).hexdigest()[:8]
                key_parts.append(f"{param_name}_arr_{param_value.shape}_{arr_hash}")

            elif isinstance(param_value, (str, int, float, bool)):
                key_parts.append(f"{param_name}_{param_value}")

            elif callable(param_value):
                # For scoring functions
                func_name = getattr(param_value, "__name__", str(type(param_value)))
                key_parts.append(f"{param_name}_func_{func_name}")

            else:
                # Fallback: try to hash string representation
                key_parts.append(f"{param_name}_{hash(str(param_value))}")

        except Exception as e:
            logger.debug(f"Failed to hash param '{param_name}': {e}")
            key_parts.append(f"{param_name}_unknown")

    # Create final hash
    combined = "_".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()


def cv_cacheable(_func=None, **kwargs):
    """
    Specialized caching decorator for cross-validation functions.

    Properly handles:
    - Sklearn classifiers (caches based on type + params, not trained state)
    - CV generators (PurgedKFold, etc.)
    - DataFrames and Series
    - Sample weights
    - Scoring functions

    Usage:
        @cv_cacheable
        def ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=None, ...):
            ...

    Performance:
    - First call: Full CV computation (minutes/hours)
    - Subsequent calls with same data: <1 second (cached)


    Dual-mode decorator that supports both old and new syntax.

    # Old syntax (backward compatible)
    @cv_cacheable
    def my_func(...)

    # New syntax
    @cv_cacheable(track_data_access=True, ...)
    def my_func(...)
    """

    def decorator(func):
        # If no kwargs provided, use old behavior
        if not kwargs:
            return _cv_cacheable_legacy(func)
        else:
            return _cv_cacheable_enhanced(func, **kwargs)

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def _cv_cacheable_legacy(func):
    """Original cv_cacheable implementation for backward compatibility."""
    from . import CACHE_DIRS, cache_stats

    func_name = f"{func.__module__}.{func.__qualname__}"
    cache_dir = CACHE_DIRS["base"] / "cv_cache"
    cache_dir.mkdir(exist_ok=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Original cache key generation (unchanged)
        cache_key = _generate_cv_cache_key(func, args, kwargs)
        cache_file = cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                cache_stats.record_hit(func_name)
                logger.info(f"CV cache hit for {func.__name__}")
                return result
            except Exception as e:
                logger.warning(f"CV cache read failed: {e}")
                cache_file.unlink()

        # Cache miss
        cache_stats.record_miss(func_name)
        result = func(*args, **kwargs)

        try:
            with open(cache_file, "wb") as f:
                pickle.load(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache CV result: {e}")

        return result

    wrapper._afml_cacheable = True
    return wrapper


def _cv_cacheable_enhanced(
    func, track_data_access=False, dataset_name=None, purpose=None, log_metrics=True
):
    """Enhanced version with tracking capabilities."""
    from . import CACHE_DIRS, cache_stats
    from .mlflow_integration import MLFLOW_AVAILABLE, get_mlflow_cache

    func_name = f"{func.__module__}.{func.__qualname__}"
    cache_dir = CACHE_DIRS["base"] / "cv_cache_enhanced"
    cache_dir.mkdir(exist_ok=True)

    def _generate_enhanced_cv_cache_key(
        base_key, track_data_access, dataset_name, purpose, log_metrics
    ):
        """Generate cache key that includes tracking parameters."""

        tracking_params = {
            "track_data_access": track_data_access,
            "dataset_name": dataset_name,
            "purpose": purpose,
            "log_metrics": log_metrics,
        }

        params_hash = hashlib.md5(json.dumps(tracking_params, sort_keys=True).encode()).hexdigest()[
            :8
        ]

        return f"{base_key}_tracking_{params_hash}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        base_key = _generate_cv_cache_key(func, args, kwargs)
        cache_key = _generate_enhanced_cv_cache_key(
            base_key, track_data_access, dataset_name, purpose, log_metrics
        )
        cache_file = cache_dir / f"{cache_key}.pkl"

        # Track data access - IMPORT HERE
        if track_data_access:
            from .data_access_tracker import get_data_tracker

            _track_cv_data_access(get_data_tracker(), args, kwargs, dataset_name, purpose)

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                cache_stats.record_hit(func_name)

                # Log cached results to MLflow
                if log_metrics and MLFLOW_AVAILABLE:
                    _log_cv_metrics_to_mlflow(result, func_name, cache_key, "cached")

                logger.info(f"Enhanced CV cache hit for {func.__name__}")
                return result
            except Exception as e:
                logger.warning(f"Enhanced CV cache read failed: {e}")
                cache_file.unlink()

        # Cache miss
        cache_stats.record_miss(func_name)
        logger.info(f"Enhanced CV cache miss for {func.__name__} - computing...")

        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Save to enhanced cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.debug(f"Cached enhanced CV result: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache enhanced CV result: {e}")

        # Log to MLflow
        if log_metrics and MLFLOW_AVAILABLE:
            _log_cv_metrics_to_mlflow(result, func_name, cache_key, "computed")
            try:
                mlflow_cache = get_mlflow_cache()
                mlflow_cache._log_metrics({"execution_time_seconds": execution_time})
            except Exception as e:
                logger.debug(f"Failed to log execution time: {e}")

        return result

    wrapper._afml_cacheable = True
    return wrapper


def _track_cv_data_access(tracker, args, kwargs, dataset_name, purpose):
    """Track data access in CV functions."""
    from .robust_cache_keys import _is_trackable_dataframe

    # Extract X, y from common CV function signatures
    X, y = None, None

    # Try to find X and y in args/kwargs
    for arg in args:
        if isinstance(arg, pd.DataFrame) and _is_trackable_dataframe(arg):
            X = arg
        elif isinstance(arg, (pd.Series, np.ndarray)) and len(arg) > 0:
            y = arg

    for key, value in kwargs.items():
        if key in ["X", "x", "features"] and _is_trackable_dataframe(value):
            X = value
        elif key in ["y", "target", "labels"] and isinstance(value, (pd.Series, np.ndarray)):
            y = value

    # Log access if we found trackable data
    if X is not None:
        tracker.log_access(
            dataset_name=dataset_name or "cv_dataset",
            start_date=X.index[0],
            end_date=X.index[-1],
            purpose=purpose or "cv",
            data_shape=X.shape,
        )


def _log_cv_metrics_to_mlflow(result, func_name, cache_key):
    """Log CV metrics to MLflow for experiment tracking."""
    from .mlflow_integration import get_mlflow_cache

    try:
        mlflow_cache = get_mlflow_cache()

        with mlflow_cache.experiment_run(
            run_name=f"cv_{func_name}_{cache_key[:8]}",
            tags={"type": "cross_validation", "function": func_name},
        ) as ctx:

            # Extract metrics from common CV result formats
            if isinstance(result, dict):
                # Direct metric dictionary
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        ctx.log_metric(f"cv_{key}", value)

            elif isinstance(result, (list, np.ndarray)):
                # Array of scores
                if len(result) > 0:
                    ctx.log_metric("cv_mean_score", np.mean(result))
                    ctx.log_metric("cv_std_score", np.std(result))
                    ctx.log_metric("cv_n_folds", len(result))

            elif hasattr(result, "cv_results_"):
                # Sklearn CV result object
                for key, value in result.cv_results_.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        ctx.log_metric(f"cv_{key}_mean", np.mean(value))

    except Exception as e:
        logger.debug(f"Failed to log CV metrics to MLflow: {e}")


def clear_cv_cache():
    """Clear all CV cache files."""
    from . import CACHE_DIRS

    cache_dir = CACHE_DIRS["base"] / "cv_cache"
    model_cache_dir = CACHE_DIRS["base"] / "cv_cache_models"

    count = 0
    for cache_dir in [cache_dir, model_cache_dir]:
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1

    logger.info(f"Cleared {count} CV cache files")
    return count


def cv_cache_with_classifier_state(func: Callable) -> Callable:
    """
    Caching decorator that also caches the trained classifier state.

    Use this if you want to cache both CV scores AND the trained models.

    Returns: (original_result, cached_classifiers)
    where cached_classifiers is a list of trained classifiers from each fold.

    Usage:
        @cv_cache_with_classifier_state
        def ml_cross_val_score_with_models(classifier, X, y, cv_gen, ...):
            # Your CV loop that returns (scores, trained_models)
            ...
    """
    from . import CACHE_DIRS, cache_stats

    func_name = f"{func.__module__}.{func.__qualname__}"
    cache_dir = CACHE_DIRS["base"] / "cv_cache_models"
    cache_dir.mkdir(exist_ok=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        try:
            cache_key = _generate_cv_cache_key(func, args, kwargs)
        except Exception as e:
            logger.warning(f"CV cache key generation failed: {e}")
            cache_stats.record_miss(func_name)
            return func(*args, **kwargs)

        # Check cache
        cache_file = cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                cache_stats.record_hit(func_name)
                logger.info(f"CV cache hit (with models) for {func.__name__}")
                return result
            except Exception as e:
                logger.warning(f"CV cache read failed: {e}")
                cache_file.unlink()

        # Cache miss - compute
        cache_stats.record_miss(func_name)
        logger.info(f"CV cache miss (with models) for {func.__name__} - computing...")
        result = func(*args, **kwargs)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.debug(f"Cached CV result with models: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache CV result: {e}")

        return result

    wrapper._afml_cacheable = True
    return wrapper


__all__ = [
    "cv_cacheable",
    "cv_cache_with_classifier_state",
    "clear_cv_cache",
]
