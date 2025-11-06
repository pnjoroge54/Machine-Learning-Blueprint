"""
Specialized caching for cross-validation functions.
Handles sklearn classifiers, CV generators, and complex ML workflows.
"""

import hashlib
import inspect
import json
import pickle
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator


def _hash_classifier(clf: BaseEstimator) -> str:
    """
    Generate stable hash for sklearn classifier.
    Uses class name + parameters (not the trained state).
    """
    try:
        # Get classifier type and parameters
        clf_type = type(clf).__name__
        params = clf.get_params(deep=True)

        # Filter out non-serializable params (like objects, functions)
        serializable_params = {}
        for k, v in params.items():
            try:
                # Test if JSON serializable
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, ValueError):
                # Use type name for non-serializable params
                serializable_params[k] = f"<{type(v).__name__}>"

        # Create stable hash
        param_str = json.dumps(serializable_params, sort_keys=True)
        combined = f"{clf_type}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    except Exception as e:
        logger.debug(f"Failed to hash classifier: {e}")
        return f"clf_{type(clf).__name__}_{id(clf)}"


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
    Handles classifiers, CV generators, DataFrames, and sample weights.
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

            elif isinstance(param_value, BaseEstimator):
                # Sklearn classifier/estimator
                clf_hash = _hash_classifier(param_value)
                key_parts.append(f"{param_name}_clf_{clf_hash}")

            elif hasattr(param_value, "split") and hasattr(param_value, "n_splits"):
                # CV generator (has split method and n_splits)
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


def cv_cacheable(func: Callable) -> Callable:
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
    """
    from . import CACHE_DIRS, cache_stats

    func_name = f"{func.__module__}.{func.__qualname__}"
    cache_dir = CACHE_DIRS["base"] / "cv_cache"
    cache_dir.mkdir(exist_ok=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        try:
            cache_key = _generate_cv_cache_key(func, args, kwargs)
        except Exception as e:
            logger.warning(f"CV cache key generation failed for {func_name}: {e}")
            cache_stats.record_miss(func_name)
            return func(*args, **kwargs)

        # Check cache
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
                cache_file.unlink()  # Remove corrupted cache

        # Cache miss - compute
        cache_stats.record_miss(func_name)
        logger.info(f"CV cache miss for {func.__name__} - computing...")
        result = func(*args, **kwargs)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.debug(f"Cached CV result: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache CV result: {e}")

        return result

    wrapper._afml_cacheable = True
    return wrapper


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


__all__ = [
    "cv_cacheable",
    "cv_cache_with_classifier_state",
    "clear_cv_cache",
]
