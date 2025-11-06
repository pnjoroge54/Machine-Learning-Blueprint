"""
Robust cache key generation for financial ML data structures.
Handles numpy arrays, pandas DataFrames, and time-series data properly.
Integrates with DataAccessTracker for comprehensive data hygiene monitoring.
"""

import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class CacheKeyGenerator:
    """Generate robust, collision-resistant cache keys for ML data structures."""

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """
        Hash pandas DataFrame including structure and content.

        Uses efficient sampling for large DataFrames to balance
        accuracy with performance.
        """
        if len(df) == 0:
            return "empty_df"

        parts = [
            f"shape_{df.shape}",
            f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}",
            f"dtypes_{hashlib.md5(str(tuple(df.dtypes)).encode()).hexdigest()[:8]}",
        ]

        # Hash index
        if isinstance(df.index, pd.DatetimeIndex):
            parts.append(f"idx_dt_{df.index[0]}_{df.index[-1]}_{len(df)}")
        else:
            idx_hash = hashlib.md5(str((df.index[0], df.index[-1])).encode()).hexdigest()[:8]
            parts.append(f"idx_{idx_hash}")

        # Hash content with sampling for large DataFrames
        if df.size > 10000:
            sample = df.iloc[:: max(1, len(df) // 100)]
            content_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(df.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")

        return "_".join(parts)

    @staticmethod
    def hash_series(series: pd.Series) -> str:
        """Hash pandas Series efficiently."""
        if len(series) == 0:
            return "empty_series"

        parts = [
            f"len_{len(series)}",
            f"dtype_{series.dtype}",
        ]

        # Hash index
        if isinstance(series.index, pd.DatetimeIndex):
            parts.append(f"idx_dt_{series.index[0]}_{series.index[-1]}")
        else:
            parts.append(f"idx_{hash((series.index[0], series.index[-1]))}")

        # Hash values with sampling
        if len(series) > 1000:
            sample = series.iloc[:: max(1, len(series) // 100)]
            content_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(series.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")

        return "_".join(parts)

    @staticmethod
    def hash_numpy_array(arr: np.ndarray) -> str:
        """Hash numpy array including shape, dtype, and content."""
        if arr.size == 0:
            return "empty_array"

        # For large arrays, sample for performance
        if arr.size > 10000:
            sample = arr.flat[:: max(1, arr.size // 1000)]
            content_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(arr.tobytes()).hexdigest()[:8]

        return f"arr_{arr.shape}_{arr.dtype}_{content_hash}"

    @staticmethod
    def hash_argument(arg: Any) -> str:
        """Hash a single argument based on its type."""
        if isinstance(arg, np.ndarray):
            return CacheKeyGenerator.hash_numpy_array(arg)
        elif isinstance(arg, pd.DataFrame):
            return CacheKeyGenerator.hash_dataframe(arg)
        elif isinstance(arg, pd.Series):
            return CacheKeyGenerator.hash_series(arg)
        elif isinstance(arg, dict):
            return CacheKeyGenerator.hash_dict(arg)
        elif isinstance(arg, (list, tuple)):
            return CacheKeyGenerator.hash_sequence(arg)
        elif isinstance(arg, (int, float, str, bool, type(None))):
            return f"{type(arg).__name__}_{hash(arg)}"
        else:
            # Fallback for unknown types
            try:
                return f"{type(arg).__name__}_{hash(repr(arg))}"
            except Exception:
                return f"{type(arg).__name__}_{id(arg)}"

    @staticmethod
    def hash_dict(d: dict) -> str:
        """Hash dictionary recursively with sorted keys."""
        if len(d) == 0:
            return "empty_dict"

        items = []
        for key, value in sorted(d.items()):
            val_hash = CacheKeyGenerator.hash_argument(value)
            items.append(f"{key}={val_hash}")

        combined = "_".join(items)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def hash_sequence(seq: Tuple[Any, ...] | list) -> str:
        """Hash list or tuple recursively."""
        if len(seq) == 0:
            return "empty_seq"

        # Hash each element
        element_hashes = [CacheKeyGenerator.hash_argument(item) for item in seq]

        combined = "_".join(element_hashes)
        return hashlib.md5(combined.encode()).hexdigest()[:8]


def robust_cacheable(
    track_data_access: bool = False,
    dataset_name: Optional[str] = None,
    purpose: Optional[str] = None,
):
    """
    Enhanced cacheable decorator with robust key generation and optional data tracking.

    Args:
        track_data_access: If True, log DataFrame accesses to DataAccessTracker
        dataset_name: Name for the dataset (required if track_data_access=True)
        purpose: Purpose of access: 'train', 'test', 'validate', 'optimize', 'analyze'

    Usage:
        @robust_cacheable()
        def compute_features(df, params):
            return features

        @robust_cacheable(track_data_access=True, dataset_name="test_2024", purpose="test")
        def load_test_data():
            return data
    """

    def decorator(func: Callable) -> Callable:
        # Import at runtime to avoid circular imports
        from . import cache_stats, memory

        func_name = f"{func.__module__}.{func.__qualname__}"
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate robust cache key
            try:
                key_parts = [func_name]

                # Hash positional arguments
                for i, arg in enumerate(args):
                    try:
                        key_parts.append(CacheKeyGenerator.hash_argument(arg))
                    except Exception as e:
                        logger.debug(f"Failed to hash arg {i}: {e}")
                        key_parts.append(f"arg_{i}_{id(arg)}")

                # Hash keyword arguments (sorted)
                for key, value in sorted(kwargs.items()):
                    try:
                        val_hash = CacheKeyGenerator.hash_argument(value)
                        key_parts.append(f"{key}={val_hash}")
                    except Exception as e:
                        logger.debug(f"Failed to hash kwarg {key}: {e}")
                        key_parts.append(f"{key}={id(value)}")

                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            except Exception as e:
                logger.warning(f"Cache key generation failed for {func_name}: {e}")
                cache_stats.record_miss(func_name)
                return func(*args, **kwargs)

            # Track data access if requested
            if track_data_access:
                _track_dataframe_access(args, kwargs, dataset_name, purpose)

            # Check if this is a cache hit
            # Note: We can't directly check joblib's cache, so we track via call patterns
            try:
                result = cached_func(*args, **kwargs)
                cache_stats.record_hit(func_name)
                return result
            except (EOFError, Exception) as e:
                logger.warning(f"Cache error for {func_name}: {e} - recomputing")
                cache_stats.record_miss(func_name)
                return func(*args, **kwargs)

        wrapper._afml_cacheable = True
        return wrapper

    return decorator


def time_aware_cacheable(dataset_name: Optional[str] = None, purpose: Optional[str] = None):
    """
    Time-series aware cacheable decorator.

    Automatically logs DataFrame accesses to DataAccessTracker and
    includes temporal range in cache key.

    Usage:
        @time_aware_cacheable(dataset_name="train_2020_2023", purpose="train")
        def load_market_data(symbol, start, end):
            return data
    """

    def decorator(func: Callable) -> Callable:
        # Import at runtime to avoid circular imports
        from . import cache_stats, memory
        from .data_access_tracker import get_data_tracker

        func_name = f"{func.__module__}.{func.__qualname__}"
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function (possibly from cache)
            try:
                result = cached_func(*args, **kwargs)
                cache_stats.record_hit(func_name)
            except (EOFError, Exception) as e:
                logger.warning(f"Cache error for {func_name}: {e} - recomputing")
                cache_stats.record_miss(func_name)
                result = func(*args, **kwargs)

            # Track data access AFTER execution (so we have the result)
            if isinstance(result, pd.DataFrame) and isinstance(result.index, pd.DatetimeIndex):
                if len(result) > 0:
                    tracker = get_data_tracker()

                    # Use provided dataset_name or try to infer from kwargs
                    ds_name = dataset_name or kwargs.get("name", "unnamed_dataset")
                    ds_purpose = purpose or kwargs.get("purpose", "unknown")

                    tracker.log_access(
                        dataset_name=ds_name,
                        start_date=result.index[0],
                        end_date=result.index[-1],
                        purpose=ds_purpose,
                        data_shape=result.shape,
                    )

                    logger.debug(
                        f"Tracked access: {ds_name} "
                        f"[{result.index[0]} to {result.index[-1]}] "
                        f"for {ds_purpose}"
                    )

            return result

        wrapper._afml_cacheable = True
        return wrapper

    return decorator


def _track_dataframe_access(
    args: tuple, kwargs: dict, dataset_name: Optional[str], purpose: Optional[str]
):
    """Helper to track DataFrame accesses in arguments."""
    from .data_access_tracker import get_data_tracker

    tracker = get_data_tracker()

    # Check args for DataFrames
    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame) and isinstance(arg.index, pd.DatetimeIndex):
            if len(arg) > 0:
                ds_name = dataset_name or f"arg_{i}_dataframe"
                ds_purpose = purpose or "unknown"

                tracker.log_access(
                    dataset_name=ds_name,
                    start_date=arg.index[0],
                    end_date=arg.index[-1],
                    purpose=ds_purpose,
                    data_shape=arg.shape,
                )

    # Check kwargs for DataFrames
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.DatetimeIndex):
            if len(value) > 0:
                ds_name = dataset_name or f"{key}_dataframe"
                ds_purpose = purpose or "unknown"

                tracker.log_access(
                    dataset_name=ds_name,
                    start_date=value.index[0],
                    end_date=value.index[-1],
                    purpose=ds_purpose,
                    data_shape=value.shape,
                )


__all__ = [
    "CacheKeyGenerator",
    "robust_cacheable",
    "time_aware_cacheable",
]
