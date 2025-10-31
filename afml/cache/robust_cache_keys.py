"""
Robust cache key generation for financial ML data structures.
Handles numpy arrays, pandas DataFrames, and time-series data properly.
"""

import hashlib
from typing import Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class CacheKeyGenerator:
    """Generate robust, collision-resistant cache keys for ML data structures."""

    @staticmethod
    def generate_key(func, args: tuple, kwargs: dict) -> str:
        """
        Generate a robust cache key for a function call.

        Args:
            func: The function being cached
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            MD5 hash string representing the unique call signature
        """
        key_parts = [
            func.__module__,
            func.__qualname__,
        ]

        # Process positional arguments
        for i, arg in enumerate(args):
            try:
                key_part = CacheKeyGenerator._hash_argument(arg, f"arg_{i}")
                key_parts.append(key_part)
            except Exception as e:
                logger.warning(f"Failed to hash argument {i} of type {type(arg)}: {e}")
                # Fallback to string representation
                key_parts.append(f"arg_{i}_{str(hash(str(arg)))}")

        # Process keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            try:
                key_part = CacheKeyGenerator._hash_argument(value, key)
                key_parts.append(f"{key}={key_part}")
            except Exception as e:
                logger.warning(f"Failed to hash kwarg '{key}' of type {type(value)}: {e}")
                # Fallback
                key_parts.append(f"{key}={str(hash(str(value)))}")

        # Combine all parts and hash
        combined = "_".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def _hash_argument(arg: Any, name: str) -> str:
        """Hash a single argument based on its type."""
        if isinstance(arg, np.ndarray):
            return CacheKeyGenerator._hash_numpy_array(arg, name)
        elif isinstance(arg, pd.DataFrame):
            return CacheKeyGenerator._hash_dataframe(arg, name)
        elif isinstance(arg, pd.Series):
            return CacheKeyGenerator._hash_series(arg, name)
        elif isinstance(arg, (list, tuple)):
            return CacheKeyGenerator._hash_sequence(arg, name)
        elif isinstance(arg, dict):
            return CacheKeyGenerator._hash_dict(arg, name)
        elif isinstance(arg, (int, float, str, bool, type(None))):
            return CacheKeyGenerator._hash_primitive(arg, name)
        else:
            # Fallback for unknown types
            return CacheKeyGenerator._hash_generic(arg, name)

    @staticmethod
    def _hash_numpy_array(arr: np.ndarray, name: str) -> str:
        """Hash numpy array including shape, dtype, and content."""
        # For large arrays, sample for performance
        if arr.size > 10000:
            # Hash shape, dtype, and a sample
            sample = arr.flat[:: max(1, arr.size // 1000)]  # Sample ~1000 points
            content_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
        else:
            # Hash full content for small arrays
            content_hash = hashlib.md5(arr.tobytes()).hexdigest()[:8]

        return f"{name}_arr_{arr.shape}_{arr.dtype}_{content_hash}"

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame, name: str) -> str:
        """Hash pandas DataFrame including index, columns, dtypes, and content."""
        parts = [
            f"shape_{df.shape}",
            f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}",
            f"dtypes_{hashlib.md5(str(tuple(df.dtypes)).encode()).hexdigest()[:8]}",
        ]

        # Hash index
        if isinstance(df.index, pd.DatetimeIndex):
            # For datetime index, hash start, end, and frequency
            parts.append(f"idx_dt_{df.index[0]}_{df.index[-1]}_{len(df.index)}")
        else:
            idx_hash = hashlib.md5(str(tuple(df.index)).encode()).hexdigest()[:8]
            parts.append(f"idx_{idx_hash}")

        # Hash content (sample for large DataFrames)
        if df.size > 10000:
            # Sample rows for hashing
            sample_rows = df.iloc[:: max(1, len(df) // 100)]  # ~100 rows
            content_hash = hashlib.md5(sample_rows.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(df.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")

        return f"{name}_df_{'_'.join(parts)}"

    @staticmethod
    def _hash_series(series: pd.Series, name: str) -> str:
        """Hash pandas Series."""
        parts = [
            f"len_{len(series)}",
            f"dtype_{series.dtype}",
        ]

        # Hash index
        if isinstance(series.index, pd.DatetimeIndex):
            parts.append(f"idx_dt_{series.index[0]}_{series.index[-1]}")
        else:
            idx_hash = hashlib.md5(str(tuple(series.index)).encode()).hexdigest()[:8]
            parts.append(f"idx_{idx_hash}")

        # Hash values
        if len(series) > 1000:
            sample = series.iloc[:: max(1, len(series) // 100)]
            content_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(series.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")

        return f"{name}_series_{'_'.join(parts)}"

    @staticmethod
    def _hash_sequence(seq: Tuple[Any, ...] | list, name: str) -> str:
        """Hash list or tuple recursively."""
        if len(seq) == 0:
            return f"{name}_empty_seq"

        # Hash each element
        element_hashes = []
        for i, item in enumerate(seq):
            elem_hash = CacheKeyGenerator._hash_argument(item, f"{name}_{i}")
            element_hashes.append(elem_hash)

        combined = "_".join(element_hashes)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def _hash_dict(d: dict, name: str) -> str:
        """Hash dictionary recursively."""
        if len(d) == 0:
            return f"{name}_empty_dict"

        # Sort keys for consistency
        items_hash = []
        for key, value in sorted(d.items()):
            val_hash = CacheKeyGenerator._hash_argument(value, f"{name}_{key}")
            items_hash.append(f"{key}={val_hash}")

        combined = "_".join(items_hash)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def _hash_primitive(value: Any, name: str) -> str:
        """Hash primitive types."""
        return f"{name}_{type(value).__name__}_{hash(value)}"

    @staticmethod
    def _hash_generic(obj: Any, name: str) -> str:
        """Fallback hashing for unknown types."""
        try:
            # Try to use object's __repr__
            return f"{name}_{type(obj).__name__}_{hash(repr(obj))}"
        except Exception:
            # Last resort: use id
            return f"{name}_{type(obj).__name__}_{id(obj)}"


class TimeSeriesCacheKey(CacheKeyGenerator):
    """
    Extended cache key generator with time-series awareness.
    Useful for financial data where lookback periods matter.
    """

    @staticmethod
    def generate_key_with_time_range(
        func, args: tuple, kwargs: dict, time_range: Tuple[pd.Timestamp, pd.Timestamp] = None
    ) -> str:
        """
        Generate cache key that includes time range information.

        Args:
            func: Function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            time_range: Optional (start, end) timestamp tuple

        Returns:
            Cache key string
        """
        base_key = CacheKeyGenerator.generate_key(func, args, kwargs)

        if time_range is None:
            # Try to extract time range from data
            time_range = TimeSeriesCacheKey._extract_time_range(args, kwargs)

        if time_range:
            start, end = time_range
            time_hash = f"time_{start}_{end}"
            return f"{base_key}_{time_hash}"

        return base_key

    @staticmethod
    def _extract_time_range(args: tuple, kwargs: dict) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
        """
        Attempt to extract time range from function arguments.
        Looks for DataFrames with DatetimeIndex or explicit start/end parameters.
        """
        # Check kwargs for explicit time parameters
        if "start_date" in kwargs and "end_date" in kwargs:
            return (pd.Timestamp(kwargs["start_date"]), pd.Timestamp(kwargs["end_date"]))

        # Check for DataFrames with DatetimeIndex in args
        for arg in args:
            if isinstance(arg, pd.DataFrame) and isinstance(arg.index, pd.DatetimeIndex):
                if len(arg.index) > 0:
                    return (arg.index[0], arg.index[-1])

            elif isinstance(arg, pd.Series) and isinstance(arg.index, pd.DatetimeIndex):
                if len(arg.index) > 0:
                    return (arg.index[0], arg.index[-1])

        return None


# =============================================================================
# Integration with existing cacheable decorator
# =============================================================================


def create_robust_cacheable(use_time_awareness: bool = False):
    """
    Factory function to create cacheable decorator with robust key generation.

    Args:
        use_time_awareness: If True, include time-series range in cache keys

    Returns:
        Decorator function
    """
    from functools import wraps

    from . import cache_stats, memory

    def cacheable(func):
        """Enhanced cacheable decorator with robust key generation."""
        func_name = f"{func.__module__}.{func.__qualname__}"
        cached_func = memory.cache(func)

        # Track seen signatures for hit detection
        seen_signatures = set()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate robust cache key
            try:
                if use_time_awareness:
                    cache_key = TimeSeriesCacheKey.generate_key_with_time_range(func, args, kwargs)
                else:
                    cache_key = CacheKeyGenerator.generate_key(func, args, kwargs)

                # Track hit/miss
                if cache_key in seen_signatures:
                    cache_stats.record_hit(func_name)
                else:
                    cache_stats.record_miss(func_name)
                    seen_signatures.add(cache_key)

            except Exception as e:
                logger.warning(f"Cache key generation failed for {func_name}: {e}")
                cache_stats.record_miss(func_name)

            # Try cached function with error handling
            try:
                return cached_func(*args, **kwargs)
            except (EOFError, Exception) as e:
                logger.warning(f"Cache error for {func_name}: {e} - recomputing")
                return func(*args, **kwargs)

        wrapper._afml_cacheable = True
        return wrapper

    return cacheable


# =============================================================================
# Convenience exports
# =============================================================================

# Standard decorator
robust_cacheable = create_robust_cacheable(use_time_awareness=False)

# Time-series aware decorator
time_aware_cacheable = create_robust_cacheable(use_time_awareness=True)


__all__ = [
    "CacheKeyGenerator",
    "TimeSeriesCacheKey",
    "robust_cacheable",
    "time_aware_cacheable",
    "create_robust_cacheable",
]
