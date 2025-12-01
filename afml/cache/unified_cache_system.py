"""
Unified cache system for AFML - eliminates all duplication.

This module replaces:
- robust_cache_keys.py (most functionality)
- cv_cache.py (CV-specific caching)
- Parts of cache_monitoring.py integration

One unified system with consistent behavior across all decorators.
"""

import hashlib
import inspect
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rv_continuous, rv_discrete

from . import cache_stats, memory
from .cache_monitoring import get_cache_monitor

# =============================================================================
# Core: Unified Cache Key Generator (with time-awareness)
# =============================================================================


class UnifiedCacheKeyGenerator:
    """
    Single cache key generator for all AFML use cases.

    Handles:
    - Pandas DataFrames (with temporal awareness)
    - NumPy arrays
    - Sklearn estimators
    - Scipy distributions (KEY FIX for clf_hyper_fit)
    - CV generators (KFold, PurgedKFold, etc.)
    - Time-series data with lookback periods
    """

    @staticmethod
    def generate_key(func: Callable, args: tuple, kwargs: dict, time_aware: bool = False) -> str:
        """
        Generate unified cache key.

        Args:
            func: Function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            time_aware: If True, include temporal bounds in key

        Returns:
            MD5 hash representing unique call signature
        """
        key_parts = [func.__module__, func.__qualname__]

        # Get function signature for proper parameter mapping
        sig = inspect.signature(func)

        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Extract time range if time-aware
            time_range = None
            if time_aware:
                time_range = UnifiedCacheKeyGenerator._extract_time_range(bound.arguments)

            # Hash each parameter
            for param_name, param_value in bound.arguments.items():
                key_part = UnifiedCacheKeyGenerator._hash_parameter(param_name, param_value)
                key_parts.append(key_part)

            # Add time range to key if present
            if time_range:
                start, end = time_range
                key_parts.append(f"time_{start}_{end}")

        except Exception as e:
            logger.debug(f"Parameter binding failed for {func.__name__}: {e}")
            # Fallback to positional hashing
            for i, arg in enumerate(args):
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(f"arg_{i}", arg))
            for k, v in kwargs.items():
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(k, v))

        combined = "_".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def _hash_parameter(name: str, value: Any) -> str:
        """Route parameter to appropriate hashing method."""

        # 1. Scipy distributions (fixes clf_hyper_fit caching)
        if isinstance(value, (rv_discrete, rv_continuous)):
            return UnifiedCacheKeyGenerator._hash_scipy_dist(name, value)

        # 2. Dictionaries (may contain scipy distributions)
        if isinstance(value, dict):
            return UnifiedCacheKeyGenerator._hash_dict(name, value)

        # 3. Sklearn estimators
        try:
            from sklearn.base import BaseEstimator

            if isinstance(value, BaseEstimator):
                return UnifiedCacheKeyGenerator._hash_estimator(name, value)
        except ImportError:
            pass

        # 4. CV generators (for cross-validation caching)
        if hasattr(value, "split") and hasattr(value, "n_splits"):
            return UnifiedCacheKeyGenerator._hash_cv_generator(name, value)

        # 5. Pandas DataFrames
        if isinstance(value, pd.DataFrame):
            return UnifiedCacheKeyGenerator._hash_dataframe(name, value)

        # 6. Pandas Series
        if isinstance(value, pd.Series):
            return UnifiedCacheKeyGenerator._hash_series(name, value)

        # 7. NumPy arrays
        if isinstance(value, np.ndarray):
            return UnifiedCacheKeyGenerator._hash_numpy_array(name, value)

        # 8. Sequences (lists, tuples)
        if isinstance(value, (list, tuple)):
            return UnifiedCacheKeyGenerator._hash_sequence(name, value)

        # 9. Primitives
        if isinstance(value, (int, float, str, bool, type(None))):
            return f"{name}_{type(value).__name__}_{hash(value)}"

        # 10. Fallback
        return UnifiedCacheKeyGenerator._hash_generic(name, value)

    @staticmethod
    def _hash_scipy_dist(name: str, dist) -> str:
        """Hash scipy distribution deterministically."""
        dist_type = type(dist).__name__
        args = dist.args if hasattr(dist, "args") else ()
        kwds = dist.kwds if hasattr(dist, "kwds") else {}

        params = {"type": dist_type, "args": args, "kwds": kwds}
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{name}_dist_{dist_type}_{param_hash}"

    @staticmethod
    def _hash_dict(name: str, d: dict) -> str:
        """Hash dictionary recursively (handles nested scipy distributions)."""
        if not d:
            return f"{name}_empty_dict"

        sorted_items = []
        for key, value in sorted(d.items()):
            val_hash = UnifiedCacheKeyGenerator._hash_parameter(f"{name}_{key}", value)
            sorted_items.append(f"{key}={val_hash}")

        combined = "_".join(sorted_items)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def _hash_estimator(name: str, estimator) -> str:
        """Hash sklearn estimator (type + parameters, not state)."""
        try:
            est_type = type(estimator).__name__
            params = estimator.get_params(deep=True)

            # Filter to serializable params
            serializable = {}
            for k, v in params.items():
                try:
                    json.dumps(v)
                    serializable[k] = v
                except (TypeError, ValueError):
                    serializable[k] = f"<{type(v).__name__}>"

            param_str = json.dumps(serializable, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

            return f"{name}_est_{est_type}_{param_hash}"
        except Exception as e:
            logger.debug(f"Estimator hashing failed: {e}")
            return f"{name}_est_{type(estimator).__name__}_{id(estimator)}"

    @staticmethod
    def _hash_cv_generator(name: str, cv_gen) -> str:
        """Hash cross-validation generator."""
        try:
            cv_type = type(cv_gen).__name__

            params = {}
            if hasattr(cv_gen, "n_splits"):
                params["n_splits"] = cv_gen.n_splits
            if hasattr(cv_gen, "pct_embargo"):
                params["pct_embargo"] = cv_gen.pct_embargo
            if hasattr(cv_gen, "t1") and isinstance(cv_gen.t1, pd.Series):
                t1 = cv_gen.t1
                params["t1_len"] = len(t1)
                params["t1_start"] = str(t1.index[0])
                params["t1_end"] = str(t1.index[-1])

            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

            return f"{name}_cv_{cv_type}_{param_hash}"
        except Exception:
            return f"{name}_cv_{type(cv_gen).__name__}_{id(cv_gen)}"

    @staticmethod
    def _hash_dataframe(name: str, df: pd.DataFrame) -> str:
        """Hash DataFrame including structure and content."""
        parts = [
            f"shape_{df.shape}",
            f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}",
        ]

        # Hash index
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            parts.append(f"idx_dt_{df.index[0]}_{df.index[-1]}_{len(df)}")
        else:
            idx_hash = hashlib.md5(str(tuple(df.index[:10])).encode()).hexdigest()[:8]
            parts.append(f"idx_{idx_hash}")

        # Sample content for large DataFrames
        if df.size > 10000:
            sample = df.iloc[:: max(1, len(df) // 100)]
            content_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(df.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")
        return f"{name}_df_{'_'.join(parts)}"

    @staticmethod
    def _hash_series(name: str, series: pd.Series) -> str:
        """Hash pandas Series."""
        parts = [f"len_{len(series)}", f"dtype_{series.dtype}"]

        if isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
            parts.append(f"idx_dt_{series.index[0]}_{series.index[-1]}")

        if len(series) > 1000:
            sample = series.iloc[:: max(1, len(series) // 100)]
            content_hash = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(series.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")
        return f"{name}_ser_{'_'.join(parts)}"

    @staticmethod
    def _hash_numpy_array(name: str, arr: np.ndarray) -> str:
        """Hash numpy array."""
        if arr.size > 10000:
            sample = arr.flat[:: max(1, arr.size // 1000)]
            content_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(arr.tobytes()).hexdigest()[:8]

        return f"{name}_arr_{arr.shape}_{arr.dtype}_{content_hash}"

    @staticmethod
    def _hash_sequence(name: str, seq) -> str:
        """Hash list or tuple recursively."""
        if not seq:
            return f"{name}_empty_seq"

        elem_hashes = [
            UnifiedCacheKeyGenerator._hash_parameter(f"{name}_{i}", item)
            for i, item in enumerate(seq)
        ]
        combined = "_".join(elem_hashes)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def _hash_generic(name: str, obj: Any) -> str:
        """Fallback for unknown types."""
        try:
            return f"{name}_{type(obj).__name__}_{hash(repr(obj))}"
        except:
            return f"{name}_{type(obj).__name__}_{id(obj)}"

    @staticmethod
    def _extract_time_range(params: dict) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Extract temporal range from parameters for time-aware caching."""
        # Check for explicit time parameters
        if "start_date" in params and "end_date" in params:
            return (pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"]))

        # Check for DataFrames with DatetimeIndex
        for param_value in params.values():
            if isinstance(param_value, pd.DataFrame):
                if isinstance(param_value.index, pd.DatetimeIndex) and len(param_value) > 0:
                    return (param_value.index[0], param_value.index[-1])
            elif isinstance(param_value, pd.Series):
                if isinstance(param_value.index, pd.DatetimeIndex) and len(param_value) > 0:
                    return (param_value.index[0], param_value.index[-1])

        return None


# =============================================================================
# Core: Unified Cache Monitor
# =============================================================================


class UnifiedCacheMonitor:
    """Single monitoring system for all cache operations."""

    def __init__(self):
        self.core_monitor = get_cache_monitor()
        self.cache_stats = cache_stats

    def track_cache_call(
        self,
        func_name: str,
        is_hit: bool,
        computation_time: Optional[float] = None,
        cache_key: Optional[str] = None,
    ):
        """Track cache operation (hit/miss + timing)."""
        # Update stats
        if is_hit:
            self.cache_stats.record_hit(func_name)
        else:
            self.cache_stats.record_miss(func_name)

        # Track access time
        self.core_monitor.track_access(func_name)

        # Track computation time for misses
        if computation_time is not None and not is_hit:
            self.core_monitor.track_computation_time(func_name, computation_time)

        # Debug logging
        status = "HIT" if is_hit else "MISS"
        log_msg = f"Cache {status}: {func_name}"
        if cache_key:
            log_msg += f" (key: {cache_key[:8]}...)"
        if computation_time:
            log_msg += f" ({computation_time:.2f}s)"
        logger.debug(log_msg)


# Global monitor instance
_unified_monitor: Optional[UnifiedCacheMonitor] = None


def get_unified_monitor() -> UnifiedCacheMonitor:
    """Get global unified monitor."""
    global _unified_monitor
    if _unified_monitor is None:
        _unified_monitor = UnifiedCacheMonitor()
    return _unified_monitor


# =============================================================================
# Core: Universal Cacheable Decorator
# =============================================================================


def cacheable(
    time_aware: bool = False,
    track_data_access: bool = False,
    dataset_name: Optional[str] = None,
    purpose: Optional[str] = None,
):
    """
    Universal caching decorator - replaces all previous decorators.

    This ONE decorator replaces:
    - robust_cacheable
    - time_aware_cacheable
    - data_tracking_cacheable
    - cv_cacheable
    - cv_cache_with_classifier_state

    Args:
        time_aware: Include temporal bounds in cache key
        track_data_access: Log DataFrame access for contamination detection
        dataset_name: Name for data tracking
        purpose: train/test/validate/optimize

    Usage:
        # Basic (replaces robust_cacheable)
        @cacheable()
        def my_function(df): ...

        # Time-aware (replaces time_aware_cacheable)
        @cacheable(time_aware=True)
        def my_function(df): ...

        # Data tracking (replaces data_tracking_cacheable)
        @cacheable(track_data_access=True, dataset_name="my_data", purpose="train")
        def my_function(df): ...

        # CV caching (replaces cv_cacheable)
        @cacheable()  # Just works automatically!
        def ml_cross_val_score(clf, X, y, cv_gen): ...
    """

    def decorator(func: Callable) -> Callable:
        func_name = f"{func.__module__}.{func.__qualname__}"
        monitor = get_unified_monitor()
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = UnifiedCacheKeyGenerator.generate_key(
                func, args, kwargs, time_aware=time_aware
            )

            # Check cache status
            is_hit = False
            computation_time = None

            try:
                cached_func.check_call_in_cache(*args, **kwargs)
                is_hit = True
            except:
                is_hit = False

            # Time the operation if it's a miss
            start_time = time.time()

            # Execute
            try:
                result = cached_func(*args, **kwargs)

                if start_time:
                    computation_time = time.time() - start_time

            except (EOFError, Exception) as e:
                # Cache corruption - clear and recompute
                logger.warning(f"Cache error for {func_name}: {e}")
                try:
                    cached_func.clear()
                except:
                    pass

                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time

            # Track in unified monitor
            monitor.track_cache_call(
                func_name=func_name,
                is_hit=is_hit,
                computation_time=computation_time,
                cache_key=cache_key,
            )

            # Track data access if requested
            if track_data_access:
                _track_data_access(args, kwargs, dataset_name, purpose)

            return result

        wrapper._afml_cacheable = True
        return wrapper

    return decorator


def _track_data_access(args, kwargs, dataset_name, purpose):
    """Track DataFrame access for contamination detection."""
    try:
        from .data_access_tracker import get_data_tracker

        tracker = get_data_tracker()

        # Check all arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame) and isinstance(arg.index, pd.DatetimeIndex):
                if len(arg) > 0:
                    tracker.log_access(
                        dataset_name=dataset_name or "unknown",
                        start_date=arg.index[0],
                        end_date=arg.index[-1],
                        purpose=purpose or "unknown",
                        data_shape=arg.shape,
                    )

        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.DatetimeIndex):
                if len(value) > 0:
                    tracker.log_access(
                        dataset_name=dataset_name or key,
                        start_date=value.index[0],
                        end_date=value.index[-1],
                        purpose=purpose or "unknown",
                        data_shape=value.shape,
                    )
    except Exception as e:
        logger.debug(f"Data tracking failed: {e}")


# =============================================================================
# Convenience Aliases (backward compatibility)
# =============================================================================

# Old names → new unified decorator
robust_cacheable = cacheable()
time_aware_cacheable = cacheable(time_aware=True)
cv_cacheable = cacheable()  # Works automatically with CV generators


def data_tracking_cacheable(dataset_name: str, purpose: str):
    """Backward compatible data tracking decorator."""
    return cacheable(track_data_access=True, dataset_name=dataset_name, purpose=purpose)


# =============================================================================
# Special: clf_hyper_fit with scipy distribution support
# =============================================================================


def create_cacheable_param_grid(param_distributions: Dict) -> Dict:
    """Convert scipy distributions to cacheable representation."""
    cacheable_params = {}

    for key, value in param_distributions.items():
        if isinstance(value, (rv_discrete, rv_continuous)):
            dist_info = (
                type(value).__name__,
                value.args if hasattr(value, "args") else (),
                value.kwds if hasattr(value, "kwds") else {},
            )
            cacheable_params[key] = dist_info
        else:
            cacheable_params[key] = value

    return cacheable_params


def reconstruct_param_grid(cacheable_params: Dict) -> Dict:
    """Reconstruct scipy distributions from cacheable representation."""
    from scipy.stats import randint, uniform

    reconstructed = {}

    for key, value in cacheable_params.items():
        if isinstance(value, tuple) and len(value) == 3:
            dist_type, args, kwds = value

            if dist_type == "randint_gen":
                reconstructed[key] = randint(*args, **kwds)
            elif dist_type == "uniform_gen":
                reconstructed[key] = uniform(*args, **kwds)
            else:
                try:
                    import scipy.stats as stats

                    dist_class = getattr(stats, dist_type.replace("_gen", ""))
                    reconstructed[key] = dist_class(*args, **kwds)
                except:
                    logger.warning(f"Could not reconstruct: {dist_type}")
                    reconstructed[key] = value
        else:
            reconstructed[key] = value

    return reconstructed


@cacheable()
def clf_hyper_fit_cached(features, labels, t1, pipe_clf, param_grid_cacheable, **kwargs):
    """Cached version of clf_hyper_fit."""
    from .hyper_fit import clf_hyper_fit

    param_grid = reconstruct_param_grid(param_grid_cacheable)

    return clf_hyper_fit(
        features=features, labels=labels, t1=t1, pipe_clf=pipe_clf, param_grid=param_grid, **kwargs
    )


def clf_hyper_fit_auto_cache(features, labels, t1, pipe_clf, param_grid, **kwargs):
    """
    Drop-in replacement for clf_hyper_fit with automatic caching.

    Just replace:
        clf_hyper_fit(features, labels, t1, pipe_clf, param_grid)
    with:
        clf_hyper_fit_auto_cache(features, labels, t1, pipe_clf, param_grid)
    """
    param_grid_cacheable = create_cacheable_param_grid(param_grid)
    return clf_hyper_fit_cached(features, labels, t1, pipe_clf, param_grid_cacheable, **kwargs)


# =============================================================================
# Convenience: Print cache report
# =============================================================================


def print_cache_report():
    """Print comprehensive cache report."""
    monitor = get_unified_monitor()

    print("\n" + "=" * 70)
    print("UNIFIED CACHE REPORT")
    print("=" * 70)

    # Get health report
    report = monitor.core_monitor.generate_health_report()

    print(f"\nOverall:")
    print(f"  Functions: {report.total_functions}")
    print(f"  Hit Rate: {report.overall_hit_rate:.1%}")
    print(f"  Total Calls: {report.total_calls:,}")
    print(f"  Cache Size: {report.total_cache_size_mb:.1f} MB")

    if report.top_performers:
        print(f"\nTop Performers:")
        for i, perf in enumerate(report.top_performers[:3], 1):
            name = perf.function_name.split(".")[-1]
            print(f"  {i}. {name}: {perf.hit_rate:.1%} ({perf.total_calls} calls)")

    print("=" * 70 + "\n")


__all__ = [
    # Core components
    "UnifiedCacheKeyGenerator",
    "UnifiedCacheMonitor",
    "get_unified_monitor",
    # Main decorator (replaces all others)
    "cacheable",
    # Backward compatibility aliases
    "robust_cacheable",
    "time_aware_cacheable",
    "cv_cacheable",
    "data_tracking_cacheable",
    # clf_hyper_fit support
    "clf_hyper_fit_auto_cache",
    "clf_hyper_fit_cached",
    "create_cacheable_param_grid",
    # Utilities
    "print_cache_report",
]
