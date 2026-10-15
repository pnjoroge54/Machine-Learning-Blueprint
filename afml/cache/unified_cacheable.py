"""
Unified cache monitoring system for AFML.
Consolidates monitoring across cv_cache, robust_cache_keys, and clf_hyper_fit.
"""

import hashlib
import inspect
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from loguru import logger
from scipy.stats import randint, rv_continuous, rv_discrete, uniform

from . import cache_stats, memory
from .cache_monitoring import get_cache_monitor
from .robust_cache_keys import stabilize_pandas_args


class UnifiedCacheKeyGenerator:
    """
    Enhanced cache key generator that handles scipy distributions.

    Key Features:
    - Handles scipy.stats distributions (randint, uniform, etc.)
    - Unified monitoring integration
    - Consistent hashing across all cache decorators
    """

    @staticmethod
    def generate_key(func: Callable, args: tuple, kwargs: dict) -> str:
        """
        Generate robust cache key handling all AFML data types.

        Special handling for:
        - scipy distributions (convert to deterministic params)
        - sklearn estimators
        - pandas DataFrames
        - numpy arrays
        """
        key_parts = [
            func.__module__,
            func.__qualname__,
        ]

        # Get function signature for parameter mapping
        sig = inspect.signature(func)

        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, param_value in bound.arguments.items():
                key_part = UnifiedCacheKeyGenerator._hash_parameter(param_name, param_value)
                key_parts.append(key_part)

        except Exception as e:
            logger.debug(f"Failed to bind arguments for {func.__name__}: {e}")
            # Fallback to positional hashing
            for i, arg in enumerate(args):
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(f"arg_{i}", arg))
            for k, v in kwargs.items():
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(k, v))

        combined = "_".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def _hash_parameter(name: str, value: Any) -> str:
        """Hash a single parameter with type-specific handling."""

        # 1. Handle scipy distributions (KEY FIX for clf_hyper_fit)
        if isinstance(value, (rv_discrete, rv_continuous)):
            return UnifiedCacheKeyGenerator._hash_scipy_distribution(name, value)

        # 2. Handle dictionaries that might contain scipy distributions
        if isinstance(value, dict):
            return UnifiedCacheKeyGenerator._hash_dict_with_distributions(name, value)

        # 3. Handle sklearn estimators
        try:
            from sklearn.base import BaseEstimator

            if isinstance(value, BaseEstimator):
                return UnifiedCacheKeyGenerator._hash_sklearn_estimator(name, value)
        except ImportError:
            pass

        # 4. Handle pandas/numpy (delegate to existing logic)
        from .robust_cache_keys import CacheKeyGenerator

        return CacheKeyGenerator._hash_argument(value, name)

    @staticmethod
    def _hash_scipy_distribution(name: str, dist) -> str:
        """
        Hash scipy distribution to deterministic string.

        This is the KEY FIX for clf_hyper_fit cache misses.
        """
        dist_type = type(dist).__name__

        # Extract distribution parameters
        if hasattr(dist, "args"):
            args = dist.args
        else:
            args = ()

        if hasattr(dist, "kwds"):
            kwds = dist.kwds
        else:
            kwds = {}

        # Create deterministic representation
        params = {"type": dist_type, "args": args, "kwds": kwds}

        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{name}_dist_{dist_type}_{param_hash}"

    @staticmethod
    def _hash_dict_with_distributions(name: str, d: dict) -> str:
        """
        Hash dictionary that may contain scipy distributions.

        Critical for param_distributions in RandomizedSearchCV.
        """
        sorted_items = []

        for key, value in sorted(d.items()):
            if isinstance(value, (rv_discrete, rv_continuous)):
                # Hash the distribution deterministically
                val_hash = UnifiedCacheKeyGenerator._hash_scipy_distribution(f"{name}_{key}", value)
            else:
                # Use standard parameter hashing
                val_hash = UnifiedCacheKeyGenerator._hash_parameter(f"{name}_{key}", value)

            sorted_items.append(f"{key}={val_hash}")

        combined = "_".join(sorted_items)
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    @staticmethod
    def _hash_sklearn_estimator(name: str, estimator) -> str:
        """Hash sklearn estimator including nested parameters."""
        try:
            estimator_type = type(estimator).__name__
            params = estimator.get_params(deep=True)

            # Filter serializable params
            serializable = {}
            for k, v in params.items():
                try:
                    json.dumps(v)
                    serializable[k] = v
                except (TypeError, ValueError):
                    serializable[k] = f"<{type(v).__name__}>"

            param_str = json.dumps(serializable, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

            return f"{name}_est_{estimator_type}_{param_hash}"

        except Exception as e:
            logger.debug(f"Estimator hashing failed: {e}")
            return f"{name}_est_{type(estimator).__name__}_{id(estimator)}"


class UnifiedCacheMonitor:
    """
    Unified monitoring for all AFML cache operations.

    Consolidates tracking from:
    - cv_cache decorators
    - robust_cacheable decorators
    - clf_hyper_fit function
    """

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
        """
        Unified tracking for all cache operations.

        Args:
            func_name: Full function name (module.function)
            is_hit: Whether this was a cache hit
            computation_time: Time spent computing (if miss)
            cache_key: The cache key used
        """
        # Track in core stats
        if is_hit:
            self.cache_stats.record_hit(func_name)
        else:
            self.cache_stats.record_miss(func_name)

        # Track access time
        self.core_monitor.track_access(func_name)

        # Track computation time if provided
        if computation_time is not None and not is_hit:
            self.core_monitor.track_computation_time(func_name, computation_time)

        # Log for debugging
        status = "HIT" if is_hit else "MISS"
        log_msg = f"Cache {status}: {func_name}"

        if cache_key:
            log_msg += f" (key: {cache_key[:8]}...)"

        if computation_time:
            log_msg += f" (computed in {computation_time:.2f}s)"

        logger.debug(log_msg)

    def get_unified_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get unified statistics for function or all functions."""
        if func_name:
            # Get stats for specific function
            func_stats = self.core_monitor.get_function_stats(func_name)
            return {
                "function": func_name,
                "stats": func_stats,
                "cache_info": self.cache_stats.get_stats().get(func_name, {}),
            }
        else:
            # Get overall stats
            return {
                "summary": self.core_monitor.generate_health_report(),
                "all_stats": self.cache_stats.get_stats(),
            }


# Global unified monitor
_unified_monitor: Optional[UnifiedCacheMonitor] = None


def get_unified_monitor() -> UnifiedCacheMonitor:
    """Get global unified monitor instance."""
    global _unified_monitor
    if _unified_monitor is None:
        _unified_monitor = UnifiedCacheMonitor()
    return _unified_monitor


def unified_cacheable(
    track_time: bool = True,
    track_data_access: bool = False,
    dataset_name: Optional[str] = None,
    purpose: Optional[str] = None,
):
    """
    Unified caching decorator with consolidated monitoring.

    Replaces both robust_cacheable and cv_cacheable with unified implementation.

    Args:
        track_time: Track computation time
        track_data_access: Track DataFrame access for contamination detection
        dataset_name: Name for data tracking
        purpose: Purpose of data access (train/test/validate)

    Usage:
        @unified_cacheable
        def my_function(data, params):
            # expensive computation
            return result
    """

    def decorator(func: Callable) -> Callable:
        func_name = f"{func.__module__}.{func.__qualname__}"
        monitor = get_unified_monitor()

        # Cache the function using joblib
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate unified cache key
            cache_key = UnifiedCacheKeyGenerator.generate_key(func, args, kwargs)

            # Check if in cache
            is_hit = False
            computation_time = None

            # check_call_in_cache returns a bool; it does not raise on a miss.
            # Key generation reads the index, so restore a canonical pandas state
            # before joblib hashes the arguments (see stabilize_pandas_args).
            stabilize_pandas_args(args, kwargs)
            try:
                is_hit = bool(cached_func.check_call_in_cache(*args, **kwargs))
            except Exception:
                is_hit = False

            # Track the call
            start_time = time.time() if track_time and not is_hit else None

            # Execute (from cache or compute)
            try:
                stabilize_pandas_args(args, kwargs)
                result = cached_func(*args, **kwargs)

                # Calculate computation time if it was a miss
                if start_time is not None:
                    computation_time = time.time() - start_time

            except Exception as e:
                # Handle cache corruption
                logger.warning(f"Cache error for {func_name}: {e}, recomputing")

                # Clear and recompute
                try:
                    cached_func.clear()
                except:
                    pass

                start_time = time.time() if track_time else None
                result = func(*args, **kwargs)

                if start_time:
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
                _track_data_access_unified(args, kwargs, dataset_name, purpose)

            return result

        wrapper._afml_cacheable = True
        wrapper._cache_key_generator = UnifiedCacheKeyGenerator
        return wrapper

    return decorator


def _track_data_access_unified(args, kwargs, dataset_name, purpose):
    """Track DataFrame access for data contamination detection."""
    try:
        import pandas as pd

        from .data_access_tracker import get_data_tracker

        tracker = get_data_tracker()

        # Check all arguments for trackable DataFrames
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
        logger.debug(f"Data access tracking failed: {e}")


# ============================================================================
# Enhanced clf_hyper_fit with proper caching
# ============================================================================


def create_cacheable_param_grid(param_distributions: Dict) -> Dict:
    """
    Convert scipy distributions to cacheable representations.

    This is the KEY FIX for clf_hyper_fit cache misses.

    Args:
        param_distributions: Dict with scipy distributions

    Returns:
        Dict with deterministic distribution representations
    """
    cacheable_params = {}

    for key, value in param_distributions.items():
        if isinstance(value, (rv_discrete, rv_continuous)):
            # Convert to tuple of (type, args, kwds)
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
    """
    Reconstruct scipy distributions from cacheable representation.

    Args:
        cacheable_params: Output from create_cacheable_param_grid

    Returns:
        Original param_distributions dict
    """
    reconstructed = {}

    for key, value in cacheable_params.items():
        if isinstance(value, tuple) and len(value) == 3:
            # This is a cached distribution
            dist_type, args, kwds = value

            # Reconstruct the distribution
            if dist_type == "randint_gen":
                reconstructed[key] = randint(*args, **kwds)
            elif dist_type == "uniform_gen":
                reconstructed[key] = uniform(*args, **kwds)
            else:
                # Try to get the distribution class dynamically
                try:
                    import scipy.stats as stats

                    dist_class = getattr(stats, dist_type.replace("_gen", ""))
                    reconstructed[key] = dist_class(*args, **kwds)
                except:
                    logger.warning(f"Could not reconstruct distribution: {dist_type}")
                    reconstructed[key] = value
        else:
            reconstructed[key] = value

    return reconstructed


@unified_cacheable(track_time=True)
def clf_hyper_fit_cached(
    features,
    labels,
    t1,
    pipe_clf,
    param_grid_cacheable,  # ← Now accepts cacheable version
    cv=5,
    bagging_n_estimators=0,
    bagging_max_samples=1.0,
    bagging_max_features=1.0,
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0,
    random_state=None,
    verbose=0,
    **fit_params,
):
    """
    Cached version of clf_hyper_fit that properly handles scipy distributions.

    This is the FIXED version that will cache properly.
    """
    # Reconstruct param_grid from cacheable version
    param_grid = reconstruct_param_grid(param_grid_cacheable)

    # Now call original clf_hyper_fit with reconstructed params
    from ..cross_validation.hyperfit import clf_hyper_fit

    return clf_hyper_fit(
        features=features,
        labels=labels,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid=param_grid,
        cv=cv,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_max_features=bagging_max_features,
        rnd_search_iter=rnd_search_iter,
        n_jobs=n_jobs,
        pct_embargo=pct_embargo,
        random_state=random_state,
        verbose=verbose,
        **fit_params,
    )


# ============================================================================
# Convenience wrapper that handles conversion automatically
# ============================================================================


def clf_hyper_fit_auto_cache(
    features, labels, t1, pipe_clf, param_grid, **kwargs  # ← Accepts scipy distributions directly
):
    """
    Wrapper that automatically converts param_grid for caching.

    Usage:
        from scipy.stats import randint, uniform

        param_grid = {
            'clf__n_estimators': randint(100, 500),
            'clf__max_depth': randint(3, 20),
        }

        # Just call this instead of clf_hyper_fit
        model, results = clf_hyper_fit_auto_cache(
            features, labels, t1, pipe_clf, param_grid
        )
    """
    # Convert to cacheable format
    param_grid_cacheable = create_cacheable_param_grid(param_grid)

    # Call cached version
    return clf_hyper_fit_cached(
        features=features,
        labels=labels,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid_cacheable=param_grid_cacheable,
        **kwargs,
    )


# ============================================================================
# Diagnostic tools
# ============================================================================


def print_unified_cache_report():
    """Print comprehensive cache report using unified monitoring."""
    monitor = get_unified_monitor()
    stats = monitor.get_unified_stats()

    print("\n" + "=" * 70)
    print("UNIFIED CACHE REPORT")
    print("=" * 70)

    # Overall summary
    summary = stats["summary"]
    print(f"\nOverall:")
    print(f"  Functions Tracked: {summary.total_functions}")
    print(f"  Overall Hit Rate: {summary.overall_hit_rate:.1%}")
    print(f"  Total Calls: {summary.total_calls:,}")
    print(f"  Cache Size: {summary.total_cache_size_mb:.1f} MB")

    # Top performers
    if summary.top_performers:
        print(f"\nTop Cache Performers:")
        for i, perf in enumerate(summary.top_performers[:3], 1):
            func_name = perf.function_name.split(".")[-1]
            print(f"  {i}. {func_name}: {perf.hit_rate:.1%} hit rate ({perf.total_calls} calls)")

    # Function-specific details
    print(f"\nFunction Details:")
    for func_name, func_data in stats["all_stats"].items():
        short_name = func_name.split(".")[-1]
        hits = func_data.get("hits", 0)
        misses = func_data.get("misses", 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0

        print(f"  {short_name}:")
        print(f"    Hit Rate: {hit_rate:.1%} ({hits}/{total})")

    print("=" * 70 + "\n")


__all__ = [
    "UnifiedCacheKeyGenerator",
    "UnifiedCacheMonitor",
    "get_unified_monitor",
    "unified_cacheable",
    "create_cacheable_param_grid",
    "clf_hyper_fit_cached",
    "clf_hyper_fit_auto_cache",
    "print_unified_cache_report",
]
