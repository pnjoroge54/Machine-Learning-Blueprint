"""
Unified, production-grade caching system for AFML.
Replaces all previous cache key generators and decorators.
"""

import hashlib
import inspect
import json
import os
import pickle
import threading
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Memory
from loguru import logger
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from appdirs import user_cache_dir

# =============================================================================
# Cache Directory & Global Setup
# =============================================================================

def _setup_cache_directories() -> Dict[str, Path]:
    cache_env = os.getenv("AFML_CACHE")
    base_dir = Path(cache_env) if cache_env else Path(user_cache_dir("afml"))
    dirs = {
        "base": base_dir,
        "joblib": base_dir / "joblib_cache",
        "numba": base_dir / "numba_cache",
        "backtest": base_dir / "backtests",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


CACHE_DIRS = _setup_cache_directories()


def _configure_numba():
    os.environ["NUMBA_CACHE_DIR"] = str(CACHE_DIRS["numba"])
    os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
    os.environ.setdefault("NUMBA_WARNINGS", "0")
    logger.debug(f"Numba cache: {CACHE_DIRS['numba']}")


# =============================================================================
# Stats Tracking (Lightweight, Thread-Safe)
# =============================================================================

class CacheStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._stats_file = CACHE_DIRS["base"] / "cache_stats.json"
        self._load()

    def _load(self):
        if self._stats_file.exists():
            try:
                with open(self._stats_file) as f:
                    self._stats.update(json.load(f))
            except Exception:
                pass

    def _save(self):
        try:
            with open(self._stats_file, "w") as f:
                json.dump(dict(self._stats), f)
        except Exception:
            pass

    def record_hit(self, func_name: str):
        with self._lock:
            self._stats[func_name]["hits"] += 1
            if self._stats[func_name]["hits"] % 25 == 0:
                self._save()

    def record_miss(self, func_name: str):
        with self._lock:
            self._stats[func_name]["misses"] += 1
            if self._stats[func_name]["misses"] % 25 == 0:
                self._save()

    def get_hit_rate(self, func_name: Optional[str] = None) -> float:
        with self._lock:
            if func_name:
                s = self._stats[func_name]
                total = s["hits"] + s["misses"]
                return s["hits"] / total if total > 0 else 0.0
            total_hits = sum(s["hits"] for s in self._stats.values())
            total_calls = sum(s["hits"] + s["misses"] for s in self._stats.values())
            return total_hits / total_calls if total_calls > 0 else 0.0

    def get_stats(self) -> Dict:
        with self._lock:
            return dict(self._stats)

    def clear(self):
        with self._lock:
            self._stats.clear()
            self._stats_file.unlink(missing_ok=True)


cache_stats = CacheStats()


# =============================================================================
# Unified Cache Key Generator (Single Source of Truth)
# =============================================================================

class UnifiedCacheKeyGenerator:
    @staticmethod
    def generate_key(
        func: Callable,
        args: tuple,
        kwargs: dict,
        time_aware: bool = False,
        auto_versioning: bool = True,
    ) -> str:
        key_parts = [func.__module__, func.__qualname__]

        if auto_versioning:
            func_hash = UnifiedCacheKeyGenerator._get_function_source_hash(func)
            if func_hash:
                key_parts.append(f"v_{func_hash}")
            else:
                mtime = UnifiedCacheKeyGenerator._get_function_file_mtime(func)
                if mtime:
                    key_parts.append(f"mtime_{int(mtime)}")

        sig = inspect.signature(func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            time_range = None
            if time_aware:
                time_range = UnifiedCacheKeyGenerator._extract_time_range(bound.arguments)

            for name, value in bound.arguments.items():
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(name, value))

            if time_range:
                start, end = time_range
                key_parts.append(f"time_{start}_{end}")
        except Exception as e:
            logger.debug(f"Signature binding failed for {func.__name__}: {e}")
            # Fallback
            for i, arg in enumerate(args):
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(f"arg_{i}", arg))
            for k, v in kwargs.items():
                key_parts.append(UnifiedCacheKeyGenerator._hash_parameter(k, v))

        return hashlib.md5("_".join(key_parts).encode()).hexdigest()

    @staticmethod
    def _get_function_source_hash(func: Callable) -> Optional[str]:
        original = func
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__
        try:
            source = inspect.getsource(original)
            h = hashlib.md5(source.encode()).hexdigest()[:12]
            if original.__closure__:
                closure_h = UnifiedCacheKeyGenerator._get_closure_hash(original)
                if closure_h:
                    return f"{h}_{closure_h}"
            return h
        except Exception:
            return None

    @staticmethod
    def _get_closure_hash(func: Callable) -> Optional[str]:
        if not func.__closure__:
            return None
        try:
            vals = []
            for cell in func.__closure__:
                try:
                    v = cell.cell_contents
                    if isinstance(v, (int, float, str, bool, type(None))):
                        vals.append(f"{type(v).__name__}:{v}")
                    else:
                        vals.append(f"{type(v).__name__}:{id(v)}")
                except Exception:
                    vals.append("empty")
            return hashlib.md5("_".join(vals).encode()).hexdigest()[:8]
        except Exception:
            return None

    @staticmethod
    def _get_function_file_mtime(func: Callable) -> Optional[float]:
        try:
            return Path(inspect.getfile(func)).stat().st_mtime
        except Exception:
            return None

    @staticmethod
    def _extract_time_range(arguments: dict) -> Optional[Tuple[str, str]]:
        # Heuristic for common time-aware params (t1, index, etc.)
        for key in ("t1", "data", "prices", "index"):
            if key in arguments:
                val = arguments[key]
                if isinstance(val, pd.DataFrame) and isinstance(val.index, pd.DatetimeIndex):
                    return str(val.index[0]), str(val.index[-1])
                if isinstance(val, pd.Series) and isinstance(val.index, pd.DatetimeIndex):
                    return str(val.index[0]), str(val.index[-1])
        return None

    @staticmethod
    def _hash_parameter(name: str, value: Any) -> str:
        try:
            if value is None:
                return f"{name}_None"
            if isinstance(value, BaseEstimator):
                return f"{name}_est_{UnifiedCacheKeyGenerator._hash_estimator(value)}"
            if isinstance(value, (rv_continuous_frozen, rv_discrete_frozen)):
                return f"{name}_dist_{UnifiedCacheKeyGenerator._hash_scipy_dist(value)}"
            if isinstance(value, pd.DataFrame):
                return f"{name}_df_{UnifiedCacheKeyGenerator._hash_dataframe_fast(value)}"
            if isinstance(value, pd.Series):
                return f"{name}_ser_{UnifiedCacheKeyGenerator._hash_series_fast(value)}"
            if isinstance(value, np.ndarray):
                h = hashlib.md5(value.tobytes()).hexdigest()[:8]
                return f"{name}_arr_{value.shape}_{h}"
            if isinstance(value, (list, tuple, dict)):
                return f"{name}_{hashlib.md5(str(sorted(str(value).encode())).encode()).hexdigest()[:12]}"
            if isinstance(value, (str, int, float, bool)):
                return f"{name}_{value}"
            # Fallback
            return f"{name}_{type(value).__name__}_{hash(str(value))}"
        except Exception as e:
            logger.debug(f"Hash failed for {name}: {e}")
            return f"{name}_unknown_{id(value)}"

    @staticmethod
    def _hash_estimator(est: BaseEstimator) -> str:
        try:
            params = est.get_params(deep=True)
            serializable = {k: v for k, v in params.items() if json_serializable(v)}
            s = json.dumps(serializable, sort_keys=True)
            return hashlib.md5(s.encode()).hexdigest()[:12]
        except Exception:
            return f"{type(est).__name__}_{id(est)}"

    @staticmethod
    def _hash_scipy_dist(dist) -> str:
        d = {"type": type(dist.dist).__name__, "args": getattr(dist, "args", ()), "kwds": getattr(dist, "kwds", {})}
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @staticmethod
    def _hash_dataframe_fast(df: pd.DataFrame) -> str:
        parts = [f"shape_{df.shape}", f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}"]
        if isinstance(df.index, pd.DatetimeIndex):
            parts.append(f"idx_{df.index[0]}_{df.index[-1]}_{len(df)}")
        sample = df.iloc[:: max(1, len(df) // 100)] if len(df) > 100 else df
        data_h = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        parts.append(f"data_{data_h}")
        return "_".join(parts)

    @staticmethod
    def _hash_series_fast(ser: pd.Series) -> str:
        parts = [f"len_{len(ser)}", f"dtype_{ser.dtype}"]
        if isinstance(ser.index, pd.DatetimeIndex):
            parts.append(f"idx_{ser.index[0]}_{ser.index[-1]}")
        sample = ser.iloc[:: max(1, len(ser) // 100)] if len(ser) > 100 else ser
        data_h = hashlib.md5(sample.values.tobytes()).hexdigest()[:8]
        parts.append(f"data_{data_h}")
        return "_".join(parts)


def json_serializable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


# =============================================================================
# Unified Decorator
# =============================================================================

memory = Memory(location=str(CACHE_DIRS["joblib"]), verbose=0)


def cacheable(
    _func: Optional[Callable] = None,
    time_aware: bool = False,
    auto_versioning: bool = True,
    cache_type: str = "joblib",  # "joblib" or "memory" (custom pickle fallback)
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = UnifiedCacheKeyGenerator.generate_key(
                func, args, kwargs,
                time_aware=time_aware,
                auto_versioning=auto_versioning,
            )
            func_name = func.__qualname__
        
            # --- Check manual pickle cache first ---
            cache_path = CACHE_DIRS["base"] / f"{key}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        result = pickle.load(f)
                    cache_stats.record_hit(func_name)
                    logger.debug(
                        f"CACHE HIT  | {func_name} | key={key[:12]}"
                    )
                    return result
                except Exception:
                    logger.debug(f"CACHE CORRUPT | {func_name} | key={key[:12]}")
        
            # --- Miss: compute and time it ---
            cache_stats.record_miss(func_name)
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.debug(
                f"CACHE MISS | {func_name} | key={key[:12]} | "
                f"computed in {elapsed:.2f}s"
            )
        
            # Store in CacheMonitor's computation_times if available
            # (imported at module level from cache_monitoring)
            try:
                from .cache_monitoring import _global_monitor
                if _global_monitor is not None:
                    _global_monitor.computation_times[func_name].append(elapsed)
            except Exception:
                pass
        
            # Persist
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"Failed to cache {func_name}: {e}")
        
            return result

        return wrapper
        
    return decorator(_func) if _func is not None else decorator


# Convenience exports
def get_cache_hit_rate(func_name: Optional[str] = None) -> float:
    return cache_stats.get_hit_rate(func_name)


def clear_afml_cache(warn: bool = True):
    if warn:
        logger.warning("Clearing AFML cache...")
    memory.clear(warn=warn)
    cache_stats.clear()


def initialize_cache_system():
    _configure_numba()
    logger.info(f"AFML Cache initialized | Joblib: {CACHE_DIRS['joblib']} | Backtest: {CACHE_DIRS['backtest']}")
    stats = cache_stats.get_stats()
    if stats:
        logger.info(f"Loaded stats: {len(stats)} functions, {cache_stats.get_hit_rate():.1%} hit rate")


# Context manager for analysis
class CacheAnalyzer:
    def __init__(self, name: str = "session"):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = cache_stats.get_stats().copy()
        return self

    def __exit__(self, *args):
        if not self.start:
            return
        end = cache_stats.get_stats()
        # Simple delta report (expand as needed)
        logger.info(f"CacheAnalyzer '{self.name}': session complete")



def create_cacheable_param_grid(param_distributions: Dict) -> Dict:
    """Convert scipy distributions to cacheable representation."""
    cacheable_params = {}

    for key, value in param_distributions.items():
        if isinstance(value, (rv_discrete_frozen, rv_continuous_frozen)):
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
    import scipy.stats as stats
    from scipy.stats import randint, uniform

    reconstructed = {}

    for key, value in cacheable_params.items():
        if isinstance(value, tuple) and len(value) == 3:
            dist_type, args, kwds = value

            if dist_type == "rv_discrete_frozen":
                reconstructed[key] = randint(*args, **kwds)
            elif dist_type == "rv_continuous_frozen":
                reconstructed[key] = uniform(*args, **kwds)
            else:
                try:
                    dist_class = getattr(stats, dist_type.replace("_frozen", ""))
                    reconstructed[key] = dist_class(*args, **kwds)
                except Exception:
                    logger.warning(f"Could not reconstruct: {dist_type}")
                    reconstructed[key] = value
        else:
            reconstructed[key] = value

    return reconstructed


# Export for other modules
__all__ = [
    "cacheable", "UnifiedCacheKeyGenerator", "cache_stats", "get_cache_hit_rate",
    "clear_afml_cache", "initialize_cache_system", "CacheAnalyzer", "CACHE_DIRS",
    "create_cacheable_param_grid", "reconstruct_param_grid",
]
