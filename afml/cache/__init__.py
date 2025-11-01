"""
Centralized caching system for AFML package.
Now with robust cache keys, MLflow integration, backtest caching, and monitoring.
"""

import json
import os
import pickle
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Union

from appdirs import user_cache_dir
from joblib import Memory
from loguru import logger

# =============================================================================
# 1) CACHE DIRECTORY SETUP - MUST BE FIRST
# =============================================================================


def _setup_cache_directories() -> Dict[str, Path]:
    """Setup centralized cache directories."""
    # Base cache directory from environment or default
    cache_env = os.getenv("AFML_CACHE")
    base_dir = Path(cache_env) if cache_env else Path(user_cache_dir("afml"))

    dirs = {
        "base": base_dir,
        "joblib": base_dir / "joblib_cache",
        "numba": base_dir / "numba_cache",
    }

    # Create directories
    for cache_dir in dirs.values():
        cache_dir.mkdir(parents=True, exist_ok=True)

    return dirs


CACHE_DIRS = _setup_cache_directories()

# =============================================================================
# 2) NUMBA CONFIGURATION
# =============================================================================


def _configure_numba():
    """Configure Numba to use centralized cache."""
    numba_dir = str(CACHE_DIRS["numba"])
    os.environ["NUMBA_CACHE_DIR"] = numba_dir

    # Performance optimizations
    os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
    os.environ.setdefault("NUMBA_WARNINGS", "0")

    logger.debug("Numba cache configured: {}", numba_dir)


# =============================================================================
# 3) SIMPLE CACHE STATISTICS
# =============================================================================


class CacheStats:
    """Lightweight cache statistics tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._stats_file = CACHE_DIRS["base"] / "cache_stats.json"
        self._load_stats()

    def _load_stats(self):
        """Load stats from disk."""
        if self._stats_file.exists():
            try:
                with open(self._stats_file, "r") as f:
                    data = json.load(f)
                    self._stats.update(data)
            except Exception:
                pass  # Start fresh if corrupted

    def _save_stats(self):
        """Save stats to disk."""
        try:
            with open(self._stats_file, "w") as f:
                json.dump(dict(self._stats), f)
        except Exception:
            pass  # Fail silently

    def record_hit(self, func_name: str):
        """Record cache hit."""
        with self._lock:
            self._stats[func_name]["hits"] += 1
            # Save every 25 hits to reduce I/O
            if self._stats[func_name]["hits"] % 25 == 0:
                self._save_stats()

    def record_miss(self, func_name: str):
        """Record cache miss."""
        with self._lock:
            self._stats[func_name]["misses"] += 1
            # Save every 25 misses
            if self._stats[func_name]["misses"] % 25 == 0:
                self._save_stats()

    def get_hit_rate(self, func_name: str = None) -> float:
        """Get hit rate for function or overall."""
        with self._lock:
            if func_name:
                stats = self._stats[func_name]
                total = stats["hits"] + stats["misses"]
                return stats["hits"] / total if total > 0 else 0.0
            else:
                total_hits = sum(s["hits"] for s in self._stats.values())
                total_calls = sum(s["hits"] + s["misses"] for s in self._stats.values())
                return total_hits / total_calls if total_calls > 0 else 0.0

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get all statistics."""
        with self._lock:
            return dict(self._stats)

    def clear(self):
        """Clear all statistics."""
        with self._lock:
            self._stats.clear()
            if self._stats_file.exists():
                self._stats_file.unlink()


# Global stats instance
cache_stats = CacheStats()

# =============================================================================
# 4) JOBLIB MEMORY INSTANCE
# =============================================================================

memory = Memory(location=str(CACHE_DIRS["joblib"]), verbose=0)

# =============================================================================
# 5) SIMPLE @cacheable DECORATOR
# =============================================================================


def cacheable(func):
    """
    Enhanced cacheable decorator with corruption handling.
    Use this for expensive computational functions.
    """
    func_name = f"{func.__module__}.{func.__qualname__}"
    cached_func = memory.cache(func)

    # Track argument signatures we've seen (approximate hit detection)
    seen_signatures = set()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create simple signature for hit detection
        try:
            sig = str(hash((str(args), str(sorted(kwargs.items())))))
        except TypeError:
            # If args aren't hashable, treat as miss
            sig = None

        if sig and sig in seen_signatures:
            cache_stats.record_hit(func_name)
        else:
            cache_stats.record_miss(func_name)
            if sig:
                seen_signatures.add(sig)

        # Try cached function with error handling
        try:
            return cached_func(*args, **kwargs)
        except (EOFError, pickle.PickleError, OSError) as e:
            # Cache corruption detected
            logger.warning("Cache corruption for {}: {} - recomputing", func_name, type(e).__name__)

            # Clear the corrupted cache entry
            try:
                cache_key = cached_func._get_cache_id(*args, **kwargs)
                cache_dir = Path(cached_func.store_backend.location)

                # Remove files matching this cache key
                for cache_file in cache_dir.rglob("*"):
                    if cache_file.is_file() and str(cache_key) in str(cache_file):
                        cache_file.unlink()
                        logger.debug("Removed corrupted file: {}", cache_file.name)

            except Exception:
                pass  # If clearing fails, just continue

            # Execute function directly
            return func(*args, **kwargs)
        except Exception as e:
            # Other unexpected errors
            logger.error("Unexpected cache error for {}: {}", func_name, e)
            return func(*args, **kwargs)

    wrapper._afml_cacheable = True
    return wrapper


# =============================================================================
# 6) UTILITY FUNCTIONS
# =============================================================================


def get_cache_hit_rate(func_name: str = None) -> float:
    """Get cache hit rate."""
    return cache_stats.get_hit_rate(func_name)


def get_cache_stats() -> Dict[str, Dict[str, int]]:
    """Get cache statistics."""
    return cache_stats.get_stats()


def clear_cache_stats():
    """Clear cache statistics."""
    cache_stats.clear()


def clear_afml_cache(warn: bool = True):
    """Clear all AFML caches."""
    if warn:
        logger.warning("Clearing AFML cache...")

    memory.clear(warn=warn)
    clear_cache_stats()


def get_cache_summary() -> Dict[str, Union[float, int]]:
    """Get simple cache performance summary."""
    stats = cache_stats.get_stats()
    total_hits = sum(s["hits"] for s in stats.values())
    total_calls = sum(s["hits"] + s["misses"] for s in stats.values())

    return {
        "hit_rate": total_hits / total_calls if total_calls > 0 else 0.0,
        "total_calls": total_calls,
        "functions_tracked": len(stats),
    }


# =============================================================================
# 7) CACHE ANALYSIS CONTEXT MANAGER
# =============================================================================


class CacheAnalyzer:
    """Simple context manager for analyzing cache performance."""

    def __init__(self, name: str = "analysis"):
        self.name = name
        self.start_stats = None

    def __enter__(self):
        self.start_stats = cache_stats.get_stats().copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            end_stats = cache_stats.get_stats()
            report = self._generate_report(end_stats)
            if report:
                logger.info("Cache analysis '{}': {}", self.name, report)

    def _generate_report(self, end_stats) -> Optional[str]:
        """Generate simple performance report."""
        if not self.start_stats:
            return None

        total_new_hits = 0
        total_new_calls = 0

        for func_name, end_data in end_stats.items():
            start_data = self.start_stats.get(func_name, {"hits": 0, "misses": 0})
            new_hits = end_data["hits"] - start_data["hits"]
            new_misses = end_data["misses"] - start_data["misses"]
            new_calls = new_hits + new_misses

            total_new_hits += new_hits
            total_new_calls += new_calls

        if total_new_calls > 0:
            hit_rate = total_new_hits / total_new_calls
            return f"{total_new_calls} calls, {hit_rate:.1%} hit rate"

        return "no cache activity"


# =============================================================================
# 8) INITIALIZATION FUNCTION
# =============================================================================


def initialize_cache_system():
    """Initialize the AFML cache system."""
    # Configure Numba first (before any @njit functions are defined)
    _configure_numba()

    # Log cache setup
    logger.info("AFML cache system initialized:")
    logger.info("  Joblib cache: {}", CACHE_DIRS["joblib"])
    logger.info("  Numba cache: {}", CACHE_DIRS["numba"])

    # Load existing stats
    stats = cache_stats.get_stats()
    if stats:
        hit_rate = cache_stats.get_hit_rate()
        logger.info("  Loaded stats: {} functions, {:.1%} hit rate", len(stats), hit_rate)


# =============================================================================
# 9) NOW SAFE TO IMPORT OTHER MODULES
# =============================================================================

# Import robust cache key generation - NOW SAFE (memory and cache_stats exist)
from .robust_cache_keys import (
    CacheKeyGenerator,
    TimeSeriesCacheKey,
    create_robust_cacheable,
    robust_cacheable,
    time_aware_cacheable,
)

# Import selective cleaner functions after base components are defined
from .selective_cleaner import (
    cache_maintenance,
    clear_changed_features_functions,
    clear_changed_labeling_functions,
    clear_changed_ml_functions,
    get_function_tracker,
    selective_cache_clear,
    smart_cacheable,
)

# MLflow integration (optional)
try:
    from .mlflow_integration import (
        MLFLOW_AVAILABLE,
        MLflowCacheIntegration,
        get_mlflow_cache,
        mlflow_cached,
        setup_mlflow_cache,
    )

    MLFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    MLFLOW_INTEGRATION_AVAILABLE = False
    logger.debug("MLflow integration not available (install mlflow)")

# Backtest caching
from .backtest_cache import (
    BacktestCache,
    BacktestMetadata,
    BacktestResult,
    cached_backtest,
    get_backtest_cache,
)

# Cache monitoring
from .cache_monitoring import (
    CacheHealthReport,
    CacheMonitor,
    FunctionCacheStats,
    analyze_cache_patterns,
    get_cache_efficiency_report,
    get_cache_monitor,
    print_cache_health,
)

# =============================================================================
# 10) AUTO-RELOAD FUNCTIONALITY (Optional)
# =============================================================================

try:
    from .auto_reload import auto_cacheable, jupyter_auto_setup, setup_auto_reloading

    AUTO_RELOAD_AVAILABLE = True
    logger.debug("Auto-reload functionality available")
except ImportError:
    AUTO_RELOAD_AVAILABLE = False
    logger.debug("Auto-reload not available (install watchdog for file watching)")

    # Provide fallbacks that work with your existing system
    def auto_cacheable(func):
        """Fallback: use smart_cacheable instead."""
        return smart_cacheable(func)

    def setup_auto_reloading(*args, **kwargs):
        logger.info("Auto-reload not available - using standard selective cache clearing")
        return None

    def jupyter_auto_setup():
        logger.info("Auto-reload not available - using standard caching")
        return None


# =============================================================================
# 11) ENHANCED CONVENIENCE FUNCTIONS
# =============================================================================


def get_comprehensive_cache_status() -> dict:
    """
    Get comprehensive cache status including all subsystems.

    Returns:
        Dict with status of all cache components
    """
    status = {
        "core": get_cache_summary(),
        "health": None,
        "backtest": None,
        "mlflow": {"available": MLFLOW_INTEGRATION_AVAILABLE},
    }

    # Get health report
    try:
        monitor = get_cache_monitor()
        report = monitor.generate_health_report()
        status["health"] = {
            "total_functions": report.total_functions,
            "hit_rate": report.overall_hit_rate,
            "total_calls": report.total_calls,
            "cache_size_mb": report.total_cache_size_mb,
        }
    except Exception as e:
        logger.debug(f"Health report failed: {e}")

    # Get backtest cache stats
    try:
        backtest_cache = get_backtest_cache()
        status["backtest"] = backtest_cache.get_cache_stats()
    except Exception as e:
        logger.debug(f"Backtest cache stats failed: {e}")

    return status


def optimize_cache_system(
    clear_changed: bool = True,
    max_size_mb: int = 1000,
    max_age_days: int = 30,
    print_report: bool = True,
) -> dict:
    """
    Comprehensive cache optimization and maintenance.

    Args:
        clear_changed: Clear caches for changed functions
        max_size_mb: Maximum total cache size in MB
        max_age_days: Remove caches older than this
        print_report: Print detailed report

    Returns:
        Dict with optimization results
    """
    logger.info("Running comprehensive cache optimization...")

    results = {
        "maintenance": None,
        "health_report": None,
        "backtest_cleanup": None,
    }

    # Run core cache maintenance
    try:
        results["maintenance"] = cache_maintenance(
            auto_clear_changed=clear_changed,
            max_cache_size_mb=max_size_mb,
            max_age_days=max_age_days,
        )
    except Exception as e:
        logger.warning(f"Cache maintenance failed: {e}")

    # Get health report
    try:
        monitor = get_cache_monitor()
        results["health_report"] = monitor.generate_health_report()

        if print_report:
            monitor.print_health_report(detailed=False)
    except Exception as e:
        logger.warning(f"Health report failed: {e}")

    # Clean old backtest caches
    try:
        backtest_cache = get_backtest_cache()
        cleared = backtest_cache.clear_old_runs(days=max_age_days)
        results["backtest_cleanup"] = {"runs_cleared": cleared}
        logger.info(f"Cleared {cleared} old backtest runs")
    except Exception as e:
        logger.warning(f"Backtest cleanup failed: {e}")

    return results


def setup_production_cache(
    enable_mlflow: bool = True,
    mlflow_experiment: str = "production",
    mlflow_uri: str = None,
    max_cache_size_mb: int = 2000,
) -> dict:
    """
    Setup cache system for production use.

    Args:
        enable_mlflow: Enable MLflow integration
        mlflow_experiment: MLflow experiment name
        mlflow_uri: MLflow tracking URI
        max_cache_size_mb: Maximum cache size

    Returns:
        Dict with initialized components
    """
    logger.info("Initializing production cache system...")

    components = {
        "core_cache": None,
        "mlflow_cache": None,
        "backtest_cache": None,
        "monitor": None,
    }

    # Initialize core cache
    initialize_cache_system()
    components["core_cache"] = True

    # Setup MLflow if available and requested
    if enable_mlflow and MLFLOW_INTEGRATION_AVAILABLE:
        try:
            components["mlflow_cache"] = setup_mlflow_cache(
                experiment_name=mlflow_experiment,
                tracking_uri=mlflow_uri,
            )
            logger.info(f"MLflow tracking enabled: {mlflow_experiment}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    # Initialize backtest cache
    try:
        components["backtest_cache"] = get_backtest_cache()
    except Exception as e:
        logger.warning(f"Backtest cache setup failed: {e}")

    # Initialize monitor
    try:
        components["monitor"] = get_cache_monitor()
    except Exception as e:
        logger.warning(f"Cache monitor setup failed: {e}")

    # Run initial maintenance
    try:
        optimize_cache_system(max_size_mb=max_cache_size_mb, print_report=False)
    except Exception as e:
        logger.warning(f"Initial optimization failed: {e}")

    logger.info("✅ Production cache system ready")
    return components


def setup_jupyter_cache(
    enable_mlflow: bool = False,
    enable_monitoring: bool = True,
) -> dict:
    """
    Setup cache system optimized for Jupyter notebooks.

    Args:
        enable_mlflow: Enable MLflow tracking
        enable_monitoring: Enable cache monitoring

    Returns:
        Dict with initialized components and helper functions
    """
    logger.info("Setting up cache for Jupyter notebook...")

    # Initialize core
    initialize_cache_system()

    components = {
        "core": True,
        "mlflow": None,
        "backtest": get_backtest_cache(),
        "monitor": get_cache_monitor() if enable_monitoring else None,
    }

    # Setup MLflow if requested
    if enable_mlflow and MLFLOW_INTEGRATION_AVAILABLE:
        try:
            components["mlflow"] = setup_mlflow_cache(experiment_name="jupyter_experiments")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    # Create helper namespace for notebook
    helpers = {
        "cache_status": lambda: get_comprehensive_cache_status(),
        "print_health": print_cache_health,
        "optimize": lambda: optimize_cache_system(print_report=True),
        "clear_all": lambda: clear_afml_cache(warn=True),
        "smart_clear": lambda module=None: selective_cache_clear(
            modules=[module] if module else None
        ),
    }

    logger.info("✅ Jupyter cache ready!")
    logger.info("   Use helpers: cache_status(), print_health(), optimize()")

    return {**components, "helpers": helpers}


# =============================================================================
# 12) EXPORTS
# =============================================================================

__all__ = [
    # Core caching (existing)
    "memory",
    "cacheable",
    "initialize_cache_system",
    "cache_stats",
    "get_cache_hit_rate",
    "get_cache_stats",
    "clear_cache_stats",
    "get_cache_summary",
    "CacheAnalyzer",
    "clear_afml_cache",
    "CACHE_DIRS",
    # Selective cache management (existing)
    "selective_cache_clear",
    "smart_cacheable",
    "cache_maintenance",
    "get_function_tracker",
    "clear_changed_ml_functions",
    "clear_changed_labeling_functions",
    "clear_changed_features_functions",
    # Auto-reload (existing)
    "auto_cacheable",
    "setup_auto_reloading",
    "jupyter_auto_setup",
    "AUTO_RELOAD_AVAILABLE",
    # NEW: Robust cache keys
    "CacheKeyGenerator",
    "TimeSeriesCacheKey",
    "robust_cacheable",
    "time_aware_cacheable",
    "create_robust_cacheable",
    # NEW: MLflow integration
    "MLflowCacheIntegration",
    "setup_mlflow_cache",
    "get_mlflow_cache",
    "mlflow_cached",
    "MLFLOW_AVAILABLE",
    "MLFLOW_INTEGRATION_AVAILABLE",
    # NEW: Backtest caching
    "BacktestCache",
    "BacktestMetadata",
    "BacktestResult",
    "get_backtest_cache",
    "cached_backtest",
    # NEW: Cache monitoring
    "CacheMonitor",
    "FunctionCacheStats",
    "CacheHealthReport",
    "get_cache_monitor",
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
    # NEW: Enhanced convenience functions
    "get_comprehensive_cache_status",
    "optimize_cache_system",
    "setup_production_cache",
    "setup_jupyter_cache",
]

# =============================================================================
# STARTUP MESSAGE UPDATE
# =============================================================================

# Add to end of file to show new features are available
logger.debug("Enhanced cache features available:")
logger.debug("  - Robust cache keys for NumPy/Pandas")
logger.debug("  - MLflow integration: {}", "✓" if MLFLOW_INTEGRATION_AVAILABLE else "✗")
logger.debug("  - Backtest caching: ✓")
logger.debug("  - Cache monitoring: ✓")
