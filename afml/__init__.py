"""
AFML - Advanced Financial Machine Learning
==========================================

Main package entry point with lazy loading and the new unified cache system.
"""

import importlib
import sys
from types import ModuleType
from typing import Dict, List

from loguru import logger

# =============================================================================
# UNIFIED CACHE IMPORTS
# =============================================================================

from .cache import (
    # Core
    cacheable,
    clear_afml_cache,
    get_cache_hit_rate,
    initialize_cache_system,
    UnifiedCacheKeyGenerator,
    CACHE_DIRS,
    # Monitoring
    print_cache_health,
    get_cache_efficiency_report,
    analyze_cache_patterns,
    diagnose_cache_issues,
    get_cache_monitor,
    # Maintenance
    selective_cleaner,
    clean_stale_cache,
    # High-level
    backtest_cache,
    BacktestCache,
)

# =============================================================================
# DATA TRACKING (convenience re-export at top level)
# =============================================================================

try:
    from .cache import (
        DataAccessTracker,
        log_data_access,
        print_contamination_report,
        clear_data_access_log,
    )
except ImportError:
    DataAccessTracker = None
    log_data_access = lambda *a, **k: None
    print_contamination_report = lambda: None
    clear_data_access_log = lambda: None

# Optional MLflow
try:
    from .cache import setup_mlflow_cache, get_mlflow_cache, mlflow_cached
    MLFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    MLFLOW_INTEGRATION_AVAILABLE = False
    logger.debug("MLflow integration not available")

# Optional Numba
try:
    from .numba_warmup import lazy_warmup, prewarm_numba_in_package, register_numba_dummy
    NUMBA_UTILS_AVAILABLE = True
except ImportError:
    NUMBA_UTILS_AVAILABLE = False
    logger.debug("Numba warmup utilities not available")

# =============================================================================
# EARLY CACHE INITIALIZATION
# =============================================================================

initialize_cache_system()

# =============================================================================
# LAZY LOADING
# =============================================================================

HEAVY_MODULES = {
    "ensemble": "ensemble",
    "clustering": "clustering",
    "feature_importance": "feature_importance",
    "cross_validation": "cross_validation",
    "portfolio_optimization": "portfolio_optimization",
    "online_portfolio_selection": "online_portfolio_selection",
    "structural_breaks": "structural_breaks",
    "backtest_statistics": "backtest_statistics",
    "microstructural_features": "microstructural_features",
    "multi_product": "multi_product",
}

# Lightweight modules
try:
    from . import (
        bet_sizing,
        data_structures,
        datasets,
        features,
        labeling,
        mt5,
        production,
        sample_weights,
        sampling,
        util,
    )
    from .filters import filters
    from .strategies import trading_strategies
    logger.debug("Lightweight modules imported successfully")
except ImportError as e:
    logger.warning(f"Some lightweight modules failed to import: {e}")

_module_cache: Dict[str, ModuleType] = {}


def __getattr__(name: str) -> ModuleType:
    """Lazy load heavy modules."""
    if name in HEAVY_MODULES:
        if name in _module_cache:
            return _module_cache[name]

        try:
            import_path = f"afml.{HEAVY_MODULES[name]}"
            logger.debug(f"Lazy loading heavy module: {name}")
            module = importlib.import_module(import_path)
            _module_cache[name] = module
            return module
        except ImportError as e:
            logger.error(f"Failed to import heavy module {name}: {e}")
            raise AttributeError(f"Module 'afml' has no attribute '{name}'") from e

    if name == "filters":
        from .filters import filters
        return filters
    if name == "strategies":
        from .strategies import trading_strategies
        return trading_strategies

    raise AttributeError(f"Module 'afml' has no attribute '{name}'")


# =============================================================================
# UTILITIES
# =============================================================================

def preload_heavy_modules(*module_names: str) -> Dict[str, ModuleType]:
    loaded = {}
    for name in module_names:
        if name in HEAVY_MODULES:
            try:
                module = getattr(sys.modules[__name__], name)
                loaded[name] = module
            except Exception as e:
                logger.warning(f"Failed to preload {name}: {e}")
    return loaded


def preload_ml_modules() -> Dict[str, ModuleType]:
    return preload_heavy_modules("ensemble", "clustering", "feature_importance", "cross_validation")


def preload_portfolio_modules() -> Dict[str, ModuleType]:
    return preload_heavy_modules("portfolio_optimization", "online_portfolio_selection")


def get_loaded_heavy_modules() -> List[str]:
    return list(_module_cache.keys())


def cache_status() -> str:
    """Return human-readable cache status."""
    hit_rate = get_cache_hit_rate()
    loaded = get_loaded_heavy_modules()

    # Fixed: properly import cache_stats
    from .cache.unified_cache import cache_stats
    tracked = len(cache_stats.get_stats())

    parts = [
        f"Cache hit rate: {hit_rate:.1%}",
        f"Tracked functions: {tracked}",
        f"Heavy modules loaded: {len(loaded)}",
    ]
    if loaded:
        parts.append(f"({', '.join(loaded)})")
    return " | ".join(parts)


def maintain_cache(auto_clear: bool = True, max_size_mb: int = 500, max_age_days: int = 30):
    """Run cache maintenance."""
    logger.info("Running AFML cache maintenance...")
    if auto_clear:
        clean_stale_cache()
    selective_cleaner.clean_old_entries(days=max_age_days)
    selective_cleaner.clean_large_files(max_size_mb=max_size_mb)
    print_cache_health()


# =============================================================================
# METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "Patrick M. Njoroge"

__all__ = [
    "cacheable", "clear_afml_cache", "get_cache_hit_rate",
    "initialize_cache_system", "UnifiedCacheKeyGenerator", "CACHE_DIRS",
    "print_cache_health", "get_cache_efficiency_report", "analyze_cache_patterns",
    "diagnose_cache_issues", "get_cache_monitor",
    "selective_cleaner", "clean_stale_cache", "maintain_cache", "cache_status",
    "backtest_cache", "BacktestCache", "DataAccessTracker",
    "log_data_access", "print_contamination_report", "clear_data_access_log",
    "preload_heavy_modules", "preload_ml_modules", "preload_portfolio_modules",
    "get_loaded_heavy_modules",
    # Lightweight
    "data_structures", "util", "datasets", "labeling", "features",
    "sample_weights", "sampling", "bet_sizing", "trading_strategies",
    "filters", "mt5", "production",
    # Heavy (lazy)
    "ensemble", "clustering", "feature_importance", "cross_validation",
    "portfolio_optimization", "online_portfolio_selection", "structural_breaks",
    "backtest_statistics", "microstructural_features", "multi_product",
]

# =============================================================================
# STARTUP
# =============================================================================

logger.info(f"AFML v{__version__} initialized successfully")
logger.info(f"Cache status: {cache_status()}")

if MLFLOW_INTEGRATION_AVAILABLE:
    logger.debug("✓ MLflow integration available")
if NUMBA_UTILS_AVAILABLE:
    logger.debug("✓ Numba utilities available")
