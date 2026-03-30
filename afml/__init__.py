"""
AFML - Advanced Financial Machine Learning
==========================================

Helps portfolio managers and traders leverage machine learning with
reproducible, interpretable, and easy-to-use tools.

This is the main package entry point with lazy loading for heavy modules
and a unified, production-grade cache system.
"""

import importlib
import sys
from types import ModuleType
from typing import Dict, List

from loguru import logger

# =============================================================================
# CORE CACHE SYSTEM IMPORTS (New Unified System)
# =============================================================================

from .cache import (
    # Core
    cacheable,
    clear_afml_cache,
    get_cache_hit_rate,
    initialize_cache_system,
    UnifiedCacheKeyGenerator,
    CACHE_DIRS,
    # Monitoring & Maintenance
    print_cache_health,
    get_cache_efficiency_report,
    analyze_cache_patterns,
    diagnose_cache_issues,
    get_cache_monitor,
    # Selective cleaning
    selective_cleaner,
    clean_stale_cache,
    # High-level APIs
    backtest_cache,
    BacktestCache,
    # Backwards compatibility shim
    cv_cacheable,
)

# Optional MLflow integration (if present)
try:
    from .cache import (
        setup_mlflow_cache,
        get_mlflow_cache,
        mlflow_cached,
        MLFLOW_INTEGRATION_AVAILABLE,
    )
except ImportError:
    MLFLOW_INTEGRATION_AVAILABLE = False
    logger.debug("MLflow integration not available in cache module")

# =============================================================================
# NUMBA WARMUP UTILITIES (if you still use them)
# =============================================================================

try:
    from .numba_warmup import lazy_warmup, prewarm_numba_in_package, register_numba_dummy
    NUMBA_UTILS_AVAILABLE = True
except ImportError:
    NUMBA_UTILS_AVAILABLE = False
    logger.debug("Numba warmup utilities not available")

# =============================================================================
# INITIALIZE CACHE SYSTEM EARLY
# =============================================================================

initialize_cache_system()

# =============================================================================
# LAZY LOADING SETUP FOR HEAVY MODULES
# =============================================================================

# Module mapping - only heavy/expensive modules
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

# Lightweight modules - import directly
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

# Cache for lazy-loaded heavy modules
_module_cache: Dict[str, ModuleType] = {}


def __getattr__(name: str) -> ModuleType:
    """Lazy load heavy modules on first access."""
    if name in HEAVY_MODULES:
        if name in _module_cache:
            return _module_cache[name]

        try:
            import_path = f"afml.{HEAVY_MODULES[name]}"
            logger.debug(f"Lazy loading heavy module: {name}")
            module = importlib.import_module(import_path)
            _module_cache[name] = module
            logger.info(f"Loaded heavy module: {name}")
            return module
        except ImportError as e:
            logger.error(f"Failed to import heavy module {name}: {e}")
            raise AttributeError(f"Module 'afml' has no attribute '{name}'") from e

    # Handle nested modules
    if name == "filters":
        from .filters import filters
        return filters
    if name == "strategies":
        from .strategies import trading_strategies
        return trading_strategies

    raise AttributeError(f"Module 'afml' has no attribute '{name}'")


def _get_module_size(module) -> str:
    """Rough estimate of module memory footprint (for logging)."""
    try:
        obj_count = len([obj for obj in vars(module).values() if not callable(obj)])
        return f"\~{obj_count // 10}0 objects"
    except Exception:
        return "unknown"


# =============================================================================
# PRELOAD CONVENIENCE FUNCTIONS
# =============================================================================

def preload_heavy_modules(*module_names: str) -> Dict[str, ModuleType]:
    """Preload specific heavy modules."""
    loaded = {}
    for name in module_names:
        if name in HEAVY_MODULES:
            try:
                module = getattr(sys.modules[__name__], name)  # triggers __getattr__
                loaded[name] = module
            except Exception as e:
                logger.warning(f"Failed to preload {name}: {e}")
        else:
            logger.warning(f"'{name}' is not a recognized heavy module")
    return loaded


def preload_ml_modules() -> Dict[str, ModuleType]:
    """Preload all ML-related heavy modules."""
    return preload_heavy_modules("ensemble", "clustering", "feature_importance", "cross_validation")


def preload_portfolio_modules() -> Dict[str, ModuleType]:
    """Preload portfolio-related heavy modules."""
    return preload_heavy_modules("portfolio_optimization", "online_portfolio_selection")


def get_loaded_heavy_modules() -> List[str]:
    """Return list of currently loaded heavy modules."""
    return list(_module_cache.keys())


# =============================================================================
# CACHE STATUS & MAINTENANCE
# =============================================================================

def cache_status() -> str:
    """Return a human-readable summary of cache + module state."""
    hit_rate = get_cache_hit_rate()
    loaded = get_loaded_heavy_modules()
    parts = [
        f"Cache hit rate: {hit_rate:.1%}",
        f"Tracked functions: {len(cache_stats.get_stats())}",
        f"Heavy modules loaded: {len(loaded)}",
    ]
    if loaded:
        parts.append(f"({', '.join(loaded)})")
    return " | ".join(parts)


def maintain_cache(auto_clear: bool = True, max_size_mb: int = 500, max_age_days: int = 30):
    """Perform intelligent cache maintenance using selective cleaner."""
    logger.info("Running AFML cache maintenance...")
    if auto_clear:
        selective_cleaner.clean_stale()
    selective_cleaner.clean_old_entries(days=max_age_days)
    selective_cleaner.clean_large_files(max_size_mb=max_size_mb)
    logger.info("Cache maintenance completed.")
    print_cache_health()


# =============================================================================
# METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "AFML Team"

__all__ = [
    # Core cache
    "cacheable",
    "cv_cacheable",
    "clear_afml_cache",
    "get_cache_hit_rate",
    "initialize_cache_system",
    "UnifiedCacheKeyGenerator",
    "CACHE_DIRS",
    # Monitoring
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
    "diagnose_cache_issues",
    "get_cache_monitor",
    # Maintenance
    "selective_cleaner",
    "clean_stale_cache",
    "maintain_cache",
    "cache_status",
    # Backtest
    "backtest_cache",
    "BacktestCache",
    # MLflow (optional)
    "setup_mlflow_cache",
    "get_mlflow_cache",
    "mlflow_cached",
    # Module management
    "preload_heavy_modules",
    "preload_ml_modules",
    "preload_portfolio_modules",
    "get_loaded_heavy_modules",
    # Numba (optional)
    "lazy_warmup",
    "prewarm_numba_in_package",
    "register_numba_dummy",
    # Lightweight modules
    "data_structures",
    "util",
    "datasets",
    "labeling",
    "features",
    "sample_weights",
    "sampling",
    "bet_sizing",
    "trading_strategies",
    "filters",
    "mt5",
    "production",
    # Heavy modules (lazy-loaded)
    "ensemble",
    "clustering",
    "feature_importance",
    "cross_validation",
    "portfolio_optimization",
    "online_portfolio_selection",
    "structural_breaks",
    "backtest_statistics",
    "microstructural_features",
    "multi_product",
]

# =============================================================================
# STARTUP LOGGING
# =============================================================================

logger.info(f"AFML v{__version__} initialized successfully")
logger.info(f"Cache status: {cache_status()}")

if MLFLOW_INTEGRATION_AVAILABLE:
    logger.debug("✓ MLflow cache integration available")
if NUMBA_UTILS_AVAILABLE:
    logger.debug("✓ Numba warmup utilities available")

logger.debug("Use maintain_cache() or print_cache_health() for cache insights")
