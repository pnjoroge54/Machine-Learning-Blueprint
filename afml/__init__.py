# afml/__init__.py - Simplified package initialization focused on performance

"""
AFML helps portfolio managers and traders leverage machine learning with
reproducible, interpretable, and easy to use tools.
"""

import importlib
import sys
from types import ModuleType
from typing import Dict, List

from loguru import logger

from .cache import (
    CACHE_DIRS,
    CacheAnalyzer,
    cache_stats,
    cacheable,
    clear_afml_cache,
    clear_cache_stats,
    get_cache_hit_rate,
    get_cache_stats,
    get_cache_summary,
    initialize_cache_system,
    memory,
)

# =============================================================================
# 1) INITIALIZE CACHE SYSTEM FIRST (before any heavy imports)
# =============================================================================


# Initialize cache system immediately
initialize_cache_system()

# =============================================================================
# 2) LIGHTWEIGHT CORE IMPORTS (always available)
# =============================================================================

from .core import AFMLApplication, pattern_jupyter_notebook

# =============================================================================
# 3) LAZY LOADING SETUP
# =============================================================================

# Module mapping - only add modules that are actually heavy to import
HEAVY_MODULES = {
    # ML modules (typically have sklearn, xgboost, etc.)
    "ensemble": "ensemble",
    "clustering": "clustering",
    "feature_importance": "feature_importance",
    "cross_validation": "cross_validation",
    # Portfolio modules (may have cvxpy, scipy.optimize)
    "portfolio_optimization": "portfolio_optimization",
    "online_portfolio_selection": "online_portfolio_selection",
    # Analysis modules (pandas, numpy heavy operations)
    "structural_breaks": "structural_breaks",
    "backtest_statistics": "backtest_statistics",
    # Data processing (may have large dependencies)
    "microstructural_features": "microstructural_features",
    "multi_product": "multi_product",
}

# Lightweight modules - import directly for better IDE support
try:
    from . import (
        bet_sizing,
        data_structures,
        datasets,
        features,
        labeling,
        sample_weights,
        sampling,
        strategies,
        util,
    )
    from .filters import filters  # Handle nested module

    logger.debug("Imported lightweight modules directly")
except ImportError as e:
    logger.warning("Some lightweight modules failed to import: {}", e)

# Cache for lazy-loaded heavy modules
_module_cache: Dict[str, ModuleType] = {}

# =============================================================================
# 4) LAZY LOADING FOR HEAVY MODULES ONLY
# =============================================================================


def __getattr__(name: str) -> ModuleType:
    """Lazy load heavy modules only."""
    if name in HEAVY_MODULES:
        # Check cache first
        if name in _module_cache:
            return _module_cache[name]

        # Import and cache
        try:
            import_path = f"afml.{HEAVY_MODULES[name]}"
            logger.debug("Lazy loading heavy module: {}", name)
            module = importlib.import_module(import_path)
            _module_cache[name] = module
            logger.info("Loaded heavy module: {} ({} MB)", name, _get_module_size(module))
            return module
        except ImportError as e:
            logger.error("Failed to import heavy module {}: {}", name, e)
            raise AttributeError(f"Module 'afml' has no attribute '{name}'") from e

    # Handle filters special case (nested module)
    if name == "filters":
        try:
            from .filters import filters

            return filters
        except ImportError as e:
            raise AttributeError(f"Module 'afml' has no attribute '{name}'") from e

    raise AttributeError(f"Module 'afml' has no attribute '{name}'")


def _get_module_size(module) -> str:
    """Rough estimate of module memory footprint."""
    try:
        import sys

        # Count objects in module namespace
        obj_count = len(
            [
                obj
                for obj in vars(module).values()
                if not callable(obj) or hasattr(obj, "__module__")
            ]
        )
        return f"~{obj_count//10}0"  # Very rough estimate
    except Exception:
        return "unknown"


# =============================================================================
# 5) SIMPLE MODULE MANAGEMENT (only what's actually useful)
# =============================================================================


def preload_heavy_modules(*module_names: str) -> Dict[str, ModuleType]:
    """
    Preload specific heavy modules. Only use this if you know you'll need them.

    Args:
        *module_names: Names of heavy modules to preload

    Returns:
        Dict of successfully loaded modules
    """
    loaded = {}
    for name in module_names:
        if name in HEAVY_MODULES:
            try:
                module = getattr(sys.modules[__name__], name)  # Triggers __getattr__
                loaded[name] = module
            except Exception as e:
                logger.warning("Failed to preload {}: {}", name, e)
        else:
            logger.warning("'{}' is not a heavy module (already imported or doesn't exist)", name)

    return loaded


def get_loaded_heavy_modules() -> List[str]:
    """Get list of currently loaded heavy modules."""
    return list(_module_cache.keys())


def preload_ml_modules() -> Dict[str, ModuleType]:
    """Convenience function to preload all ML-related modules."""
    ml_modules = ["ensemble", "clustering", "feature_importance", "cross_validation"]
    return preload_heavy_modules(*ml_modules)


def preload_portfolio_modules() -> Dict[str, ModuleType]:
    """Convenience function to preload portfolio-related modules."""
    portfolio_modules = ["portfolio_optimization", "online_portfolio_selection"]
    return preload_heavy_modules(*portfolio_modules)


# =============================================================================
# 6) JUPYTER INTEGRATION
# =============================================================================


def setup_jupyter(preload_ml: bool = False, preload_portfolio: bool = False) -> AFMLApplication:
    """
    Setup AFML for Jupyter notebook use.

    Args:
        preload_ml: Preload ML modules (ensemble, clustering, etc.)
        preload_portfolio: Preload portfolio modules

    Returns:
        Configured AFMLApplication
    """
    logger.info("Setting up AFML for Jupyter notebook...")

    # Preload heavy modules if requested
    if preload_ml:
        loaded_ml = preload_ml_modules()
        logger.info("Preloaded ML modules: {}", list(loaded_ml.keys()))

    if preload_portfolio:
        loaded_portfolio = preload_portfolio_modules()
        logger.info("Preloaded portfolio modules: {}", list(loaded_portfolio.keys()))

    # Setup notebook environment
    setup_nb = pattern_jupyter_notebook()
    app = setup_nb()

    # Show cache status
    summary = get_cache_summary()
    logger.info(
        "Cache ready: {:.1%} hit rate, {} functions tracked",
        summary["hit_rate"],
        summary["functions_tracked"],
    )

    return app


# =============================================================================
# 7) CACHE MONITORING UTILITIES
# =============================================================================


def cache_status() -> str:
    """Get human-readable cache status string."""
    summary = get_cache_summary()
    loaded = get_loaded_heavy_modules()

    status_parts = [
        f"Hit rate: {summary['hit_rate']:.1%}",
        f"Tracked functions: {summary['functions_tracked']}",
        f"Heavy modules loaded: {len(loaded)}",
    ]

    if loaded:
        status_parts.append(f"({', '.join(loaded)})")

    return " | ".join(status_parts)


def cache_info():
    """Print detailed cache information."""
    print("AFML Cache System Status")
    print("=" * 50)

    # Cache directories
    print("\nCache Directories:")
    for name, path in CACHE_DIRS.items():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
        print(f"  {name:>10}: {path} ({size/1024/1024:.1f} MB)")

    # Statistics
    summary = get_cache_summary()
    print(f"\nCache Performance:")
    print(f"  Overall hit rate: {summary['hit_rate']:.1%}")
    print(f"  Total calls: {summary['total_calls']:,}")
    print(f"  Functions tracked: {summary['functions_tracked']}")

    # Module status
    loaded = get_loaded_heavy_modules()
    print(f"\nModule Status:")
    print(f"  Heavy modules loaded: {len(loaded)}/{len(HEAVY_MODULES)}")
    if loaded:
        print(f"  Loaded: {', '.join(loaded)}")


# =============================================================================
# 8) __all__ AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "AFML Team"

# Clean __all__ - only essential items
__all__ = [
    # Cache system
    "memory",
    "cacheable",
    "get_cache_hit_rate",
    "get_cache_stats",
    "clear_cache_stats",
    "clear_afml_cache",
    "get_cache_summary",
    "CacheAnalyzer",
    # Core
    "AFMLApplication",
    "pattern_jupyter_notebook",
    # Jupyter setup
    "setup_jupyter",
    # Module management (simplified)
    "preload_heavy_modules",
    "preload_ml_modules",
    "preload_portfolio_modules",
    "get_loaded_heavy_modules",
    # Utilities
    "cache_status",
    "cache_info",
    # Lightweight modules (directly imported)
    "data_structures",
    "util",
    "datasets",
    "labeling",
    "features",
    "sample_weights",
    "sampling",
    "bet_sizing",
    "strategies",
    "filters",
    # Heavy modules (lazy loaded)
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
# 9) STARTUP
# =============================================================================

logger.info(
    "AFML v{} ready - {} heavy modules available for lazy loading", __version__, len(HEAVY_MODULES)
)
logger.debug("Cache status: {}", cache_status())
logger.info(
    "AFML v{} ready - {} heavy modules available for lazy loading", __version__, len(HEAVY_MODULES)
)
logger.debug("Cache status: {}", cache_status())
