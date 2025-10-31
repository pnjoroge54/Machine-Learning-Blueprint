"""
AFML helps portfolio managers and traders leverage machine learning with
reproducible, interpretable, and easy to use tools.
"""

import importlib
import sys
from types import ModuleType
from typing import Dict, List

import numpy as np
from loguru import logger

from .cache import (
    AUTO_RELOAD_AVAILABLE,
    CACHE_DIRS,
    CacheAnalyzer,
    auto_cacheable,
    cache_maintenance,
    cache_stats,
    cacheable,
    clear_afml_cache,
    clear_cache_stats,
    clear_changed_features_functions,
    clear_changed_labeling_functions,
    clear_changed_ml_functions,
    get_cache_hit_rate,
    get_cache_stats,
    get_cache_summary,
    get_function_tracker,
    initialize_cache_system,
    jupyter_auto_setup,
    memory,
    selective_cache_clear,
    setup_auto_reloading,
    smart_cacheable,
)

# at top-level of afml/__init__.py add:
from .numba_warmup import lazy_warmup, prewarm_numba_in_package, register_numba_dummy

# Register realistic dummy signatures for critical functions
# Adjust these as needed after inspecting actual function signatures in your codebase.
# register_numba_dummy(
#     "_precompute_active_indices_nopython",
#     args=(np.array([np.int64(0)]), np.array([np.int64(0)]), np.array([np.int64(0)])),
# )

# Optional: export helpers for convenience
# __all__ = ["register_numba_dummy", "lazy_warmup", "prewarm_numba_in_package"]

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
        mt5,
        sample_weights,
        sampling,
        util,
    )

    # Handle nested modules
    from .filters import filters
    from .strategies import strategies

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
    if name == "strategies":
        try:
            from .strategies import strategies

            return strategies
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


def setup_jupyter(
    preload_ml: bool = False, preload_portfolio: bool = False, enable_auto_reload: bool = True
):
    """Safe Jupyter setup that doesn't fail on AFMLApplication issues."""
    logger.info("Setting up AFML for Jupyter notebook...")

    # Initialize cache system first
    initialize_cache_system()

    # Setup auto-reload
    if enable_auto_reload and AUTO_RELOAD_AVAILABLE:
        try:
            setup_auto_reloading(watch_paths=["afml/", "."])
            logger.info("Auto-reload enabled")
        except Exception as e:
            logger.warning("Auto-reload setup failed: {}", e)

    # Try to preload modules (optional)
    if preload_ml:
        try:
            loaded_ml = preload_ml_modules()
            logger.info("Preloaded ML modules: {}", list(loaded_ml.keys()))
        except Exception as e:
            logger.warning("ML module preload failed: {}", e)

    # Skip AFMLApplication for now - just return cache status
    summary = get_cache_summary()
    logger.info("Cache ready: {:.1%} hit rate", summary["hit_rate"])

    return summary  # Return something useful instead of the failing app


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


def smart_cache_clear(modules: str = None, dry_run: bool = False):
    """
    Intelligently clear cache for changed functions.

    Args:
        modules: Module name to check (e.g., 'labeling', 'features')
        dry_run: If True, only report what would be cleared

    Examples:
        smart_cache_clear('labeling')  # Clear changed labeling functions
        smart_cache_clear(dry_run=True)  # See what would be cleared
    """
    if modules:
        module_name = f"afml.{modules}" if not modules.startswith("afml.") else modules
        result = selective_cache_clear(modules=[module_name], dry_run=dry_run)
    else:
        result = selective_cache_clear(dry_run=dry_run)

    if not dry_run and result["cleared"]:
        logger.info("Smart cache clear completed - cleared {} functions", len(result["cleared"]))

    return result


def maintain_cache(auto_clear: bool = True, max_size_mb: int = 500, max_age_days: int = 30):
    """
    Perform intelligent cache maintenance.

    Args:
        auto_clear: Automatically clear changed functions
        max_size_mb: Maximum cache size in MB
        max_age_days: Remove cache files older than this
    """
    logger.info("Running cache maintenance...")
    report = cache_maintenance(
        auto_clear_changed=auto_clear, max_cache_size_mb=max_size_mb, max_age_days=max_age_days
    )
    return report


# =============================================================================
# 8) __all__ AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "AFML Team"

__all__ = [
    # Cache system
    "memory",
    "cacheable",
    "smart_cacheable",  # NEW
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
    # NEW: Smart cache management
    "smart_cache_clear",
    "maintain_cache",
    "selective_cache_clear",
    "cache_maintenance",
    "clear_changed_ml_functions",
    "clear_changed_labeling_functions",
    "clear_changed_features_functions",
    "get_function_tracker",  # Updated from function_tracker
    "auto_cacheable",
    "setup_auto_reloading",
    "jupyter_auto_setup",
    "AUTO_RELOAD_AVAILABLE",
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
    "mt5",
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
