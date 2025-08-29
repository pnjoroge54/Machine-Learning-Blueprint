# __init__.py - Improved dynamic module access patterns

"""
AFML helps portfolio managers and traders who want to leverage the power of machine learning by providing
reproducible, interpretable, and easy to use tools.
"""

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import Dict, Optional

from loguru import logger

from .cache import (
    CacheAnalyzer,
    clear_afml_cache,
    clear_cache_stats,
    get_cache_hit_rate,
    get_cache_stats,
    memory,
)
from .core import AFMLApplication, pattern_jupyter_notebook

PACKAGE_NAME = "afml"

# =============================================================================
# Lazy Loading with __getattr__ (Python 3.7+) - RECOMMENDED
# =============================================================================

# Define submodules with their import paths
SUBMODULES = {
    "cross_validation": "cross_validation",
    "data_structures": "data_structures",
    "datasets": "datasets",
    "multi_product": "multi_product",
    "filters": "filters.filters",  # Note the nested module
    "labeling": "labeling",
    "features": "features",
    "sample_weights": "sample_weights",
    "sampling": "sampling",
    "bet_sizing": "bet_sizing",
    "util": "util",
    "structural_breaks": "structural_breaks",
    "feature_importance": "feature_importance",
    "ensemble": "ensemble",
    "portfolio_optimization": "portfolio_optimization",
    "clustering": "clustering",
    "microstructural_features": "microstructural_features",
    "backtest_statistics": "backtest_statistics",
    "online_portfolio_selection": "online_portfolio_selection",
    "strategies": "strategies",
}

# Cache for loaded modules
_module_cache: Dict[str, ModuleType] = {}


def __getattr__(name: str) -> ModuleType:
    """
    Lazy loading of submodules using __getattr__.
    This is called when an attribute is not found in the module.
    """
    if name in SUBMODULES:
        # Check cache first
        if name in _module_cache:
            return _module_cache[name]

        # Import the module
        try:
            import_path = f"{PACKAGE_NAME}.{SUBMODULES[name]}"
            module = importlib.import_module(import_path)
            _module_cache[name] = module
            logger.debug("Lazy loaded module: {}", name)
            return module
        except ImportError as e:
            logger.error("Failed to import {}: {}", name, e)
            raise AttributeError(f"Module '{PACKAGE_NAME}' has no attribute '{name}'") from e

    raise AttributeError(f"Module '{PACKAGE_NAME}' has no attribute '{name}'")


def __dir__():
    """
    Customize dir() output to include dynamically available modules.
    """
    # Standard attributes
    standard_attrs = [
        "memory",
        "init_cache_plugins",
        "get_cache_hit_rate",
        "get_cache_stats",
        "clear_cache_stats",
        "CacheAnalyzer",
        "get_cache_performance_summary",
        "preload_modules",
        "get_loaded_modules",
        "reload_module",
    ]

    # Add submodules
    return standard_attrs + list(SUBMODULES.keys())


# =============================================================================
# Utility Functions
# =============================================================================


# Could be loaded from a config file or environment
MODULE_CONFIG = {
    "core_modules": [
        "data_structures",
        "util",
    ],
    "analysis_modules": ["structural_breaks"],
    "ml_modules": ["cross_validation", "ensemble", "feature_importance", "clustering"],
    "portfolio_modules": ["portfolio_optimization", "bet_sizing", "online_portfolio_selection"],
    "labeling_modules": [
        "labeling",
        "filters",
        "features",
        "microstructural_features",
        "backtest_statistics",
        "strategies",
    ],
}


def load_module_group(group_name: str, fail_fast: bool = False) -> Dict[str, ModuleType]:
    """Load a specific group of modules"""
    if group_name not in MODULE_CONFIG:
        raise ValueError(f"Unknown module group: {group_name}")

    loaded = {}
    for module_name in MODULE_CONFIG[group_name]:
        if module_name in SUBMODULES:
            try:
                import_path = f"{PACKAGE_NAME}.{SUBMODULES[module_name]}"
                module = importlib.import_module(import_path)
                loaded[module_name] = module
                _module_cache[module_name] = module
            except ImportError as e:
                logger.warning("Failed to load {} from group {}: {}", module_name, group_name, e)
                if fail_fast:
                    raise

    return loaded


def init_cache_plugins(skip_modules: list[str] = None, skip_prefixes: list[str] = None):
    """
    Auto-discover and cache all functions marked with @cacheable.
    Now works with lazy loading.
    """
    skip_modules = set(skip_modules or [])
    skip_prefixes = set(skip_prefixes or ["afml.tests", "afml.test_"])

    # Get the actual package module
    pkg = importlib.import_module(__name__)
    cached_functions = 0

    for finder, fullname, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if fullname in skip_modules or any(fullname.startswith(p) for p in skip_prefixes):
            continue

        try:
            mod = importlib.import_module(fullname)
        except ImportError as e:
            logger.warning("Skipping {} (import failed: {})", fullname, e)
            continue

        # Find cacheable functions
        for attr in dir(mod):
            fn = getattr(mod, attr)
            if getattr(fn, "_mlfl_cache", False):
                cached_functions += 1
                logger.debug("Found cacheable function: {}.{}", fullname, attr)

    logger.info("Initialized caching for {} functions", cached_functions)


def preload_modules(*module_names: str) -> Dict[str, ModuleType]:
    """Preload specific modules by name"""
    loaded = {}
    for name in module_names:
        if name in SUBMODULES:
            try:
                # This will trigger __getattr__ and cache the module
                module = getattr(sys.modules[__name__], name)
                loaded[name] = module
            except (ImportError, AttributeError) as e:
                logger.warning("Failed to preload {}: {}", name, e)
    return loaded


def get_loaded_modules() -> list:
    """Get list of currently loaded module names"""
    return list(_module_cache.keys())


def reload_module(name: str) -> Optional[ModuleType]:
    """Reload a specific module"""
    if name not in SUBMODULES:
        raise ValueError(f"Unknown module: {name}")

    # Remove from cache
    if name in _module_cache:
        del _module_cache[name]

    # Remove from sys.modules if present
    import_path = f"{PACKAGE_NAME}.{SUBMODULES[name]}"
    if import_path in sys.modules:
        del sys.modules[import_path]

    # Reload
    try:
        return getattr(sys.modules[__name__], name)
    except (ImportError, AttributeError) as e:
        logger.error("Failed to reload {}: {}", name, e)
        return None


def get_cache_performance_summary() -> dict:
    """Get a summary of cache performance across all functions"""
    stats = get_cache_stats()
    overall_hit_rate = get_cache_hit_rate()

    total_hits = sum(s["hits"] for s in stats.values())
    total_misses = sum(s["misses"] for s in stats.values())
    total_calls = total_hits + total_misses

    return {
        "overall_hit_rate": overall_hit_rate,
        "total_calls": total_calls,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "functions_tracked": len(stats),
        "top_performers": sorted(
            [
                {
                    "function": fname,
                    "hit_rate": (
                        s["hits"] / (s["hits"] + s["misses"])
                        if (s["hits"] + s["misses"]) > 0
                        else 0
                    ),
                    "total_calls": s["hits"] + s["misses"],
                }
                for fname, s in stats.items()
                if (s["hits"] + s["misses"]) > 0
            ],
            key=lambda x: x["hit_rate"],
            reverse=True,
        )[:5],
    }


def load_jupyter_notebook():
    setup_nb = pattern_jupyter_notebook()
    app = setup_nb()
    return app


# =============================================================================
# __all__ definition - what gets imported with "from afml import *"
# =============================================================================

__all__ = [
    # Core functionality
    "memory",
    "init_cache_plugins",
    # Cache monitoring
    "get_cache_hit_rate",
    "get_cache_stats",
    "clear_cache_stats",
    "clear_afml_cache",
    "CacheAnalyzer",
    "get_cache_performance_summary",
    # Module management
    "AFMLApplication",
    "preload_modules",
    "get_loaded_modules",
    "reload_module",
    "load_module_group",
    "load_jupyter_notebook",
    # All submodules (available via lazy loading)
    *SUBMODULES.keys(),
]
