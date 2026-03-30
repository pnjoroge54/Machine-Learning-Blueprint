"""
AFML Unified Cache System
=========================
Single, production-ready caching solution for all ML, backtesting, and CV workflows.
"""

from .unified_cache import (
    cacheable,
    UnifiedCacheKeyGenerator,
    cache_stats,
    get_cache_hit_rate,
    clear_afml_cache,
    initialize_cache_system,
    CacheAnalyzer,
    CACHE_DIRS,
)

# High-level APIs
from .backtest_cache import backtest_cache, BacktestCache
from .cv_cache import cv_cacheable  # shim for backwards compatibility

# Updated supporting modules
from .selective_cleaner import (
    SelectiveCacheCleaner,
    selective_cleaner,
    clean_stale_cache,
    clean_module_cache,
    get_cache_summary,
)
from .cache_monitoring import get_cache_monitor, CacheMonitor
from .startup_script import run_cache_startup

__all__ = [
    "cacheable",
    "UnifiedCacheKeyGenerator",
    "cache_stats",
    "get_cache_hit_rate",
    "clear_afml_cache",
    "initialize_cache_system",
    "CacheAnalyzer",
    "CACHE_DIRS",
    "backtest_cache",
    "BacktestCache",
    "cv_cacheable",
    "get_cache_monitor",
    "CacheMonitor",
    "run_cache_startup",
    "SelectiveCacheCleaner",
    "selective_cleaner",
    "clean_stale_cache",
]

# Auto-initialize on first import (optional but recommended)
initialize_cache_system()


