"""
AFML Unified Cache System
=========================
Production-grade caching + data access tracking for financial ML workflows.
"""

from loguru import logger

from .backtest_cache import BacktestCache, backtest_cache
from .cache_monitoring import (
    CacheMonitor,
    analyze_cache_patterns,
    diagnose_cache_issues,
    get_cache_efficiency_report,
    get_cache_monitor,
    print_cache_health,
)
from .incremental_bar_cache import IncrementalBarCache
from .selective_cleaner import (
    SelectiveCacheCleaner,
    clean_module_cache,
    clean_stale_cache,
    get_cache_summary,
    selective_cleaner,
)
from .unified_cache import (
    CACHE_DIRS,
    CacheAnalyzer,
    UnifiedCacheKeyGenerator,
    cache_stats,
    cacheable,
    clear_afml_cache,
    create_cacheable_param_grid,
    get_cache_hit_rate,
    initialize_cache_system,
    reconstruct_param_grid,
)

# =============================================================================
# DATA ACCESS TRACKER (exactly matching your file)
# =============================================================================

try:
    from .data_access_tracker import (
        DataAccessTracker,
        clear_data_access_log,
        get_data_tracker,
        log_data_access,
        print_contamination_report,
    )

    DATA_TRACKING_AVAILABLE = True
except ImportError:
    DATA_TRACKING_AVAILABLE = False
    logger.debug("data_access_tracker.py not found — data tracking disabled")
    # Safe no-op fallbacks so nothing breaks
    DataAccessTracker = None
    get_data_tracker = lambda: None
    log_data_access = lambda *a, **k: None
    print_contamination_report = lambda: None
    clear_data_access_log = lambda: None

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core caching
    "cacheable",
    "UnifiedCacheKeyGenerator",
    "cache_stats",
    "get_cache_hit_rate",
    "clear_afml_cache",
    "initialize_cache_system",
    "CacheAnalyzer",
    "CACHE_DIRS",
    "create_cacheable_param_grid",
    "reconstruct_param_grid",
    # Monitoring
    "CacheMonitor",
    "get_cache_monitor",
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
    "diagnose_cache_issues",
    # Maintenance
    "SelectiveCacheCleaner",
    "selective_cleaner",
    "clean_stale_cache",
    "clean_module_cache",
    "get_cache_summary",
    # Backtest
    "backtest_cache",
    "BacktestCache",
    # Data Access Tracking (your exact functions)
    "DataAccessTracker",
    "get_data_tracker",
    "log_data_access",
    "print_contamination_report",
    "clear_data_access_log",
    # Flags
    "DATA_TRACKING_AVAILABLE",
]

# Auto-initialize the cache system when the package is imported
initialize_cache_system()

if DATA_TRACKING_AVAILABLE:
    logger.debug("✓ Data access tracking enabled (anti-contamination protection)")
