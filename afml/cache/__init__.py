### 4. `afml/cache/__init__.py`
```python
"""
AFML Unified Cache System
Single production-ready caching solution.
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

from .backtest_cache import backtest_cache, BacktestCache
from .cv_cache import cv_cacheable  # backwards compatibility shim
from .selective_cleaner import SelectiveCacheCleaner, selective_cleaner, clean_stale_cache
from .cache_monitoring import (
    CacheMonitor,
    get_cache_monitor,
    print_cache_health,
    get_cache_efficiency_report,
    analyze_cache_patterns,
    diagnose_cache_issues,
)

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
    "SelectiveCacheCleaner",
    "selective_cleaner",
    "clean_stale_cache",
    "CacheMonitor",
    "get_cache_monitor",
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
    "diagnose_cache_issues",
]

# Auto-initialize when package is imported
initialize_cache_system()
