# cache.py - Enhanced version with hit rate tracking
import json
import os
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from appdirs import user_cache_dir
from joblib import Memory

# 1) Setup cache directory
_cache_env = os.getenv("AFML_CACHE")
_cache_dir = Path(_cache_env) if _cache_env else Path(user_cache_dir("afml"))
_cache_dir.mkdir(parents=True, exist_ok=True)

# 2) Cache statistics file
_stats_file = _cache_dir / "cache_stats.json"


# 3) Thread-safe statistics tracking
class CacheStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._load_stats()

    def _load_stats(self):
        """Load existing statistics from disk"""
        if _stats_file.exists():
            try:
                with open(_stats_file, "r") as f:
                    data = json.load(f)
                    self._stats.update({k: v for k, v in data.items()})
            except (json.JSONDecodeError, IOError):
                pass  # Start fresh if file is corrupted

    def _save_stats(self):
        """Save statistics to disk"""
        try:
            with open(_stats_file, "w") as f:
                json.dump(dict(self._stats), f)
        except IOError:
            pass  # Fail silently if can't write

    def record_hit(self, func_name: str):
        """Record a cache hit"""
        with self._lock:
            self._stats[func_name]["hits"] += 1
            if self._stats[func_name]["hits"] % 10 == 0:  # Save every 10 hits
                self._save_stats()

    def record_miss(self, func_name: str):
        """Record a cache miss"""
        with self._lock:
            self._stats[func_name]["misses"] += 1
            if self._stats[func_name]["misses"] % 10 == 0:  # Save every 10 misses
                self._save_stats()

    def get_hit_rate(self, func_name: str = None) -> float:
        """Get hit rate for specific function or overall"""
        with self._lock:
            if func_name:
                stats = self._stats[func_name]
                total = stats["hits"] + stats["misses"]
                return stats["hits"] / total if total > 0 else 0.0
            else:
                # Overall hit rate
                total_hits = sum(s["hits"] for s in self._stats.values())
                total_calls = sum(s["hits"] + s["misses"] for s in self._stats.values())
                return total_hits / total_calls if total_calls > 0 else 0.0

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get all statistics"""
        with self._lock:
            return dict(self._stats)

    def clear_stats(self):
        """Clear all statistics"""
        with self._lock:
            self._stats.clear()
            if _stats_file.exists():
                _stats_file.unlink()


# Global stats instance
_cache_stats = CacheStats()

# 4) Create Memory instance
memory = Memory(location=str(_cache_dir), verbose=0)


# 5) Enhanced decorator that tracks cache hits/misses
def cacheable(func):
    """
    Mark a function for auto-caching with hit rate tracking.
    Will be picked up by init_cache_plugins().
    """
    # Store original function name for stats
    func_name = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key to check if result exists
        cached_func = memory.cache(func)

        # Check if we have a cached result by trying to load it
        try:
            # This is a bit of a hack - we create the cache key manually
            # to check if the result exists without computing it
            cache_key = cached_func._get_cache_key(*args, **kwargs)
            cache_path = cached_func.cache_dir / cache_key

            if cache_path.exists():
                _cache_stats.record_hit(func_name)
            else:
                _cache_stats.record_miss(func_name)

        except (AttributeError, Exception):
            # If we can't determine cache status, assume it's a miss
            _cache_stats.record_miss(func_name)

        # Execute the cached function
        return cached_func(*args, **kwargs)

    # Mark for caching
    setattr(wrapper, "_mlfl_cache", True)
    setattr(wrapper, "_original_func", func)
    return wrapper


# 6) Alternative approach using filesystem analysis
def _estimate_cache_hit_rate_filesystem() -> float:
    """
    Estimate cache hit rate based on filesystem access patterns.
    This analyzes file modification times to infer usage patterns.
    """
    if not _cache_dir.exists():
        return 0.0

    try:
        import time

        current_time = time.perf_counter()
        recent_threshold = 3600  # 1 hour in seconds

        cache_files = []
        for item in _cache_dir.rglob("*"):
            if item.is_file() and not item.name.startswith("."):
                cache_files.append(item)

        if not cache_files:
            return 0.0

        # Files accessed recently are likely hits
        recent_access = sum(
            1 for f in cache_files if (current_time - f.stat().st_atime) < recent_threshold
        )

        # Rough estimate: recent accesses / total files
        return min(recent_access / len(cache_files), 1.0)

    except (OSError, AttributeError):
        return 0.0


# 7) Main hit rate estimation function
def _estimate_cache_hit_rate() -> float:
    """
    Estimate cache hit rate using tracked statistics.
    Falls back to filesystem analysis if no stats available.
    """
    hit_rate = _cache_stats.get_hit_rate()

    if hit_rate == 0.0:
        # Fallback to filesystem analysis
        return _estimate_cache_hit_rate_filesystem()

    return hit_rate


# 8) Utility functions
def get_cache_hit_rate(func_name: str = None) -> float:
    """Get cache hit rate for a specific function or overall"""
    hit_rate = _cache_stats.get_hit_rate(func_name)

    if hit_rate == 0.0:
        # Fallback to filesystem analysis
        return _estimate_cache_hit_rate_filesystem()

    return hit_rate


def get_cache_stats() -> Dict[str, Dict[str, int]]:
    """Get detailed cache statistics"""
    return _cache_stats.get_stats()


def clear_cache_stats():
    """Clear cache statistics (not the cache itself)"""
    _cache_stats.clear_stats()


def clear_afml_cache(warn: bool = False, clear_stats: bool = True):
    """Wipe out all entries in the afml cache directory."""
    memory.clear(warn=warn)
    if clear_stats:
        clear_cache_stats()


# 9) Context manager for cache analysis
class CacheAnalyzer:
    """Context manager to analyze cache performance for a block of code"""

    def __init__(self):
        self.start_stats = None
        self.end_stats = None

    def __enter__(self):
        self.start_stats = _cache_stats.get_stats().copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = _cache_stats.get_stats().copy()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for the analyzed period"""
        if not self.start_stats or not self.end_stats:
            return {}

        report = {}
        for func_name in self.end_stats:
            start = self.start_stats.get(func_name, {"hits": 0, "misses": 0})
            end = self.end_stats[func_name]

            period_hits = end["hits"] - start["hits"]
            period_misses = end["misses"] - start["misses"]
            period_total = period_hits + period_misses

            if period_total > 0:
                report[func_name] = {
                    "hits": period_hits,
                    "misses": period_misses,
                    "hit_rate": period_hits / period_total,
                    "total_calls": period_total,
                }

        return report
