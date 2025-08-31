# afml/cache/__init__.py - Simplified but effective cache system

"""
Centralized caching system for AFML package.
Handles both Numba JIT compilation caching and function result caching.
"""

import json
import os
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Union

from appdirs import user_cache_dir
from joblib import Memory
from loguru import logger

# =============================================================================
# 1) CACHE DIRECTORY SETUP
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
    Simple cacheable decorator with hit/miss tracking.
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

        return cached_func(*args, **kwargs)

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
# 9) EXPORTS
# =============================================================================

__all__ = [
    # Core caching
    "memory",
    "cacheable",
    "initialize_cache_system",
    # Statistics
    "cache_stats",
    "get_cache_hit_rate",
    "get_cache_stats",
    "clear_cache_stats",
    "get_cache_summary",
    # Analysis
    "CacheAnalyzer",
    # Cleanup
    "clear_afml_cache",
    # Directory info
    "CACHE_DIRS",
]
