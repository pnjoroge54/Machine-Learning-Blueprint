"""
Advanced Cache Monitoring and Performance Analysis for AFML
============================================================

Provides detailed insights into cache efficiency, usage patterns,
hit rates, cache sizes, and optimization recommendations.

Fully compatible with the new unified_cache.py (cloudpickle + joblib).
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from joblib import Memory
from loguru import logger

# Import from unified core (no circular imports)
from .unified_cache import cache_stats, CACHE_DIRS


@dataclass
class FunctionCacheStats:
    """Statistics for a single cached function."""

    function_name: str
    total_calls: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_computation_time: Optional[float] = None
    cache_size_mb: Optional[float] = None
    last_accessed: Optional[float] = None


@dataclass
class CacheHealthReport:
    """Overall cache system health report."""

    total_functions: int
    overall_hit_rate: float
    total_calls: int
    total_cache_size_mb: float
    top_performers: List[FunctionCacheStats]
    worst_performers: List[FunctionCacheStats]
    stale_caches: List[str]
    recommendations: List[str]


class CacheMonitor:
    """
    Advanced cache monitoring and analysis system.

    Tracks performance, identifies bottlenecks, and suggests optimizations.
    Works seamlessly with @cacheable decorator and UnifiedCacheKeyGenerator.
    """

    def __init__(self):
        """Initialize the cache monitor."""
        self.cache_stats = cache_stats
        self.cache_dirs = CACHE_DIRS

        # Joblib Memory instance (lazy-loaded to avoid circular imports)
        self._memory = None

        # Track computation times and access patterns
        self.computation_times: Dict[str, List[float]] = defaultdict(list)
        self.access_log: Dict[str, List[float]] = defaultdict(list)

    @property
    def memory(self) -> Optional[Memory]:
        """Lazy load joblib Memory instance."""
        if self._memory is None:
            try:
                self._memory = Memory(location=str(self.cache_dirs["joblib"]), verbose=0)
            except Exception as e:
                logger.warning(f"Failed to initialize joblib Memory: {e}")
                self._memory = None
        return self._memory

    def get_function_stats(self, function_name: str) -> Optional[FunctionCacheStats]:
        """Get detailed statistics for a specific function."""
        all_stats = self.cache_stats.get_stats()

        if function_name not in all_stats:
            return None

        stats = all_stats[function_name]
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        total = hits + misses

        # Average computation time
        avg_time = None
        if function_name in self.computation_times and self.computation_times[function_name]:
            times = self.computation_times[function_name]
            avg_time = sum(times) / len(times)

        # Cache size on disk
        cache_size = self._get_function_cache_size(function_name)

        # Last access time
        last_access = max(self.access_log[function_name]) if self.access_log[function_name] else None

        return FunctionCacheStats(
            function_name=function_name,
            total_calls=total,
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hits / total if total > 0 else 0.0,
            avg_computation_time=avg_time,
            cache_size_mb=cache_size,
            last_accessed=last_access,
        )

    def get_all_function_stats(self) -> List[FunctionCacheStats]:
        """Get statistics for all tracked functions."""
        all_stats = self.cache_stats.get_stats()
        return [self.get_function_stats(name) for name in all_stats.keys() if self.get_function_stats(name)]

    def generate_health_report(
        self, top_n: int = 5, stale_days: int = 7
    ) -> CacheHealthReport:
        """Generate a comprehensive cache health report."""
        all_stats = self.get_all_function_stats()

        if not all_stats:
            return CacheHealthReport(
                total_functions=0,
                overall_hit_rate=0.0,
                total_calls=0,
                total_cache_size_mb=0.0,
                top_performers=[],
                worst_performers=[],
                stale_caches=[],
                recommendations=["No cached functions found. Start decorating with @cacheable."],
            )

        total_calls = sum(s.total_calls for s in all_stats)
        total_hits = sum(s.cache_hits for s in all_stats)
        overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0.0
        total_size = sum(s.cache_size_mb or 0 for s in all_stats)

        sorted_by_hit = sorted(all_stats, key=lambda x: x.hit_rate, reverse=True)
        top_performers = sorted_by_hit[:top_n]
        worst_performers = sorted_by_hit[-top_n:]

        stale_cutoff = time.time() - (stale_days * 86400)
        stale_caches = [
            s.function_name for s in all_stats
            if s.last_accessed and s.last_accessed < stale_cutoff
        ]

        recommendations = self._generate_recommendations(
            all_stats, overall_hit_rate, total_size, stale_caches
        )

        return CacheHealthReport(
            total_functions=len(all_stats),
            overall_hit_rate=overall_hit_rate,
            total_calls=total_calls,
            total_cache_size_mb=round(total_size, 2),
            top_performers=top_performers,
            worst_performers=worst_performers,
            stale_caches=stale_caches,
            recommendations=recommendations,
        )

    def get_efficiency_report(self) -> pd.DataFrame:
        """Return a pandas DataFrame with per-function cache efficiency."""
        all_stats = self.get_all_function_stats()
        if not all_stats:
            return pd.DataFrame()

        data = []
        for s in all_stats:
            data.append({
                "function": s.function_name,
                "calls": s.total_calls,
                "hits": s.cache_hits,
                "misses": s.cache_misses,
                "hit_rate": f"{s.hit_rate:.1%}",
                "avg_time_ms": f"{s.avg_computation_time * 1000:.2f}" if s.avg_computation_time else "N/A",
                "cache_size_mb": f"{s.cache_size_mb:.2f}" if s.cache_size_mb else "N/A",
                "last_access": (
                    pd.Timestamp.fromtimestamp(s.last_accessed).strftime("%Y-%m-%d %H:%M")
                    if s.last_accessed else "N/A"
                ),
            })

        df = pd.DataFrame(data)
        return df.sort_values("hit_rate", ascending=False)

    def analyze_cache_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns and flag optimization opportunities."""
        all_stats = self.get_all_function_stats()
        patterns: Dict[str, List[Dict]] = {
            "high_miss_rate_functions": [],
            "unused_caches": [],
            "large_caches": [],
            "frequently_accessed": [],
            "optimization_candidates": [],
        }

        for s in all_stats:
            if s.hit_rate < 0.5 and s.total_calls > 10:
                patterns["high_miss_rate_functions"].append({
                    "function": s.function_name, "hit_rate": s.hit_rate, "calls": s.total_calls
                })

            if s.last_accessed:
                days_since = (time.time() - s.last_accessed) / 86400
                if days_since > 7:
                    patterns["unused_caches"].append({
                        "function": s.function_name, "days": int(days_since)
                    })

            if s.cache_size_mb and s.cache_size_mb > 100:
                patterns["large_caches"].append({
                    "function": s.function_name, "size_mb": s.cache_size_mb, "hit_rate": s.hit_rate
                })

            if s.total_calls > 100:
                patterns["frequently_accessed"].append({
                    "function": s.function_name, "calls": s.total_calls
                })

            if s.total_calls > 50 and s.hit_rate < 0.3:
                patterns["optimization_candidates"].append({
                    "function": s.function_name, "calls": s.total_calls, "hit_rate": s.hit_rate
                })

        return patterns

    def track_computation_time(self, function_name: str, duration: float):
        """Record computation time (call from @cacheable on miss)."""
        self.computation_times[function_name].append(duration)
        if len(self.computation_times[function_name]) > 100:
            self.computation_times[function_name] = self.computation_times[function_name][-100:]

    def track_access(self, function_name: str):
        """Record cache access time (hit or miss)."""
        self.access_log[function_name].append(time.time())
        if len(self.access_log[function_name]) > 1000:
            self.access_log[function_name] = self.access_log[function_name][-1000:]

    def print_health_report(self, detailed: bool = False):
        """Print a nicely formatted health report to console."""
        report = self.generate_health_report()

        print("\n" + "=" * 70)
        print("AFML CACHE HEALTH REPORT")
        print("=" * 70)

        print(f"\nOverall Statistics:")
        print(f"  Total Functions : {report.total_functions}")
        print(f"  Total Calls     : {report.total_calls:,}")
        print(f"  Overall Hit Rate: {report.overall_hit_rate:.1%}")
        print(f"  Total Cache Size: {report.total_cache_size_mb:.2f} MB")

        if report.top_performers:
            print(f"\nTop Performers:")
            for i, s in enumerate(report.top_performers, 1):
                print(f"  {i}. {s.function_name.split('.')[-1]}: {s.hit_rate:.1%} ({s.total_calls} calls)")

        if report.worst_performers:
            print(f"\nWorst Performers:")
            for i, s in enumerate(report.worst_performers, 1):
                print(f"  {i}. {s.function_name.split('.')[-1]}: {s.hit_rate:.1%} ({s.total_calls} calls)")

        if report.stale_caches:
            print(f"\nStale Caches (>7 days):")
            for func in report.stale_caches[:5]:
                print(f"  - {func.split('.')[-1]}")

        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        if detailed:
            print(f"\nDetailed Efficiency Report:")
            print(self.get_efficiency_report().to_string(index=False))

        print("\n" + "=" * 70 + "\n")

    def export_report(self, output_path: Union[str, Path]):
        """Export efficiency report to CSV, JSON, or HTML."""
        output_path = Path(output_path)
        df = self.get_efficiency_report()

        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        elif output_path.suffix == ".html":
            df.to_html(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}. Use .csv, .json or .html")

        logger.info(f"Cache report exported to {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_function_cache_size(self, function_name: str) -> float:
        """Calculate on-disk cache size for a function (supports joblib + manual .pkl)."""
        try:
            total_size = 0
            patterns = [
                function_name.replace(".", "_").lower(),
                function_name.split(".")[-1].lower(),
            ]

            # Scan joblib cache
            if self.memory and hasattr(self.memory, "location"):
                cache_dir = Path(self.memory.location)
                if cache_dir.exists():
                    for file in cache_dir.rglob("*"):
                        if file.is_file() and any(p in str(file).lower() for p in patterns):
                            total_size += file.stat().st_size

            # Scan manual .pkl files in base cache dir
            for file in self.cache_dirs["base"].glob("*.pkl"):
                if any(p in file.name.lower() for p in patterns):
                    total_size += file.stat().st_size

            return total_size / (1024 * 1024)  # MB
        except Exception as e:
            logger.debug(f"Could not compute cache size for {function_name}: {e}")
            return 0.0

    def _generate_recommendations(
        self, all_stats: List[FunctionCacheStats], overall_hit_rate: float,
        total_size: float, stale_caches: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if overall_hit_rate < 0.5:
            recs.append("Low overall hit rate (<50%). Review cache keys or add time_aware=True where appropriate.")
        elif overall_hit_rate > 0.9:
            recs.append("Excellent hit rate! Cache system is performing very well.")

        if total_size > 1000:
            recs.append(f"Cache is large ({total_size:.0f} MB). Consider running selective_cleaner.clean_old_entries() or clean_large_files().")

        if len(stale_caches) > 5:
            recs.append(f"{len(stale_caches)} stale cache entries detected. Run clean_stale_cache().")

        low_hit = [s for s in all_stats if s.hit_rate < 0.3 and s.total_calls > 20]
        if low_hit:
            names = [s.function_name.split(".")[-1] for s in low_hit[:3]]
            recs.append(f"Low hit-rate functions: {', '.join(names)}. Consider improving key generation or parameters.")

        if not recs:
            recs.append("Cache system looks healthy. No immediate actions needed.")

        return recs


# =============================================================================
# Global instance + convenience functions
# =============================================================================

_global_monitor: Optional[CacheMonitor] = None


def get_cache_monitor() -> CacheMonitor:
    """Return the global CacheMonitor instance (singleton)."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CacheMonitor()
    return _global_monitor


def print_cache_health(detailed: bool = False):
    """Quick way to print health report."""
    get_cache_monitor().print_health_report(detailed=detailed)


def get_cache_efficiency_report() -> pd.DataFrame:
    """Return efficiency DataFrame."""
    return get_cache_monitor().get_efficiency_report()


def analyze_cache_patterns() -> Dict[str, Any]:
    """Return pattern analysis dict."""
    return get_cache_monitor().analyze_cache_patterns()


def diagnose_cache_issues():
    """Run a full diagnostics printout (useful in notebooks/scripts)."""
    monitor = get_cache_monitor()
    monitor.print_health_report(detailed=True)


__all__ = [
    "CacheMonitor",
    "FunctionCacheStats",
    "CacheHealthReport",
    "get_cache_monitor",
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
    "diagnose_cache_issues",
]
