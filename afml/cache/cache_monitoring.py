"""
Advanced cache monitoring and performance analysis.
Provides detailed insights into cache efficiency and usage patterns.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


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
    Tracks performance, identifies issues, and provides optimization recommendations.
    """

    def __init__(self):
        """Initialize cache monitor."""
        try:
            from . import cache_stats, memory

            self.cache_stats = cache_stats
            self.memory = memory
        except ImportError as e:
            logger.warning(f"Failed to import cache dependencies: {e}")
            # Create fallback attributes
            self.cache_stats = None
            self.memory = None

        # Import at runtime to avoid circular imports
        from . import CACHE_DIRS

        self.cache_dirs = CACHE_DIRS

        # Track computation times
        self.computation_times: Dict[str, List[float]] = defaultdict(list)

        # Track access patterns
        self.access_log: Dict[str, List[float]] = defaultdict(list)

    def get_function_stats(self, function_name: str) -> Optional[FunctionCacheStats]:
        """
        Get detailed statistics for a specific function.

        Args:
            function_name: Full function name (module.function)

        Returns:
            FunctionCacheStats or None if function not tracked
        """
        all_stats = self.cache_stats.get_stats()

        if function_name not in all_stats:
            return None

        stats = all_stats[function_name]
        hits = stats["hits"]
        misses = stats["misses"]
        total = hits + misses

        # Calculate average computation time if available
        avg_time = None
        if function_name in self.computation_times:
            times = self.computation_times[function_name]
            avg_time = sum(times) / len(times) if times else None

        # Get cache size
        cache_size = self._get_function_cache_size(function_name)

        # Get last access time
        last_access = None
        if function_name in self.access_log:
            last_access = max(self.access_log[function_name])

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
        result = []

        for function_name in all_stats.keys():
            stats = self.get_function_stats(function_name)
            if stats:
                result.append(stats)

        return result

    def generate_health_report(self, top_n: int = 5, stale_days: int = 7) -> CacheHealthReport:
        """
        Generate comprehensive cache health report.

        Args:
            top_n: Number of top/worst performers to include
            stale_days: Days to consider cache stale

        Returns:
            CacheHealthReport with analysis and recommendations
        """
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
                recommendations=["No cached functions found. Start using @cacheable decorators."],
            )

        # Calculate overall metrics
        total_calls = sum(s.total_calls for s in all_stats)
        total_hits = sum(s.cache_hits for s in all_stats)
        overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

        # Calculate total cache size
        total_size = sum(s.cache_size_mb or 0 for s in all_stats)

        # Sort by hit rate for top/worst performers
        sorted_by_hit_rate = sorted(all_stats, key=lambda x: x.hit_rate, reverse=True)
        top_performers = sorted_by_hit_rate[:top_n]
        worst_performers = sorted_by_hit_rate[-top_n:]

        # Find stale caches
        stale_cutoff = time.time() - (stale_days * 24 * 3600)
        stale_caches = [
            s.function_name for s in all_stats if s.last_accessed and s.last_accessed < stale_cutoff
        ]

        # Generate recommendations
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
        """
        Get detailed efficiency report as DataFrame.

        Returns:
            DataFrame with per-function statistics
        """
        all_stats = self.get_all_function_stats()

        if not all_stats:
            return pd.DataFrame()

        data = []
        for stats in all_stats:
            data.append(
                {
                    "function": stats.function_name,
                    "calls": stats.total_calls,
                    "hits": stats.cache_hits,
                    "misses": stats.cache_misses,
                    "hit_rate": f"{stats.hit_rate:.1%}",
                    "avg_time_ms": (
                        f"{stats.avg_computation_time * 1000:.2f}"
                        if stats.avg_computation_time
                        else "N/A"
                    ),
                    "cache_size_mb": (
                        f"{stats.cache_size_mb:.2f}" if stats.cache_size_mb else "N/A"
                    ),
                    "last_access": (
                        pd.Timestamp.fromtimestamp(stats.last_accessed).strftime("%Y-%m-%d %H:%M")
                        if stats.last_accessed
                        else "N/A"
                    ),
                }
            )

        df = pd.DataFrame(data)
        return df.sort_values("hit_rate", ascending=False)

    def analyze_cache_patterns(self) -> Dict[str, Any]:
        """
        Analyze cache access patterns to identify optimization opportunities.

        Returns:
            Dict with pattern analysis results
        """
        all_stats = self.get_all_function_stats()

        patterns = {
            "high_miss_rate_functions": [],
            "unused_caches": [],
            "large_caches": [],
            "frequently_accessed": [],
            "optimization_candidates": [],
        }

        for stats in all_stats:
            # High miss rate (< 50%)
            if stats.hit_rate < 0.5 and stats.total_calls > 10:
                patterns["high_miss_rate_functions"].append(
                    {
                        "function": stats.function_name,
                        "hit_rate": stats.hit_rate,
                        "calls": stats.total_calls,
                    }
                )

            # Unused caches (no hits in last 7 days)
            if stats.last_accessed:
                days_since_access = (time.time() - stats.last_accessed) / (24 * 3600)
                if days_since_access > 7:
                    patterns["unused_caches"].append(
                        {"function": stats.function_name, "days": int(days_since_access)}
                    )

            # Large caches (> 100 MB)
            if stats.cache_size_mb and stats.cache_size_mb > 100:
                patterns["large_caches"].append(
                    {
                        "function": stats.function_name,
                        "size_mb": stats.cache_size_mb,
                        "hit_rate": stats.hit_rate,
                    }
                )

            # Frequently accessed (> 100 calls)
            if stats.total_calls > 100:
                patterns["frequently_accessed"].append(
                    {"function": stats.function_name, "calls": stats.total_calls}
                )

            # Optimization candidates (high calls, low hit rate)
            if stats.total_calls > 50 and stats.hit_rate < 0.3:
                patterns["optimization_candidates"].append(
                    {
                        "function": stats.function_name,
                        "calls": stats.total_calls,
                        "hit_rate": stats.hit_rate,
                    }
                )

        return patterns

    def track_computation_time(self, function_name: str, duration: float):
        """
        Track computation time for a function call.

        Args:
            function_name: Full function name
            duration: Execution time in seconds
        """
        self.computation_times[function_name].append(duration)

        # Keep only last 100 measurements to limit memory
        if len(self.computation_times[function_name]) > 100:
            self.computation_times[function_name] = self.computation_times[function_name][-100:]

    def track_access(self, function_name: str):
        """
        Track cache access time.

        Args:
            function_name: Full function name
        """
        self.access_log[function_name].append(time.time())

        # Keep only last 1000 accesses
        if len(self.access_log[function_name]) > 1000:
            self.access_log[function_name] = self.access_log[function_name][-1000:]

    def get_time_series_analysis(
        self, function_name: str, hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """
        Get time-series analysis of cache access patterns.

        Args:
            function_name: Function to analyze
            hours: Number of hours to analyze

        Returns:
            DataFrame with hourly access patterns
        """
        if function_name not in self.access_log:
            return None

        access_times = self.access_log[function_name]
        cutoff = time.time() - (hours * 3600)

        # Filter to requested time range
        recent_accesses = [t for t in access_times if t > cutoff]

        if not recent_accesses:
            return None

        # Convert to timestamps and aggregate by hour
        timestamps = pd.to_datetime(recent_accesses, unit="s")
        df = pd.DataFrame({"timestamp": timestamps})
        df["hour"] = df["timestamp"].dt.floor("H")

        # Count accesses per hour
        hourly = df.groupby("hour").size().reset_index(name="access_count")

        return hourly

    def print_health_report(self, detailed: bool = False):
        """
        Print formatted health report to console.

        Args:
            detailed: If True, include detailed statistics
        """
        report = self.generate_health_report()

        print("\n" + "=" * 70)
        print("CACHE HEALTH REPORT")
        print("=" * 70)

        print(f"\nOverall Statistics:")
        print(f"  Total Functions:     {report.total_functions}")
        print(f"  Total Calls:         {report.total_calls:,}")
        print(f"  Overall Hit Rate:    {report.overall_hit_rate:.1%}")
        print(f"  Total Cache Size:    {report.total_cache_size_mb:.2f} MB")

        if report.top_performers:
            print(f"\nTop Performers (by hit rate):")
            for i, stats in enumerate(report.top_performers, 1):
                print(
                    f"  {i}. {stats.function_name.split('.')[-1]}: "
                    f"{stats.hit_rate:.1%} ({stats.total_calls} calls)"
                )

        if report.worst_performers:
            print(f"\nWorst Performers (by hit rate):")
            for i, stats in enumerate(report.worst_performers, 1):
                print(
                    f"  {i}. {stats.function_name.split('.')[-1]}: "
                    f"{stats.hit_rate:.1%} ({stats.total_calls} calls)"
                )

        if report.stale_caches:
            print(f"\nStale Caches (not accessed recently):")
            for func in report.stale_caches[:5]:
                print(f"  - {func.split('.')[-1]}")

        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        if detailed:
            print(f"\nDetailed Statistics:")
            df = self.get_efficiency_report()
            print(df.to_string(index=False))

        print("\n" + "=" * 70 + "\n")

    def export_report(self, output_path: Path):
        """
        Export detailed report to file.

        Args:
            output_path: Path to save report (supports .csv, .json, .html)
        """
        df = self.get_efficiency_report()

        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        elif output_path.suffix == ".html":
            df.to_html(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logger.info(f"Exported cache report to {output_path}")

    # Private methods

    def _get_function_cache_size(self, function_name: str) -> Optional[float]:
        """Get disk size of cache for a function in MB."""
        try:
            cache_dir = Path(self.memory.location)
            logger.debug(f"Looking for cache in: {cache_dir}")

            if not cache_dir.exists():
                logger.debug(f"Cache directory does not exist: {cache_dir}")
                return None

            # Find cache files for this function
            total_size = 0
            found_files = []

            # Joblib uses a different directory structure based on function fingerprint
            for cache_file in cache_dir.rglob("*.pkl"):  # Look for pickle files
                if cache_file.is_file():
                    # Check if this might belong to our function
                    # Joblib stores functions in module/function_name subdirectories
                    if function_name.replace(".", "/") in str(cache_file):
                        file_size = cache_file.stat().st_size
                        total_size += file_size
                        found_files.append((cache_file, file_size))

            if found_files:
                logger.debug(f"Found {len(found_files)} cache files for {function_name}")
                for file_path, size in found_files:
                    logger.debug(f"  {file_path.name}: {size / 1024:.2f} KB")

                size_mb = total_size / (1024 * 1024)
                logger.debug(f"Total cache size for {function_name}: {size_mb:.2f} MB")
                return size_mb
            else:
                logger.debug(f"No cache files found for {function_name}")
                return 0.0  # Return 0 instead of None for no cache files

        except Exception as e:
            logger.warning(f"Error calculating cache size for {function_name}: {e}")
            return None

    def _generate_recommendations(
        self,
        all_stats: List[FunctionCacheStats],
        overall_hit_rate: float,
        total_size: float,
        stale_caches: List[str],
    ) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Overall hit rate recommendations
        if overall_hit_rate < 0.5:
            recommendations.append(
                "Overall hit rate is low (<50%). Consider reviewing cache key generation "
                "or function parameter patterns."
            )
        elif overall_hit_rate > 0.9:
            recommendations.append("Excellent hit rate (>90%)! Cache system is performing well.")

        # Cache size recommendations
        if total_size > 1000:  # > 1 GB
            recommendations.append(
                f"Cache size is large ({total_size:.0f} MB). Consider implementing TTL-based "
                "cleanup or reducing cached data size."
            )

        # Stale cache recommendations
        if len(stale_caches) > 5:
            recommendations.append(
                f"Found {len(stale_caches)} stale caches. Run cache_maintenance() to clean up."
            )

        # Function-specific recommendations
        low_hit_rate_funcs = [s for s in all_stats if s.hit_rate < 0.3 and s.total_calls > 20]
        if low_hit_rate_funcs:
            func_names = [f.function_name.split(".")[-1] for f in low_hit_rate_funcs[:3]]
            recommendations.append(
                f"Functions with low hit rate: {', '.join(func_names)}. "
                "Review cache key generation for these functions."
            )

        # Large cache recommendations
        large_caches = [s for s in all_stats if s.cache_size_mb and s.cache_size_mb > 100]
        if large_caches:
            func_names = [f.function_name.split(".")[-1] for f in large_caches[:3]]
            recommendations.append(
                f"Large caches detected: {', '.join(func_names)}. "
                "Consider compressing cached data or implementing selective caching."
            )

        if not recommendations:
            recommendations.append("Cache system is healthy. No issues detected.")

        return recommendations


# =============================================================================
# Global instance and convenience functions
# =============================================================================

_global_monitor: Optional[CacheMonitor] = None


def get_cache_monitor() -> CacheMonitor:
    """Get global cache monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CacheMonitor()
    return _global_monitor


def print_cache_health():
    """Print cache health report to console."""
    monitor = get_cache_monitor()
    monitor.print_health_report(detailed=False)


def get_cache_efficiency_report() -> pd.DataFrame:
    """Get cache efficiency report as DataFrame."""
    monitor = get_cache_monitor()
    return monitor.get_efficiency_report()


def analyze_cache_patterns() -> Dict[str, Any]:
    """Analyze cache access patterns."""
    monitor = get_cache_monitor()
    return monitor.analyze_cache_patterns()


__all__ = [
    "CacheMonitor",
    "FunctionCacheStats",
    "CacheHealthReport",
    "get_cache_monitor",
    "print_cache_health",
    "get_cache_efficiency_report",
    "analyze_cache_patterns",
]
