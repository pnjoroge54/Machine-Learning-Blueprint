"""
Cache cleanup utility for orphaned function versions.

This module is now focused on MAINTENANCE rather than active cache invalidation:
- Identifies orphaned cache entries from old function versions
- Provides size-based and age-based cleanup
- Works as a scheduled/manual maintenance tool

It NO LONGER does active change detection (that's handled by auto_versioning).
"""

import hashlib
import inspect
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger


class FunctionVersionTracker:
    """
    Tracks function versions to identify orphaned caches.

    This is now a PASSIVE tracker - it records current versions
    and helps identify old versions that can be cleaned up.
    """

    def __init__(self):
        # Import CACHE_DIRS at runtime to avoid circular import
        from . import CACHE_DIRS

        self.tracker_file = CACHE_DIRS["base"] / "function_versions.json"
        self.current_versions: Dict[str, Dict] = {}
        self._load_version_data()

    def _load_version_data(self):
        """Load existing version tracking data."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r") as f:
                    self.current_versions = json.load(f)
                logger.debug(
                    "Loaded version data for {} functions", len(self.current_versions)
                )
            except Exception as e:
                logger.warning("Failed to load version tracker: {}", e)
                self.current_versions = {}

    def _save_version_data(self):
        """Save version tracking data."""
        try:
            with open(self.tracker_file, "w") as f:
                json.dump(self.current_versions, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save version tracker: {}", e)

    def _get_function_hash(self, func) -> Optional[str]:
        """Get hash of function source code."""
        try:
            source = inspect.getsource(func)
            return hashlib.md5(source.encode()).hexdigest()[:12]
        except (OSError, TypeError):
            return None

    def _get_file_mtime(self, func) -> Optional[float]:
        """Get modification time of function's source file."""
        try:
            file_path = inspect.getfile(func)
            return Path(file_path).stat().st_mtime
        except (OSError, TypeError):
            return None

    def record_current_version(self, func) -> Dict[str, any]:
        """
        Record the current version of a function.

        Returns:
            Dict with version info: {hash, mtime, module, timestamp}
        """
        func_name = f"{func.__module__}.{func.__qualname__}"

        version_info = {
            "hash": self._get_function_hash(func),
            "mtime": self._get_file_mtime(func),
            "module": func.__module__,
            "last_seen": time.time(),
        }

        self.current_versions[func_name] = version_info
        self._save_version_data()

        return version_info

    def get_all_current_versions(self) -> Dict[str, Dict]:
        """Get all currently tracked function versions."""
        return self.current_versions.copy()

    def scan_cacheable_functions(self) -> Dict[str, Dict]:
        """
        Scan for all @cacheable functions and record their versions.

        Returns:
            Dict of function_name -> version_info
        """
        import gc
        import types

        scanned = {}

        try:
            for obj in gc.get_objects():
                try:
                    if (
                        isinstance(obj, types.FunctionType)
                        and hasattr(obj, "_afml_cacheable")
                        and obj._afml_cacheable
                    ):
                        func_name = f"{obj.__module__}.{obj.__qualname__}"
                        version_info = self.record_current_version(obj)
                        scanned[func_name] = version_info
                except (ReferenceError, AttributeError):
                    continue
        except Exception as e:
            logger.debug("Error during function scan: {}", e)

        logger.info("Scanned {} cacheable functions", len(scanned))
        return scanned


# Global version tracker
_version_tracker: Optional[FunctionVersionTracker] = None


def get_version_tracker() -> FunctionVersionTracker:
    """Get global version tracker instance."""
    global _version_tracker
    if _version_tracker is None:
        _version_tracker = FunctionVersionTracker()
    return _version_tracker


# =============================================================================
# Cache Cleanup Utilities
# =============================================================================


def find_orphaned_caches(
    modules: Optional[Union[str, List[str]]] = None, dry_run: bool = True
) -> Dict[str, List[str]]:
    """
    Find cache entries for old function versions.

    This identifies caches that were created by previous versions
    of functions (when auto_versioning is enabled).

    Args:
        modules: Module name(s) to check (None = all modules)
        dry_run: If True, only report (don't delete)

    Returns:
        Dict with 'orphaned_files', 'current_functions', 'total_size_mb'
    """
    if isinstance(modules, str):
        modules = [modules]

    from . import memory

    cache_dir = Path(memory.location)
    if not cache_dir.exists():
        return {"orphaned_files": [], "current_functions": [], "total_size_mb": 0.0}

    # Scan current function versions
    tracker = get_version_tracker()
    current_versions = tracker.scan_cacheable_functions()

    # Filter by modules if specified
    if modules:
        current_versions = {
            name: info
            for name, info in current_versions.items()
            if any(name.startswith(mod) for mod in modules)
        }

    logger.info(
        "Checking for orphaned caches across {} functions", len(current_versions)
    )

    # Build list of current version hashes
    current_hashes = {
        info["hash"] for info in current_versions.values() if info["hash"]
    }

    # Scan cache directory for versioned cache files
    orphaned_files = []
    total_size = 0.0

    for cache_file in cache_dir.rglob("*"):
        if not cache_file.is_file():
            continue

        # Look for version markers in cache filenames/paths
        cache_path_str = str(cache_file)

        # Check if this is a versioned cache (contains v_<hash>)
        if "_v_" in cache_path_str:
            # Extract version hash from path
            for current_hash in current_hashes:
                if f"v_{current_hash}" in cache_path_str:
                    break  # Found current version
            else:
                # No current version found - this is orphaned
                file_size = cache_file.stat().st_size / (1024 * 1024)
                orphaned_files.append(
                    {
                        "path": str(cache_file),
                        "size_mb": file_size,
                        "mtime": cache_file.stat().st_mtime,
                    }
                )
                total_size += file_size

    result = {
        "orphaned_files": orphaned_files,
        "current_functions": list(current_versions.keys()),
        "total_size_mb": round(total_size, 2),
        "orphaned_count": len(orphaned_files),
    }

    if orphaned_files:
        logger.info(
            "Found {} orphaned cache files ({:.2f} MB)", len(orphaned_files), total_size
        )
    else:
        logger.info("No orphaned caches found")

    return result


def clean_orphaned_caches(
    modules: Optional[Union[str, List[str]]] = None,
    min_age_hours: int = 24,
) -> Dict[str, any]:
    """
    Remove orphaned cache entries from old function versions.

    Args:
        modules: Module name(s) to clean (None = all)
        min_age_hours: Only remove orphaned caches older than this

    Returns:
        Dict with cleanup results
    """
    # Find orphaned caches
    orphaned = find_orphaned_caches(modules=modules, dry_run=True)

    if not orphaned["orphaned_files"]:
        logger.info("No orphaned caches to clean")
        return {
            "removed_count": 0,
            "removed_size_mb": 0.0,
            "kept_count": 0,
        }

    # Filter by age
    cutoff_time = time.time() - (min_age_hours * 3600)
    to_remove = [f for f in orphaned["orphaned_files"] if f["mtime"] < cutoff_time]

    # Remove files
    removed_count = 0
    removed_size = 0.0
    errors = []

    for file_info in to_remove:
        try:
            file_path = Path(file_info["path"])
            if file_path.exists():
                file_path.unlink()
                removed_count += 1
                removed_size += file_info["size_mb"]
        except Exception as e:
            errors.append(f"{file_info['path']}: {e}")
            logger.debug("Failed to remove {}: {}", file_info["path"], e)

    kept_count = len(orphaned["orphaned_files"]) - removed_count

    result = {
        "removed_count": removed_count,
        "removed_size_mb": round(removed_size, 2),
        "kept_count": kept_count,  # Too recent to remove
        "errors": errors,
    }

    logger.info(
        "Cleaned {} orphaned caches ({:.2f} MB), kept {} recent",
        removed_count,
        removed_size,
        kept_count,
    )

    return result


def cleanup_by_size(max_size_mb: int) -> float:
    """Remove oldest cache files if total size exceeds limit."""
    from . import memory

    cache_dir = Path(memory.location)
    if not cache_dir.exists():
        return 0.0

    # Get all cache files with sizes and modification times
    cache_files = []
    total_size = 0

    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            cache_files.append((file_path, size_mb, stat.st_mtime))
            total_size += size_mb

    if total_size <= max_size_mb:
        logger.info(
            "Cache size {:.1f} MB is under limit {:.1f} MB", total_size, max_size_mb
        )
        return 0.0

    # Sort by modification time (oldest first)
    cache_files.sort(key=lambda x: x[2])

    # Remove oldest files until under size limit
    size_to_remove = total_size - max_size_mb
    removed_size = 0.0

    for file_path, size_mb, _ in cache_files:
        if removed_size >= size_to_remove:
            break
        try:
            file_path.unlink()
            removed_size += size_mb
        except Exception as e:
            logger.debug("Failed to remove {}: {}", file_path, e)

    logger.info(
        "Removed {:.1f} MB of old caches (limit: {:.1f} MB)", removed_size, max_size_mb
    )
    return removed_size


def cleanup_by_age(max_age_days: int) -> int:
    """Remove cache files older than specified age."""
    from . import memory

    cache_dir = Path(memory.location)
    if not cache_dir.exists():
        return 0

    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    removed_count = 0

    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            except Exception as e:
                logger.debug("Failed to remove {}: {}", file_path, e)

    logger.info("Removed {} caches older than {} days", removed_count, max_age_days)
    return removed_count


# =============================================================================
# Main Cache Maintenance Function
# =============================================================================


def cache_maintenance(
    clean_orphaned: bool = True,
    max_cache_size_mb: Optional[int] = None,
    max_age_days: Optional[int] = None,
    min_orphan_age_hours: int = 24,
) -> Dict[str, Union[int, float, List[str]]]:
    """
    Perform comprehensive cache maintenance.

    This is now focused on CLEANUP rather than change detection:
    - Removes orphaned caches from old function versions
    - Enforces size limits
    - Removes stale old caches

    Args:
        clean_orphaned: Clean orphaned caches from old versions
        max_cache_size_mb: Clear oldest caches if total size exceeds this
        max_age_days: Clear caches older than this many days
        min_orphan_age_hours: Only remove orphaned caches older than this

    Returns:
        Maintenance report
    """
    report = {
        "orphaned_removed": 0,
        "orphaned_size_mb": 0.0,
        "size_cleared_mb": 0.0,
        "old_files_removed": 0,
        "functions_scanned": 0,
    }

    try:
        # Scan current functions
        tracker = get_version_tracker()
        versions = tracker.scan_cacheable_functions()
        report["functions_scanned"] = len(versions)

        # Clean orphaned caches
        if clean_orphaned:
            try:
                result = clean_orphaned_caches(min_age_hours=min_orphan_age_hours)
                report["orphaned_removed"] = result["removed_count"]
                report["orphaned_size_mb"] = result["removed_size_mb"]
            except Exception as e:
                logger.warning("Orphaned cache cleanup failed: {}", e)

        # Size-based cleanup
        if max_cache_size_mb:
            try:
                size_cleared = cleanup_by_size(max_cache_size_mb)
                report["size_cleared_mb"] = size_cleared
            except Exception as e:
                logger.warning("Size-based cleanup failed: {}", e)

        # Age-based cleanup
        if max_age_days:
            try:
                files_removed = cleanup_by_age(max_age_days)
                report["old_files_removed"] = files_removed
            except Exception as e:
                logger.warning("Age-based cleanup failed: {}", e)

        logger.info(
            "Cache maintenance completed: {}", _format_maintenance_report(report)
        )

    except Exception as e:
        logger.error("Cache maintenance failed: {}", e)
        report["error"] = str(e)

    return report


def _format_maintenance_report(report: Dict) -> str:
    """Format maintenance report for logging."""
    parts = []

    if report["functions_scanned"]:
        parts.append(f"{report['functions_scanned']} functions scanned")

    if report["orphaned_removed"]:
        parts.append(
            f"{report['orphaned_removed']} orphaned removed ({report['orphaned_size_mb']:.1f}MB)"
        )

    if report["size_cleared_mb"]:
        parts.append(f"{report['size_cleared_mb']:.1f}MB size-cleared")

    if report["old_files_removed"]:
        parts.append(f"{report['old_files_removed']} old files removed")

    return ", ".join(parts) if parts else "no cleanup needed"


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_cache_versions() -> Dict[str, any]:
    """
    Analyze cache fragmentation by function versions.

    Returns info about how many versions exist for each function.
    """
    from . import memory

    cache_dir = Path(memory.location)
    if not cache_dir.exists():
        return {}

    # Count versions per function
    version_counts = defaultdict(lambda: {"versions": set(), "total_size_mb": 0.0})

    for cache_file in cache_dir.rglob("*"):
        if not cache_file.is_file():
            continue

        cache_path = str(cache_file)

        # Extract function name and version from path
        # Path typically contains: module_name/function_name/hash/...
        parts = cache_path.split(os.sep)

        for i, part in enumerate(parts):
            if "_v_" in part:
                # Found a versioned cache
                func_name = parts[i - 1] if i > 0 else "unknown"
                version_hash = part.split("_v_")[1].split("_")[0]

                version_counts[func_name]["versions"].add(version_hash)
                file_size = cache_file.stat().st_size / (1024 * 1024)
                version_counts[func_name]["total_size_mb"] += file_size

    # Format results
    analysis = {}
    for func_name, data in version_counts.items():
        analysis[func_name] = {
            "version_count": len(data["versions"]),
            "total_size_mb": round(data["total_size_mb"], 2),
            "avg_size_per_version_mb": (
                round(data["total_size_mb"] / len(data["versions"]), 2)
                if data["versions"]
                else 0.0
            ),
        }

    return analysis


def print_version_analysis():
    """Print analysis of cache versions."""
    analysis = analyze_cache_versions()

    if not analysis:
        print("\nNo versioned caches found.")
        return

    print("\n" + "=" * 70)
    print("CACHE VERSION ANALYSIS")
    print("=" * 70)

    # Sort by version count
    sorted_funcs = sorted(
        analysis.items(), key=lambda x: x[1]["version_count"], reverse=True
    )

    total_versions = sum(info["version_count"] for _, info in sorted_funcs)
    total_size = sum(info["total_size_mb"] for _, info in sorted_funcs)

    print(f"\nOverall:")
    print(f"  Functions with versions: {len(sorted_funcs)}")
    print(f"  Total versions: {total_versions}")
    print(f"  Total size: {total_size:.2f} MB")

    print(f"\nTop fragmented functions:")
    for i, (func_name, info) in enumerate(sorted_funcs[:10], 1):
        if info["version_count"] > 1:
            print(f"  {i}. {func_name}")
            print(f"     Versions: {info['version_count']}")
            print(f"     Size: {info['total_size_mb']:.2f} MB")
            print(f"     Avg per version: {info['avg_size_per_version_mb']:.2f} MB")

    print("=" * 70 + "\n")


# =============================================================================
# Convenience functions for common use cases
# =============================================================================


def clear_orphaned_ml_caches():
    """Clear orphaned caches for ML-related functions."""
    ml_modules = [
        "afml.ensemble",
        "afml.clustering",
        "afml.feature_importance",
        "afml.cross_validation",
        "afml.backtester",
    ]
    return clean_orphaned_caches(modules=ml_modules)


def clear_orphaned_labeling_caches():
    """Clear orphaned caches for labeling functions."""
    return clean_orphaned_caches(modules=["afml.labeling"])


def clear_orphaned_features_caches():
    """Clear orphaned caches for feature functions."""
    return clean_orphaned_caches(modules=["afml.features", "afml.strategies"])


__all__ = [
    # Core tracker
    "FunctionVersionTracker",
    "get_version_tracker",
    # Main maintenance function
    "cache_maintenance",
    # Cleanup functions
    "find_orphaned_caches",
    "clean_orphaned_caches",
    "cleanup_by_size",
    "cleanup_by_age",
    # Analysis
    "analyze_cache_versions",
    "print_version_analysis",
    # Convenience functions
    "clear_orphaned_ml_caches",
    "clear_orphaned_labeling_caches",
    "clear_orphaned_features_caches",
]
