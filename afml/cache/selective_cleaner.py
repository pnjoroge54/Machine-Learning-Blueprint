# afml/cache/selective_cleaner.py - Targeted cache invalidation

"""
Selective cache cleaning system that only invalidates functions that have changed.
Uses source code hashing and modification times to detect changes.
"""

import hashlib
import inspect
import json
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from loguru import logger


class FunctionTracker:
    """Tracks function signatures and source code hashes."""

    def __init__(self):
        # Import CACHE_DIRS at runtime to avoid circular import
        from . import CACHE_DIRS

        self.tracker_file = CACHE_DIRS["base"] / "function_tracker.json"
        self.tracked_functions: Dict[str, Dict] = {}
        # Registry to keep strong references to tracked functions
        self.function_registry: Dict[str, callable] = {}
        self._load_tracking_data()

    def _load_tracking_data(self):
        """Load existing function tracking data."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r") as f:
                    self.tracked_functions = json.load(f)
                logger.debug("Loaded tracking data for {} functions", len(self.tracked_functions))
            except Exception as e:
                logger.warning("Failed to load function tracker: {}", e)
                self.tracked_functions = {}

    def _save_tracking_data(self):
        """Save function tracking data."""
        try:
            with open(self.tracker_file, "w") as f:
                json.dump(self.tracked_functions, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save function tracker: {}", e)

    def _get_function_hash(self, func) -> Optional[str]:
        """Get hash of function source code."""
        try:
            source = inspect.getsource(func)
            return hashlib.md5(source.encode()).hexdigest()
        except (OSError, TypeError):
            # Can't get source (built-in, dynamically created, etc.)
            return None

    def _get_file_mtime(self, func) -> Optional[float]:
        """Get modification time of function's source file."""
        try:
            file_path = inspect.getfile(func)
            return Path(file_path).stat().st_mtime
        except (OSError, TypeError):
            return None

    def register_function(self, func):
        """Register a function to keep a strong reference to it."""
        func_name = f"{func.__module__}.{func.__qualname__}"
        self.function_registry[func_name] = func

    def track_function(self, func) -> bool:
        """
        Track a function and return True if it has changed.

        Returns:
            True if function has changed or is new, False if unchanged
        """
        func_name = f"{func.__module__}.{func.__qualname__}"

        # Register the function to prevent garbage collection
        self.register_function(func)

        # Get current function metadata
        current_hash = self._get_function_hash(func)
        current_mtime = self._get_file_mtime(func)

        # Get stored metadata
        stored = self.tracked_functions.get(func_name, {})
        stored_hash = stored.get("hash")
        stored_mtime = stored.get("mtime")

        # Check if function has changed
        has_changed = (
            current_hash != stored_hash
            or current_mtime != stored_mtime
            or stored_hash is None  # New function
        )

        if has_changed:
            # Update tracking data
            self.tracked_functions[func_name] = {
                "hash": current_hash,
                "mtime": current_mtime,
                "module": func.__module__,
            }
            self._save_tracking_data()

            if stored_hash is not None:  # Not a new function
                logger.debug("Function changed: {}", func_name)

        return has_changed

    def get_changed_functions(self, module_names: Optional[List[str]] = None) -> List[str]:
        """Get list of functions that have changed in specified modules."""
        changed = []

        try:
            # First try to get functions from our registry (more reliable)
            for func_name, func in self.function_registry.items():
                if module_names and not any(func_name.startswith(mod) for mod in module_names):
                    continue
                try:
                    if self.track_function(func):
                        changed.append(func_name)
                except (ReferenceError, AttributeError):
                    # Function reference is stale, remove from registry
                    logger.debug("Removing stale function reference: {}", func_name)
                    continue

            # Fallback to garbage collection method for any missed functions
            try:
                for name, obj in _get_cacheable_functions().items():
                    if name in self.function_registry:
                        continue  # Already processed above

                    if module_names and not any(name.startswith(mod) for mod in module_names):
                        continue

                    if self.track_function(obj):
                        changed.append(name)
            except Exception as e:
                logger.debug("Error in fallback function discovery: {}", e)

        except Exception as e:
            logger.debug("Error getting changed functions: {}", e)

        return changed


def _get_cacheable_functions() -> Dict[str, callable]:
    """Find all functions decorated with @cacheable."""
    import gc
    import types
    import weakref

    cacheable_funcs = {}

    try:
        # Search through all function objects in memory
        for obj in gc.get_objects():
            try:
                if (
                    isinstance(obj, types.FunctionType)
                    and hasattr(obj, "_afml_cacheable")
                    and obj._afml_cacheable
                ):
                    func_name = f"{obj.__module__}.{obj.__qualname__}"
                    cacheable_funcs[func_name] = obj
            except (ReferenceError, AttributeError):
                # Skip objects that are being garbage collected or have weak references
                continue

    except Exception as e:
        logger.debug("Error during cacheable function discovery: {}", e)

    return cacheable_funcs


# Global function tracker - initialize lazily to avoid circular import
_function_tracker = None


def get_function_tracker():
    """Get the global function tracker, creating it if needed."""
    global _function_tracker
    if _function_tracker is None:
        _function_tracker = FunctionTracker()
    return _function_tracker


# For backward compatibility
@property
def function_tracker():
    return get_function_tracker()


def selective_cache_clear(
    modules: Optional[Union[str, List[str]]] = None,
    functions: Optional[Union[str, List[str]]] = None,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """
    Selectively clear cache for changed functions.

    Args:
        modules: Module name(s) to check for changes (e.g., 'afml.labeling')
        functions: Specific function name(s) to check/clear
        dry_run: If True, only report what would be cleared

    Returns:
        Dict with 'changed', 'cleared', and 'errors' lists
    """
    if isinstance(modules, str):
        modules = [modules]
    if isinstance(functions, str):
        functions = [functions]

    result = {"changed": [], "cleared": [], "errors": []}

    try:
        tracker = get_function_tracker()
    except Exception as e:
        result["errors"].append(f"Failed to initialize function tracker: {e}")
        return result

    try:
        # Find changed functions
        if functions:
            # Check specific functions
            try:
                cacheable_funcs = _get_cacheable_functions()
                for func_name in functions:
                    if func_name in cacheable_funcs:
                        try:
                            if tracker.track_function(cacheable_funcs[func_name]):
                                result["changed"].append(func_name)
                        except (ReferenceError, AttributeError) as e:
                            result["errors"].append(
                                f"Function reference error for {func_name}: {e}"
                            )
                    else:
                        result["errors"].append(f"Function not found: {func_name}")
            except Exception as e:
                result["errors"].append(f"Error checking specific functions: {e}")
        else:
            # Check modules
            try:
                result["changed"] = tracker.get_changed_functions(modules)
            except Exception as e:
                result["errors"].append(f"Error checking module functions: {e}")

        # Clear cache for changed functions
        if result["changed"] and not dry_run:
            try:
                result["cleared"] = _clear_function_caches(result["changed"])
            except Exception as e:
                result["errors"].append(f"Error clearing caches: {e}")

        # Log results
        if result["changed"]:
            action = "Would clear" if dry_run else "Cleared"
            logger.info(
                "{} cache for {} changed functions: {}",
                action,
                len(result["changed"]),
                result["changed"][:3],
            )
            if len(result["changed"]) > 3:
                logger.debug("Full list: {}", result["changed"])
        else:
            logger.info("No function changes detected")

        if result["errors"]:
            logger.warning("Errors: {}", result["errors"])

    except Exception as e:
        logger.error("Selective cache clear failed: {}", e)
        result["errors"].append(str(e))

    return result


def _clear_function_caches(func_names: List[str]) -> List[str]:
    """Clear joblib cache for specific functions."""
    # Import at runtime to avoid circular import
    from . import memory

    cleared = []

    # Get joblib cache store
    cache_location = Path(memory.location)

    for func_name in func_names:
        try:
            # Joblib stores cache with function name as part of path
            # Look for directories matching the function pattern
            func_pattern = func_name.replace(".", os.sep)

            for cache_dir in cache_location.rglob("*"):
                if cache_dir.is_dir() and func_pattern in str(cache_dir):
                    # Remove the cache directory
                    import shutil

                    shutil.rmtree(cache_dir)
                    cleared.append(func_name)
                    logger.debug("Cleared cache directory: {}", cache_dir)
                    break

        except Exception as e:
            logger.warning("Failed to clear cache for {}: {}", func_name, e)

    return cleared


def smart_cacheable(func):
    """
    Enhanced @cacheable decorator that auto-tracks function changes.

    This wrapper automatically detects when a function has changed
    and invalidates its cache accordingly.
    """
    # Import at runtime to avoid circular import
    from . import cacheable

    # Apply the original cacheable decorator
    cached_func = cacheable(func)

    try:
        tracker = get_function_tracker()
        # Register the function immediately to prevent garbage collection issues
        tracker.register_function(func)
    except Exception as e:
        logger.debug("Failed to register function with tracker: {}", e)
        # Fall back to basic cacheable behavior
        return cached_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if function has changed before calling
        try:
            if tracker.track_function(func):
                # Function changed - clear its cache
                func_name = f"{func.__module__}.{func.__qualname__}"
                try:
                    cleared = _clear_function_caches([func_name])
                    if cleared:
                        logger.info("Auto-cleared changed function cache: {}", func_name)
                except Exception as e:
                    logger.debug("Auto-clear failed for {}: {}", func_name, e)
        except (ReferenceError, AttributeError) as e:
            logger.debug("Function reference error in smart_cacheable: {}", e)
        except Exception as e:
            logger.debug("Error in smart_cacheable change detection: {}", e)

        return cached_func(*args, **kwargs)

    wrapper._afml_cacheable = True
    wrapper._original_func = func
    return wrapper


def cache_maintenance(
    auto_clear_changed: bool = True,
    max_cache_size_mb: Optional[int] = None,
    max_age_days: Optional[int] = None,
) -> Dict[str, Union[int, List[str]]]:
    """
    Perform comprehensive cache maintenance.

    Args:
        auto_clear_changed: Automatically clear caches for changed functions
        max_cache_size_mb: Clear oldest caches if total size exceeds this
        max_age_days: Clear caches older than this many days

    Returns:
        Maintenance report
    """
    report = {
        "functions_checked": 0,
        "changed_functions": [],
        "cleared_functions": [],
        "size_cleared_mb": 0,
        "old_files_removed": 0,
    }

    try:
        # Auto-clear changed functions
        if auto_clear_changed:
            try:
                result = selective_cache_clear(dry_run=False)
                report["changed_functions"] = result["changed"]
                report["cleared_functions"] = result["cleared"]

                # Count functions checked from registry + GC scan
                try:
                    tracker = get_function_tracker()
                    registry_count = len(tracker.function_registry)
                    gc_count = len(_get_cacheable_functions())
                    report["functions_checked"] = max(registry_count, gc_count)
                except Exception:
                    report["functions_checked"] = len(result["changed"]) + len(result["cleared"])

            except Exception as e:
                logger.warning("Auto-clear failed during maintenance: {}", e)

        # Size-based cleanup
        if max_cache_size_mb:
            try:
                size_cleared = _cleanup_by_size(max_cache_size_mb)
                report["size_cleared_mb"] = size_cleared
            except Exception as e:
                logger.warning("Size-based cleanup failed: {}", e)

        # Age-based cleanup
        if max_age_days:
            try:
                files_removed = _cleanup_by_age(max_age_days)
                report["old_files_removed"] = files_removed
            except Exception as e:
                logger.warning("Age-based cleanup failed: {}", e)

        logger.info("Cache maintenance completed: {}", _format_maintenance_report(report))

    except Exception as e:
        logger.error("Cache maintenance failed: {}", e)
        report["error"] = str(e)

    return report


def _cleanup_by_size(max_size_mb: int) -> float:
    """Remove oldest cache files if total size exceeds limit."""
    # Import at runtime to avoid circular import
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
        except Exception:
            pass

    return removed_size


def _cleanup_by_age(max_age_days: int) -> int:
    """Remove cache files older than specified age."""
    import time

    # Import at runtime to avoid circular import
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
            except Exception:
                pass

    return removed_count


def _format_maintenance_report(report: Dict) -> str:
    """Format maintenance report for logging."""
    parts = []

    if report["functions_checked"]:
        parts.append(f"{report['functions_checked']} functions checked")

    if report["changed_functions"]:
        parts.append(f"{len(report['changed_functions'])} changed")

    if report["cleared_functions"]:
        parts.append(f"{len(report['cleared_functions'])} cleared")

    if report["size_cleared_mb"]:
        parts.append(f"{report['size_cleared_mb']:.1f}MB size-cleared")

    if report["old_files_removed"]:
        parts.append(f"{report['old_files_removed']} old files removed")

    return ", ".join(parts) if parts else "no changes"


# Convenience functions for common use cases
def clear_changed_ml_functions():
    """Clear cache for changed ML-related functions."""
    ml_modules = [
        "afml.ensemble",
        "afml.clustering",
        "afml.feature_importance",
        "afml.cross_validation",
        "afml.backtester",
    ]
    return selective_cache_clear(modules=ml_modules)


def clear_changed_labeling_functions():
    """Clear cache for changed labeling functions."""
    return selective_cache_clear(modules=["afml.labeling"])


def clear_changed_features_functions():
    """Clear cache for changed feature functions."""
    return selective_cache_clear(modules=["afml.features", "afml.strategies"])


# Export new functionality
__all__ = [
    "selective_cache_clear",
    "smart_cacheable",
    "cache_maintenance",
    "get_function_tracker",
    "clear_changed_ml_functions",
    "clear_changed_labeling_functions",
    "clear_changed_features_functions",
]
