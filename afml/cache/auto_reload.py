# Enhanced caching system with automatic change detection

import importlib
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Set

from loguru import logger

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not available - install with: pip install watchdog")


class AutoReloadingCache:
    """Cache system that automatically reloads functions when files change."""

    def __init__(self, watch_paths: Optional[list] = None):
        self.function_registry: Dict[str, callable] = {}
        self.module_paths: Dict[str, Path] = {}
        self.observer = None
        self.watched_paths: Set[Path] = set()

        if watch_paths:
            self.start_watching(watch_paths)

    def start_watching(self, paths: list):
        """Start watching specified paths for changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File watching not available - falling back to periodic checks")
            return

        if self.observer:
            self.stop_watching()

        self.observer = Observer()
        handler = FileChangeHandler(self)

        for path_str in paths:
            path = Path(path_str).resolve()
            if path.exists():
                self.observer.schedule(handler, str(path), recursive=True)
                self.watched_paths.add(path)
                logger.info("Watching for changes: {}", path)

        self.observer.start()

    def stop_watching(self):
        """Stop file watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

    def register_function(self, func):
        """Register a function for auto-reloading."""
        func_name = f"{func.__module__}.{func.__qualname__}"
        self.function_registry[func_name] = func

        # Track the module file
        try:
            module_file = Path(func.__code__.co_filename)
            self.module_paths[func.__module__] = module_file
        except (AttributeError, OSError):
            pass

    def reload_module(self, module_name: str):
        """Reload a specific module."""
        try:
            if module_name in sys.modules:
                logger.info("Reloading module: {}", module_name)
                importlib.reload(sys.modules[module_name])

                # Update function references
                self._update_function_references(module_name)

                # Clear related caches
                self._clear_module_caches(module_name)

        except Exception as e:
            logger.error("Failed to reload module {}: {}", module_name, e)

    def _update_function_references(self, module_name: str):
        """Update function references after module reload."""
        module = sys.modules.get(module_name)
        if not module:
            return

        for func_name in list(self.function_registry.keys()):
            if func_name.startswith(module_name + "."):
                try:
                    # Get the function name without module prefix
                    attr_name = func_name.split(".")[-1]
                    new_func = getattr(module, attr_name, None)

                    if new_func and callable(new_func):
                        self.function_registry[func_name] = new_func
                        logger.debug("Updated function reference: {}", func_name)

                except AttributeError:
                    pass

    def _clear_module_caches(self, module_name: str):
        """Clear caches for functions in a reloaded module."""
        from . import memory  # Import your existing memory instance

        try:
            # Clear joblib caches for this module
            cache_location = Path(memory.location)
            module_pattern = module_name.replace(".", "/")

            for cache_dir in cache_location.rglob("*"):
                if cache_dir.is_dir() and module_pattern in str(cache_dir):
                    import shutil

                    shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.debug("Cleared cache for module: {}", module_name)

        except Exception as e:
            logger.debug("Cache clear failed for {}: {}", module_name, e)


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system events for auto-reloading."""

    def __init__(self, cache_system: AutoReloadingCache):
        self.cache_system = cache_system
        self._debounce_timer = None
        self._changed_files = set()

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix == ".py":
            self._changed_files.add(file_path)
            self._debounce_reload()

    def _debounce_reload(self):
        """Debounce rapid file changes to avoid excessive reloads."""
        if self._debounce_timer:
            self._debounce_timer.cancel()

        self._debounce_timer = threading.Timer(0.5, self._process_changes)
        self._debounce_timer.start()

    def _process_changes(self):
        """Process accumulated file changes."""
        for file_path in self._changed_files:
            self._handle_file_change(file_path)

        self._changed_files.clear()

    def _handle_file_change(self, file_path: Path):
        """Handle a single file change."""
        # Find which modules might be affected
        affected_modules = []

        for module_name, module_path in self.cache_system.module_paths.items():
            if module_path == file_path:
                affected_modules.append(module_name)

        # Reload affected modules
        for module_name in affected_modules:
            self.cache_system.reload_module(module_name)


# Global auto-reloading cache instance
auto_cache = None


def get_auto_cache():
    """Get or create the global auto-reloading cache."""
    global auto_cache
    if auto_cache is None:
        # Default paths - adjust for your project structure
        default_paths = [
            "afml/",  # Your main package
            ".",  # Current directory
        ]
        auto_cache = AutoReloadingCache(default_paths)
    return auto_cache


def auto_cacheable(func):
    """Enhanced cacheable decorator with auto-reloading."""
    from . import cacheable  # Your existing cacheable decorator

    # Apply original caching
    cached_func = cacheable(func)

    # Register with auto-reload system
    cache_system = get_auto_cache()
    cache_system.register_function(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return cached_func(*args, **kwargs)

    wrapper._afml_cacheable = True
    wrapper._auto_reloadable = True
    return wrapper


# Periodic checking fallback (when watchdog isn't available)
class PeriodicChecker:
    """Fallback periodic checking for environments without file watching."""

    def __init__(self, check_interval: float = 2.0):
        self.check_interval = check_interval
        self.file_mtimes: Dict[Path, float] = {}
        self.running = False
        self.thread = None

    def start(self, paths: list):
        """Start periodic checking."""
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, args=(paths,))
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started periodic file checking ({}s interval)", self.check_interval)

    def stop(self):
        """Stop periodic checking."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _check_loop(self, paths: list):
        """Main checking loop."""
        while self.running:
            try:
                self._check_files(paths)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.debug("Error in periodic check: {}", e)

    def _check_files(self, paths: list):
        """Check files for modifications."""
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                continue

            for py_file in path.rglob("*.py"):
                try:
                    current_mtime = py_file.stat().st_mtime

                    if py_file in self.file_mtimes:
                        if current_mtime > self.file_mtimes[py_file]:
                            logger.info("File changed (periodic check): {}", py_file)
                            self._handle_file_change(py_file)

                    self.file_mtimes[py_file] = current_mtime

                except OSError:
                    pass

    def _handle_file_change(self, file_path: Path):
        """Handle file change detection."""
        # Trigger selective cache clear
        try:
            from . import selective_cache_clear

            result = selective_cache_clear()
            if result["cleared"]:
                logger.info("Auto-cleared {} functions after file change", len(result["cleared"]))
        except Exception as e:
            logger.debug("Auto-clear failed: {}", e)


# Usage examples and setup functions
def setup_auto_reloading(watch_paths: Optional[list] = None, use_periodic_fallback: bool = True):
    """Setup automatic cache reloading."""
    paths = watch_paths or ["afml/", "."]

    cache_system = get_auto_cache()

    if WATCHDOG_AVAILABLE:
        cache_system.start_watching(paths)
    elif use_periodic_fallback:
        periodic = PeriodicChecker()
        periodic.start(paths)
        logger.info("Using periodic file checking (install watchdog for better performance)")
    else:
        logger.warning("No auto-reloading available - install watchdog or enable periodic checking")

    return cache_system


def jupyter_auto_setup():
    """Convenient setup for Jupyter notebooks."""
    return setup_auto_reloading(watch_paths=[".", "afml/"])


# Export the new functionality
__all__ = [
    "auto_cacheable",
    "setup_auto_reloading",
    "jupyter_auto_setup",
    "get_auto_cache",
    "AutoReloadingCache",
]
