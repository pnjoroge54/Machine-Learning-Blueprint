# app_startup.py - Complete implementation of Pattern 3

import time
from typing import Any, Dict, List, Optional

from loguru import logger

import afml


class AFMLApplication:
    """
    Application wrapper that handles MLFinLab initialization optimally.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.startup_time = None
        self.loaded_modules: List[str] = []
        self.cache_initialized = False

    def _get_default_config(self) -> Dict[str, Any]:
        """Default startup configuration"""
        return {
            "preload_core": True,
            "preload_groups": ["core_modules"],  # Load these groups at startup
            "lazy_groups": ["optional_modules"],  # Load these on-demand
            "enable_caching": True,
            "cache_warm_functions": [],  # Functions to warm cache for
            "startup_timeout": 30,  # Max startup time in seconds
            "log_performance": True,
        }

    def startup(self) -> "AFMLApplication":
        """
        Optimized startup sequence for MLFinLab.
        Call this once at application initialization.
        """
        start_time = time.perf_counter()
        logger.info("Starting MLFinLab application initialization...")

        try:
            # Step 1: Preload core modules
            if self.config.get("preload_core", False):
                self._preload_core_modules()

            # Step 2: Load configured module groups
            for group in self.config.get("preload_groups", False):
                self._load_module_group(group)

            # Step 3: Initialize caching system
            if self.config.get("enable_caching", False):
                self._initialize_caching()

            # Step 4: Warm cache if configured
            if self.config.get("cache_warm_functions", False):
                self._warm_cache()

            # Step 5: Verify critical functionality
            self._verify_startup()

            self.startup_time = time.perf_counter() - start_time

            if self.config.get("log_performance", False):
                self._log_startup_performance()

            logger.info(f"MLFinLab startup completed in {self.startup_time:.2f}s")
            return self

        except Exception as e:
            logger.error(f"MLFinLab startup failed: {e}")
            raise RuntimeError(f"Application startup failed: {e}") from e

    def _preload_core_modules(self):
        """Load essential modules that are always needed"""
        core_modules = ["data_structures", "util"]
        logger.info("Preloading core modules...")

        loaded = mlfinlab.preload_modules(*core_modules)
        self.loaded_modules.extend(loaded.keys())
        logger.debug(f"Loaded core modules: {list(loaded.keys())}")

    def _load_module_group(self, group_name: str):
        """Load a specific module group"""
        logger.info(f"Loading module group: {group_name}")

        try:
            loaded = mlfinlab.load_module_group(group_name, fail_fast=False)
            self.loaded_modules.extend(loaded.keys())
            logger.debug(f"Loaded from {group_name}: {list(loaded.keys())}")
        except ValueError as e:
            logger.warning(f"Unknown module group {group_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load group {group_name}: {e}")
            raise

    def _initialize_caching(self):
        """Initialize the caching system"""
        logger.info("Initializing caching system...")

        # Skip test modules and any problematic modules
        skip_prefixes = ["mlfinlab.tests", "mlfinlab.test_", "mlfinlab.examples"]

        mlfinlab.init_cache_plugins(skip_prefixes=skip_prefixes)
        self.cache_initialized = True
        logger.debug("Caching system initialized")

    def _warm_cache(self):
        """Warm cache for frequently used functions"""
        logger.info("Warming cache for configured functions...")

        # This would call specific functions with common parameters
        # Implementation depends on your specific use case
        for func_config in self.config["cache_warm_functions"]:
            try:
                self._warm_function_cache(func_config)
            except Exception as e:
                logger.warning(f"Failed to warm cache for {func_config}: {e}")

    def _warm_function_cache(self, func_config: Dict[str, Any]):
        """Warm cache for a specific function"""
        # Example implementation
        module_name = func_config.get("module")
        function_name = func_config.get("function")
        parameters = func_config.get("parameters", [])

        if module_name and function_name:
            module = getattr(mlfinlab, module_name)
            func = getattr(module, function_name)

            for params in parameters:
                try:
                    func(**params)
                except Exception as e:
                    logger.debug(f"Cache warming failed for {function_name}: {e}")

    def _verify_startup(self):
        """Verify that critical functionality is working"""
        logger.debug("Verifying startup integrity...")

        # Basic checks
        loaded_count = len(mlfinlab.get_loaded_modules())
        if loaded_count == 0:
            raise RuntimeError("No modules were loaded successfully")

        # Cache system check
        if self.config["enable_caching"] and not self.cache_initialized:
            raise RuntimeError("Caching system failed to initialize")

        logger.debug(f"Startup verification passed ({loaded_count} modules loaded)")

    def _log_startup_performance(self):
        """Log detailed startup performance metrics"""
        stats = {
            "startup_time": f"{self.startup_time:.2f}s",
            "modules_loaded": len(self.loaded_modules),
            "loaded_modules": self.loaded_modules,
            "cache_enabled": self.cache_initialized,
        }

        if self.cache_initialized:
            cache_stats = mlfinlab.get_cache_performance_summary()
            stats["cacheable_functions"] = cache_stats.get("functions_tracked", 0)

        logger.info(f"Startup performance: {stats}")

    def get_module(self, name: str):
        """Get a module, loading it lazily if needed"""
        return getattr(mlfinlab, name)

    def preload_additional(self, *module_names: str):
        """Preload additional modules after startup"""
        loaded = mlfinlab.preload_modules(*module_names)
        self.loaded_modules.extend(loaded.keys())
        return loaded

    def get_status(self) -> Dict[str, Any]:
        """Get current application status"""
        return {
            "startup_time": self.startup_time,
            "modules_loaded": len(mlfinlab.get_loaded_modules()),
            "cache_enabled": self.cache_initialized,
            "cache_stats": (
                mlfinlab.get_cache_performance_summary() if self.cache_initialized else None
            ),
            "loaded_modules": mlfinlab.get_loaded_modules(),
        }


# =============================================================================
# Usage Patterns and Examples
# =============================================================================


def pattern_basic_usage():
    """Basic usage pattern for simple applications"""
    # Initialize with defaults
    app = AFMLApplication()
    app.startup()

    # Now use mlfinlab normally
    data_structures = app.get_module("data_structures")
    features = app.get_module("features")  # Loaded lazily

    return app


def pattern_custom_config():
    """Usage with custom configuration"""
    config = {
        "preload_core": True,
        "preload_groups": ["core_modules", "analysis_modules"],  # Load more upfront
        "enable_caching": True,
        "log_performance": True,
        "cache_warm_functions": [
            {
                "module": "features",
                "function": "some_common_function",
                "parameters": [{"param1": "common_value1"}, {"param1": "common_value2"}],
            }
        ],
    }

    app = AFMLApplication(config)
    app.startup()

    return app


def pattern_web_application():
    """Usage pattern for web applications (FastAPI, Flask, etc.)"""

    # Global app instance
    ml_app = None

    def create_app():
        """Application factory pattern"""
        global ml_app

        config = {
            "preload_groups": ["core_modules", "analysis_modules", "ml_modules"],
            "lazy_groups": ["optional_modules"],
            "enable_caching": True,
            "log_performance": True,
        }

        ml_app = AFMLApplication(config)
        ml_app.startup()

        return ml_app

    def get_ml_app() -> AFMLApplication:
        """Get the initialized ML application"""
        if ml_app is None:
            create_app()
        return ml_app

    # Example FastAPI integration
    try:
        from fastapi import FastAPI

        def create_fastapi_app():
            fastapi_app = FastAPI()

            @fastapi_app.on_event("startup")
            async def startup_event():
                create_app()
                logger.info("MLFinLab initialized for FastAPI")

            @fastapi_app.get("/ml-status")
            async def ml_status():
                return get_ml_app().get_status()

            return fastapi_app

    except ImportError:
        logger.info("FastAPI not available, skipping FastAPI integration example")


def pattern_data_processing_pipeline():
    """Usage for data processing pipelines"""

    config = {
        "preload_groups": ["core_modules", "analysis_modules"],
        "enable_caching": True,
        "cache_warm_functions": [
            # Warm commonly used functions
            {
                "module": "data_structures",
                "function": "get_daily_vol",
                "parameters": [{"span": 100}, {"span": 252}],
            }
        ],
    }

    app = AFMLApplication(config)
    app.startup()

    # Processing pipeline
    def process_data(data):
        # Use preloaded modules
        ds = app.get_module("data_structures")
        features = app.get_module("features")  # Lazy loaded

        # Process with cached functions
        processed = ds.some_function(data)  # Benefits from cache
        features_extracted = features.extract_features(processed)

        return features_extracted

    return app, process_data


def pattern_jupyter_notebook():
    """Usage pattern for Jupyter notebooks"""

    def setup_notebook_environment():
        """Call this at the top of your notebook"""
        config = {
            "preload_groups": ["core_modules", "analysis_modules"],
            "enable_caching": True,
            "log_performance": False,  # Less verbose for notebooks
        }

        app = AFMLApplication(config)
        app.startup()

        # Make modules easily accessible in notebook
        globals().update(
            {
                "ds": app.get_module("data_structures"),
                "features": app.get_module("features"),
                "labeling": app.get_module("labeling"),
                "utils": app.get_module("util"),
                "ml_app": app,
            }
        )

        print("✅ MLFinLab environment ready!")
        print(f"📊 Loaded modules: {', '.join(app.loaded_modules)}")

        return app

    return setup_notebook_environment


# =============================================================================
# Production Deployment Example
# =============================================================================


def production_startup():
    """Production-ready startup configuration"""

    # Production config - optimized for performance and reliability
    config = {
        "preload_core": True,
        "preload_groups": ["core_modules", "analysis_modules", "ml_modules"],
        "enable_caching": True,
        "log_performance": True,
        "startup_timeout": 60,  # Allow more time for production startup
        "cache_warm_functions": [
            # Add your most commonly used functions here
        ],
    }

    try:
        app = AFMLApplication(config)
        app.startup()

        # Health check
        status = app.get_status()
        if status["modules_loaded"] < 5:  # Adjust threshold as needed
            raise RuntimeError("Too few modules loaded for production")

        logger.info("✅ Production MLFinLab startup successful")
        return app

    except Exception as e:
        logger.error(f"❌ Production startup failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("=== Pattern 3 App Startup Examples ===\n")

    # Basic usage
    print("1. Basic Usage:")
    app1 = pattern_basic_usage()
    print(f"   Status: {app1.get_status()}\n")

    # Custom config
    print("2. Custom Config:")
    app2 = pattern_custom_config()
    print(f"   Loaded modules: {app2.loaded_modules}\n")

    # Jupyter setup
    print("3. Jupyter Notebook Setup:")
    setup_nb = pattern_jupyter_notebook()
    app3 = setup_nb()
    print()

    # Production example
    print("4. Production Startup:")
    app4 = production_startup()
    print(f"   Production ready: {len(app4.loaded_modules)} modules loaded")
    print(f"   Production ready: {len(app4.loaded_modules)} modules loaded")
    print(f"   Production ready: {len(app4.loaded_modules)} modules loaded")
