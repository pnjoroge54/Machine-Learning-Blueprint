# usage_examples.py - How to use the improved dynamic module loading

import afml

# =============================================================================
# Example 1: Lazy Loading (Default Behavior)
# =============================================================================


def lazy_loading_example():
    """Demonstrate lazy loading - modules loaded only when accessed"""
    print("=== Lazy Loading Example ===")

    # Initially, no modules are loaded
    print(f"Initially loaded modules: {afml.get_loaded_modules()}")

    # Access a module - this triggers loading
    print("Accessing data_structures module...")
    ds_module = afml.data_structures
    print(f"Loaded modules after accessing data_structures: {afml.get_loaded_modules()}")

    # Access another module
    print("Accessing features module...")
    features_module = afml.features
    print(f"Loaded modules after accessing features: {afml.get_loaded_modules()}")

    # Subsequent access uses cached version (no additional loading)
    ds_module_again = afml.data_structures
    print(f"Same module instance: {ds_module is ds_module_again}")


# =============================================================================
# Example 2: Preloading Modules
# =============================================================================


def preloading_example():
    """Demonstrate preloading modules for better performance"""
    print("\n=== Preloading Example ===")

    # Preload specific modules you know you'll need
    print("Preloading core modules...")
    loaded = afml.preload_modules("data_structures", "util", "datasets")
    print(f"Preloaded modules: {list(loaded.keys())}")
    print(f"All loaded modules: {afml.get_loaded_modules()}")

    # Now accessing these modules is instant (already loaded)
    util_module = afml.util  # No loading time
    datasets_module = afml.datasets  # No loading time


# =============================================================================
# Example 3: Group-based Loading
# =============================================================================


def group_loading_example():
    """Demonstrate loading modules by functional groups"""
    print("\n=== Group Loading Example ===")

    # Load all ML-related modules at once
    print("Loading ML modules group...")
    ml_modules = afml.load_module_group("ml_modules")
    print(f"Loaded ML modules: {list(ml_modules.keys())}")

    # Load portfolio-related modules
    print("Loading portfolio modules group...")
    portfolio_modules = afml.load_module_group("portfolio_modules")
    print(f"Loaded portfolio modules: {list(portfolio_modules.keys())}")

    print(f"Total loaded modules: {afml.get_loaded_modules()}")


# =============================================================================
# Example 4: Error Handling and Fallbacks
# =============================================================================


def error_handling_example():
    """Demonstrate error handling in module loading"""
    print("\n=== Error Handling Example ===")

    # Try to access a non-existent module
    try:
        non_existent = afml.non_existent_module
    except AttributeError as e:
        print(f"Expected error for non-existent module: {e}")

    # Load optional modules with graceful failure
    try:
        optional_modules = afml.load_module_group("optional_modules", fail_fast=False)
        print(f"Successfully loaded optional modules: {list(optional_modules.keys())}")
    except Exception as e:
        print(f"Error loading optional modules: {e}")


# =============================================================================
# Example 5: Module Reloading (Development/Testing)
# =============================================================================


def module_reloading_example():
    """Demonstrate module reloading for development"""
    print("\n=== Module Reloading Example ===")

    # Load a module
    original_module = afml.util
    print(f"Original module: {id(original_module)}")

    # Reload it (useful during development)
    reloaded_module = afml.reload_module("util")
    print(f"Reloaded module: {id(reloaded_module)}")

    # Access it again - should get the reloaded version
    current_module = afml.util
    print(f"Current module: {id(current_module)}")
    print(f"Reloaded successfully: {current_module is reloaded_module}")


# =============================================================================
# Example 6: Performance Monitoring
# =============================================================================


def performance_monitoring_example():
    """Demonstrate monitoring module loading performance"""
    print("\n=== Performance Monitoring Example ===")

    import time

    # Time lazy loading
    start_time = time.time()
    labeling_module = afml.strategies
    lazy_time = time.time() - start_time
    print(f"Lazy loading time: {lazy_time:.4f} seconds")

    # Time cached access
    start_time = time.time()
    labeling_module_cached = afml.strategies
    cached_time = time.time() - start_time
    print(f"Cached access time: {cached_time:.6f} seconds")
    print(f"Speedup: {lazy_time / cached_time:.1f}x")


# =============================================================================
# Example 7: Integration with Caching System
# =============================================================================


def caching_integration_example():
    """Show how module loading works with the caching system"""
    print("\n=== Caching Integration Example ===")

    # Load modules and initialize caching
    afml.preload_modules("features", "data_structures", "util", "labeling")

    # Initialize cache plugins (this scans all loaded modules)
    afml.init_cache_plugins()

    # Show cache performance summary
    summary = afml.get_cache_performance_summary()
    print(f"Cache system initialized with {summary.get('functions_tracked', 0)} tracked functions")

    # Use a cached function (if available)
    # This would depend on your actual module implementations
    print("Modules available for cached operations:")
    for module_name in afml.get_loaded_modules():
        print(f"  - {module_name}")


# =============================================================================
# Example 8: Production Optimization Pattern
# =============================================================================


def production_optimization_example():
    """Show optimal pattern for production use"""
    print("\n=== Production Optimization Pattern ===")

    # Strategy 1: Preload only what you need
    critical_modules = ["data_structures", "util", "features", "labeling", "filters"]
    loaded = afml.preload_modules(*critical_modules)
    print(f"Preloaded critical modules: {list(loaded.keys())}")

    # Strategy 2: Load by functional groups
    analysis_modules = afml.load_module_group("analysis_modules")
    print(f"Loaded analysis modules: {list(analysis_modules.keys())}")

    # Strategy 3: Initialize caching for loaded modules only
    afml.init_cache_plugins()

    # Strategy 4: Monitor performance
    print(f"Total modules loaded: {len(afml.get_loaded_modules())}")
    print("Ready for high-performance operations!")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    lazy_loading_example()
    preloading_example()
    group_loading_example()
    error_handling_example()
    module_reloading_example()
    # performance_monitoring_example()
    caching_integration_example()
    production_optimization_example()

    print(f"\n=== Final State ===")
    print(f"Total loaded modules: {len(afml.get_loaded_modules())}")
    print(f"Loaded modules: {afml.get_loaded_modules()}")

# # =============================================================================
# # Alternative usage patterns
# # =============================================================================

# # Pattern 1: Import specific modules upfront
# def pattern_1():
#     from afml import data_structures, features, util
#     # These are loaded immediately when imported

# # Pattern 2: Conditional loading
# def pattern_2():
#     import afml

#     if some_condition():
#         heavy_module = afml.portfolio_optimization  # Only load if needed

#     # Always load lightweight utilities
#     afml.preload_modules("util", "data_structures")

# # Pattern 3: Application startup optimization
# def pattern_3_app_startup():
#     import afml

#     # Load core modules at startup
#     afml.load_module_group("core_modules")

#     # Initialize caching
#     afml.init_cache_plugins()

#     # Lazy load everything else as needed
#     return afml
