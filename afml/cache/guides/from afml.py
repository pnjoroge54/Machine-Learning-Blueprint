from afml.cache import (
    cached_backtest,
    get_comprehensive_cache_status,
    mlflow_cached,
    print_cache_health,
    robust_cacheable,
    setup_production_cache,
    time_aware_cacheable,
)


@robust_cacheable
def my_function(df):
    # Expensive computation
    return result
