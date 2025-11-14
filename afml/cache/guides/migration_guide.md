# Migration Guide: Upgrading Your AFML Cache System

Step-by-step guide to integrate the enhanced caching features into your existing codebase.

## Overview

All enhancements are **backward compatible**. You can:

- Use new features immediately without breaking existing code
- Migrate gradually, module by module
- Keep using `@cacheable` while adopting `@robust_cacheable` for new code

---

## Installation Steps

### Step 1: Add New Files

Copy these new files to your `afml/cache/` directory:

```
afml/cache/
├── __init__.py                 # UPDATE (add new imports)
├── robust_cache_keys.py        # NEW
├── mlflow_integration.py       # NEW
├── backtest_cache.py           # NEW
├── cache_monitoring.py         # NEW
├── selective_cleaner.py        # EXISTING (no changes needed)
├── auto_reload.py              # EXISTING (no changes needed)
└── ...
```

### Step 2: Update `__init__.py`

Replace your `afml/cache/__init__.py` with the updated version, or manually add these imports:

```python
# Add after existing imports

# Robust cache keys
from .robust_cache_keys import (
    robust_cacheable,
    time_aware_cacheable,
)

# MLflow integration (optional)
try:
    from .mlflow_integration import (
        setup_mlflow_cache,
        mlflow_cached,
        MLFLOW_AVAILABLE,
    )
except ImportError:
    MLFLOW_AVAILABLE = False

# Backtest caching
from .backtest_cache import (
    BacktestCache,
    cached_backtest,
    get_backtest_cache,
)

# Cache monitoring
from .cache_monitoring import (
    get_cache_monitor,
    print_cache_health,
    get_cache_efficiency_report,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    "robust_cacheable",
    "time_aware_cacheable",
    "setup_mlflow_cache",
    "mlflow_cached",
    "cached_backtest",
    "get_cache_monitor",
    "print_cache_health",
]
```

### Step 3: Install Dependencies (Optional)

For MLflow integration:

```bash
pip install mlflow
```

For better performance:

```bash
pip install pyarrow  # For fast parquet in backtest cache
```

---

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

**Keep existing code working, add enhancements incrementally.**

#### Phase 1: Add Monitoring (No code changes)

```python
# Add to your startup script or notebook
from afml.cache import print_cache_health

# Check cache health anytime
print_cache_health()
```

**Benefits:** Immediate visibility into cache performance. No changes to existing code.

#### Phase 2: Upgrade High-Value Functions

Identify functions that would benefit most:

```python
from afml.cache import get_cache_efficiency_report

# Find functions with low hit rates or high call counts
df = get_cache_efficiency_report()
print(df.sort_values('calls', ascending=False).head(10))
```

Upgrade these functions one-by-one:

```python
# BEFORE
from afml.cache import cacheable

@cacheable
def compute_features(df, params):
    # ...
    
# AFTER
from afml.cache import robust_cacheable

@robust_cacheable  # Just change the decorator!
def compute_features(df, params):
    # Same code, better caching
```

#### Phase 3: Add MLflow for New Projects

```python
# For new workflows, use MLflow tracking
from afml.cache import setup_mlflow_cache, mlflow_cached

# Setup once
setup_mlflow_cache("my_new_project")

# Use on new training functions
@mlflow_cached(tags={"version": "v1"})
def train_new_model(data, params):
    # Automatically tracked + cached
    return model, metrics
```

#### Phase 4: Add Backtest Caching

```python
# Wrap existing backtest functions
from afml.cache import cached_backtest

@cached_backtest("my_strategy")
def run_my_backtest(data, params):
    # Existing backtest code unchanged
    return metrics, trades, equity
```

### Strategy 2: Fresh Start (New Projects)

**For new projects, use the complete enhanced system from day 1.**

```python
# project_startup.py

from afml.cache import setup_production_cache

# Initialize everything
components = setup_production_cache(
    enable_mlflow=True,
    mlflow_experiment="my_project",
    max_cache_size_mb=2000,
)

print("✅ Cache system ready")
```

Then use enhanced decorators throughout:

```python
from afml.cache import (
    time_aware_cacheable,
    robust_cacheable,
    mlflow_cached,
    cached_backtest,
)

@time_aware_cacheable
def load_data(symbol, start, end):
    # ...

@robust_cacheable
def compute_features(df, params):
    # ...

@mlflow_cached(tags={"model": "rf"})
def train_model(features, labels, params):
    # ...

@cached_backtest("strategy_v1")
def backtest(data, model, params):
    # ...
```

---

## Backward Compatibility Testing

### Verify Nothing Breaks

```python
# test_backward_compatibility.py

from afml.cache import cacheable, get_cache_stats

# Your existing cached function
@cacheable
def my_existing_function(x, y):
    return x + y

# Test it still works
result1 = my_existing_function(5, 10)
result2 = my_existing_function(5, 10)  # Should use cache

assert result1 == result2 == 15

# Verify cache is working
stats = get_cache_stats()
assert 'my_existing_function' in str(stats)

print("✅ Backward compatibility verified")
```

---

## Common Migration Patterns

### Pattern 1: DataFrame-Heavy Functions

**Problem:** Functions with DataFrames have low cache hit rates.

```python
# BEFORE (weak cache keys)
@cacheable
def process_dataframe(df, window):
    return df.rolling(window).mean()

# Different DataFrame objects = cache miss even with same data

# AFTER (robust cache keys)
@robust_cacheable
def process_dataframe(df, window):
    return df.rolling(window).mean()

# Same data = cache hit, even if different objects
```

### Pattern 2: Time-Series Functions

**Problem:** Functions don't distinguish between date ranges.

```python
# BEFORE (caches all date ranges together)
@cacheable
def compute_indicators(prices, window):
    return prices.rolling(window).mean()

# AFTER (date-range aware)
@time_aware_cacheable
def compute_indicators(prices, window):
    return prices.rolling(window).mean()

# Different date ranges = different cache entries
```

### Pattern 3: Expensive Labeling

**Problem:** Triple-barrier labeling takes hours, slows iteration.

```python
# BEFORE (no caching)
def triple_barrier_labels(prices, barriers):
    # 2 hours of computation
    return labels

# AFTER (add caching with one line)
@robust_cacheable
def triple_barrier_labels(prices, barriers):
    # First call: 2 hours
    # Subsequent calls: <1 second
    return labels
```

### Pattern 4: Model Training

**Problem:** No tracking of experiments, hard to reproduce.

```python
# BEFORE (no tracking)
@cacheable
def train_model(features, labels, params):
    model = RandomForest(**params)
    model.fit(features, labels)
    return model

# AFTER (tracked + cached)
@mlflow_cached(tags={"model": "rf"})
def train_model(features, labels, params):
    model = RandomForest(**params)
    model.fit(features, labels)
    
    metrics = {
        'accuracy': model.score(features, labels)
    }
    
    return model, metrics  # Automatically logged to MLflow
```

### Pattern 5: Parameter Optimization

**Problem:** Testing 100 parameter combinations takes days.

```python
# BEFORE (recomputes everything each time)
for params in param_grid:
    metrics = run_backtest(data, params)
    results.append(metrics)

# AFTER (caches each combination)
@cached_backtest("strategy_name")
def run_backtest(data, params):
    # Backtest logic
    return metrics, trades, equity

# First run: hours
# Adding more parameters: only new ones computed
# Resuming after crash: picks up where left off
```

---

## Troubleshooting Migration Issues

### Issue 1: Import Errors

**Problem:**

```python
ImportError: cannot import name 'robust_cacheable' from 'afml.cache'
```

**Solution:**

```bash
# Make sure new files are in place
ls afml/cache/

# Should see:
# robust_cache_keys.py
# mlflow_integration.py
# backtest_cache.py
# cache_monitoring.py
```

### Issue 2: MLflow Not Available

**Problem:**

```python
WARNING: MLflow not available - install with: pip install mlflow
```

**Solution A (Install MLflow):**

```bash
pip install mlflow
```

**Solution B (Use without MLflow):**

```python
# System still works, just without MLflow tracking
from afml.cache import MLFLOW_AVAILABLE

if MLFLOW_AVAILABLE:
    setup_mlflow_cache("my_experiment")
else:
    print("Using local caching only")
```

### Issue 3: Cache Performance Degradation

**Problem:** Cache hit rate dropped after migration.

**Solution:**

```python
# Diagnose the issue
from afml.cache import analyze_cache_patterns

patterns = analyze_cache_patterns()

if patterns['high_miss_rate_functions']:
    print("Functions with low hit rates:")
    for func in patterns['high_miss_rate_functions']:
        print(f"  {func['function']}: {func['hit_rate']:.1%}")
        
# Fix by using robust_cacheable for these functions
```

### Issue 4: Cache Directory Conflicts

**Problem:** Multiple cache systems interfering.

**Solution:**

```python
# Clear and rebuild
from afml.cache import clear_afml_cache, initialize_cache_system

clear_afml_cache(warn=True)
initialize_cache_system()

# Verify clean state
from afml.cache import get_cache_summary
print(get_cache_summary())
```

---

## Testing Your Migration

### Test Suite

```python
# tests/test_cache_migration.py

import pytest
import pandas as pd
import numpy as np
from afml.cache import (
    cacheable,
    robust_cacheable,
    time_aware_cacheable,
    get_cache_stats,
    clear_afml_cache,
)

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test."""
    clear_afml_cache(warn=False)
    yield
    clear_afml_cache(warn=False)

def test_existing_cacheable_still_works():
    """Verify @cacheable decorator still works."""
    @cacheable
    def simple_func(x, y):
        return x + y
    
    result1 = simple_func(5, 10)
    result2 = simple_func(5, 10)
    
    assert result1 == result2 == 15
    
    stats = get_cache_stats()
    assert len(stats) > 0

def test_robust_cacheable_with_dataframes():
    """Verify @robust_cacheable handles DataFrames correctly."""
    @robust_cacheable
    def process_df(df):
        return df.sum().sum()
    
    df1 = pd.DataFrame({'a': [1, 2, 3]})
    df2 = pd.DataFrame({'a': [1, 2, 3]})  # Same data, different object
    
    result1 = process_df(df1)
    result2 = process_df(df2)  # Should be cache hit
    
    assert result1 == result2 == 6

def test_time_aware_cacheable():
    """Verify @time_aware_cacheable handles date ranges."""
    @time_aware_cacheable
    def process_time_series(df):
        return df['value'].mean()
    
    # Create two DataFrames with different date ranges
    dates1 = pd.date_range('2024-01-01', periods=10)
    dates2 = pd.date_range('2024-02-01', periods=10)
    
    df1 = pd.DataFrame({'value': range(10)}, index=dates1)
    df2 = pd.DataFrame({'value': range(10)}, index=dates2)
    
    result1 = process_time_series(df1)
    result2 = process_time_series(df2)
    
    # Different date ranges = both should compute (not cache hit)
    assert result1 == result2 == 4.5

def test_cache_monitoring():
    """Verify monitoring features work."""
    from afml.cache import get_cache_monitor
    
    @cacheable
    def monitored_func(x):
        return x * 2
    
    # Generate some activity
    for i in range(10):
        monitored_func(i)
    
    # Check monitoring
    monitor = get_cache_monitor()
    stats = monitor.get_all_function_stats()
    
    assert len(stats) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run tests:

```bash
pytest tests/test_cache_migration.py -v
```

---

## Performance Validation

### Benchmark Script

```python
# benchmark_migration.py

import time
import pandas as pd
import numpy as np
from afml.cache import (
    cacheable,
    robust_cacheable,
    clear_afml_cache,
)

def benchmark_decorator(decorator_name, decorator, func, args):
    """Benchmark a specific decorator."""
    # Clear cache
    clear_afml_cache(warn=False)
    
    # Apply decorator
    cached_func = decorator(func)
    
    # First call (cache miss)
    start = time.time()
    result1 = cached_func(*args)
    first_call_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = cached_func(*args)
    second_call_time = time.time() - start
    
    speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
    
    return {
        'decorator': decorator_name,
        'first_call_ms': first_call_time * 1000,
        'second_call_ms': second_call_time * 1000,
        'speedup': f"{speedup:.0f}x",
    }

# Test function
def expensive_computation(df):
    """Simulate expensive computation."""
    time.sleep(0.1)  # Simulate work
    return df.sum().sum()

# Create test data
test_df = pd.DataFrame(np.random.randn(1000, 10))

# Benchmark
results = []

results.append(benchmark_decorator(
    "cacheable (original)",
    cacheable,
    expensive_computation,
    (test_df,)
))

results.append(benchmark_decorator(
    "robust_cacheable (new)",
    robust_cacheable,
    expensive_computation,
    (test_df,)
))

# Print results
print("\n" + "=" * 70)
print("CACHE DECORATOR BENCHMARK")
print("=" * 70)
for r in results:
    print(f"\n{r['decorator']}:")
    print(f"  First call:  {r['first_call_ms']:.1f} ms")
    print(f"  Second call: {r['second_call_ms']:.1f} ms")
    print(f"  Speedup:     {r['speedup']}")
print("\n" + "=" * 70)
```

---

## Rollback Plan

If you encounter issues:

### Step 1: Revert `__init__.py`

```bash
# Restore from git
git checkout afml/cache/__init__.py

# Or remove new imports manually
```

### Step 2: Remove New Files (Optional)

```bash
cd afml/cache/
rm robust_cache_keys.py
rm mlflow_integration.py
rm backtest_cache.py
rm cache_monitoring.py
```

### Step 3: Clear Cache

```python
from afml.cache import clear_afml_cache
clear_afml_cache(warn=True)
```

### Step 4: Restart Python

```python
# Exit and restart your Python session
exit()

# Start fresh
python
```

---

## Success Criteria

Migration is successful when:

✅ All existing tests pass
✅ Cache hit rates are maintained or improved  
✅ No performance degradation in cached calls  
✅ Monitoring dashboard shows healthy metrics  
✅ MLflow integration works (if enabled)  
✅ Backtest cache speeds up parameter optimization  

---

## Getting Help

If you encounter issues:

1. **Check logs:**

   ```python
   from loguru import logger
   logger.add("cache_debug.log", level="DEBUG")
   ```

2. **Analyze cache:**

   ```python
   from afml.cache import print_cache_health
   print_cache_health()
   ```

3. **Report patterns:**

   ```python
   from afml.cache import analyze_cache_patterns
   patterns = analyze_cache_patterns()
   print(patterns)
   ```

4. **Open issue:** Include cache stats and log files

---

## Next Steps After Migration

1. **Monitor performance:**
   - Run `print_cache_health()` weekly
   - Check for low hit rates
   - Identify optimization opportunities

2. **Maintain cache:**
   - Schedule monthly cleanup
   - Remove old backtests
   - Archive MLflow experiments

3. **Optimize further:**
   - Profile remaining slow functions
   - Add caching strategically
   - Tune cache size limits

4. **Document your setup:**
   - Record decorator choices
   - Note cache size requirements
   - Track performance improvements

---

## Summary

The migration is designed to be **gradual, safe, and backward-compatible**:

- ✅ Keep existing `@cacheable` decorators working
- ✅ Add new features incrementally
- ✅ Monitor performance continuously
- ✅ Rollback if needed

Start with monitoring, then upgrade high-value functions, and finally add MLflow and backtest caching for new workflows.
