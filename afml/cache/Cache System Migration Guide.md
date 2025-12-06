# Cache System Migration Guide

## 🎯 TL;DR - What Changed

**Auto-versioning is now ENABLED BY DEFAULT.**

Your cache will automatically invalidate when function code changes. This prevents stale cache bugs.

**Most users need to do nothing** - just update and enjoy automatic cache invalidation.

**Only opt-out if:**

- Function takes hours/days to compute AND
- Function is stable/won't change AND  
- You understand the risk of stale results

---

## Overview of Changes

1. **`auto_versioning=True` by default**: Cache keys include function source hash
2. **One decorator to rule them all**: `@cacheable()` replaces multiple decorators
3. **Removed `smart_cacheable`**: Now redundant (built into default behavior)
4. **Selective cleaner refocused**: Maintenance tool for orphaned caches

---

## Quick Migration Table

| Old Code | New Code | Notes |
|----------|----------|-------|
| `@robust_cacheable` | `@cacheable()` | Now has auto-versioning by default |
| `@time_aware_cacheable` | `@cacheable(time_aware=True)` | Now has auto-versioning by default |
| `@cv_cacheable` | `@cacheable()` | Now has auto-versioning by default |
| `@smart_cacheable` | `@cacheable()` | **REMOVED - now default behavior** |
| `@cacheable()` (old) | `@cacheable(auto_versioning=False)` | **Only if you need old behavior** |

---

## What is Auto-Versioning?

### The Problem It Solves

```python
# Without auto-versioning
@cacheable(auto_versioning=False)
def calculate_returns(prices):
    return prices.pct_change()

calculate_returns(df)  # Cache miss, stores result
calculate_returns(df)  # Cache hit ✓

# Developer fixes a bug
def calculate_returns(prices):
    return prices.pct_change().fillna(0)  # Bug fix!

calculate_returns(df)  # Cache HIT - WRONG RESULT! ❌
# Returns OLD buggy result from cache
```

### With Auto-Versioning (Now Default)

```python
# With auto-versioning (NEW DEFAULT)
@cacheable()  # auto_versioning=True by default
def calculate_returns(prices):
    return prices.pct_change()

calculate_returns(df)  # Cache miss, stores at key "v_abc123..."
calculate_returns(df)  # Cache hit ✓

# Developer fixes bug
def calculate_returns(prices):
    return prices.pct_change().fillna(0)  # Bug fix!

calculate_returns(df)  # Cache MISS - new key "v_def456..." ✓
# Computes with NEW correct code
```

---

## Migration Steps

### Step 1: Update `smart_cacheable` (REQUIRED)

**Old code:**

```python
from afml.cache import smart_cacheable

@smart_cacheable
def my_function(data):
    return data.mean()
```

**New code:**

```python
from afml.cache import cacheable

@cacheable()  # That's it! auto_versioning is now default
def my_function(data):
    return data.mean()
```

### Step 2: Review Expensive Functions (OPTIONAL)

If you have functions that take **hours to compute** and **rarely change**:

```python
@cacheable(auto_versioning=False)  # Explicit opt-out
def train_huge_model(data):
    """Takes 48 hours, changes once per year"""
    return expensive_training(data)
```

⚠️ **Warning**: With `auto_versioning=False`, adding a comment invalidates cache:

```python
@cacheable(auto_versioning=False)
def train_huge_model(data):
    """Added this docstring"""  # THIS CHANGE WON'T INVALIDATE CACHE
    return expensive_training(data)  # May return stale result!
```

### Step 3: Clean Up Old Caches (RECOMMENDED)

After migration, clean up orphaned caches:

```python
from afml.cache import cache_maintenance

# One-time cleanup after migration
cache_maintenance(
    clean_orphaned=True,
    max_cache_size_mb=1000,
    max_age_days=30
)
```

---

## Understanding Auto-Versioning Behavior

### How Cache Keys Work

**Without auto-versioning:**

```
cache_key = md5("module.function_name" + "arg_hashes")
           = "a1b2c3d4..."
```

**With auto-versioning (default):**

```
cache_key = md5("module.function_name" + "v_abc123" + "arg_hashes")
                                         ^^^^^^^^^^
                                    function source hash
           = "e5f6g7h8..."  # Different key!
```

### When Cache Invalidates

Cache invalidates when:

- ✅ Function body changes
- ✅ Function name changes  
- ✅ Default parameters change
- ✅ Decorators change
- ❌ Comments change (graceful: uses file mtime as fallback)
- ❌ Docstrings change (graceful: uses file mtime as fallback)

### Graceful Fallback

For built-in/dynamic functions where source is unavailable:

```python
# Can't get source for built-ins
import numpy as np

@cacheable()  # Gracefully falls back to file mtime
def use_builtin(data):
    return np.mean(data)  # np.mean has no source

# Warning logged, but doesn't crash
```

---

## Common Scenarios

### Scenario 1: Development (Default - No Changes Needed)

```python
from afml.cache import cacheable

@cacheable()  # Just use defaults!
def my_feature(data, window):
    """Feature under active development"""
    return data.rolling(window).mean()

# Work normally - cache auto-invalidates on changes
result1 = my_feature(df, 20)
result2 = my_feature(df, 20)  # Cache hit

# ... modify my_feature ...

result3 = my_feature(df, 20)  # Cache miss (automatic!)
```

### Scenario 2: Expensive Computation (Explicit Opt-Out)

```python
from afml.cache import cacheable

@cacheable(auto_versioning=False)  # Explicit opt-out
def train_production_model(data):
    """Takes 24 hours, changes rarely, want to preserve cache"""
    return expensive_training(data)
```

### Scenario 3: Bulk Opt-Out for Stable Functions

```python
from afml.cache import disable_auto_versioning

# Create custom decorator without versioning
cacheable_stable = disable_auto_versioning()

@cacheable_stable()
def stable_func_1(data): ...

@cacheable_stable()
def stable_func_2(data): ...

@cacheable_stable(time_aware=True)  # Can combine with other options
def stable_func_3(data): ...
```

### Scenario 4: Mixed Strategy

```python
from afml.cache import cacheable

# Under development - auto-versioning
@cacheable()
def experimental_feature(data):
    return data.ewm(span=20).mean()

# Production stable - opt-out
@cacheable(auto_versioning=False)
def load_data(symbol, start, end):
    return expensive_data_load(symbol, start, end)
```

---

## Maintenance & Cleanup

### Periodic Cleanup (Recommended)

Set up weekly/monthly cleanup:

```python
from afml.cache import cache_maintenance

# Run weekly via cron/scheduler
cache_maintenance(
    clean_orphaned=True,      # Remove old function versions
    max_cache_size_mb=2000,   # Enforce size limit
    max_age_days=90,          # Remove very old caches
    min_orphan_age_hours=48   # Keep recent orphans (grace period)
)
```

### Analyze Cache Fragmentation

Check if auto-versioning is creating too many versions:

```python
from afml.cache import print_version_analysis

print_version_analysis()
# Output:
# ========================================
# CACHE VERSION ANALYSIS
# ========================================
# Functions with versions: 12
# Total versions: 34
# Total size: 1.2 GB
# 
# Top fragmented functions:
#   1. calculate_feature
#      Versions: 8
#      Size: 450 MB
```

If fragmentation is high, consider opting out for those functions.

---

## Performance Implications

### Overhead of Auto-Versioning

**Minimal overhead** - hash computed once at decorator application:

```python
# Old smart_cacheable: 0.5ms PER CALL
@smart_cacheable  # Read source + hash on EVERY call
def fast_func(x):
    return x + 1

# New auto_versioning: 0ms per call
@cacheable()  # Hash computed ONCE at import time
def fast_func(x):
    return x + 1
```

### Storage Implications

With auto-versioning, multiple versions can coexist temporarily:

```bash
cache/
  my_module/
    my_function/
      v_abc123_args_xyz/  # Version 1 (orphaned)
      v_def456_args_xyz/  # Version 2 (current)
      v_ghi789_args_xyz/  # Version 3 (current)
```

**Mitigation**: Run `cache_maintenance()` periodically to clean orphans.

---

## Testing Your Migration

### 1. Check for `smart_cacheable` usage

```bash
# This should find zero results after migration
grep -r "smart_cacheable" your_project/
```

### 2. Test auto-versioning behavior

```python
from afml.cache import cacheable

@cacheable()
def test_func(x):
    return x * 2

# First call
result1 = test_func(5)  # Cache miss

# Second call (should hit)
result2 = test_func(5)  # Cache hit

# Change function
def test_func(x):
    return x * 3  # Changed!

# Third call (should miss due to version change)
result3 = test_func(5)  # Cache miss (automatic!)

assert result3 == 15  # New result
```

### 3. Verify cleanup works

```python
from afml.cache import find_orphaned_caches

orphans = find_orphaned_caches()
print(f"Found {orphans['orphaned_count']} orphaned caches")
print(f"Total size: {orphans['total_size_mb']} MB")
```

---

## Troubleshooting

### Issue: Cache not invalidating on changes

**Cause**: Function source unavailable (built-in/dynamic)

**Solution**: Check logs for warnings:

```python
# Look for:
# "Cannot hash source for my_func, using file mtime for versioning"
```

If file mtime also fails, explicitly use `auto_versioning=False` and manage manually.

### Issue: Too many cache versions

**Cause**: Rapid development with many changes

**Solution**: Run cleanup more frequently:

```python
from afml.cache import cache_maintenance

cache_maintenance(
    clean_orphaned=True,
    min_orphan_age_hours=12  # More aggressive
)
```

### Issue: Expensive function cache lost

**Cause**: Auto-versioning invalidated cache on minor change

**Solution**: Opt-out for that specific function:

```python
@cacheable(auto_versioning=False)
def expensive_stable_function(data):
    return days_of_computation(data)
```

---

## Backward Compatibility

### Old Decorator Aliases

These still work (no changes needed):

```python
from afml.cache import (
    robust_cacheable,      # = cacheable()
    time_aware_cacheable,  # = cacheable(time_aware=True)
    cv_cacheable,          # = cacheable()
)

# All now have auto_versioning=True by default
```

### Disabling Auto-Versioning Globally

If you want old behavior everywhere (not recommended):

```python
# In your __init__.py or main module
from afml.cache import disable_auto_versioning

# Use this instead of cacheable
cacheable = disable_auto_versioning()

# Now all @cacheable() calls have auto_versioning=False
```

---

## Getting Help

### Check Cache Health

```python
from afml.cache import print_cache_report
print_cache_report()
```

### Debug Specific Function

```python
from afml.cache import debug_function_cache
debug_function_cache("afml.features.my_func")
```

### Analyze Version Fragmentation

```python
from afml.cache import analyze_cache_versions, print_version_analysis

analysis = analyze_cache_versions()
print_version_analysis()
```

---

## Summary

✅ **What You Need to Do:**

1. Replace `@smart_cacheable` with `@cacheable()` (required)
2. Review expensive functions and opt-out if needed (optional)
3. Set up periodic cache maintenance (recommended)

✅ **What's Better Now:**

- Automatic cache invalidation on code changes (correctness)
- No per-call overhead (performance)
- Complete invalidation for all args (reliability)
- Simpler mental model (clarity)

✅ **Default is Correct:**

- `auto_versioning=True` prevents stale cache bugs
- Only opt-out for specific expensive stable functions
- When in doubt, use the default
