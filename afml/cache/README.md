# AFML Unified Cache Module

Production-grade caching for financial ML, backtesting, and cross-validation workflows.

## Features
- Single `@cacheable` decorator with auto-versioning and time-awareness.
- Robust hashing for DataFrames, sklearn objects, scipy distributions, etc.
- Joblib + cloudpickle persistence (cloudpickle preferred for ML objects).
- Advanced monitoring, selective cleaning, and stats.
- MQL5 bridge compatibility.
- Automatic invalidation on code changes.

## Quick Start
```python
from afml.cache import cacheable, initialize_cache_system, print_cache_health

initialize_cache_system()

@cacheable(time_aware=True, auto_versioning=True)
def expensive_backtest(...):
    ...
