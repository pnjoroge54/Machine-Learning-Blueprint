# Enhanced AFML Cache System - Usage Guide

Complete guide to using the improved caching system with all new features.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Robust Cache Keys](#robust-cache-keys)
3. [MLflow Integration](#mlflow-integration)
4. [Backtest Caching](#backtest-caching)
5. [Cache Monitoring](#cache-monitoring)
6. [Best Practices](#best-practices)

---

## Basic Usage

### Standard Caching (Original System)

```python
from afml.cache import cacheable

@cacheable
def compute_triple_barrier_labels(closes, barriers, ptsl):
    """Expensive computation - will be cached."""
    # Your implementation
    return labels

# First call: computes and caches
labels1 = compute_triple_barrier_labels(prices, [0.02, 0.02], [2, 3])

# Second call with same args: returns cached result (instant)
labels2 = compute_triple_barrier_labels(prices, [0.02, 0.02], [2, 3])
```

---

## Robust Cache Keys

### Why Use Robust Cache Keys?

The original caching system can miss cache hits due to weak key generation. The new system properly handles:
- NumPy arrays (considers shape, dtype, and content)
- Pandas DataFrames (considers index, columns, and data)
- Time-series data (considers date ranges)

### Installation

Add to your existing code:

```python
# In afml/cache/__init__.py, add:
from .robust_cache_keys import robust_cacheable, time_aware_cacheable

__all__ = [
    # ... existing exports ...
    "robust_cacheable",
    "time_aware_cacheable",
]
```

### Using Robust Cacheable

```python
from afml.cache import robust_cacheable

@robust_cacheable
def extract_features(prices_df, indicators):
    """
    Properly caches based on DataFrame content.
    Even if you pass a different DataFrame object with same data,
    it will recognize as cache hit.
    """
    # Feature extraction
    return features

# These will use the same cache (same data, different objects)
df1 = pd.DataFrame({'close': [100, 101, 102]})
df2 = pd.DataFrame({'close': [100, 101, 102]})  # Different object

features1 = extract_features(df1, ['sma', 'rsi'])  # Computes
features2 = extract_features(df2, ['sma', 'rsi'])  # Cache hit!
```

### Time-Series Aware Caching

Perfect for financial data where date ranges matter:

```python
from afml.cache import time_aware_cacheable

@time_aware_cacheable
def compute_volatility(prices_df, window=20):
    """
    Cache is aware of time range.
    Different date ranges = different cache entries.
    """
    return prices_df.rolling(window).std()

# Different time ranges = different cache entries
jan_data = prices['2024-01-01':'2024-01-31']
feb_data = prices['2024-02-01':'2024-02-28']

vol_jan = compute_volatility(jan_data)  # Cached separately
vol_feb = compute_volatility(feb_data)  # Cached separately
```

---

## MLflow Integration

### Setup

```python
from afml.cache.mlflow_integration import setup_mlflow_cache

# Initialize once at startup
mlflow_cache = setup_mlflow_cache(
    experiment_name="mt5_strategy_optimization",
    tracking_uri="http://localhost:5000"  # Or None for local ./mlruns
)
```

### Basic Experiment Tracking

```python
from afml.cache.mlflow_integration import mlflow_cached

@mlflow_cached(tags={"strategy": "momentum", "market": "EURUSD"})
def train_model(data, params):
    """
    Automatically:
    - Checks local cache first (fast)
    - Tracks parameters in MLflow
    - Logs metrics to MLflow
    - Caches results locally
    """
    model = RandomForestClassifier(**params)
    model.fit(data['features'], data['labels'])
    
    metrics = {
        'accuracy': 0.85,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.15
    }
    
    return model, metrics

# First call: trains and logs to MLflow
model, metrics = train_model(mt5_data, {'n_estimators': 100})

# Second call: uses cached result (still logs to MLflow for tracking)
model2, metrics2 = train_model(mt5_data, {'n_estimators': 100})
```

### Manual Experiment Context

For more control:

```python
from afml.cache.mlflow_integration import get_mlflow_cache

mlflow_cache = get_mlflow_cache()

with mlflow_cache.experiment_run("parameter_optimization") as ctx:
    # Check cache
    cached = ctx.get_cached("opt_results_v1")
    if cached:
        results = cached
    else:
        # Run optimization
        results = optimize_parameters(data, param_grid)
        
        # Cache for next time
        ctx.cache_result("opt_results_v1", results)
    
    # Log metrics
    ctx.log_metric("best_sharpe", results['best_sharpe'])
    ctx.log_param("param_grid_size", len(param_grid))
```

### Finding Best Runs

```python
# Get best performing run
best_run_id = mlflow_cache.get_best_run("sharpe_ratio", maximize=True)

# Load model from best run
best_model = mlflow_cache.load_model_from_run(best_run_id)

# Compare multiple runs
comparison = mlflow_cache.compare_runs(
    run_ids=["abc123", "def456", "ghi789"],
    metrics=["sharpe_ratio", "max_drawdown", "win_rate"]
)
print(comparison)
```

---

## Backtest Caching

### Setup

```python
from afml.cache.backtest_cache import BacktestCache, cached_backtest

# Initialize backtest cache
backtest_cache = BacktestCache()
```

### Using the Decorator

```python
@cached_backtest("momentum_strategy", save_trades=True)
def run_momentum_backtest(data, params):
    """
    Automatically caches:
    - Complete backtest results
    - Individual trades
    - Equity curve
    - Performance metrics
    """
    # Your backtest logic
    trades = []
    equity = []
    
    # ... backtest implementation ...
    
    metrics = {
        'total_return': 0.45,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.12,
        'win_rate': 0.58,
        'profit_factor': 2.1
    }
    
    return metrics, pd.DataFrame(trades), pd.Series(equity)

# First run: computes and caches
params = {'lookback': 20, 'threshold': 0.02}
metrics, trades, equity = run_momentum_backtest(mt5_data, params)

# Second run with same params: instant from cache
metrics2, trades2, equity2 = run_momentum_backtest(mt5_data, params)
```

### Manual Caching

For full control:

```python
# Cache a backtest
run_id = backtest_cache.cache_backtest(
    strategy_name="mean_reversion",
    parameters={'window': 50, 'std_dev': 2},
    data=mt5_data,
    metrics={'sharpe': 1.5, 'return': 0.3},
    trades=trades_df,
    equity_curve=equity_series
)

# Retrieve cached backtest
cached = backtest_cache.get_cached_backtest(
    strategy_name="mean_reversion",
    parameters={'window': 50, 'std_dev': 2},
    data=mt5_data
)

if cached:
    print(f"Sharpe Ratio: {cached.metrics['sharpe']}")
    print(f"Total Trades: {len(cached.trades)}")
```

### Parameter Optimization with Cache

```python
# Test multiple parameter combinations
param_grid = [
    {'lookback': 10, 'threshold': 0.01},
    {'lookback': 20, 'threshold': 0.02},
    {'lookback': 30, 'threshold': 0.03},
]

results = []
for params in param_grid:
    # Uses cache if available
    metrics, _, _ = run_momentum_backtest(mt5_data, params)
    results.append((params, metrics))

# Find best parameters
best_params = backtest_cache.find_best_parameters(
    strategy_name="momentum_strategy",
    metric="sharpe_ratio",
    maximize=True,
    top_n=3
)

print("Best Parameter Combinations:")
print(best_params)
```

### Walk-Forward Analysis

```python
# Cache walk-forward splits
split_id = "2024_walk_forward"

for fold in range(5):
    train_data = get_train_fold(fold)
    test_data = get_test_fold(fold)
    
    # Cache the split
    backtest_cache.cache_walk_forward_split(
        split_id=split_id,
        train_data=train_data,
        test_data=test_data,
        fold_number=fold,
        total_folds=5
    )
    
    # Run backtest on this fold
    # (uses cached split if running again)
    metrics, _, _ = run_momentum_backtest(test_data, best_params)
```

### Comparing Strategies

```python
# Run multiple strategies
strategies = [
    ("momentum", momentum_params),
    ("mean_reversion", mr_params),
    ("breakout", breakout_params)
]

run_ids = []
for strategy_name, params in strategies:
    metrics, _, _ = eval(f"run_{strategy_name}_backtest")(mt5_data, params)
    run_ids.append(backtest_cache.index[...])  # Get run_id

# Compare all strategies
comparison = backtest_cache.compare_runs(
    run_ids=run_ids,
    metrics=["sharpe_ratio", "max_drawdown", "win_rate"]
)

print("\nStrategy Comparison:")
print(comparison.sort_values("sharpe_ratio", ascending=False))
```

---

## Cache Monitoring

### Health Report

```python
from afml.cache.cache_monitoring import print_cache_health

# Print comprehensive health report
print_cache_health()

# Output:
# ======================================================================
# CACHE HEALTH REPORT
# ======================================================================
# 
# Overall Statistics:
#   Total Functions:     15
#   Total Calls:         1,234
#   Overall Hit Rate:    78.5%
#   Total Cache Size:    245.67 MB
# 
# Top Performers (by hit rate):
#   1. compute_features: 95.2% (450 calls)
#   2. triple_barrier: 89.1% (320 calls)
#   ...
```

### Detailed Analysis

```python
from afml.cache.cache_monitoring import get_cache_efficiency_report

# Get DataFrame with detailed stats
df = get_cache_efficiency_report()
print(df)

# Output:
#                     function  calls  hits  misses hit_rate  avg_time_ms  cache_size_mb     last_access
# 0         compute_features    450   428      22    95.2%        234.56          45.23  2024-01-15 14:30
# 1       triple_barrier       320   285      35    89.1%        567.89          67.45  2024-01-15 14:25
# ...
```

### Pattern Analysis

```python
from afml.cache.cache_monitoring import analyze_cache_patterns

patterns = analyze_cache_patterns()

# Check for issues
if patterns['high_miss_rate_functions']:
    print("Functions with poor cache performance:")
    for func in patterns['high_miss_rate_functions']:
        print(f"  - {func['function']}: {func['hit_rate']:.1%} hit rate")

if patterns['large_caches']:
    print("\nLarge caches consuming disk space:")
    for func in patterns['large_caches']:
        print(f"  - {func['function']}: {func['size_mb']:.1f} MB")
```

### Export Reports

```python
from afml.cache.cache_monitoring import get_cache_monitor

monitor = get_cache_monitor()

# Export to CSV for analysis
monitor.export_report(Path("cache_report.csv"))

# Export to HTML for sharing
monitor.export_report(Path("cache_report.html"))
```

### Function-Specific Analysis

```python
from afml.cache.cache_monitoring import get_cache_monitor

monitor = get_cache_monitor()

# Get stats for specific function
stats = monitor.get_function_stats("afml.labeling.triple_barrier")

print(f"Function: {stats.function_name}")
print(f"Total Calls: {stats.total_calls}")
print(f"Hit Rate: {stats.hit_rate:.1%}")
print(f"Avg Computation Time: {stats.avg_computation_time:.2f}s")
print(f"Cache Size: {stats.cache_size_mb:.2f} MB")
```

---

## Best Practices

### 1. Choose the Right Decorator

```python
# For simple functions with basic types
@cacheable
def simple_computation(x, y):
    return x + y

# For functions with NumPy/Pandas
@robust_cacheable
def data_processing(df, params):
    return process(df)

# For time-series functions
@time_aware_cacheable
def rolling_features(prices_df, window):
    return prices_df.rolling(window).mean()

# For experiment tracking + caching
@mlflow_cached(tags={"type": "training"})
def train_model(data, params):
    return model, metrics

# For backtesting
@cached_backtest("strategy_name")
def run_backtest(data, params):
    return metrics, trades, equity
```

### 2. Cache Maintenance

```python
from afml.cache import maintain_cache, smart_cache_clear

# Run maintenance regularly (e.g., weekly)
report = maintain_cache(
    auto_clear=True,      # Clear changed functions
    max_size_mb=1000,     # Keep cache under 1GB
    max_age_days=30       # Remove old caches
)

print(f"Cleared {len(report['cleared_functions'])} functions")
print(f"Freed {report['size_cleared_mb']:.1f} MB")

# Or manually clear specific modules
smart_cache_clear('labeling')  # Clear changed labeling functions
```

### 3. Monitor Regularly

```python
# Add to your startup script or notebook
from afml.cache import cache_status

print(cache_status())
# Output: Hit rate: 82.5% | Tracked functions: 15 | Heavy modules loaded: 3

# Periodic health checks
import schedule

def check_cache_health():
    from afml.cache.cache_monitoring import print_cache_health
    print_cache_health()

schedule.every().week.do(check_cache_health)
```

### 4. Production Setup

```python
# production_startup.py

from afml.cache import initialize_cache_system
from afml.cache.mlflow_integration import setup_mlflow_cache
from afml.cache.backtest_cache import BacktestCache

def initialize_production_cache():
    """Setup cache system for production."""
    
    # Initialize base cache
    initialize_cache_system()
    
    # Setup MLflow tracking
    mlflow_cache = setup_mlflow_cache(
        experiment_name="production_trading",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    )
    
    # Initialize backtest cache
    backtest_cache = BacktestCache()
    
    # Run initial maintenance
    from afml.cache import maintain_cache
    maintain_cache(auto_clear=True, max_size_mb=2000, max_age_days=90)
    
    print("✅ Production cache system initialized")
    
    return mlflow_cache, backtest_cache

# Call at startup
if __name__ == "__main__":
    initialize_production_cache()
```

### 5. Debugging Cache Issues

```python
# Enable debug logging
from loguru import logger
logger.add("cache_debug.log", level="DEBUG")

# Check what's being cached
from afml.cache import get_cache_stats
stats = get_cache_stats()
print(stats)

# Analyze patterns
from afml.cache.cache_monitoring import analyze_cache_patterns
patterns = analyze_cache_patterns()

if patterns['high_miss_rate_functions']:
    print("⚠️ These functions have poor cache performance:")
    for func in patterns['high_miss_rate_functions']:
        print(f"  {func}")

# Clear and rebuild if needed
from afml.cache import clear_afml_cache
clear_afml_cache(warn=True)
```

---

## Complete Example: MT5 Strategy Development

```python
# Complete workflow using all enhanced features

import MetaTrader5 as mt5
from afml.cache import initialize_cache_system
from afml.cache.mlflow_integration import setup_mlflow_cache
from afml.cache.backtest_cache import cached_backtest
from afml.cache.robust_cache_keys import time_aware_cacheable

# 1. Initialize systems
initialize_cache_system()
mlflow_cache = setup_mlflow_cache("mt5_momentum_strategy")

# 2. Data preparation (time-aware caching)
@time_aware_cacheable
def prepare_mt5_data(symbol, timeframe, start, end):
    """Load and prepare MT5 data with caching."""
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.set_index('time')

# 3. Feature engineering (robust caching)
@robust_cacheable
def compute_features(df, params):
    """Compute features with proper DataFrame caching."""
    features = pd.DataFrame(index=df.index)
    features['sma'] = df['close'].rolling(params['sma_window']).mean()
    features['rsi'] = compute_rsi(df['close'], params['rsi_period'])
    return features

# 4. Model training (MLflow tracking)
@mlflow_cached(tags={"model": "rf", "version": "v2"})
def train_model(features, labels, params):
    """Train with caching and experiment tracking."""
    model = RandomForestClassifier(**params)
    model.fit(features, labels)
    
    metrics = {
        'accuracy': accuracy_score(labels, model.predict(features)),
        'precision': precision_score(labels, model.predict(features))
    }
    
    return model, metrics

# 5. Backtesting (specialized backtest cache)
@cached_backtest("momentum_v2", save_trades=True)
def run_backtest(data, model, params):
    """Backtest with comprehensive caching."""
    # Backtest logic
    trades = execute_strategy(data, model, params)
    equity = calculate_equity_curve(trades)
    
    metrics = {
        'total_return': 0.35,
        'sharpe_ratio': 1.65,
        'max_drawdown': -0.11,
        'win_rate': 0.62
    }
    
    return metrics, trades, equity

# 6. Execution
if __name__ == "__main__":
    # Load data (cached)
    data = prepare_mt5_data("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date)
    
    # Compute features (cached)
    features = compute_features(data, {'sma_window': 20, 'rsi_period': 14})
    
    # Train model (cached + tracked)
    model, train_metrics = train_model(features, labels, {'n_estimators': 100})
    
    # Backtest (cached with all details)
    metrics, trades, equity = run_backtest(data, model, {'threshold': 0.02})
    
    # Monitor cache performance
    from afml.cache.cache_monitoring import print_cache_health
    print_cache_health()
    
    print(f"\n✅ Strategy Performance:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Total Return: {metrics['total_return']:.1%}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
```

---

## Troubleshooting

### Cache Not Working?

```python
# Check if function is decorated
from afml.cache import get_cache_stats
stats = get_cache_stats()
print(stats)  # Should show your function

# Verify cache directory
from afml.cache import CACHE_DIRS
print(f"Cache location: {CACHE_DIRS['joblib']}")

# Clear and retry
from afml.cache import clear_afml_cache
clear_afml_cache()
```

### Low Hit Rate?

```python
# Analyze why
from afml.cache.cache_monitoring import analyze_cache_patterns
patterns = analyze_cache_patterns()

# Check if cache keys are stable
# Use robust_cacheable instead of cacheable for DataFrames/arrays
```

### Cache Too Large?

```python
# Run maintenance
from afml.cache import maintain_cache
report = maintain_cache(max_size_mb=500, max_age_days=14)
print(f"Freed {report['size_cleared_mb']:.1f} MB")
```

---

## Summary

The enhanced cache system provides:

1. **Robust Cache Keys**: Proper handling of NumPy/Pandas data
2. **MLflow Integration**: Experiment tracking + local caching
3. **Backtest Caching**: Specialized caching for trading strategies
4. **Monitoring**: Detailed analytics and health reports
5. **Maintenance**: Automated cleanup and optimization

All improvements are **backward compatible** with your existing code!