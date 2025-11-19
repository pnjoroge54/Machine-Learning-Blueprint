# AFML Cache + MQL5 Integration - Installation Guide

## Prerequisites

### Python Side

- Python 3.8+
- Required packages:

  ```bash
  pip install numpy pandas loguru joblib appdirs scikit-learn
  ```

### MQL5 Side

- MetaTrader 5 terminal
- MQL5 editor
- **No external libraries needed** - all JSON handling is built-in

## Installation Steps

### 1. Python Setup

**Copy the cache modules to your AFML package:**

```ini
afml/
├── cache/
│   ├── __init__.py
│   ├── backtest_cache.py
│   ├── cache_monitoring.py
│   ├── cv_cache.py
│   ├── data_access_tracker.py
│   ├── robust_cache_keys.py
│   ├── selective_cleaner.py
│   └── mql5_bridge.py  # NEW MODULE
```

**Initialize the cache system:**

```python
from afml.cache import initialize_cache_system
initialize_cache_system()
```

### 2. MQL5 Setup

**a) Copy the EA file:**

1. Open MetaEditor (F4 in MetaTrader 5)
2. Navigate to `MQL5/Experts/`
3. Create new file: `PythonBridgeEA.mq5`
4. Copy the complete code from the artifact
5. Save and compile (F7)

**b) Verify compilation:**

- You should see "0 errors, 0 warnings"
- If you see errors, make sure you're using the latest version without JAson.mqh

### 3. Test Connection

**a) Start Python server:**

```python
from afml.cache.mql5_bridge import MQL5Bridge

# Create and start bridge
bridge = MQL5Bridge(
    host="http://127.0.0.1",
    port=80,
    mode="live"
)
bridge.start_server()

print("✅ Python server running. Start your MQL5 EA now.")
```

**b) Start MQL5 EA:**

1. In MetaTrader 5, drag the `PythonBridgeEA` onto any chart
2. Check inputs:
   - PythonHost: "localhost"
   - PythonPort: 80
   - EnableTrading: false (for testing)
3. Enable AutoTrading (Ctrl+E)
4. Check "Experts" tab for connection message

## Common Issues & Solutions

### Issue 1: "file 'JAson.mqh' not found"

**Problem:** Old version of EA that requires external JSON library

**Solution:** Use the updated EA code that has built-in JSON handling (no external libraries needed)

### Issue 2: Socket connection fails

**Symptoms:**

- MQL5: "Failed to connect to Python server"
- Python: No connection messages

**Solutions:**

1. **Check firewall:**

   ```bash
   # Windows: Allow Python through firewall
   # Or temporarily disable firewall for testing
   ```

2. **Verify port is available:**

   ```python
   import socket
   s = socket.socket()
   try:
       s.bind(('http://127.0.0.1', 80))
       print("✅ Port 80 is available")
   except OSError:
       print("❌ Port 80 is in use")
   finally:
       s.close()
   ```

3. **Check MetaTrader settings:**
   - Tools → Options → Expert Advisors
   - ✅ Enable "Allow WebRequest for listed URL"
   - ✅ Enable "Allow DLL imports"

### Issue 3: Compilation errors in MQL5

**Common errors and fixes:**

| Error | Solution |
|-------|----------|
| `undeclared identifier` | Make sure all function declarations are before their usage |
| `'&' - comma expected` | Check function parameter syntax |
| `implicit conversion` | Add explicit type casts: `(double)value` |

### Issue 4: Messages not received

**Debugging steps:**

**Python side:**

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check if client connected
print(f"Connected: {bridge.client_socket is not None}")
```

**MQL5 side:**

```mql5
// Add to OnTick()
if(is_connected)
{
    Print("✓ Connected to Python");
}
else
{
    Print("✗ Not connected");
}
```

### Issue 5: Cache not working

**Check cache initialization:**

```python
from afml.cache import get_cache_stats

stats = get_cache_stats()
print(f"Functions tracked: {len(stats)}")
print(f"Cache stats: {stats}")
```

**Verify decorator usage:**

```python
# Correct
@robust_cacheable
def my_function(data):
    return process(data)

# Also correct
from afml.cache import cached_backtest

@cached_backtest("my_strategy")
def backtest(data, params):
    return metrics, trades, equity
```

## Performance Testing

### Test 1: Cache Speedup

```python
import time
import pandas as pd
import numpy as np

from afml.cache import robust_cacheable

@robust_cacheable
def expensive_calculation(data):
    time.sleep(2)  # Simulate expensive operation
    return data.rolling(50).mean()

# Generate test data
data = pd.Series(np.random.randn(1000))

# First run (slow)
start = time.time()
result1 = expensive_calculation(data)
time1 = time.time() - start

# Second run (fast - cached)
start = time.time()
result2 = expensive_calculation(data)
time2 = time.time() - start

print(f"First run: {time1:.2f}s")
print(f"Second run: {time2:.4f}s")
print(f"Speedup: {time1/time2:.0f}x")
```

### Test 2: MQL5 Connection

```python
from afml.cache.mql5_bridge import MQL5Bridge, SignalPacket
from datetime import datetime

# Start bridge
bridge = MQL5Bridge(port=80)
bridge.start_server()

# Wait for connection
import time
time.sleep(5)

# Send test signal
signal = SignalPacket(
    timestamp=datetime.now().isoformat(),
    symbol="EURUSD",
    signal_type="BUY",
    entry_price=1.1000,
    stop_loss=1.0950,
    take_profit=1.1100,
    position_size=0.01
)

success = bridge.send_signal(signal)
print(f"Signal sent: {success}")

# Check stats
stats = bridge.get_performance_stats()
print(f"Bridge stats: {stats}")
```

## Production Deployment

### Step 1: Optimize Cache Settings

```python
from afml.cache import setup_production_cache

components = setup_production_cache(
    enable_mlflow=False,  # Set True if using MLflow
    max_cache_size_mb=2000,
    mlflow_experiment="production"
)

print("✅ Production cache initialized")
```

### Step 2: Setup Monitoring

```python
from afml.cache.mql5_bridge import setup_mql5_monitoring

print_report = setup_mql5_monitoring(bridge)

# Print report every hour
import schedule
schedule.every(1).hour.do(print_report)
```

### Step 3: Enable Error Logging

```python
from loguru import logger

# Configure logging
logger.add(
    "mql5_bridge_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)
```

### Step 4: MQL5 Production Settings

```mql5
// Production settings in EA inputs
input bool   EnableTrading = true;      // Enable real trading
input double RiskPercent = 1.0;         // Risk 1% per trade
input int    MagicNumber = 12345;       // Unique magic number
```

## Advanced Configuration

### Custom Cache Directory

```python
import os
os.environ["AFML_CACHE"] = "/path/to/custom/cache"

from afml.cache import initialize_cache_system
initialize_cache_system()
```

### Multiple Symbol Support

```python
# MQL5 side: Run multiple EAs on different charts
# Each EA connects to same Python bridge
# Python automatically handles multiple symbols

symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

for symbol in symbols:
    # Get market data for symbol
    data = bridge.get_market_data(symbol)
    if data is not None:
        # Generate signals
        signals = strategy.generate_signals(data, params)
```

### Backtest Mode

```python
# Use cached historical data
bridge = MQL5Bridge(
    port=80,
    mode="backtest"
)

# Replay cached signals
bridge._load_cached_signals()
print(f"Loaded {len(bridge.signal_history)} historical signals")
```

## Monitoring Dashboard

```python
def print_system_status():
    from afml.cache import get_comprehensive_cache_status
    
    status = get_comprehensive_cache_status()
    
    print("\n" + "="*70)
    print("SYSTEM STATUS")
    print("="*70)
    
    # Cache performance
    print(f"\nCache:")
    print(f"  Hit Rate: {status['core']['hit_rate']:.1%}")
    print(f"  Total Calls: {status['core']['total_calls']}")
    
    # Bridge status
    bridge_stats = bridge.get_performance_stats()
    print(f"\nMQL5 Bridge:")
    print(f"  Connected: {bridge_stats['connected']}")
    print(f"  Signals Sent: {bridge_stats['signals_sent']}")
    print(f"  Execution Rate: {bridge_stats['execution_rate']:.1%}")
    print(f"  Uptime: {bridge_stats['uptime_seconds']:.0f}s")
    
    print("="*70 + "\n")

# Call periodically
import schedule
schedule.every(10).minutes.do(print_system_status)
```

## Support & Debugging

### Enable Debug Mode

```python
# Python
import logging
logging.basicConfig(level=logging.DEBUG)

# MQL5 - add to EA
#define DEBUG_MODE

#ifdef DEBUG_MODE
    #define DebugPrint(x) Print(x)
#else
    #define DebugPrint(x)
#endif

// Use in code
DebugPrint("Processing signal: " + signal_type);
```

### Check Logs

**Python logs location:**

```text
~/.cache/afml/  (Linux/Mac)
%LOCALAPPDATA%/afml/  (Windows)
```

**MQL5 logs location:**

```text
MetaTrader 5/MQL5/Logs/
```

### Get Help

- Check cache stats: `get_cache_stats()`
- Check bridge stats: `bridge.get_performance_stats()`
- Check data access: `print_contamination_report()`
- Full system report: `optimize_cache_system()`

## Next Steps

1. ✅ Install and test connection
2. ✅ Run backtest with caching
3. ✅ Test live signal generation
4. ✅ Setup monitoring
5. ✅ Deploy to production
6. 📊 Monitor performance
7. 🔧 Optimize as needed

## FAQ

**Q: Can I use this with other brokers?**  
A: Yes! The EA works with any MetaTrader 5 broker.

**Q: Does this work on Mac?**  
A: Python side: Yes. MQL5 side: Requires MetaTrader 5 (Windows or Wine on Mac).

**Q: How much speedup can I expect?**  
A: Typical speedups: 10-100x for features, 50-500x for backtests with same data.

**Q: Is my data safe?**  
A: All communication is local (localhost). No external connections.

**Q: Can I use this for multiple accounts?**  
A: Yes! Run multiple EA instances with different magic numbers.
