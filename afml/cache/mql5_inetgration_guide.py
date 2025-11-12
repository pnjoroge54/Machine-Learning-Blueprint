"""
Complete example: Using AFML cache system with MQL5 for live/backtest trading.
Demonstrates cached ML strategies, data tracking, and performance monitoring.
"""

from datetime import datetime, timedelta
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Import AFML cache components
from afml.cache import (
    cached_backtest,
    cv_cacheable,
    get_backtest_cache,
    get_comprehensive_cache_status,
    get_data_tracker,
    initialize_cache_system,
    optimize_cache_system,
    print_contamination_report,
    robust_cacheable,
    setup_production_cache,
)

# Import MQL5 bridge
from afml.cache.mql5_bridge import (
    MQL5Bridge,
    MQL5CachedStrategy,
    SignalPacket,
    setup_mql5_monitoring,
)

# =============================================================================
# Step 1: Define ML Strategy with Caching
# =============================================================================


@robust_cacheable
def calculate_features(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features (cached).
    This function's results are cached based on data content.
    """
    features = ohlcv_data.copy()

    # Moving averages
    features["sma_20"] = ohlcv_data["close"].rolling(20).mean()
    features["sma_50"] = ohlcv_data["close"].rolling(50).mean()
    features["sma_200"] = ohlcv_data["close"].rolling(200).mean()

    # RSI
    delta = ohlcv_data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))

    # Volatility
    features["atr"] = ohlcv_data["high"] - ohlcv_data["low"]
    features["atr_14"] = features["atr"].rolling(14).mean()

    # Price momentum
    features["momentum_5"] = ohlcv_data["close"].pct_change(5)
    features["momentum_20"] = ohlcv_data["close"].pct_change(20)

    return features.dropna()


@robust_cacheable
def ml_predict_signals(features: pd.DataFrame, model_params: Dict) -> pd.Series:
    """
    ML model predictions (cached).
    Simulates a trained model making predictions.
    """
    # Simulate ML model prediction
    # In reality, you'd load a trained sklearn/xgboost model here

    signals = pd.Series(0, index=features.index)

    # Simple rule-based logic (replace with real ML model)
    buy_condition = (
        (features["sma_20"] > features["sma_50"])
        & (features["rsi"] < 30)
        & (features["momentum_5"] > 0)
    )

    sell_condition = (
        (features["sma_20"] < features["sma_50"])
        & (features["rsi"] > 70)
        & (features["momentum_5"] < 0)
    )

    signals[buy_condition] = 1
    signals[sell_condition] = -1

    return signals


@cv_cacheable
def optimize_strategy_parameters(
    train_data: pd.DataFrame, param_grid: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    Optimize strategy parameters using cross-validation (cached).
    This is expensive and benefits greatly from caching.
    """
    best_params = {}
    best_score = -np.inf

    # Grid search (simplified)
    for sma_fast in param_grid["sma_fast"]:
        for sma_slow in param_grid["sma_slow"]:
            params = {"sma_fast": sma_fast, "sma_slow": sma_slow}

            # Calculate backtest score
            features = calculate_features(train_data)
            signals = ml_predict_signals(features, params)
            score = calculate_sharpe_ratio(train_data, signals)

            if score > best_score:
                best_score = score
                best_params = params

    best_params["best_sharpe"] = best_score
    return best_params


def calculate_sharpe_ratio(data: pd.DataFrame, signals: pd.Series) -> float:
    """Calculate Sharpe ratio from signals."""
    returns = data["close"].pct_change()
    strategy_returns = returns * signals.shift(1)

    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        if strategy_returns.std() > 0
        else 0
    )

    return sharpe


# =============================================================================
# Step 2: Strategy Function for MQL5
# =============================================================================


@cached_backtest("momentum_strategy", save_trades=True)
def momentum_strategy(market_data: pd.DataFrame, params: Dict[str, Any]) -> tuple:
    """
    Complete trading strategy with caching.
    Returns (metrics, trades, equity_curve).
    """
    # Calculate features (cached)
    features = calculate_features(market_data)

    # Generate signals (cached)
    signals = ml_predict_signals(features, params)

    # Generate trades
    trades = []
    position = 0
    entry_price = 0

    for i in range(len(signals)):
        if signals.iloc[i] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = market_data["close"].iloc[i]
            trades.append(
                {
                    "timestamp": market_data.index[i],
                    "type": "BUY",
                    "price": entry_price,
                    "size": 1.0,
                }
            )

        elif signals.iloc[i] == -1 and position == 1:  # Sell signal
            exit_price = market_data["close"].iloc[i]
            pnl = exit_price - entry_price

            trades.append(
                {
                    "timestamp": market_data.index[i],
                    "type": "SELL",
                    "price": exit_price,
                    "size": 1.0,
                    "pnl": pnl,
                }
            )
            position = 0

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        total_pnl = trades_df["pnl"].sum()
        win_rate = (trades_df["pnl"] > 0).sum() / len(trades_df[trades_df["pnl"].notna()])
        sharpe = calculate_sharpe_ratio(market_data, signals)
    else:
        total_pnl = 0
        win_rate = 0
        sharpe = 0

    metrics = {
        "total_return": total_pnl,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "num_trades": len(trades_df),
    }

    # Calculate equity curve
    equity_curve = pd.Series(0.0, index=market_data.index)
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        for trade in trades_df[trades_df["pnl"].notna()].itertuples():
            equity_curve.loc[trade.timestamp :] += trade.pnl

    return metrics, trades_df, equity_curve


# =============================================================================
# Step 3: MQL5 Signal Generator
# =============================================================================


def generate_mql5_signals(market_data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """
    Generate signals for MQL5 from cached strategy.
    Returns list of signal dictionaries.
    """
    # Get cached predictions
    features = calculate_features(market_data)
    signals = ml_predict_signals(features, params)

    # Convert to MQL5-compatible format
    mql5_signals = []

    # Get latest signal
    if len(signals) > 0:
        latest_signal = signals.iloc[-1]
        latest_price = market_data["close"].iloc[-1]

        if latest_signal == 1:  # Buy
            atr = features["atr_14"].iloc[-1]

            mql5_signals.append(
                {
                    "timestamp": market_data.index[-1].isoformat(),
                    "symbol": "EURUSD",  # Configurable
                    "type": "BUY",
                    "price": latest_price,
                    "stop_loss": latest_price - (2 * atr),
                    "take_profit": latest_price + (3 * atr),
                    "size": 0.01,  # 0.01 lots
                    "confidence": abs(features["rsi"].iloc[-1] - 50) / 50,
                    "metadata": {
                        "rsi": features["rsi"].iloc[-1],
                        "sma_20": features["sma_20"].iloc[-1],
                        "atr": atr,
                    },
                }
            )

        elif latest_signal == -1:  # Sell
            atr = features["atr_14"].iloc[-1]

            mql5_signals.append(
                {
                    "timestamp": market_data.index[-1].isoformat(),
                    "symbol": "EURUSD",
                    "type": "SELL",
                    "price": latest_price,
                    "stop_loss": latest_price + (2 * atr),
                    "take_profit": latest_price - (3 * atr),
                    "size": 0.01,
                    "confidence": abs(features["rsi"].iloc[-1] - 50) / 50,
                    "metadata": {
                        "rsi": features["rsi"].iloc[-1],
                        "sma_20": features["sma_20"].iloc[-1],
                        "atr": atr,
                    },
                }
            )

    return mql5_signals


# =============================================================================
# Step 4: Complete Integration Examples
# =============================================================================


def example_backtest_with_cache():
    """
    Example: Run backtest with full caching.
    Subsequent runs with same data are instant.
    """
    print("\n" + "=" * 70)
    print("BACKTEST WITH CACHING EXAMPLE")
    print("=" * 70)

    # Initialize cache system
    initialize_cache_system()

    # Generate sample data
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="1H")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": 1.1000 + np.random.randn(len(dates)) * 0.001,
            "high": 1.1000 + np.random.randn(len(dates)) * 0.001 + 0.0005,
            "low": 1.1000 + np.random.randn(len(dates)) * 0.001 - 0.0005,
            "close": 1.1000 + np.random.randn(len(dates)) * 0.001,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # Strategy parameters
    params = {"sma_fast": 20, "sma_slow": 50}

    # Run backtest (first run computes, second run uses cache)
    print("\n1st run (computing)...")
    import time

    start = time.time()
    metrics1, trades1, equity1 = momentum_strategy(data, params)
    time1 = time.time() - start

    print("\n2nd run (cached)...")
    start = time.time()
    metrics2, trades2, equity2 = momentum_strategy(data, params)
    time2 = time.time() - start

    print(f"\nFirst run: {time1:.2f}s")
    print(f"Second run: {time2:.2f}s")
    print(f"Speedup: {time1/time2:.1f}x")

    print(f"\nBacktest Results:")
    print(f"  Total Return: {metrics1['total_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics1['win_rate']:.1%}")
    print(f"  Number of Trades: {metrics1['num_trades']}")


def example_live_trading_with_mql5():
    """
    Example: Live trading with MQL5 integration.
    Demonstrates real-time signal generation with caching.
    """
    print("\n" + "=" * 70)
    print("LIVE TRADING WITH MQL5 EXAMPLE")
    print("=" * 70)

    # Setup production cache
    print("\nInitializing production cache system...")
    setup_production_cache(
        enable_mlflow=False, max_cache_size_mb=2000  # Set to True if using MLflow
    )

    # Create MQL5 bridge
    print("\nStarting MQL5 bridge...")
    bridge = MQL5Bridge(host="localhost", port=9090, mode="live")
    bridge.start_server()

    # Create cached strategy
    strategy = MQL5CachedStrategy(
        strategy_func=generate_mql5_signals, bridge=bridge, use_cache=True, track_data=True
    )

    # Setup monitoring
    print_report = setup_mql5_monitoring(bridge)

    print("\nâœ… Bridge ready. Waiting for MQL5 connection...")
    print("   Start your MQL5 EA to connect.")
    print("   Press Ctrl+C to stop.\n")

    try:
        # Simulation loop (in practice, this would run continuously)
        params = {"sma_fast": 20, "sma_slow": 50}

        for i in range(10):  # Simulate 10 ticks
            # Generate sample market data
            dates = pd.date_range(datetime.now() - timedelta(hours=200), datetime.now(), freq="1H")

            data = pd.DataFrame(
                {
                    "open": 1.1000 + np.random.randn(len(dates)) * 0.001,
                    "high": 1.1000 + np.random.randn(len(dates)) * 0.001 + 0.0005,
                    "low": 1.1000 + np.random.randn(len(dates)) * 0.001 - 0.0005,
                    "close": 1.1000 + np.random.randn(len(dates)) * 0.001,
                    "volume": np.random.randint(1000, 10000, len(dates)),
                },
                index=dates,
            )

            # Generate and send signals
            signals = strategy.generate_signals(data, params)

            if signals:
                print(f"✓ Generated {len(signals)} signal(s)")

            time.sleep(2)  # Wait 2 seconds

        # Print integrated report
        print("\n" + "=" * 70)
        print_report()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        bridge.stop()


def example_parameter_optimization():
    """
    Example: Parameter optimization with caching.
    Shows massive speedup from CV caching.
    """
    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION WITH CACHING EXAMPLE")
    print("=" * 70)

    initialize_cache_system()

    # Generate training data
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="1H")
    np.random.seed(42)

    train_data = pd.DataFrame(
        {
            "open": 1.1000 + np.random.randn(len(dates)) * 0.001,
            "high": 1.1000 + np.random.randn(len(dates)) * 0.001 + 0.0005,
            "low": 1.1000 + np.random.randn(len(dates)) * 0.001 - 0.0005,
            "close": 1.1000 + np.random.randn(len(dates)) * 0.001,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # Parameter grid
    param_grid = {"sma_fast": [10, 20, 30], "sma_slow": [50, 100, 200]}

    print("\n1st optimization run (computing)...")
    start = time.time()
    best_params1 = optimize_strategy_parameters(train_data, param_grid)
    time1 = time.time() - start

    print("\n2nd optimization run (cached)...")
    start = time.time()
    best_params2 = optimize_strategy_parameters(train_data, param_grid)
    time2 = time.time() - start

    print(f"\nFirst run: {time1:.2f}s")
    print(f"Second run: {time2:.2f}s")
    print(f"Speedup: {time1/time2:.1f}x")

    print(f"\nBest parameters:")
    for key, value in best_params1.items():
        print(f"  {key}: {value}")


def example_data_contamination_check():
    """
    Example: Check for data contamination during development.
    Critical for detecting data snooping.
    """
    print("\n" + "=" * 70)
    print("DATA CONTAMINATION CHECK EXAMPLE")
    print("=" * 70)

    initialize_cache_system()
    tracker = get_data_tracker()

    # Simulate multiple accesses to test set
    dates = pd.date_range("2024-01-01", "2024-02-01", freq="1H")
    test_data = pd.DataFrame({"close": 1.1000 + np.random.randn(len(dates)) * 0.001}, index=dates)

    # Log multiple accesses (simulating iterative development)
    for i in range(5):
        tracker.log_access(
            dataset_name="test_set_Q1_2024",
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            purpose="test",
            data_shape=test_data.shape,
        )

    # Print contamination report
    print_contamination_report()

    # Save for audit trail
    tracker.save_log()


def example_cache_maintenance():
    """
    Example: Regular cache maintenance and optimization.
    """
    print("\n" + "=" * 70)
    print("CACHE MAINTENANCE EXAMPLE")
    print("=" * 70)

    initialize_cache_system()

    # Run comprehensive optimization
    print("\nRunning cache optimization...")
    results = optimize_cache_system(
        clear_changed=True, max_size_mb=1000, max_age_days=30, print_report=True
    )

    # Get comprehensive status
    print("\n" + "=" * 70)
    print("CACHE SYSTEM STATUS")
    print("=" * 70)

    status = get_comprehensive_cache_status()

    print(f"\nCore Cache:")
    print(f"  Hit Rate: {status['core']['hit_rate']:.1%}")
    print(f"  Total Calls: {status['core']['total_calls']}")

    if status["health"]:
        print(f"\nHealth:")
        print(f"  Functions: {status['health']['total_functions']}")
        print(f"  Cache Size: {status['health']['cache_size_mb']:.1f} MB")

    if status["backtest"]:
        print(f"\nBacktest Cache:")
        print(f"  Total Runs: {status['backtest']['total_runs']}")
        print(f"  Strategies: {list(status['backtest']['strategies'].keys())}")


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("AFML CACHE + MQL5 INTEGRATION DEMO")
    print("=" * 70)

    examples = {
        "1": ("Backtest with Caching", example_backtest_with_cache),
        "2": ("Live Trading with MQL5", example_live_trading_with_mql5),
        "3": ("Parameter Optimization", example_parameter_optimization),
        "4": ("Data Contamination Check", example_data_contamination_check),
        "5": ("Cache Maintenance", example_cache_maintenance),
    }

    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\nRun all examples? (y/n): ", end="")
    choice = input().strip().lower()

    if choice == "y":
        for _, (name, func) in examples.items():
            func()
            input("\nPress Enter to continue...")
    else:
        print("Enter example number: ", end="")
        num = input().strip()
        if num in examples:
            examples[num][1]()
        else:
            print("Invalid choice")
