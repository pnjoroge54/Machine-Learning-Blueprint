"""
Usage examples for streamlined caching system with data access tracking.
Demonstrates how to prevent test set contamination while maintaining fast iteration.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from afml.cache import (
    print_contamination_report,
    robust_cacheable,
    save_access_log,
    time_aware_cacheable,
)

# =============================================================================
# Example 1: Time-Series Data Loading with Tracking
# =============================================================================


@time_aware_cacheable(dataset_name="train_2020_2023", purpose="train")
def load_training_data(symbol: str, start_date, end_date):
    """
    Load training data with automatic tracking.

    - First call: Loads from MT5, caches result
    - Subsequent calls: Returns cached data instantly
    - All accesses logged to DataAccessTracker
    """
    import MetaTrader5 as mt5

    print(f"📊 Loading {symbol} data from {start_date} to {end_date}...")

    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    # Fetch data
    rates = mt5.copy_rates_range(
        symbol, mt5.TIMEFRAME_H1, pd.to_datetime(start_date), pd.to_datetime(end_date)
    )

    if rates is None or len(rates) == 0:
        raise ValueError(f"No data returned for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")

    mt5.shutdown()

    print(f"   Loaded {len(df)} bars")
    return df


@time_aware_cacheable(dataset_name="test_2024", purpose="test")
def load_test_data(symbol: str, start_date, end_date):
    """Load test data - accesses are tracked separately from training data."""
    # Same implementation as load_training_data
    return load_training_data(symbol, start_date, end_date)


# =============================================================================
# Example 2: Feature Computation with Robust Caching
# =============================================================================


@robust_cacheable()
def compute_technical_features(df: pd.DataFrame, config: dict):
    """
    Compute technical indicators with robust DataFrame caching.

    Even if you pass a different DataFrame object with identical data,
    the cache will recognize it and return cached results.

    Performance:
    - First call: ~5 minutes
    - Subsequent calls: <1 second (1800x speedup!)
    """
    print(f"🔧 Computing features with config: {config}")

    features = pd.DataFrame(index=df.index)

    # SMA features
    for window in config["sma_windows"]:
        features[f"sma_{window}"] = df["close"].rolling(window).mean()
        features[f"std_{window}"] = df["close"].rolling(window).std()

    # Momentum features
    for lag in config["momentum_lags"]:
        features[f"momentum_{lag}"] = df["close"].pct_change(lag)

    # RSI
    for period in config.get("rsi_periods", [14]):
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    features = features.dropna()

    print(f"   Computed {len(features.columns)} features for {len(features)} bars")
    return features


# =============================================================================
# Example 3: Complete ML Workflow with Contamination Tracking
# =============================================================================


def run_ml_workflow_with_tracking():
    """
    Complete ML workflow demonstrating data hygiene monitoring.
    """
    print("\n" + "=" * 70)
    print("ML WORKFLOW WITH DATA CONTAMINATION TRACKING")
    print("=" * 70)

    # Step 1: Load training data
    print("\n[Step 1] Loading training data...")
    train_data = load_training_data("EURUSD", "2020-01-01", "2023-12-31")
    print(f"✓ Training data: {len(train_data)} bars")

    # Step 2: Load test data
    print("\n[Step 2] Loading test data...")
    test_data = load_test_data("EURUSD", "2024-01-01", "2024-09-30")
    print(f"✓ Test data: {len(test_data)} bars")

    # Step 3: Compute features (cached)
    print("\n[Step 3] Computing features...")
    feature_config = {"sma_windows": [10, 20, 50], "momentum_lags": [1, 5, 10], "rsi_periods": [14]}

    train_features = compute_technical_features(train_data, feature_config)
    test_features = compute_technical_features(test_data, feature_config)

    print(f"✓ Features computed: {train_features.shape[1]} columns")

    # Step 4: Simulate iterative development (multiple test set accesses)
    print("\n[Step 4] Simulating iterative development...")

    for iteration in range(1, 6):
        print(f"\n  Iteration {iteration}:")

        # Different feature configs
        config = {
            "sma_windows": [10 + iteration * 5, 20 + iteration * 10],
            "momentum_lags": [1, 5 + iteration],
            "rsi_periods": [14],
        }

        # Recompute features (or get from cache)
        features = compute_technical_features(test_data, config)
        print(f"    Tested config: {config}")
        print(f"    Features: {features.shape}")

        # This simulates checking test set performance multiple times
        # (Each access is being logged!)

    # Step 5: Save access log
    print("\n[Step 5] Saving access log...")
    save_access_log()

    # Step 6: Analyze contamination
    print("\n[Step 6] Analyzing data contamination...")
    print_contamination_report()

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)


# =============================================================================
# Example 4: Parameter Optimization with Contamination Awareness
# =============================================================================


@robust_cacheable(track_data_access=True, purpose="optimize")
def evaluate_strategy(data: pd.DataFrame, params: dict):
    """
    Evaluate strategy with parameters.

    Each evaluation is cached, and data access is logged with
    purpose='optimize' to track optimization-related contamination.
    """
    # Simulate strategy evaluation
    features = compute_technical_features(data, params)

    # Simple mock evaluation
    sharpe_ratio = np.random.uniform(0.5, 2.0)  # Replace with real backtest

    return {"sharpe_ratio": sharpe_ratio, "params": params, "n_features": features.shape[1]}


def parameter_optimization_example():
    """
    Demonstrate parameter optimization with contamination tracking.
    """
    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION WITH CONTAMINATION TRACKING")
    print("=" * 70)

    # Load test data
    test_data = load_test_data("EURUSD", "2024-01-01", "2024-09-30")

    # Parameter grid
    param_grid = [
        {"sma_windows": [10, 20], "momentum_lags": [1, 5], "rsi_periods": [14]},
        {"sma_windows": [15, 30], "momentum_lags": [2, 10], "rsi_periods": [14]},
        {"sma_windows": [20, 50], "momentum_lags": [5, 10], "rsi_periods": [14, 21]},
        # Add 47 more configurations...
    ]

    print(f"\nTesting {len(param_grid)} parameter combinations...")
    print("(Each test is logged - accumulating test set exposure)\n")

    results = []
    for i, params in enumerate(param_grid, 1):
        print(f"[{i}/{len(param_grid)}] Testing: {params}")

        # Evaluate (cached if already tested)
        result = evaluate_strategy(test_data, params)
        results.append(result)

        print(f"   Sharpe: {result['sharpe_ratio']:.3f}")

    # After optimization, check contamination
    print("\n" + "=" * 70)
    print("POST-OPTIMIZATION CONTAMINATION ANALYSIS")
    print("=" * 70)
    print_contamination_report()

    print("\n⚠️  IMPORTANT:")
    print("⚠️  Your 'best' parameters were chosen after multiple test set exposures.")
    print("⚠️  Validate on truly held-out data before deployment!")


# =============================================================================
# Example 5: Checking Contamination Before Final Validation
# =============================================================================


def final_validation_with_hygiene_check():
    """
    Proper final validation with contamination awareness.
    """
    from afml.cache.data_access_tracker import get_data_tracker

    print("\n" + "=" * 70)
    print("FINAL VALIDATION WITH HYGIENE CHECK")
    print("=" * 70)

    tracker = get_data_tracker()

    # Check test set contamination before proceeding
    test_accesses, warning_level, _ = tracker.analyze_contamination("test_2024")

    print(f"\nTest Set Contamination Check:")
    print(f"  Dataset: test_2024")
    print(f"  Total Accesses: {test_accesses}")
    print(f"  Warning Level: {warning_level}")

    if warning_level in ["WARNING", "CONTAMINATED"]:
        print("\n⚠️  WARNING: Test set is contaminated!")
        print("⚠️  Results on this data are unreliable.")
        print("⚠️  Loading truly held-out validation set instead...")

        # Load fresh validation data (first time)
        validation_data = load_validation_data_once()
        print("\n✓ Using clean validation set for final assessment")
    else:
        print("\n✓ Test set contamination acceptable for final validation")
        validation_data = load_test_data("EURUSD", "2024-01-01", "2024-09-30")

    # Continue with validation...
    print("\n[Proceeding with final model validation...]")


@time_aware_cacheable(dataset_name="validation_2024_q4", purpose="validate")
def load_validation_data_once():
    """
    Load validation data that should only be accessed 1-2 times maximum.

    This is your truly held-out data for honest final assessment.
    """
    return load_training_data("EURUSD", "2024-10-01", "2024-12-31")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    # Example 1: Basic workflow with tracking
    run_ml_workflow_with_tracking()

    # Example 2: Parameter optimization (shows contamination building up)
    # parameter_optimization_example()

    # Example 3: Final validation with hygiene check
    # final_validation_with_hygiene_check()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Every data access is logged with temporal boundaries")
    print("2. Test set contamination is automatically detected")
    print("3. Caching speeds up iteration while maintaining audit trail")
    print("4. Use truly held-out validation for final honest assessment")
    print("=" * 70 + "\n")
