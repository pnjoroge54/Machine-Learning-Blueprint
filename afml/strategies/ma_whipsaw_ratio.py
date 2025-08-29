import numpy as np
import pandas as pd


def calculate_ma_whipsaw_ratio(
    prices, short_ma_period, long_ma_period, whipsaw_threshold_multiplier=2.0
):
    """
    Calculate Whipsaw Ratio for Moving Average Crossover System

    Parameters:
    prices: pandas Series or array of price data
    short_ma_period: Period for short MA (e.g., 20)
    long_ma_period: Period for long MA (e.g., 50)
    whipsaw_threshold_multiplier: Multiplier for reversal detection window

    Returns:
    whipsaw_ratio: Float between 0-1
    trade_details: DataFrame with trade analysis
    """

    # Calculate moving averages
    short_ma = pd.Series(prices).rolling(short_ma_period).mean()
    long_ma = pd.Series(prices).rolling(long_ma_period).mean()

    # Generate signals
    signal = np.where(short_ma > long_ma, 1, -1)  # 1=long, -1=short
    signal_changes = np.diff(signal)

    # Find crossover points (non-zero signal changes)
    crossover_indices = np.where(signal_changes != 0)[0] + 1

    if len(crossover_indices) == 0:
        return 0.0, pd.DataFrame()

    # Set whipsaw detection window
    whipsaw_window = int(short_ma_period * whipsaw_threshold_multiplier)

    trades = []
    whipsaw_count = 0

    for i, entry_idx in enumerate(crossover_indices):
        entry_signal = signal[entry_idx]
        entry_price = prices[entry_idx]

        # Look for next signal change (exit)
        exit_idx = None
        for j in range(i + 1, len(crossover_indices)):
            if signal[crossover_indices[j]] != entry_signal:
                exit_idx = crossover_indices[j]
                break

        # If no exit found, use last price
        if exit_idx is None:
            exit_idx = len(prices) - 1

        exit_price = prices[exit_idx]
        trade_duration = exit_idx - entry_idx

        # Calculate return
        if entry_signal == 1:  # Long trade
            trade_return = (exit_price - entry_price) / entry_price
        else:  # Short trade
            trade_return = (entry_price - exit_price) / entry_price

        # Check if this is a whipsaw
        is_whipsaw = trade_duration <= whipsaw_window
        if is_whipsaw:
            whipsaw_count += 1

        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_signal": entry_signal,
                "duration": trade_duration,
                "return": trade_return,
                "is_whipsaw": is_whipsaw,
            }
        )

    # Calculate whipsaw ratio
    total_trades = len(trades)
    whipsaw_ratio = whipsaw_count / total_trades if total_trades > 0 else 0.0

    trade_details = pd.DataFrame(trades)

    return whipsaw_ratio, trade_details


def calculate_enhanced_whipsaw_metrics(prices, short_ma, long_ma):
    """
    Calculate additional whipsaw-related metrics for comprehensive analysis
    """
    whipsaw_ratio, trade_details = calculate_ma_whipsaw_ratio(prices, short_ma, long_ma)

    if trade_details.empty:
        return {
            "whipsaw_ratio": 0.0,
            "avg_whipsaw_loss": 0.0,
            "consecutive_whipsaw_streaks": 0,
            "whipsaw_economic_impact": 0.0,
        }

    whipsaw_trades = trade_details[trade_details["is_whipsaw"] == True]

    # Average loss from whipsaw trades
    avg_whipsaw_loss = whipsaw_trades["return"].mean() if len(whipsaw_trades) > 0 else 0.0

    # Count consecutive whipsaw streaks
    consecutive_whipsaws = 0
    current_streak = 0
    for is_whip in trade_details["is_whipsaw"]:
        if is_whip:
            current_streak += 1
        else:
            if current_streak >= 3:  # 3+ consecutive whipsaws
                consecutive_whipsaws += 1
            current_streak = 0

    # Economic impact: total whipsaw losses vs total trend gains
    whipsaw_losses = whipsaw_trades["return"].sum()
    non_whipsaw_net = trade_details[trade_details["is_whipsaw"] == False]["return"].sum()

    # Only positive non-whipsaw returns
    non_whipsaw_gains_positive = trade_details[
        (trade_details["is_whipsaw"] == False) & (trade_details["return"] > 0)
    ]["return"].sum()

    return {
        "whipsaw_ratio": whipsaw_ratio,
        "avg_whipsaw_loss": avg_whipsaw_loss,
        "consecutive_whipsaw_streaks": consecutive_whipsaws,
        "whipsaw_impact_on_profitable_trades": abs(whipsaw_losses)
        / (non_whipsaw_gains_positive + 1e-8),
        "whipsaw_impact_on_all_trades": abs(whipsaw_losses) / (non_whipsaw_net + 1e-8),
        "total_trades": len(trade_details),
        "whipsaw_trades": len(whipsaw_trades),
    }


# Example usage and testing
if __name__ == "__main__":
    # Generate sample price data
    np.random.seed(42)
    trend = np.cumsum(np.random.randn(1000) * 0.01)  # Random walk with trend
    noise = np.random.randn(1000) * 0.005  # Add some noise
    sample_prices = 100 * np.exp(trend + noise)  # Convert to price series

    # Test the function
    short_period, long_period = 20, 50

    whipsaw_ratio, trade_details = calculate_ma_whipsaw_ratio(
        sample_prices, short_period, long_period
    )

    enhanced_metrics = calculate_enhanced_whipsaw_metrics(sample_prices, short_period, long_period)

    print(f"Whipsaw Ratio: {whipsaw_ratio:.3f}")
    print(f"Total Trades: {len(trade_details)}")
    print(f"Whipsaw Trades: {enhanced_metrics['whipsaw_trades']}")
    print(f"Average Whipsaw Loss: {enhanced_metrics['avg_whipsaw_loss']:.4f}")
    print(f"Economic Impact: {enhanced_metrics['whipsaw_impact_on_all_trades']:.3f}")

    # Show first few trades
    print("\nFirst 10 trades:")
    print(trade_details.head(10)[["duration", "return", "is_whipsaw"]])
