from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..backtest_statistics.performance_analysis import (
    analyze_trading_behavior,
    calculate_performance_metrics,
)


def backtest_model(
    model,
    X_test: pd.DataFrame,
    close_prices: pd.Series,
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    position_sizing: str = "fixed",
    max_position: float = 1.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest a trained model on out-of-sample data.

    Parameters:
    model: Trained model (e.g., Random Forest)
    X_test: Test features
    close_prices: Series of close prices
    initial_capital: Starting capital
    transaction_cost: Transaction cost per trade (percentage)
    position_sizing: 'fixed' or 'confidence'
    max_position: Maximum position size
    risk_free_rate: Annual risk-free rate for Sharpe ratio

    Returns:
    portfolio_history: DataFrame with portfolio values
    metrics: Dictionary of performance metrics
    """
    # Generate predictions
    predictions = model.predict(X_test)
    predictions = pd.Series(predictions, index=X_test.index)

    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        confidence = pd.Series(confidence, index=X_test.index)
    else:
        confidence = pd.Series(1.0, index=X_test.index)

    # Calculate returns (current period, not forward-looking)
    returns = close_prices.pct_change().reindex(X_test.index)

    # Create positions based on predictions and position sizing
    positions = pd.Series(0, index=X_test.index)

    if position_sizing == "fixed":
        # Convert predictions to directional signals if they aren't already
        # Assuming predictions are probabilities, convert to signals
        if predictions.min() >= 0 and predictions.max() <= 1:
            # Predictions are probabilities, convert to signals
            signals = np.where(predictions > 0.5, 1, -1)
        else:
            # Predictions are already signals
            signals = predictions
        positions = pd.Series(signals, index=X_test.index) * max_position

    elif position_sizing == "confidence":
        # For confidence-based sizing, use probability * direction
        if predictions.min() >= 0 and predictions.max() <= 1:
            # Predictions are probabilities
            signals = np.where(predictions > 0.5, 1, -1)
            # Use confidence as position size
            positions = pd.Series(signals, index=X_test.index) * confidence * max_position
        else:
            # Predictions are signals, use as-is
            positions = predictions * confidence * max_position

    # Calculate strategy returns (positions from previous period * current returns)
    strategy_returns = (positions.shift(1) * returns).fillna(0)

    # Apply transaction costs when position changes
    position_changes = positions.diff().fillna(0)
    transaction_costs = abs(position_changes) * transaction_cost
    strategy_returns_net = strategy_returns - transaction_costs

    # Calculate equity curve
    equity_curve = (1 + strategy_returns_net.fillna(0)).cumprod() * initial_capital

    # Create portfolio history
    portfolio_history = pd.DataFrame(
        {
            "close": close_prices.reindex(X_test.index),
            "position": positions,
            "return": strategy_returns_net,
            "equity": equity_curve,
        }
    )

    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        returns=strategy_returns_net,
        data_index=X_test.index,
        positions=positions,
        trading_days_per_year=252,
        trading_hours_per_day=24,
    )

    # Add custom metrics
    metrics["final_capital"] = equity_curve.iloc[-1]
    metrics["total_return"] = (equity_curve.iloc[-1] / initial_capital) - 1
    metrics["sharpe_ratio_annualized"] = metrics["sharpe_ratio"]  # Already annualized
    metrics["transaction_costs"] = transaction_costs.sum()

    # Analyze trading behavior
    trading_behavior = analyze_trading_behavior(positions, strategy_returns_net)
    metrics.update(trading_behavior)

    return portfolio_history, metrics


def backtest_with_labeling(
    labels: pd.DataFrame,
    close_prices: pd.Series,
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
) -> Dict:
    """
    Backtest using the entry/exit points defined by labeling functions

    Parameters:
    labels: DataFrame from triple_barrier_labels or get_bins_from_trend
    close_prices: Series of close prices
    initial_capital: Starting capital
    transaction_cost: Transaction cost per trade

    Returns:
    Dictionary with backtest results
    """
    # Extract trade information from labels
    trade_entries = labels.index
    trade_exits = labels["t1"]
    trade_returns = labels["ret"]
    trade_directions = labels.get("side", 1)  # Default to long if side not specified

    # Initialize tracking variables
    equity_curve = pd.Series(initial_capital, index=close_prices.index)
    current_capital = initial_capital
    active_trades = {}

    # Create a position series for compatibility with performance metrics
    positions = pd.Series(0, index=close_prices.index)

    # Process each trading period
    for timestamp, price in close_prices.items():
        # Check for new trade entries at this timestamp
        if timestamp in trade_entries:
            trade_idx = trade_entries.get_loc(timestamp)
            exit_time = trade_exits.iloc[trade_idx]
            trade_ret = trade_returns.iloc[trade_idx]
            direction = trade_directions.iloc[trade_idx] if hasattr(trade_directions, "iloc") else 1

            # Record this trade
            active_trades[timestamp] = {
                "exit_time": exit_time,
                "expected_return": trade_ret,
                "direction": direction,
                "entry_price": price,
            }

            # Update positions
            positions.loc[timestamp] = direction

        # Check for trade exits at this timestamp
        trades_to_remove = []
        for entry_time, trade_info in active_trades.items():
            if timestamp >= trade_info["exit_time"]:
                # Calculate actual return (might differ from expected due to execution)
                exit_price = close_prices.loc[trade_info["exit_time"]]
                actual_return = (exit_price / trade_info["entry_price"] - 1) * trade_info[
                    "direction"
                ]

                # Apply transaction costs
                actual_return_net = actual_return - transaction_cost * 2  # Entry and exit

                # Update capital
                current_capital *= 1 + actual_return_net

                # Mark for removal
                trades_to_remove.append(entry_time)

                # Update positions
                positions.loc[timestamp] = 0

        # Remove completed trades
        for entry_time in trades_to_remove:
            del active_trades[entry_time]

        # Update equity curve
        equity_curve.loc[timestamp] = current_capital

    # Calculate strategy returns from equity curve
    strategy_returns = equity_curve.pct_change().fillna(0)

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        returns=strategy_returns,
        data_index=close_prices.index,
        positions=positions,
        trading_days_per_year=252,
    )

    # Add trade-based metrics
    performance_metrics["final_capital"] = equity_curve.iloc[-1]
    performance_metrics["total_return"] = (equity_curve.iloc[-1] / initial_capital) - 1

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "returns": strategy_returns,
        "performance_metrics": performance_metrics,
    }
