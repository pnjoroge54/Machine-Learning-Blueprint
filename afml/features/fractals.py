"""
Comprehensive Fractal-Based Features for Financial Market Analysis

This module provides advanced fractal analysis tools for identifying significant market
structure points, trend validation, and whipsaw filtering in financial time series.

Fractals represent natural support/resistance levels and trend reversal points that
occur across multiple timeframes due to the self-similar nature of financial markets.

References:
- Williams, Bill (1998). "Trading Chaos"
- Mandelbrot, Benoit (2004). "The (Mis)Behavior of Markets"
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Define type aliases for clarity
FractalIndicators = Dict[str, pd.Series]
FractalLevels = Dict[str, pd.Series]
FractalTrendFeatures = Dict[str, pd.Series]
FractalSignals = Dict[str, pd.Series]


def calculate_basic_fractals(high: pd.Series, low: pd.Series, n: int = 2) -> FractalIndicators:
    """
    Calculate basic fractal patterns in price data.

    Parameters:
    -----------
    high : pd.Series
        Series of high prices with datetime index
    low : pd.Series
        Series of low prices with datetime index
    n : int, default 2
        Number of bars to look on each side (total pattern length = 2n + 1)

    Returns:
    --------
    FractalIndicators
        Dictionary with keys:
        - 'fractal_high': Bearish fractal indicators (1 where pattern exists)
        - 'fractal_low': Bullish fractal indicators (1 where pattern exists)
    """
    if len(high) != len(low):
        raise ValueError("High and low series must have same length")
    if not isinstance(high.index, pd.DatetimeIndex):
        raise ValueError("High series must have a DatetimeIndex")
    if not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("Low series must have a DatetimeIndex")

    # Calculate fractal patterns
    fractal_high = (high == high.rolling(2 * n + 1, center=True).max()).astype(int)
    fractal_low = (low == low.rolling(2 * n + 1, center=True).min()).astype(int)

    return {"fractal_high": fractal_high, "fractal_low": fractal_low}


def calculate_enhanced_fractals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 2,
    volatility_threshold: float = 0.001,
) -> FractalIndicators:
    """
    Calculate enhanced fractal features with strength measurement and validation.

    Parameters:
    -----------
    high : pd.Series
        Series of high prices with datetime index
    low : pd.Series
        Series of low prices with datetime index
    close : pd.Series
        Series of close prices with datetime index
    n : int, default 2
        Number of bars to look on each side
    volatility_threshold : float, default 0.001
        Minimum price movement threshold for valid fractals (percentage)

    Returns:
    --------
    FractalIndicators
        Dictionary with enhanced fractal features:
        - 'fractal_high': Basic bearish fractal indicators
        - 'fractal_low': Basic bullish fractal indicators
        - 'fractal_high_strength': Strength measurement of bearish fractals
        - 'fractal_low_strength': Strength measurement of bullish fractals
        - 'valid_fractal_high': Validated bearish fractals
        - 'valid_fractal_low': Validated bullish fractals
        - 'fractal_breakout_up': Upward breakout from bullish fractal pattern
        - 'fractal_breakout_down': Downward breakout from bearish fractal pattern
    """
    # Validate inputs
    for series, name in [(high, "high"), (low, "low"), (close, "close")]:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} series must have a DatetimeIndex")

    if len({len(high), len(low), len(close)}) > 1:
        raise ValueError("All input series must have the same length")

    # Calculate basic fractals
    basic_fractals = calculate_basic_fractals(high, low, n)

    # Calculate fractal strength
    fractal_high_strength = np.where(
        basic_fractals["fractal_high"] == 1,
        high / high.rolling(2 * n + 1, center=True).mean() - 1,
        0,
    )

    fractal_low_strength = np.where(
        basic_fractals["fractal_low"] == 1, 1 - low / low.rolling(2 * n + 1, center=True).mean(), 0
    )

    # Validate fractals based on strength threshold
    valid_fractal_high = (basic_fractals["fractal_high"] == 1) & (
        fractal_high_strength > volatility_threshold
    )
    valid_fractal_low = (basic_fractals["fractal_low"] == 1) & (
        fractal_low_strength > volatility_threshold
    )

    # Identify fractal breakouts
    fractal_breakout_up = (
        (close > high.shift(n)) & (basic_fractals["fractal_low"].shift(n) == 1)
    ).astype(int)

    fractal_breakout_down = (
        (close < low.shift(n)) & (basic_fractals["fractal_high"].shift(n) == 1)
    ).astype(int)

    return {
        "fractal_high": basic_fractals["fractal_high"],
        "fractal_low": basic_fractals["fractal_low"],
        "fractal_high_strength": pd.Series(fractal_high_strength, index=high.index),
        "fractal_low_strength": pd.Series(fractal_low_strength, index=low.index),
        "valid_fractal_high": pd.Series(valid_fractal_high.astype(int), index=high.index),
        "valid_fractal_low": pd.Series(valid_fractal_low.astype(int), index=low.index),
        "fractal_breakout_up": pd.Series(fractal_breakout_up, index=close.index),
        "fractal_breakout_down": pd.Series(fractal_breakout_down, index=close.index),
    }


def calculate_fractal_levels(
    high: pd.Series,
    low: pd.Series,
    valid_fractal_high: pd.Series,
    valid_fractal_low: pd.Series,
    lookback_period: int = 20,
) -> FractalLevels:
    """
    Calculate dynamic support and resistance levels based on recent fractals.

    Parameters:
    -----------
    high : pd.Series
        Series of high prices with datetime index
    low : pd.Series
        Series of low prices with datetime index
    valid_fractal_high : pd.Series
        Series indicating valid bearish fractals (1 where valid)
    valid_fractal_low : pd.Series
        Series indicating valid bullish fractals (1 where valid)
    lookback_period : int, default 20
        Number of periods to look back for fractal levels

    Returns:
    --------
    FractalLevels
        Dictionary with fractal-based support/resistance levels:
        - 'resistance_level': Dynamic resistance based on recent bearish fractals
        - 'support_level': Dynamic support based on recent bullish fractals
        - 'distance_to_resistance': Percentage distance to resistance level
        - 'distance_to_support': Percentage distance to support level
    """
    # Validate inputs
    for series, name in [
        (high, "high"),
        (low, "low"),
        (valid_fractal_high, "valid_fractal_high"),
        (valid_fractal_low, "valid_fractal_low"),
    ]:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} series must have a DatetimeIndex")

    # Extract recent valid fractals
    recent_high_fractals = high[valid_fractal_high == 1]
    recent_low_fractals = low[valid_fractal_low == 1]

    # Calculate rolling resistance and support levels
    resistance_level = recent_high_fractals.rolling(lookback_period, min_periods=1).max()
    support_level = recent_low_fractals.rolling(lookback_period, min_periods=1).min()

    # Forward fill to current time
    resistance_level = resistance_level.reindex(high.index, method="ffill")
    support_level = support_level.reindex(low.index, method="ffill")

    # Calculate distances to levels
    current_close = (high + low) / 2  # Use midpoint as reference
    distance_to_resistance = (resistance_level - current_close) / current_close
    distance_to_support = (current_close - support_level) / current_close

    return {
        "resistance_level": resistance_level,
        "support_level": support_level,
        "distance_to_resistance": distance_to_resistance,
        "distance_to_support": distance_to_support,
    }


def calculate_fractal_trend_features(
    close: pd.Series,
    fractal_breakout_up: pd.Series,
    fractal_breakout_down: pd.Series,
    ma_period: int = 20,
) -> FractalTrendFeatures:
    """
    Calculate trend-related features based on fractal analysis.

    Parameters:
    -----------
    close : pd.Series
        Series of close prices with datetime index
    fractal_breakout_up : pd.Series
        Series indicating upward fractal breakouts (1 where breakout occurs)
    fractal_breakout_down : pd.Series
        Series indicating downward fractal breakouts (1 where breakout occurs)
    ma_period : int, default 20
        Period for moving average calculations

    Returns:
    --------
    FractalTrendFeatures
        Dictionary with fractal-based trend features:
        - 'fractal_trend_strength': Strength of trend based on fractal patterns
        - 'fractal_trend_direction': Direction of trend based on fractal patterns
        - 'fractal_ma_ratio': Ratio of price to fractal-based moving average
    """
    # Validate inputs
    for series, name in [
        (close, "close"),
        (fractal_breakout_up, "fractal_breakout_up"),
        (fractal_breakout_down, "fractal_breakout_down"),
    ]:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} series must have a DatetimeIndex")

    # Calculate fractal-based moving average
    fractal_ma = close.rolling(ma_period).mean()

    # Calculate trend strength based on fractal density
    fractal_density = (fractal_breakout_up + fractal_breakout_down).rolling(ma_period).sum()
    trend_strength = fractal_density / ma_period

    # Determine trend direction based on fractal breakouts
    breakout_balance = (fractal_breakout_up - fractal_breakout_down).rolling(ma_period).sum()
    trend_direction = np.sign(breakout_balance)

    # Calculate ratio of price to fractal MA
    ma_ratio = close / fractal_ma - 1

    return {
        "fractal_trend_strength": trend_strength,
        "fractal_trend_direction": trend_direction,
        "fractal_ma_ratio": ma_ratio,
    }


def generate_fractal_signals(
    close: pd.Series,
    fractal_breakout_up: pd.Series,
    fractal_breakout_down: pd.Series,
    trend_direction: pd.Series,
    fractal_low_strength: pd.Series,
    fractal_high_strength: pd.Series,
    volatility: pd.Series,
) -> FractalSignals:
    """
    Generate fractal-based entry signals with built-in whipsaw protection.

    Parameters:
    -----------
    close : pd.Series
        Series of close prices with datetime index
    fractal_breakout_up : pd.Series
        Series indicating upward fractal breakouts
    fractal_breakout_down : pd.Series
        Series indicating downward fractal breakouts
    trend_direction : pd.Series
        Series indicating trend direction from calculate_fractal_trend_features
    fractal_low_strength : pd.Series
        Series with bullish fractal strength measurements
    fractal_high_strength : pd.Series
        Series with bearish fractal strength measurements
    volatility : pd.Series
        Series of volatility measurements (e.g., ATR)

    Returns:
    --------
    FractalSignals
        Dictionary with fractal-based entry signals:
        - 'fractal_buy_signal': Buy signals based on fractal breakouts
        - 'fractal_sell_signal': Sell signals based on fractal breakouts
        - 'signal_strength': Strength of the signal (0-1)
    """
    # Validate inputs
    for series, name in [
        (close, "close"),
        (fractal_breakout_up, "fractal_breakout_up"),
        (fractal_breakout_down, "fractal_breakout_down"),
        (trend_direction, "trend_direction"),
        (fractal_low_strength, "fractal_low_strength"),
        (fractal_high_strength, "fractal_high_strength"),
        (volatility, "volatility"),
    ]:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} series must have a DatetimeIndex")

    # Basic breakout signals
    buy_signals = (fractal_breakout_up == 1) & (trend_direction >= 0)
    sell_signals = (fractal_breakout_down == 1) & (trend_direction <= 0)

    # Calculate signal strength based on fractal strength and volatility
    signal_strength = np.zeros(len(close))
    signal_strength[buy_signals] = fractal_low_strength.shift(2)[buy_signals] / (
        volatility[buy_signals] + 1e-8
    )
    signal_strength[sell_signals] = fractal_high_strength.shift(2)[sell_signals] / (
        volatility[sell_signals] + 1e-8
    )

    # Normalize signal strength
    signal_strength = np.clip(signal_strength, 0, 1)

    return {
        "fractal_buy_signal": pd.Series(buy_signals.astype(int), index=close.index),
        "fractal_sell_signal": pd.Series(sell_signals.astype(int), index=close.index),
        "signal_strength": pd.Series(signal_strength, index=close.index),
    }


# Example usage and comprehensive analysis function
def comprehensive_fractal_analysis(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volatility: pd.Series,
    n: int = 2,
    lookback_period: int = 20,
    ma_period: int = 20,
) -> Dict[str, Union[FractalIndicators, FractalLevels, FractalTrendFeatures, FractalSignals]]:
    """
    Perform comprehensive fractal analysis on price data.

    Parameters:
    -----------
    high : pd.Series
        Series of high prices with datetime index
    low : pd.Series
        Series of low prices with datetime index
    close : pd.Series
        Series of close prices with datetime index
    volatility : pd.Series
        Series of volatility measurements (e.g., ATR) with datetime index
    n : int, default 2
        Number of bars to look on each side for fractal detection
    lookback_period : int, default 20
        Number of periods to look back for fractal levels
    ma_period : int, default 20
        Period for moving average calculations in trend features

    Returns:
    --------
    Dict
        Dictionary containing all fractal analysis results:
        - 'indicators': Basic and enhanced fractal indicators
        - 'levels': Fractal-based support/resistance levels
        - 'trend_features': Trend-related features based on fractals
        - 'signals': Fractal-based trading signals
    """
    # Calculate all fractal components
    indicators = calculate_enhanced_fractals(high, low, close, n)

    levels = calculate_fractal_levels(
        high,
        low,
        indicators["valid_fractal_high"],
        indicators["valid_fractal_low"],
        lookback_period,
    )

    trend_features = calculate_fractal_trend_features(
        close, indicators["fractal_breakout_up"], indicators["fractal_breakout_down"], ma_period
    )

    signals = generate_fractal_signals(
        close,
        indicators["fractal_breakout_up"],
        indicators["fractal_breakout_down"],
        trend_features["fractal_trend_direction"],
        indicators["fractal_low_strength"],
        indicators["fractal_high_strength"],
        volatility,
    )

    return {
        "indicators": indicators,
        "levels": levels,
        "trend_features": trend_features,
        "signals": signals,
    }


if __name__ == "__main__":
    # Example usage
    print("Fractal Analysis Module")
    print("======================")
    print("This module provides comprehensive fractal-based features for market analysis.")
    print("\nAvailable functions:")
    print("- calculate_basic_fractals(): Identify basic fractal patterns")
    print("- calculate_enhanced_fractals(): Enhanced fractals with strength measurement")
    print("- calculate_fractal_levels(): Dynamic support/resistance levels")
    print("- calculate_fractal_trend_features(): Trend analysis using fractals")
    print("- generate_fractal_signals(): Generate entry signals with whipsaw protection")
    print("- comprehensive_fractal_analysis(): Complete fractal analysis pipeline")
