"""
Various volatility estimators with comprehensive documentation

This module provides multiple volatility estimation methods, each designed to capture
different aspects of price movement and risk characteristics. The choice of estimator
should align with your trading strategy's time horizon and risk profile.
"""

import numpy as np
import pandas as pd

# pylint: disable=redefined-builtin


def get_daily_vol(close: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Daily Volatility Estimates using Exponentially Weighted Moving Average

    Advances in Financial Machine Learning, Snippet 3.1, page 44.

    Computes daily volatility at intraday estimation points using close-to-close returns.
    This is the most common volatility measure used in López de Prado's triple barrier method.

    **What it measures:**
    - Day-to-day price volatility based on closing prices
    - Captures overnight gaps and daily price movements
    - Gives more weight to recent observations via exponential weighting

    **Best used for:**
    - Daily or multi-day trading strategies
    - When overnight risk is relevant to your strategy
    - Triple barrier horizontal thresholds (most common approach)
    - General-purpose volatility estimation

    **Limitations:**
    - Misses intraday volatility patterns
    - May underestimate true volatility for intraday strategies
    - Sensitive to outlier days

    :param close: (pd.Series) Closing prices, datetime indexed
    :param lookback: (int) EWM span parameter - higher values = smoother estimates
    :return: (pd.Series) Daily volatility estimates aligned with close prices
    """
    # Find previous valid trading day for each date
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prev_idx = prev_idx[prev_idx > 0]  # Drop indices before the start

    # Align current and previous closes
    curr_idx = close.index[close.shape[0] - prev_idx.shape[0] :]
    prev_close = close.iloc[prev_idx - 1].values  # Previous day's close
    ret = close.loc[curr_idx] / prev_close - 1
    vol = ret.ewm(span=lookback).std()
    return vol


def get_period_vol(close: pd.Series, lookback: int = 100, **time_delta_kwargs) -> pd.Series:
    """
    Periodic Volatility Estimates with Custom Time Intervals

    Generalizes daily volatility to any time period (hourly, weekly, etc.).
    Useful for strategies operating on non-daily frequencies.

    **What it measures:**
    - Volatility over custom time periods
    - Period-to-period price changes with exponential weighting
    - Flexible time horizon adaptation

    **Best used for:**
    - Non-daily trading frequencies (hourly, 4-hour, weekly)
    - Matching volatility measurement to strategy horizon
    - Cross-timeframe analysis

    **Example usage:**
    - Hourly vol: get_period_vol(close, hours=1)
    - Weekly vol: get_period_vol(close, days=7)
    - 4-hour vol: get_period_vol(close, hours=4)

    :param close: (pd.Series) Closing prices, datetime indexed
    :param lookback: (int) EWM span parameter
    :param time_delta_kwargs: Time components (days, hours, minutes, seconds)
    :return: (pd.Series) Period-specific volatility estimates
    """
    # Find previous valid period for each timestamp
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(**time_delta_kwargs))
    prev_idx = prev_idx[prev_idx > 0]

    # Align current and previous closes
    curr_idx = close.index[close.shape[0] - prev_idx.shape[0] :]
    prev_close = close.iloc[prev_idx - 1].array

    ret = close.loc[curr_idx] / prev_close - 1
    vol = ret.ewm(span=lookback).std()

    return vol


def get_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson Volatility Estimator

    Uses only high and low prices to estimate volatility. More efficient than
    close-to-close volatility as it captures intraday price range information.

    **What it measures:**
    - Intraday volatility based on high-low range
    - True price variation within each period
    - Eliminates overnight gap effects

    **Mathematical foundation:**
    - Based on the range of Brownian motion
    - Approximately 5x more efficient than close-to-close volatility
    - Formula: (1/4ln(2)) * ln(High/Low)²

    **Best used for:**
    - Intraday trading strategies
    - When you want to ignore overnight gaps
    - Continuous trading hours (forex, crypto)
    - More stable volatility estimates

    **Limitations:**
    - Requires high/low data
    - Assumes no price jumps within the period
    - May underestimate volatility in trending markets

    :param high: (pd.Series) High prices for each period
    :param low: (pd.Series) Low prices for each period
    :param window: (int) Rolling window for averaging
    :return: (pd.Series) Parkinson volatility estimates
    """
    ret = np.log(high / low)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret**2)
    return np.sqrt(estimator.rolling(window=window).mean())


def get_garman_klass_vol(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Garman-Klass Volatility Estimator

    Incorporates open, high, low, and close prices for more accurate volatility estimation.
    Extends Parkinson by adding information about opening and closing price relationship.

    **What it measures:**
    - Intraday volatility using full OHLC information
    - Both range-based and close-to-open movements
    - More comprehensive than Parkinson volatility

    **Mathematical foundation:**
    - Combines high-low range with close-open information
    - Formula: 0.5*ln(H/L)² - (2ln(2)-1)*ln(C/O)²
    - Theoretically more efficient than simple range estimators

    **Best used for:**
    - When you have full OHLC data available
    - Intraday strategies requiring precise volatility estimates
    - Markets with significant opening gaps
    - Risk management applications

    **Advantages over Parkinson:**
    - More accurate when opening gaps are significant
    - Better handles markets with auction-based openings
    - Lower estimation error in most market conditions

    **Limitations:**
    - Requires complete OHLC data
    - Still assumes no jumps within periods
    - More complex to compute

    :param open: (pd.Series) Opening prices
    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param close: (pd.Series) Closing prices
    :param window: (int) Rolling window for averaging
    :return: (pd.Series) Garman-Klass volatility estimates
    """
    ret = np.log(high / low)  # High/Low return
    close_open_ret = np.log(close / open)  # Close/Open return
    estimator = 0.5 * ret**2 - (2 * np.log(2) - 1) * close_open_ret**2
    return np.sqrt(estimator.rolling(window=window).mean())


def get_yang_zhang_vol(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Yang-Zhang Volatility Estimator

    The most comprehensive OHLC volatility estimator that handles both overnight
    gaps and intraday movements. Combines overnight, intraday, and Rogers-Satchell
    components for maximum accuracy.

    **What it measures:**
    - Complete volatility including overnight gaps and intraday movements
    - Separates overnight risk from intraday trading risk
    - Drift-independent volatility estimation

    **Mathematical foundation:**
    - Decomposes volatility into: overnight + k*close + (1-k)*Rogers-Satchell
    - k is optimally chosen to minimize estimation variance
    - Handles non-zero drift in underlying price process

    **Best used for:**
    - Daily+ timeframe strategies exposed to overnight risk
    - Markets with significant overnight gaps (equity markets)
    - Triple barrier setting when overnight gaps matter
    - Risk management requiring complete volatility picture
    - Academic research and backtesting

    **Key advantages:**
    - Most accurate volatility estimator for daily+ data
    - Explicitly accounts for overnight gap risk
    - Robust to price drift
    - Lower bias than other OHLC estimators

    **When to prefer over EWM:**
    - Strategy holds positions overnight
    - Significant overnight news/gap risk in your market
    - Need to separate intraday vs overnight volatility components
    - Higher accuracy requirements justify computational complexity

    **Limitations:**
    - Most computationally complex
    - Requires complete OHLC data
    - Less intuitive than simple measures

    :param open: (pd.Series) Opening prices
    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param close: (pd.Series) Closing prices
    :param window: (int) Rolling window for estimation
    :return: (pd.Series) Yang-Zhang volatility estimates
    """
    # Optimal k parameter minimizes estimation variance
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Overnight component: open relative to previous close
    open_prev_close_ret = np.log(open / close.shift(1))

    # Close-to-close component for comparison
    close_prev_open_ret = np.log(close / open.shift(1))

    # Rogers-Satchell components (intraday, drift-free)
    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    # Three volatility components
    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret**2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret**2).rolling(window=window).sum()
    sigma_rs_sq = (
        1
        / (window - 1)
        * (high_close_ret * high_open_ret + low_close_ret * low_open_ret)
        .rolling(window=window)
        .sum()
    )

    # Yang-Zhang combines all components with optimal weighting
    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)


def two_time_scale_realized_vol(tick_prices: pd.Series, slow_freq: str = "5min") -> float:
    """
    Two-Time-Scale Realized Volatility Estimator

    Advanced estimator for tick data that removes microstructure noise while
    preserving information. Combines high-frequency and low-frequency sampling
    to extract true volatility signal.

    **What it measures:**
    - True underlying volatility from noisy tick data
    - Removes bid-ask bounce and other microstructure effects
    - Preserves information lost in simple low-frequency sampling

    **Mathematical foundation:**
    - TSRV = RV_slow - (n_slow/n_fast) * (RV_fast - RV_slow)
    - Uses ratio of observation counts to properly scale noise estimate
    - Asymptotically consistent under jump-diffusion models

    **Best used for:**
    - High-frequency trading strategies
    - When you have access to tick data
    - Precision-critical applications (research, risk management)
    - Markets with significant microstructure noise

    **Advantages:**
    - Most accurate volatility estimate for tick data
    - Removes upward bias from microstructure noise
    - Retains more information than sparse sampling
    - Theoretically well-founded

    **Computational considerations:**
    - More intensive than simple realized volatility
    - Requires choice of slow sampling frequency
    - Benefits increase with data quality and frequency

    **Typical slow frequencies:**
    - 1 minute: Very liquid assets, high precision needed
    - 5 minutes: Most common, good noise reduction
    - 15-30 minutes: Less liquid assets

    :param tick_prices: (pd.Series) Tick-level price data, datetime indexed
    :param slow_freq: (str) Slow sampling frequency ('5min', '1min', etc.)
    :return: (float) Two-time-scale realized volatility estimate
    """
    # Fast scale (tick-by-tick)
    tick_returns = np.log(tick_prices / tick_prices.shift(1)).dropna()
    rv_fast = (tick_returns**2).sum()
    n_fast = len(tick_returns)

    # Slow scale (e.g., 5-minute)
    slow_prices = tick_prices.resample(slow_freq).last().dropna()
    slow_returns = np.log(slow_prices / slow_prices.shift(1)).dropna()
    rv_slow = (slow_returns**2).sum()
    n_slow = len(slow_returns)

    # Two-time-scale estimator with proper scaling
    if n_fast > 0 and n_slow > 0:
        tsrv = rv_slow - (n_slow / n_fast) * (rv_fast - rv_slow)
        return max(tsrv, 0)  # Ensure non-negative result
    else:
        return rv_slow


# Usage guide and selection matrix
"""
VOLATILITY ESTIMATOR SELECTION GUIDE:

╔══════════════════╦═══════════════╦═══════════════════╦═══════════════════════╗
║   Strategy Type  ║   Data Freq   ║   Time Horizon    ║ Recommended Method    ║
╠══════════════════╬═══════════════╬═══════════════════╬═══════════════════════╣
║ Daily+ Swing     ║ Daily OHLC    ║ Days to Weeks     ║ Yang-Zhang            ║
║ Daily+ Trend     ║ Daily Close   ║ Days to Months    ║ get_daily_vol (EWM)   ║
║ Intraday Mean    ║ Intraday OHLC ║ Hours             ║ Garman-Klass          ║
║ Scalping         ║ Minute/Tick   ║ Minutes           ║ Two-Time-Scale RV     ║
║ Crypto/Forex     ║ Any frequency ║ Any               ║ Parkinson (24/7)      ║
║ Risk Management  ║ Daily OHLC    ║ Portfolio level   ║ Yang-Zhang            ║
║ Research/Backtest║ Best available║ Strategy dependent║ Highest quality avail ║
╚══════════════════╩═══════════════╩═══════════════════╩═══════════════════════╝

KEY DECISION FACTORS:
1. **Overnight Risk**: Yang-Zhang if overnight gaps matter, Parkinson if not
2. **Data Availability**: Use highest quality estimator your data supports  
3. **Computational Cost**: Simple EWM for speed, TSRV for accuracy
4. **Market Type**: Equity (Yang-Zhang), Crypto/Forex (Parkinson), HFT (TSRV)
5. **Strategy Horizon**: Match volatility lookback to strategy timeframe
"""
