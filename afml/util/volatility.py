"""
Various volatility estimators
"""

import numpy as np
import pandas as pd

# pylint: disable=redefined-builtin


# Snippet 3.1, page 44, Daily Volatility Estimates
def get_daily_vol(close: pd.Series, lookback: int = 100):
    """
    Advances in Financial Machine Learning, Snippet 3.1, page 44.

    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.
    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: (pd.Series) Closing prices
    :param lookback: (int) Lookback period to compute volatility
    :return: (pd.Series) Daily volatility value
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
    Compute the exponentially weighted moving volatility of periodic returns.

    This function first calculates periodic returns using an
    Exponentially Weighted Moving (EWM) standard deviation
    to these returns to estimate volatility.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param lookback: (int) lookback window (default is 100)
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) Periodic volatility values
    """
    # Find previous valid trading day for each date
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(**time_delta_kwargs))

    # Drop indices that are before the start of the 'close' Series
    prev_idx = prev_idx[prev_idx > 0]

    # Align current and previous closes
    curr_idx = close.index[close.shape[0] - prev_idx.shape[0] :]
    prev_close = close.iloc[prev_idx - 1].array

    ret = close.loc[curr_idx] / prev_close - 1
    vol = ret.ewm(span=lookback).std()

    return vol


def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """
    ret = np.log(high / low)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret**2)
    return np.sqrt(estimator.rolling(window=window).mean())


def get_garman_class_vol(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Garman-Class volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """
    ret = np.log(high / low)  # High/Low return
    close_open_ret = np.log(close / open)  # Close/Open return
    estimator = 0.5 * ret**2 - (2 * np.log(2) - 1) * close_open_ret**2
    return np.sqrt(estimator.rolling(window=window).mean())


def get_yang_zhang_vol(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret**2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret**2).rolling(window=window).sum()
    sigma_rs_sq = (
        1
        / (window - 1)
        * (high_close_ret * high_open_ret + low_close_ret * low_open_ret)
        .rolling(window=window)
        .sum()
    )

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)
