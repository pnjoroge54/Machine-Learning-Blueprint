from typing import Union

import numpy as np
import pandas as pd
from numba import njit, prange


def get_period_returns(close: pd.Series, **time_delta_kwargs) -> pd.Series:
    """
    Compute periodic returns for a given time period, robust to non-consecutive trading days.

    This function calculates returns by finding the closing price from a specified
    time duration (days, hours, minutes) in the past. It handles cases where
    the prior period might not be a trading day by using `searchsorted` to find
    the nearest valid previous index.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) Periodic returns (percentage changes), aligned to the prior valid trading period
    """
    # Find previous valid trading day for each date
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(**time_delta_kwargs))

    # Drop indices that are before the start of the 'close' Series
    prev_idx = prev_idx[prev_idx > 0]

    # Align current and previous closes
    curr_idx = close.index[close.shape[0] - prev_idx.shape[0] :]
    prev_close = close.iloc[prev_idx - 1].values

    ret = close.loc[curr_idx] / prev_close - 1
    return ret


@njit(parallel=True)
def rolling_autocorr_numba(data: np.ndarray, lookback: int) -> np.ndarray:
    """
    Computes rolling autocorrelation for a 1D NumPy array using Numba for performance.

    This function calculates the autocorrelation between `data[t]` and `data[t-1]`
    within a rolling window of `lookback` size. It leverages Numba's `njit` and
    `prange` for parallel execution, making it efficient for large datasets.

    Args:
        data: A 1D NumPy array of numerical data (e.g., returns).
        lookback: The size of the rolling window for autocorrelation calculation.

    Returns:
        A NumPy array containing the rolling autocorrelation values.
        The initial `lookback - 1` values will be NaN as there isn't enough data.
    """
    result = np.full(len(data), np.nan)
    for i in prange(lookback - 1, len(data)):
        window = data[i - lookback + 1 : i + 1]
        # [0, 1] extracts the correlation between the two series (not self-correlation)
        result[i] = np.corrcoef(window[:-1], window[1:])[0, 1]
    return result


def get_period_autocorr(close: pd.Series, lookback: int = 100, **time_delta_kwargs) -> pd.Series:
    """
    Estimates rolling periodic autocorrelation of closing prices.

    This function first calculates the periodic returns using `get_period_returns`
    and then computes the rolling autocorrelation of these returns using the
    Numba-optimized `rolling_autocorr_numba` function.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param lookback: (int) The window equivalent of the Simple Moving Average for the Exponentially Weighted Moving
                average calculation (default is 100)
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) of rolling periodic autocorrelation values, indexed by the datetime index of the input `close` Series.
    """
    ret = get_period_returns(close, **time_delta_kwargs)
    acorr = rolling_autocorr_numba(ret.to_numpy(), lookback)
    df0 = pd.Series(acorr, index=ret.index)
    return df0


def get_lagged_returns(
    prices: Union[pd.Series, pd.DataFrame],
    lags: list,
    nperiods: int = 3,
) -> pd.DataFrame:
    """
    Generates a DataFrame of various lagged returns and optionally forward target returns.

    This function calculates returns for specified lag periods, clips extreme
    values based on quantiles, and then creates additional lagged features
    (e.g., `returns_X_lag_Y`). It can also generate forward returns
    as a target variable.

    Args:
        prices: A pandas Series or DataFrame of close prices. If a Series, it's
                treated as a single instrument. If a DataFrame, each column
                represents a different instrument or asset. The index should
                be datetime-based.
        lags: A list of integers, where each integer represents a lag period
              for which returns should be calculated (e.g., `[1, 5, 20]` for
              daily, weekly, and monthly returns).
        nperiods: The number of additional lagged versions to create for each
                 return series. For example, if `nperiods=3` and `lags=[1]`,
                 it will create `returns_1_lag_1`, `returns_1_lag_2`,
                 `returns_1_lag_3`. Defaults to 3.

    Returns:
        A pandas DataFrame containing the calculated returns and their lagged versions.
        If `target` is True, it will also include forward target returns.
    """
    q = 0.0001  # Quantile cut-off for winsorizing extreme prices
    df = pd.DataFrame()

    for lag in lags:
        # Calculate 1-period geometric mean return of the lag period and
        # winsorize extreme values by clipping.
        df[f"returns_{lag}"] = (
            prices.pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))  # winsorize
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )

    # Create additional lagged versions of the calculated returns
    for t in range(1, nperiods + 1):
        for lag in lags:
            df[f"returns_{lag}_lag_{t}"] = df[f"returns_{lag}"].shift(t * lag)

    df.rename(columns={"returns_1": "returns"}, inplace=True)
    return df


def get_return_dist_features(close, window=10):
    """Distribution of return features"""
    df = pd.DataFrame(index=close.index)
    ret = np.log(close).diff()
    sma_returns = ret.rolling(window, min_periods=3)
    df["returns_normalized"] = (ret - sma_returns.mean()) / sma_returns.std()
    df[f"returns_skew"] = sma_returns.skew()
    df[f"returns_kurtosis"] = sma_returns.kurt()
    return df
