"""
Filters are used to filter events based on some kind of trigger. For example a structural break filter can be
used to filter events where a structural break occurs. This event is then used to measure the return from the event
to some event horizon, say a day.
"""

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit


# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
@njit(cache=True)
def _cusum_filter_numba_core(
    log_returns_np: np.ndarray, thresholds_np: np.ndarray, index_values_np: np.ndarray
) -> list:
    """
    Core CUSUM filter logic implemented with Numba for speed.
    """
    t_events_val = []  # Numba will infer the type for the list
    s_pos = 0.0
    s_neg = 0.0

    # Determine if threshold is a single scalar value or an array per data point
    is_threshold_scalar = thresholds_np.shape[0] == 1

    scalar_thresh_val = 0.0
    if is_threshold_scalar:
        scalar_thresh_val = thresholds_np[0]

    for i in range(len(log_returns_np)):
        log_ret = log_returns_np[i]

        current_thresh = scalar_thresh_val if is_threshold_scalar else thresholds_np[i]

        # Calculate potential new s_pos and s_neg
        s_pos_candidate = s_pos + log_ret
        s_neg_candidate = s_neg + log_ret

        # Apply CUSUM logic: reset if sum crosses zero from the wrong direction
        s_pos = max(0.0, s_pos_candidate)
        s_neg = min(0.0, s_neg_candidate)

        # Check for events
        if s_neg < -current_thresh:
            s_neg = 0.0  # Reset the sum that triggered
            t_events_val.append(index_values_np[i])
        elif s_pos > current_thresh:  # `elif` ensures only one event type per step
            s_pos = 0.0  # Reset the sum that triggered
            t_events_val.append(index_values_np[i])

    return t_events_val


def cusum_filter(
    raw_time_series: pd.Series, threshold: Union[float, int, pd.Series], time_stamps: bool = True
):
    """
    Advances in Financial Machine Learning, Snippet 2.4, page 39.

    The Symmetric Dynamic/Fixed CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity
    away from a target value. The filter is set up to identify a sequence of upside or downside divergences from any
    reset level zero. We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by raw_time_series
    hovering around a threshold level, which is a flaw suffered by popular market signals such as Bollinger Bands.
    It will require a full run of length threshold for raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine whether the occurrence
    of such events constitutes actionable intelligence. Below is an implementation of the Symmetric CUSUM filter.

    Note: As per the book this filter is applied to closing prices but we extended it to also work on other
    time series such as volatility.

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, the function captures
                      it as an event, can be dynamic if threshold is pd.Series
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    if not isinstance(raw_time_series, pd.Series):
        if isinstance(raw_time_series, pd.DataFrame) and raw_time_series.shape[1] == 1:
            # If DataFrame with one column, use that column
            ts = raw_time_series.iloc[:, 0].copy()
        else:
            try:
                # Attempt to convert other array-like inputs
                ts = pd.Series(raw_time_series, name="price").copy()
            except Exception as e:
                raise ValueError(
                    "raw_time_series must be a pandas Series, a single-column pandas DataFrame, or convertible to a pandas Series."
                ) from e
    else:
        ts = raw_time_series.copy()  # Work on a copy

    if ts.empty:
        return pd.DatetimeIndex([]) if time_stamps else []

    if (ts <= 0).any():  # Check for non-positive values before log
        raise ValueError(
            "raw_time_series contains non-positive values, cannot compute log returns."
        )

    # Calculate log returns (vectorized)
    log_returns = np.log(ts).diff()

    # Prepare indices and log_returns numpy array (dropping the initial NaN)
    valid_indices = log_returns.index[1:]  # Indices corresponding to actual log return values
    log_returns_np = log_returns.dropna().to_numpy()

    if len(log_returns_np) == 0:  # If series had 0 or 1 element (so no valid log returns)
        return pd.DatetimeIndex([]) if time_stamps else []

    # Prepare thresholds numpy array
    if isinstance(threshold, (float, int)):
        if threshold <= 0:
            raise ValueError("Scalar threshold must be positive.")
        # Numba function expects an array, so wrap scalar threshold
        thresholds_np = np.array([float(threshold)])
    elif isinstance(threshold, pd.Series):
        # Align threshold Series with the valid_indices of log_returns
        # These are the indices for which we have actual log return values
        aligned_threshold = threshold.reindex(valid_indices)
        if aligned_threshold.isnull().any():
            raise ValueError(
                "Threshold series contains NaNs after alignment with log_return series' valid indices, "
                "or does not cover its range. Ensure threshold series index matches raw_time_series index."
            )
        thresholds_np = aligned_threshold.to_numpy()
        if (thresholds_np <= 0).any():
            raise ValueError("Threshold series contains non-positive values.")
        if len(thresholds_np) != len(log_returns_np):
            # This check is mostly a safeguard; reindex should handle lengths correctly if indices are sound.
            raise ValueError(
                f"Threshold series length ({len(thresholds_np)}) does not match "
                f"log returns length ({len(log_returns_np)}) after alignment."
            )
    else:
        raise ValueError("threshold must be a float, int, or pandas Series.")

    # Get the actual index values (e.g., datetime64 objects, integers, etc.) to pass to Numba
    index_values_np = valid_indices.to_numpy()

    # Call the Numba-jitted core function 🚀
    event_indices_values = _cusum_filter_numba_core(log_returns_np, thresholds_np, index_values_np)
    logger.info(f"{len(event_indices_values):,} CUSUM-filtered events")

    if time_stamps:
        # pd.Index can correctly form DatetimeIndex if original index was datetime-like
        return pd.Index(event_indices_values)
    else:
        # event_indices_values is already a list from the Numba function
        return event_indices_values


def z_score_filter(
    raw_time_series: pd.Series,
    mean_window: int,
    std_window: int,
    z_score: float = 3,
    time_stamps: bool = True,
):
    """
    Filter which implements z_score filter
    (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data)

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param mean_window: (int): Rolling mean window
    :param std_window: (int): Rolling std window
    :param z_score: (float): Number of standard deviations to trigger the event
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    t_events = raw_time_series[
        raw_time_series
        >= raw_time_series.rolling(window=mean_window).mean()
        + z_score * raw_time_series.rolling(window=std_window).std()
    ].index
    logger.info(f"{len(t_events):,} Z-Score filtered events")
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps
    return t_events
