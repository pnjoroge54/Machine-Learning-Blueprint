from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from ..util.misc import (
    flatten_column_names,
    log_df_info,
    optimize_dtypes,
    set_resampling_freq,
)


def calculate_ticks_per_period(
    df: pd.DataFrame,
    timeframe: str = "M1",
    method: str = "median",
    verbose: bool = True,
) -> int:
    """
    Compute the number of ticks per period for dynamic bar sizing using either mean or median.

    Args:
        df (pd.DataFrame): Tick data with a datetime index.
        timeframe (str): Timeframe using MetaTrader5 convention (e.g., 'M1').
        method (str): Calculation method from ['median', 'mean']
        verbose (bool): Whether to log the result.

    Returns:
        int: Rounded number of ticks per period.
    """
    freq = set_resampling_freq(timeframe)
    resampled = (
        df.resample(freq).size().values
    )  # Count all rows, not just non-NaN values
    fn = getattr(np, method)  # function used for getting ticks in period
    num_ticks = fn(resampled)
    num_rounded = int(round(num_ticks))

    # Round dynamically based on magnitude
    num_digits = min(2, len(str(num_rounded)) - 1)
    rounded_ticks = int(round(num_rounded, -num_digits))
    rounded_ticks = max(10, rounded_ticks)  # Make 10 ticks the minimum bar size

    if verbose:
        t0, t1 = (x.date() for x in df.index[[0, -1]])
        logger.info(
            f"{method.title()} {timeframe} ticks = {num_rounded:,} -> "
            f"{rounded_ticks:,} ({t0} to {t1})"
        )

    return rounded_ticks


def _make_bar_type_grouper(
    df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size: Union[int, str] = 100,
) -> tuple[pd.DataFrame.groupby, int]:
    """
    Create a grouped object for aggregating tick data into time/tick/dollar/volume bars.

    Args:
        df: DataFrame with tick data (index should be datetime for time bars).
        bar_type: Type of bar ('time', 'tick', 'dollar', 'volume').
        bar_size:
            - Timeframe for resampling (e.g., 'H1', 'D1', 'W1') for time bars.
            - Number of ticks/dollars/volume per bar (ignored for time bars).

    Returns:
        - GroupBy object for aggregation
        - Calculated bar_size (for tick/dollar/volume bars)
        - Bar ids
    """
    df = df.copy(deep=False)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.set_index("time", inplace=True)
        except KeyError as e:
            raise TypeError("Could not set 'time' as index") from e

    # Sort if needed
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    # Time bars
    if bar_type == "time":
        freq = set_resampling_freq(bar_size)
        bar_group = (
            df.resample(freq, closed="left", label="right")
            if not freq.startswith(("B", "W"))
            else df.resample(freq)
        )
        return bar_group, bar_size, None

    # Dynamic bar sizing
    if bar_type == "tick" and isinstance(bar_size, str):
        bar_size = calculate_ticks_per_period(df, bar_size)

    if not isinstance(bar_size, int):
        raise NotImplementedError(
            f"{bar_type} bars require integer bar_size, but you input '{bar_size}'"
        )
    elif bar_size == 0:
        raise NotImplementedError(f"{bar_type} bars require non-zero bar_size")

    # Non-time bars
    df["time"] = df.index  # Add without copying

    if bar_type == "tick":
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type in ("volume", "dollar"):
        if "volume" not in df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")

        # Optimized cumulative sum
        cum_metric = df["volume"] * df["bid"] if bar_type == "dollar" else df["volume"]
        cumsum = cum_metric.cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"{bar_type} bars not implemented")

    return df.groupby(bar_id), bar_size, bar_id


def make_bars(
    tick_df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size: Union[int, str] = 100,
    price: str = "mid_price",
    tick_num: bool = True,
    verbose: bool = False,
):
    """
    Constructs OHLC bars from tick data.

    Args:
        tick_df (pd.DataFrame): Tick data.
        bar_type (str): Bar type ('tick', 'time', 'volume', 'dollar').
        bar_size (int | str): For non-time bars; if str, dynamic calculation is used.
        timeframe (str): Timeframe for calculation.
        price (str): Price field strategy ('bid', 'ask', 'mid_price', 'bid_ask').
        tick_num (bool): Add column with index of which tick where each bar was formed if True.
        verbose (bool): Prints runtime details if True.

    Returns:
        pd.DataFrame: OHLC bars with additional metrics.
    """
    tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2
    if "spread" not in tick_df.columns:
        tick_df["spread"] = tick_df["ask"] - tick_df["bid"]
        tick_df["spread_bps"] = tick_df["spread"] / tick_df["mid_price"] * 10000

    price_cols = ["bid", "ask"] if price == "bid_ask" else [price]
    price_cols += ["spread", "spread_bps"]
    if bar_type in ("volume", "dollar"):
        if "volume" not in tick_df:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        price_cols.append("volume")  # Add volume for dollar- and volume- bars

    bar_group, bar_size, bar_id = _make_bar_type_grouper(
        tick_df[price_cols], bar_type, bar_size
    )

    if price != "bid_ask":
        ohlc_df = bar_group[price].ohlc()
    else:
        ohlc_df = bar_group.agg({k: "ohlc" for k in ("bid", "ask")})
        ohlc_df.columns = flatten_column_names(ohlc_df)
        # Make OHLC using mid-price
        for col in ["open", "high", "low", "close"]:
            ohlc_df[col] = ohlc_df.filter(regex=col).sum(axis=1).div(2)

    ohlc_df["spread"] = bar_group["spread"].mean()
    ohlc_df["spread_bps"] = bar_group["spread_bps"].mean()
    ohlc_df["tick_volume"] = bar_group.size() if bar_type != "tick" else bar_size

    if "volume" in tick_df.columns:
        ohlc_df["volume"] = bar_group["volume"].sum()

    if bar_type == "time":
        eq_zero = ohlc_df["tick_volume"] == 0
        ohlc_df = ohlc_df[~eq_zero]

        nzeros = eq_zero.sum()
        if nzeros > 0:
            nrows = ohlc_df.shape[0]
            msg = f"{nzeros:,} of {nrows:,} ({nzeros / nrows:.2%}) rows with zero tick volume."
            logger.info(f"Dropped {msg}")

        if tick_num:
            ohlc_df["tick_num"] = ohlc_df["tick_volume"].cumsum()  # 1-based index

    else:
        ohlc_df.index = bar_group["time"].last() + pd.Timedelta(
            microseconds=1
        )  # Ensure end time is after last tick

        if len(tick_df) % bar_size > 0:
            ohlc_df = ohlc_df.iloc[:-1]

        if tick_num:
            ohlc_df["tick_num"] = _get_bar_tick_indices(tick_df, bar_size, bar_id)

    try:
        ohlc_df = ohlc_df.tz_convert(None)  # Remove timezone information from index
    except TypeError:
        logger.warning(
            "The tick data used to construct 'ohlc_df' lacks timezone information; skipping tz conversion. \
                Ensure source data is timezone-aware to avoid downstream ambiguity."
        )

    ohlc_df = optimize_dtypes(ohlc_df)  # Save memory

    if verbose:
        bar_info = (
            f"{bar_type}-{bar_size:,}"
            if (bar_type != "time")
            else f"{bar_size.upper()}"
        )
        logger.info(f"{bar_info} bars contain {ohlc_df.shape[0]:,} rows.")
        logger.info(f"Tick data contains {tick_df.shape[0]:,} rows.")
        log_df_info(ohlc_df)

    return ohlc_df


def _get_bar_tick_indices(tick_df, bar_size, bar_id) -> pd.Series:
    """
    Return the tick indices that form each bar.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with datetime index (or 'time' column).
    bar_type : str, default 'tick'
        Bar type ('tick', 'time', 'volume', 'dollar').
    bar_size : int or str, default 100
        Bar size. If str and bar_type='tick', dynamic calculation is used.

    Returns
    -------
    pd.Series
        Series indexed by bar end time with tick number on which bar was formed
    """
    n_ticks = len(tick_df)

    # Find where bar_id changes (new bar starts)
    # diff > 0 indicates a bar boundary
    diff = np.diff(bar_id, prepend=-1)
    boundary_indices = np.where(diff > 0)[0]

    # Last tick indices are one before each boundary
    last_indices = boundary_indices - 1

    # Add final bar if complete
    if n_ticks % bar_size == 0 and n_ticks > 0:
        last_indices = np.append(last_indices, n_ticks - 1)

    # Filter valid indices and set to 1-based index
    last_indices = last_indices[last_indices >= 0] + 1

    return last_indices
