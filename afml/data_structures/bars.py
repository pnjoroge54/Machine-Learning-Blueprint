import numpy as np
import pandas as pd
from loguru import logger

from ..cache import smart_cacheable
from ..util.misc import (
    flatten_column_names,
    log_df_info,
    optimize_dtypes,
    set_resampling_freq,
)


@smart_cacheable
def calculate_ticks_per_period(
    df: pd.DataFrame,
    timeframe: str = "M1",
    method: str = "mean",
    verbose: bool = True,
) -> int:
    """
    Compute the number of ticks per period for dynamic bar sizing using either mean or median.

    Args:
        df (pd.DataFrame): Tick data with a datetime index.
        timeframe (str): Timeframe using MetaTrader5 convention (e.g., 'M1').
        method (str): Calculation method from ['median', 'mean']
        verbose (bool): Whether to logger the result.

    Returns:
        int: Rounded number of ticks per period.
    """
    freq = set_resampling_freq(timeframe)
    resampled = df.resample(freq).size().values  # Count all rows, not just non-NaN values
    fn = getattr(np, method)  # function used for getting ticks in period
    num_ticks = fn(resampled)
    num_rounded = int(round(num_ticks))

    # Round dynamically based on magnitude
    num_digits = len(str(num_rounded)) - 1
    rounded_ticks = int(round(num_rounded, -num_digits))
    rounded_ticks = max(10, rounded_ticks)  # Make 10 ticks the minimum bar size

    if verbose:
        t0, t1 = (df.index[i].date() for i in (0, -1))
        logger.info(
            f"{method.title()} {timeframe} ticks = {num_rounded:,} -> "
            f"{rounded_ticks:,} ({t0} to {t1})"
        )

    return rounded_ticks


def _make_bar_type_grouper(
    df: pd.DataFrame, bar_type: str = "tick", bar_size: int = 100, timeframe: str = "M1"
) -> tuple[pd.core.groupby.generic.DataFrameGroupBy, int]:
    """
    Create a grouped object for aggregating tick data into time/tick/dollar/volume bars.

    Args:
        df: DataFrame with tick data (index should be datetime for time bars).
        bar_type: Type of bar ('time', 'tick', 'dollar', 'volume').
        bar_size: Number of ticks/dollars/volume per bar (ignored for time bars).
        timeframe: Timeframe for resampling (e.g., 'H1', 'D1', 'W1').

    Returns:
        - GroupBy object for aggregation
        - Calculated bar_size (for tick/dollar/volume bars)
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
        freq = set_resampling_freq(timeframe)
        bar_group = (
            df.resample(freq, closed="left", label="right")
            if not freq.startswith(("B", "W"))
            else df.resample(freq)
        )
        return bar_group, 0  # bar_size not used

    # Dynamic bar sizing
    if bar_size == 0:
        if bar_type == "tick":
            bar_size = calculate_ticks_per_period(df, timeframe)
        else:
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

    return df.groupby(bar_id), bar_size


@smart_cacheable
def make_bars(
    tick_df: pd.DataFrame,
    bar_type: str = "tick",
    timeframe: str = "M1",
    price: str = "mid_price",
    bar_size: int = 0,
    drop_zero_volume: bool = True,
    verbose: bool = False,
):
    """
    Constructs OHLC bars from tick data.

    Args:
        tick_df (pd.DataFrame): Tick data.
        bar_type (str): Bar type ('tick', 'time', 'volume', 'dollar').
        timeframe (str): Timeframe for calculation.
        price (str): Price field strategy ('bid', 'ask', 'mid_price', 'bid_ask').
        bar_size (int): For non-time bars; if 0, dynamic calculation is used.
        drop_zero_volume (bool): If True, drops bars with zero tick volume.
        verbose (bool): Prints runtime details if True.

    Returns:
        pd.DataFrame: OHLC bars with additional metrics.
    """
    if price == "mid_price" and price not in tick_df.columns:
        # Modifies the tick_df, which is OK as we'll not want to keep
        # repeating this calculation when making other bars.
        tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2

    price_cols = ["bid", "ask"] if price == "bid_ask" else [price]
    if bar_type in ("volume", "dollar"):
        if "volume" not in tick_df:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        price_cols.append("volume")  # Add volume for dollar- and volume- bars

    bar_group, bar_size_ = _make_bar_type_grouper(
        tick_df[price_cols], bar_type, bar_size, timeframe
    )

    if price != "bid_ask":
        ohlc_df = bar_group[price].ohlc().astype("float64")
    else:
        ohlc_df = bar_group.agg({k: "ohlc" for k in ("bid", "ask")}).astype("float64")
        ohlc_df.columns = flatten_column_names(ohlc_df)
        # Make OHLC using mid-price
        for col in ["open", "high", "low", "close"]:
            ohlc_df[col] = ohlc_df.filter(regex=col).sum(axis=1).div(2)
        ohlc_df["spread"] = ohlc_df.ask_close - ohlc_df.bid_close

    ohlc_df["tick_volume"] = bar_group.size() if bar_type != "tick" else bar_size_

    if "volume" in tick_df.columns:
        ohlc_df["volume"] = bar_group["volume"].sum()

    if bar_type == "time":
        eq_zero = ohlc_df["tick_volume"] == 0
        nzeros = eq_zero.sum()
        nrows = ohlc_df.shape[0]
        msg = f"{nzeros:,} of {nrows:,} ({nzeros / nrows:.2%}) rows with zero tick volume."
        if drop_zero_volume:
            # Drop bars with zero tick volume
            ohlc_df = ohlc_df[~eq_zero]  # drop bars with zero ticks
            if nzeros > 0:
                logger.info(f"Dropped {msg}")
        else:
            ohlc_df = ohlc_df.ffill().dropna()  # Forward fill to ensure no NaNs
            if nzeros > 0:
                logger.info(f"Forward filled {msg}")
    else:
        ohlc_df.index = bar_group["time"].last() + pd.Timedelta(
            microseconds=1
        )  # Ensure end time is after last tick
        if len(tick_df) % bar_size_ > 0:
            ohlc_df = ohlc_df.iloc[:-1]

    try:
        ohlc_df = ohlc_df.tz_convert(None)  # Remove timezone information from index
    except TypeError:
        logger.warning(
            "The tick data used to construct 'ohlc_df' lacks timezone information; skipping tz conversion. \
                Ensure source data is timezone-aware to avoid downstream ambiguity."
        )

    ohlc_df = optimize_dtypes(ohlc_df)  # Save memory

    if verbose:
        tm_info = f"{bar_type}-{bar_size_:,}" if (bar_type != "time") else f"{timeframe}"
        logger.info(f"{tm_info} bars contain {ohlc_df.shape[0]:,} rows.")
        logger.info(f"Tick data contains {tick_df.shape[0]:,} rows.")
        log_df_info(ohlc_df)

    return ohlc_df
