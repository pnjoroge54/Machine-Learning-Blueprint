import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

from ..cache import cacheable
from ..util.misc import (
    log_df_info,
    optimize_dtypes,
    set_resampling_freq,
)


@cacheable(time_aware=True)
def calculate_ticks_per_period(
    df: pd.DataFrame,
    timeframe: str = "M1",
    method: str = "median",
    verbose: bool = True,
) -> int:
    """
    Compute the number of ticks per period for dynamic bar sizing.

    Args:
        df (pd.DataFrame): Tick data with a datetime index.
        timeframe (str): Timeframe using MetaTrader5 convention (e.g., 'M1').
        method (str): Calculation method from ['median', 'mean'].
        verbose (bool): Whether to log the result.

    Returns:
        int: Rounded number of ticks per period.
    """
    freq = set_resampling_freq(timeframe)
    resampled = df.resample(freq).size().values
    fn = getattr(np, method)
    num_ticks = fn(resampled)
    num_rounded = int(round(num_ticks))

    num_digits = min(2, len(str(num_rounded)) - 1)
    rounded_ticks = int(round(num_rounded, -num_digits))
    rounded_ticks = max(10, rounded_ticks)

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
    bar_size: int | str = 100,
) -> tuple[pd.DataFrame.groupby, int, np.ndarray | None]:
    """
    Create a grouped object for aggregating tick data into time/tick/dollar/volume bars.

    Args:
        df: DataFrame with tick data (index should be datetime for time bars).
        bar_type: Type of bar ('time', 'tick', 'dollar', 'volume').
        bar_size: Timeframe string for time bars or integer count for others.

    Returns:
        - GroupBy object for aggregation
        - Resolved bar_size
        - Bar id array (None for time bars)
    """
    df = df.copy(deep=False)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.set_index("time", inplace=True)
        except KeyError as e:
            raise TypeError("Could not set 'time' as index") from e

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    if bar_type == "time":
        freq = set_resampling_freq(bar_size)
        bar_group = (
            df.resample(freq, closed="left", label="right")
            if not freq.startswith(("B", "W"))
            else df.resample(freq)
        )
        return bar_group, bar_size, None

    if bar_type == "tick" and isinstance(bar_size, str):
        bar_size = calculate_ticks_per_period(df, bar_size)

    if not isinstance(bar_size, int):
        raise NotImplementedError(f"{bar_type} bars require integer bar_size, got '{bar_size}'")
    if bar_size == 0:
        raise NotImplementedError(f"{bar_type} bars require non-zero bar_size")

    df["time"] = df.index

    if bar_type == "tick":
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type in ("volume", "dollar"):
        if "volume" not in df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        cum_metric = df["volume"] * df["mid_price"] if bar_type == "dollar" else df["volume"]
        cumsum = cum_metric.cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"{bar_type} bars not implemented")

    return df.groupby(bar_id), bar_size, bar_id


def _postprocess_bars(
    ohlc_df: pd.DataFrame,
    verbose: bool,
    bar_label: str,
    n_ticks: int,
) -> pd.DataFrame:
    """
    Apply post-processing steps shared by all bar types.

    Steps:
        1. Strip timezone from DatetimeIndex.
        2. Downcast dtypes to save memory (suppress stdout).
        3. Optionally log shape and column info.

    Parameters
    ----------
    ohlc_df   : completed OHLC DataFrame before finalisation
    verbose   : whether to log bar/tick counts and df.info()
    bar_label : human-readable label for log messages (e.g. 'tick-100')
    n_ticks   : number of source ticks (for verbose logging)

    Returns
    -------
    pd.DataFrame
        Finalised OHLC DataFrame.
    """
    try:
        ohlc_df = ohlc_df.tz_convert(None)
    except TypeError:
        logger.warning(
            "Tick data lacks timezone information; skipping tz conversion. "
            "Ensure source data is timezone-aware to avoid downstream ambiguity."
        )

    ohlc_df = optimize_dtypes(ohlc_df, verbose=False)

    if verbose:
        logger.info(f"{bar_label} bars contain {ohlc_df.shape[0]:,} rows.")
        logger.info(f"Tick data contains {n_ticks:,} rows.")
        log_df_info(ohlc_df)

    return ohlc_df


@njit(cache=True)
def _get_bar_tick_indices(
    n_ticks: int,
    bar_size: int,
    bar_id: np.ndarray,
) -> np.ndarray:
    """
    Return the 1-based global tick indices at which each standard bar closes.

    Parameters
    ----------
    n_ticks  : number of source ticks
    bar_size : ticks per bar
    bar_id   : per-tick bar membership array

    Returns
    -------
    np.ndarray
        1-based tick indices, one per completed bar.
    """
    diff_ = np.diff(bar_id, prepend=-1)
    boundary_indices = np.where(diff_ > 0)[0]
    last_indices = boundary_indices - 1

    if n_ticks % bar_size == 0 and n_ticks > 0:
        last_indices = np.append(last_indices, n_ticks - 1)

    last_indices = last_indices[last_indices >= 0] + 1
    return last_indices


@njit(cache=True)
def _majority_tick_sign_njit(
    bar_ids: np.ndarray, tick_signs: np.ndarray, n_bars: int
) -> np.ndarray:
    """Compiled kernel: count signs per bar and return the majority sign.

    bar_ids   : int64 array, values in [0, n_bars)
    tick_signs: int8  array, values in {-1, 0, 1}
    Returns   : int8  array of length n_bars; -128 = no ticks for that bar
    """
    counts = np.zeros((n_bars, 3), dtype=np.int64)
    for i in range(len(bar_ids)):
        counts[bar_ids[i], tick_signs[i] + 1] += 1

    result = np.full(n_bars, np.int8(-128), dtype=np.int8)
    for b in range(n_bars):
        best_count = np.int64(0)
        for j in range(3):
            if counts[b, j] > best_count:
                best_count = counts[b, j]
                result[b] = np.int8(j - 1)
    return result
