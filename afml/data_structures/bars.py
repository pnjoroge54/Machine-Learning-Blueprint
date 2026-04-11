from typing import Literal, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..util.misc import (
    flatten_column_names,
    log_df_info,
    optimize_dtypes,
    set_resampling_freq,
)
from .information_bars import (
    _aggregate_bars,
    _compute_metric,
    _detect_imbalance_boundaries,
    _detect_runs_boundaries,
    _tick_rule,
    BarInfoType,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STANDARD_BAR_TYPES = frozenset({"tick", "time", "volume", "dollar"})
_INFO_BAR_TYPES = frozenset({
    "tick_imbalance", "volume_imbalance", "dollar_imbalance",
    "tick_runs",      "volume_runs",      "dollar_runs",
})
_ALL_BAR_TYPES = _STANDARD_BAR_TYPES | _INFO_BAR_TYPES


# ---------------------------------------------------------------------------
# Tick counting helper (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Grouper for standard bars (unchanged)
# ---------------------------------------------------------------------------


def _make_bar_type_grouper(
    df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size: Union[int, str] = 100,
) -> tuple[pd.DataFrame.groupby, int, Union[np.ndarray, None]]:
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
        raise NotImplementedError(
            f"{bar_type} bars require integer bar_size, got '{bar_size}'"
        )
    if bar_size == 0:
        raise NotImplementedError(f"{bar_type} bars require non-zero bar_size")

    df["time"] = df.index

    if bar_type == "tick":
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type in ("volume", "dollar"):
        if "volume" not in df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        cum_metric = df["volume"] * df["bid"] if bar_type == "dollar" else df["volume"]
        cumsum = cum_metric.cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"{bar_type} bars not implemented")

    return df.groupby(bar_id), bar_size, bar_id


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tick index helper (unchanged)
# ---------------------------------------------------------------------------


def _get_bar_tick_indices(
    tick_df: pd.DataFrame,
    bar_size: int,
    bar_id: np.ndarray,
) -> np.ndarray:
    """
    Return the 1-based global tick indices at which each standard bar closes.

    Parameters
    ----------
    tick_df  : source tick DataFrame
    bar_size : ticks per bar
    bar_id   : per-tick bar membership array

    Returns
    -------
    np.ndarray
        1-based tick indices, one per completed bar.
    """
    n_ticks = len(tick_df)

    diff = np.diff(bar_id, prepend=-1)
    boundary_indices = np.where(diff > 0)[0]
    last_indices = boundary_indices - 1

    if n_ticks % bar_size == 0 and n_ticks > 0:
        last_indices = np.append(last_indices, n_ticks - 1)

    last_indices = last_indices[last_indices >= 0] + 1
    return last_indices


# ---------------------------------------------------------------------------
# Standard bar builder (internal)
# ---------------------------------------------------------------------------


def _make_standard_bars(
    tick_df: pd.DataFrame,
    bar_type: str,
    bar_size: Union[int, str],
    price: str,
    tick_num: bool,
) -> pd.DataFrame:
    """
    Build standard (time / tick / volume / dollar) OHLC bars.

    This is the original make_bars logic extracted verbatim so that
    make_bars can cleanly dispatch between standard and information bars.

    Parameters
    ----------
    tick_df  : prepared tick DataFrame (mid_price / spread already added)
    bar_type : 'time', 'tick', 'volume', or 'dollar'
    bar_size : timeframe string or integer count
    price    : price column strategy
    tick_num : whether to add tick_num column

    Returns
    -------
    pd.DataFrame
        OHLC bars (not yet post-processed).
    """
    price_cols = ["bid", "ask"] if price == "bid_ask" else [price]
    price_cols += ["spread", "spread_bps"]

    if bar_type in ("volume", "dollar"):
        if "volume" not in tick_df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        price_cols.append("volume")

    bar_group, bar_size, bar_id = _make_bar_type_grouper(
        tick_df[price_cols], bar_type, bar_size
    )

    # --- OHLC ---
    if price != "bid_ask":
        ohlc_df = bar_group[price].ohlc()
    else:
        ohlc_df = bar_group.agg({k: "ohlc" for k in ("bid", "ask")})
        ohlc_df.columns = flatten_column_names(ohlc_df)
        for col in ["open", "high", "low", "close"]:
            ohlc_df[col] = ohlc_df.filter(regex=col).sum(axis=1).div(2)

    # --- Additional columns ---
    ohlc_df["spread"]       = bar_group["spread"].mean()
    ohlc_df["spread_bps"]   = bar_group["spread_bps"].mean()
    ohlc_df["tick_volume"]  = bar_group.size() if bar_type != "tick" else bar_size

    if "volume" in tick_df.columns:
        ohlc_df["volume"] = bar_group["volume"].sum()

    # --- Bar-type specific index / tick_num ---
    if bar_type == "time":
        eq_zero = ohlc_df["tick_volume"] == 0
        ohlc_df = ohlc_df[~eq_zero]

        nzeros = eq_zero.sum()
        if nzeros > 0:
            nrows = ohlc_df.shape[0]
            logger.info(
                f"Dropped {nzeros:,} of {nrows:,} "
                f"({nzeros / nrows:.2%}) rows with zero tick volume."
            )

        if tick_num:
            ohlc_df["tick_num"] = ohlc_df["tick_volume"].cumsum()

    else:
        ohlc_df.index = bar_group["time"].last() + pd.Timedelta(microseconds=1)

        if len(tick_df) % bar_size > 0:
            ohlc_df = ohlc_df.iloc[:-1]

        if tick_num:
            ohlc_df["tick_num"] = _get_bar_tick_indices(tick_df, bar_size, bar_id)

    return ohlc_df


# ---------------------------------------------------------------------------
# Information bar builder (internal)
# ---------------------------------------------------------------------------


def _make_information_bars(
    tick_df: pd.DataFrame,
    bar_info_type: BarInfoType,
    exp_ticks_init: Union[int, float],
    exp_imbalance_init: float,
    ewm_span: int,
    price: str,
    tick_num: bool,
) -> pd.DataFrame:
    """
    Build information (imbalance / runs) OHLC bars.

    Parameters
    ----------
    tick_df            : prepared tick DataFrame (mid_price / spread already added)
    bar_info_type      : one of the six information bar type strings
    exp_ticks_init     : initial E_0[T]
    exp_imbalance_init : initial E_0[imbalance or run per tick]
    ewm_span           : EWM span in bars for threshold adaptation
    price              : price column for OHLC
    tick_num           : whether to add tick_num column

    Returns
    -------
    pd.DataFrame
        OHLC bars (not yet post-processed).
    """
    needs_volume = bar_info_type not in ("tick_imbalance", "tick_runs")
    if needs_volume and "volume" not in tick_df.columns:
        raise KeyError(f"'volume' column required for '{bar_info_type}' bars.")

    prices = tick_df[price].to_numpy()
    b = _tick_rule(prices)

    if needs_volume:
        volumes      = tick_df["volume"].to_numpy()
        dollar_values = volumes * tick_df["mid_price"].to_numpy()
    else:
        n = len(tick_df)
        volumes       = np.ones(n, dtype=np.float64)
        dollar_values = np.ones(n, dtype=np.float64)

    metric = _compute_metric(b, volumes, dollar_values, bar_info_type)

    is_runs = bar_info_type.endswith("runs")

    if is_runs:
        boundaries = _detect_runs_boundaries(
            metric          = metric,
            b               = b,
            exp_ticks_init  = float(exp_ticks_init),
            exp_runs_init   = exp_imbalance_init,
            ewm_span        = ewm_span,
        )
    else:
        boundaries = _detect_imbalance_boundaries(
            metric              = metric,
            exp_ticks_init      = float(exp_ticks_init),
            exp_imbalance_init  = exp_imbalance_init,
            ewm_span            = ewm_span,
        )

    n_bars = len(boundaries)
    n_ticks = len(tick_df)
    logger.info(
        f"{bar_info_type}: {n_bars:,} bars from {n_ticks:,} ticks "
        f"(avg {n_ticks / max(n_bars, 1):.0f} ticks/bar)"
    )

    return _aggregate_bars(
        tick_df    = tick_df,
        boundaries = boundaries,
        price_col  = price,
        tick_num   = tick_num,
    )


# ---------------------------------------------------------------------------
# Public API — single entry point for all bar types
# ---------------------------------------------------------------------------


def make_bars(
    tick_df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size: Union[int, str] = 100,
    price: str = "mid_price",
    tick_num: bool = True,
    verbose: bool = False,
    # --- Information bar parameters (ignored for standard bar types) ---
    exp_ticks_init: Union[int, float, None] = None,
    exp_imbalance_init: float = 0.1,
    ewm_span: int = 20,
) -> pd.DataFrame:
    """
    Construct OHLC bars from tick data for all supported bar types.

    Standard bar types
    ------------------
    'tick'    : fixed number of ticks per bar
    'time'    : fixed time interval per bar  (bar_size = MT5 timeframe, e.g. 'M5')
    'volume'  : fixed cumulative volume per bar
    'dollar'  : fixed cumulative dollar value per bar

    Information bar types  (AFML Ch. 1)
    ------------------------------------
    'tick_imbalance'   : sample on cumulative tick imbalance
    'volume_imbalance' : sample on cumulative volume imbalance
    'dollar_imbalance' : sample on cumulative dollar imbalance
    'tick_runs'        : sample on dominant tick run
    'volume_runs'      : sample on dominant volume run
    'dollar_runs'      : sample on dominant dollar run

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with DatetimeIndex.  Required columns:
            bid, ask            (always)
            volume              (volume_*, dollar_* bar types only)

    bar_type : str
        Any of the ten bar types listed above.

    bar_size : int or str
        Standard bars  : ticks / volume / dollar count (int), or MT5 timeframe
                         string for time bars and dynamic tick sizing.
        Information bars: ignored — use exp_ticks_init instead.

    price : str
        Price column for OHLC.  One of:
            'mid_price' (default), 'bid', 'ask', 'bid_ask'.
        'bid_ask' is not supported for information bars.

    tick_num : bool
        Add 'tick_num' column with the 1-based global tick index at bar close.

    verbose : bool
        Log bar count, tick count, and DataFrame structure.

    exp_ticks_init : int, float, or None
        Information bars only.  Initial E_0[T] (expected ticks per bar).
        When None, ``calculate_ticks_per_period`` is called automatically
        using the default timeframe 'M1' and method 'median'.

    exp_imbalance_init : float
        Information bars only.  Initial expected imbalance / run per tick.
        Typical range: 0.01 – 0.5.  Default: 0.1.

    ewm_span : int
        Information bars only.  EWM span in bars for threshold adaptation.
        Default: 20.

    Returns
    -------
    pd.DataFrame
        OHLC bars indexed by bar-close time with columns:
            open, high, low, close
            spread, spread_bps
            tick_volume
            volume          (if source data contains 'volume')
            tick_num        (if tick_num=True)

    Raises
    ------
    NotImplementedError
        If bar_type is not one of the ten supported types.
    KeyError
        If a required column is missing for the chosen bar type.

    Examples
    --------
    >>> # Standard tick bars
    >>> make_bars(tick_df, bar_type="tick", bar_size=500)

    >>> # Time bars
    >>> make_bars(tick_df, bar_type="time", bar_size="M5")

    >>> # Dollar imbalance bars with automatic E_0[T] estimation
    >>> make_bars(tick_df, bar_type="dollar_imbalance", exp_imbalance_init=0.05)

    >>> # Tick imbalance bars with manual E_0[T]
    >>> make_bars(tick_df, bar_type="tick_imbalance", exp_ticks_init=300)
    """
    # ------------------------------------------------------------------
    # 1. Validate bar_type
    # ------------------------------------------------------------------
    if bar_type not in _ALL_BAR_TYPES:
        raise NotImplementedError(
            f"bar_type must be one of {sorted(_ALL_BAR_TYPES)}, got '{bar_type}'"
        )

    # ------------------------------------------------------------------
    # 2. Shared tick data preparation
    # ------------------------------------------------------------------
    tick_df = tick_df.copy(deep=False)

    if not isinstance(tick_df.index, pd.DatetimeIndex):
        try:
            tick_df.set_index("time", inplace=True)
        except KeyError as e:
            raise TypeError("Could not set 'time' as index.") from e

    if not tick_df.index.is_monotonic_increasing:
        tick_df.sort_index(inplace=True)

    # Compute derived columns once, before dispatch
    tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2
    if "spread" not in tick_df.columns:
        tick_df["spread"]     = tick_df["ask"] - tick_df["bid"]
        tick_df["spread_bps"] = tick_df["spread"] / tick_df["mid_price"] * 10_000

    # ------------------------------------------------------------------
    # 3. Dispatch
    # ------------------------------------------------------------------
    is_info_bar = bar_type in _INFO_BAR_TYPES

    if is_info_bar:
        if price == "bid_ask":
            raise NotImplementedError(
                "'bid_ask' price mode is not supported for information bars."
            )

        # Auto-estimate E_0[T] when caller omits it
        if exp_ticks_init is None:
            exp_ticks_init = calculate_ticks_per_period(
                tick_df, timeframe="M1", method="median", verbose=verbose
            )

        ohlc_df = _make_information_bars(
            tick_df            = tick_df,
            bar_info_type      = bar_type,      # bar_type IS the bar_info_type here
            exp_ticks_init     = exp_ticks_init,
            exp_imbalance_init = exp_imbalance_init,
            ewm_span           = ewm_span,
            price              = price,
            tick_num           = tick_num,
        )

    else:
        ohlc_df = _make_standard_bars(
            tick_df  = tick_df,
            bar_type = bar_type,
            bar_size = bar_size,
            price    = price,
            tick_num = tick_num,
        )

    if ohlc_df.empty:
        logger.warning(f"make_bars returned an empty DataFrame for bar_type='{bar_type}'.")
        return ohlc_df

    # ------------------------------------------------------------------
    # 4. Shared post-processing
    # ------------------------------------------------------------------
    bar_label = (
        bar_type if is_info_bar
        else (f"{bar_type}-{bar_size:,}" if bar_type != "time" else bar_size.upper())
    )

    return _postprocess_bars(
        ohlc_df   = ohlc_df,
        verbose   = verbose,
        bar_label = bar_label,
        n_ticks   = len(tick_df),
    )
