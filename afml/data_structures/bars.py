import numpy as np
import pandas as pd
from loguru import logger

from .calibration import (
    _calibrate_information_bar_params,
    _calibrate_runs_bar_params,
)
from .information_bars import (
    BarInfoType,
    make_information_bars,
)
from .standard_bars import make_standard_bars
from .utils import (
    _majority_tick_sign_njit,
    _postprocess_bars,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STANDARD_BAR_TYPES = frozenset({"tick", "time", "volume", "dollar"})
_INFO_BAR_TYPES = frozenset(BarInfoType.__args__)

_ALL_BAR_TYPES = _STANDARD_BAR_TYPES | _INFO_BAR_TYPES

# Row count above which make_standard_bars switches to the Polars path.
_POLARS_THRESHOLD = 500_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_tick_sign_per_bar(bars_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `tick_sign` column to bars with the majority tick rule.

    Parameters
    ----------
    bars_df : OHLC bars with DatetimeIndex (bar close times)
    tick_df : ticks with DatetimeIndex and integer 'tick_sign' column (1,-1,0)

    Returns
    -------
    pd.DataFrame with new column 'tick_sign' (str: 'a','b','c')
    """
    if not bars_df.index.is_monotonic_increasing:
        bars_df.sort_index(inplace=True)
    if not tick_df.index.is_monotonic_increasing:
        tick_df.sort_index(inplace=True)

    bar_times = bars_df.index.to_numpy(np.datetime64)
    tick_times = tick_df.index.to_numpy(np.datetime64)
    tick_signs = tick_df["tick_sign"].to_numpy(np.int8)

    # side='right': tick at time T belongs to the bar whose close time >= T
    bar_ids = np.searchsorted(bar_times, tick_times, side="right").astype(np.int64)
    mask = bar_ids < len(bars_df)

    majority = _majority_tick_sign_njit(bar_ids[mask], tick_signs[mask], len(bars_df))

    # Map: -1->'b', 0->'c', 1->'a'; sentinel -128 stays NaN
    lookup = np.array(["b", "c", "a"], dtype=object)
    res_array = np.full(len(bars_df), np.nan, dtype=object)
    has_ticks = majority != np.int8(-128)
    res_array[has_ticks] = lookup[majority[has_ticks].astype(np.int64) + 1]

    bars_df["tick_sign"] = res_array
    return bars_df


def make_bars(
    tick_df: pd.DataFrame,
    bar_type: str = "tick",
    bar_size: int | str = 100,
    price: str = "mid_price",
    tick_num: bool = True,
    verbose: bool = False,
    # --- Information bar calibration (preferred entry point) ---
    target_timeframe: str | None = None,
    ewm_span: int = 20,
    # --- Escape hatches (mutually exclusive with target_timeframe) ---
    exp_ticks_init: float | None = None,
    exp_imbalance_init: float | None = None,
    # --- Run bar manual seeds ---
    exp_runs_buy_init: float | None = None,
    exp_runs_sell_init: float | None = None,
    add_tick_sign: bool = False,
) -> pd.DataFrame:
    """
    Construct OHLC bars from tick data for all supported bar types.

    Standard bar types
    ------------------
    'tick'    : fixed number of ticks per bar
    'time'    : fixed time interval per bar (bar_size = MT5 timeframe, e.g. 'M5')
    'volume'  : fixed cumulative volume per bar
    'dollar'  : fixed cumulative dollar value per bar

    Information bar types (AFML Ch. 1)
    ----------------------------------
    'tick_imbalance'   : sample on cumulative tick imbalance
    'volume_imbalance' : sample on cumulative volume imbalance
    'dollar_imbalance' : sample on cumulative dollar imbalance
    'tick_runs'        : sample on dominant tick run
    'volume_runs'      : sample on dominant volume run
    'dollar_runs'      : sample on dominant dollar run

    Information bar calibration
    ---------------------------
    Information bars are sized by the threshold

        |θ_T| ≥ E_0[T] · |E_0[imbalance per tick]|

    The default entry point is ``target_timeframe``, which derives both
    initial values from a sample of the tick data so that bars close at
    roughly the target clock-time cadence on average. Because the
    threshold adapts via EWM, the initial values only influence the first
    ``ewm_span`` bars; after that the data drives the equilibrium.

    ``target_timeframe`` is an anchor, not a guarantee. Realized bar
    cadence fluctuates with volatility and volume.

    Advanced users can bypass auto-calibration by passing ``exp_ticks_init``
    and optionally ``exp_imbalance_init`` directly. Mixing ``target_timeframe``
    with either raw parameter raises ``ValueError``.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with DatetimeIndex. Required columns:
            bid, ask            (always)
            volume              (volume_*, dollar_* bar types only)

    bar_type : str
        One of the ten bar types listed above.

    bar_size : int or str
        Standard bars: ticks/volume/dollar count (int) or MT5 timeframe
        string for time bars and dynamic tick sizing. Ignored for
        information bars.

    price : str
        Price column for OHLC. One of 'mid_price' (default), 'bid', 'ask',
        'bid_ask'. 'bid_ask' is not supported for information bars.

    tick_num : bool
        Add 'tick_num' column with the 1-based global tick index at bar close.

    verbose : bool
        Log bar count, tick count, and DataFrame structure.

    target_timeframe : str, optional
        Information bars only. MT5 timeframe string expressing the target
        bar cadence. When provided, ``exp_ticks_init`` and
        ``exp_imbalance_init`` are derived automatically from the data.

    ewm_span : int
        Information bars only. EWM span in bars for threshold adaptation.

    exp_ticks_init : int, float, or None
        Information bars only. Initial E_0[T]. Mutually exclusive with
        ``target_timeframe``.

    exp_imbalance_init : float, optional
        Information bars only. Initial E_0[|imbalance per tick|].
        Only meaningful when ``exp_ticks_init`` is also provided.
        Mutually exclusive with ``target_timeframe``.

    exp_runs_buy_init : float, optional
        Run bars only. Initial E_0[theta+/T] (buy-side per-bar expectation).
        If omitted and ``exp_imbalance_init`` is given, that value is used.

    exp_runs_sell_init : float, optional
        Run bars only. Initial E_0[theta-/T] (sell-side per-bar expectation).
        If omitted and ``exp_imbalance_init`` is given, that value is used.

    add_tick_sign : bool, default False
        If True, append a ``'tick_sign'`` column to the returned OHLC DataFrame.
        The column contains the **majority tick rule** for each bar, encoded as:
            ``'a'``  (buy ticks dominate),
            ``'b'``  (sell ticks dominate),
            ``'c'``  (equal or zero ticks).
        This provides a bar‑level categorical sequence ready for entropy
        estimation (e.g., Shannon entropy of order flow direction).
        Requires that ``tick_df`` has a ``'tick_sign'`` column; if missing it
        is automatically computed as ``np.sign(mid_price.diff())``.

    Returns
    -------
    pd.DataFrame
        OHLC bars indexed by bar-close time.

    Raises
    ------
    NotImplementedError
        If bar_type is not recognized.
    ValueError
        If ``target_timeframe`` is combined with raw calibration parameters,
        or if neither is supplied for an information bar type.
    KeyError
        If a required column is missing for the chosen bar type.

    Examples
    --------
    Preferred — auto-calibrated dollar imbalance bars targeting M15 cadence:

    >>> bars = make_bars(tick_df, bar_type="dollar_imbalance", target_timeframe="M15")

    Advanced — explicit initialization (skips auto-calibration):

    >>> bars = make_bars(
    ...     tick_df,
    ...     bar_type="tick_imbalance",
    ...     exp_ticks_init=11_700,
    ...     exp_imbalance_init=0.02,
    ... )

    Standard tick bars are unchanged:

    >>> bars = make_bars(tick_df, bar_type="tick", bar_size=500)
    """
    # ------------------------------------------------------------------
    # 1. Validate bar_type
    # ------------------------------------------------------------------
    if bar_type not in _ALL_BAR_TYPES:
        raise NotImplementedError(
            f"bar_type must be one of {sorted(_ALL_BAR_TYPES)}, got '{bar_type}'"
        )

    is_info_bar = bar_type in _INFO_BAR_TYPES

    # ------------------------------------------------------------------
    # 2. Validate information bar calibration inputs
    # ------------------------------------------------------------------
    if is_info_bar:
        raw_params_given = (exp_ticks_init is not None) or (exp_imbalance_init is not None)

        if target_timeframe is not None and raw_params_given:
            raise ValueError(
                "Pass either 'target_timeframe' (auto-calibration) OR "
                "'exp_ticks_init'/'exp_imbalance_init' (manual), not both."
            )

        if target_timeframe is None and exp_ticks_init is None:
            raise ValueError(
                f"Information bar type '{bar_type}' requires either "
                f"'target_timeframe' (recommended) or 'exp_ticks_init' "
                f"(advanced). Example: target_timeframe='M15'."
            )

    # ------------------------------------------------------------------
    # 3. Shared tick data preparation
    # ------------------------------------------------------------------
    tick_df = tick_df.copy(deep=False)

    if not isinstance(tick_df.index, pd.DatetimeIndex):
        try:
            tick_df.set_index("time", inplace=True)
        except KeyError as e:
            raise TypeError("Could not set 'time' as index.") from e

    if not tick_df.index.is_monotonic_increasing:
        tick_df.sort_index(inplace=True)

    if "mid_price" not in tick_df.columns:
        tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2

    if "spread" not in tick_df.columns:
        tick_df["spread"] = tick_df["ask"] - tick_df["bid"]
        tick_df["spread_bps"] = tick_df["spread"] / tick_df["mid_price"] * 10_000

    # ------------------------------------------------------------------
    # 4. Dispatch
    # ------------------------------------------------------------------
    if is_info_bar:
        if price == "bid_ask":
            raise NotImplementedError("'bid_ask' price mode is not supported for information bars.")

        if target_timeframe is not None:
            if bar_type.endswith("runs"):
                exp_ticks_init, exp_runs_buy_init, exp_runs_sell_init = _calibrate_runs_bar_params(
                    tick_df=tick_df,
                    bar_type=bar_type,
                    target_timeframe=target_timeframe,
                    price_col=price,
                    verbose=verbose,
                )
            else:
                exp_ticks_init, exp_imbalance_init = _calibrate_information_bar_params(
                    tick_df=tick_df,
                    bar_info_type=bar_type,
                    target_timeframe=target_timeframe,
                    price_col=price,
                    verbose=verbose,
                )
        elif exp_ticks_init is not None:
            if bar_type.endswith("runs"):
                if exp_runs_buy_init is None:
                    exp_runs_buy_init = (
                        exp_imbalance_init if exp_imbalance_init is not None else 0.1
                    )
                if exp_runs_sell_init is None:
                    exp_runs_sell_init = (
                        exp_imbalance_init if exp_imbalance_init is not None else 0.1
                    )
            if exp_imbalance_init is None:
                exp_imbalance_init = 0.1

        ohlc_df = make_information_bars(
            tick_df=tick_df,
            bar_info_type=bar_type,
            exp_ticks_init=exp_ticks_init,
            exp_imbalance_init=exp_imbalance_init,
            ewm_span=ewm_span,
            price=price,
            tick_num=tick_num,
            exp_runs_buy_init=exp_runs_buy_init if bar_type.endswith("runs") else None,
            exp_runs_sell_init=exp_runs_sell_init if bar_type.endswith("runs") else None,
        )
    else:
        ohlc_df = make_standard_bars(
            tick_df=tick_df,
            bar_type=bar_type,
            bar_size=bar_size,
            price=price,
            tick_num=tick_num,
        )

    if ohlc_df.empty:
        logger.warning(f"make_bars returned an empty DataFrame for bar_type='{bar_type}'.")
        return ohlc_df

    # ------------------------------------------------------------------
    # 5. Shared post-processing
    # ------------------------------------------------------------------
    bar_label = (
        bar_type
        if is_info_bar
        else (f"{bar_type}-{bar_size}" if bar_type != "time" else bar_size.upper())
    )

    if add_tick_sign:
        if "tick_sign" not in tick_df.columns:
            tick_df["tick_sign"] = np.sign(tick_df["mid_price"].diff()).fillna(0)
        ohlc_df = add_tick_sign_per_bar(ohlc_df, tick_df)

    return _postprocess_bars(
        ohlc_df=ohlc_df,
        verbose=verbose,
        bar_label=bar_label,
        n_ticks=len(tick_df),
    )
