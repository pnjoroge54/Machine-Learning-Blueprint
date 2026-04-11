# src/bars/information_bars.py

from typing import Literal, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..util.misc import log_df_info, optimize_dtypes


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BarInfoType = Literal[
    "tick_imbalance",
    "volume_imbalance",
    "dollar_imbalance",
    "tick_runs",
    "volume_runs",
    "dollar_runs",
]


# ---------------------------------------------------------------------------
# Tick rule
# ---------------------------------------------------------------------------


def _tick_rule(prices: np.ndarray) -> np.ndarray:
    """
    Apply the tick rule to produce a directional series b_t ∈ {-1, +1}.

    Rule (AFML Ch. 1):
        b_t = sign(Δp_t)   if Δp_t ≠ 0
        b_t = b_{t-1}       otherwise  (carry forward)

    The first tick defaults to +1 when Δp_0 = 0.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices in chronological order.

    Returns
    -------
    np.ndarray
        Array of tick directions (+1.0 / -1.0), same length as prices.
    """
    diff = np.diff(prices, prepend=prices[0])   # Δp_t; first diff = 0
    signed = np.sign(diff).astype(float)

    # Replace zeros with NaN so ffill carries the last non-zero sign forward
    signed[signed == 0.0] = np.nan

    b = (
        pd.Series(signed)
        .ffill()
        .fillna(1.0)    # First tick defaults to buy when no prior direction exists
        .to_numpy()
    )
    return b


# ---------------------------------------------------------------------------
# EWM helper
# ---------------------------------------------------------------------------


def _ewm_scalar(values: list[float], span: int) -> float:
    """
    Compute the last value of an EWM mean over a list of scalars.

    Used to update E_0[T] and E_0[|θ|] after each completed bar,
    matching the adaptive threshold update in AFML Ch. 1.

    Parameters
    ----------
    values : list of float
        Historical observations (e.g., bar tick counts or imbalance magnitudes).
    span : int
        EWM span parameter (controls decay rate; larger = slower adaptation).

    Returns
    -------
    float
        The final EWM mean value.
    """
    if not values:
        return 0.0
    return float(
        pd.Series(values, dtype=float)
        .ewm(span=span, adjust=False)
        .mean()
        .iloc[-1]
    )


# ---------------------------------------------------------------------------
# Per-tick metric
# ---------------------------------------------------------------------------


def _compute_metric(
    b: np.ndarray,
    volumes: np.ndarray,
    dollar_values: np.ndarray,
    bar_info_type: BarInfoType,
) -> np.ndarray:
    """
    Compute the signed per-tick metric used to accumulate θ_T.

    Maps each bar type to its corresponding AFML formulation:

        tick_imbalance / tick_runs      : b_t
        volume_imbalance / volume_runs  : b_t · v_t
        dollar_imbalance / dollar_runs  : b_t · p_t · v_t

    Parameters
    ----------
    b            : tick directions (+1/-1), shape (n,)
    volumes      : tick volumes,            shape (n,)
    dollar_values: pre-computed p_t * v_t,  shape (n,)
    bar_info_type: one of the six information bar types

    Returns
    -------
    np.ndarray
        Signed metric per tick, shape (n,).
    """
    if bar_info_type in ("tick_imbalance", "tick_runs"):
        return b
    elif bar_info_type in ("volume_imbalance", "volume_runs"):
        return b * volumes
    elif bar_info_type in ("dollar_imbalance", "dollar_runs"):
        return b * dollar_values
    else:
        raise NotImplementedError(f"Unknown bar_info_type: '{bar_info_type}'")


# ---------------------------------------------------------------------------
# Boundary detection — imbalance bars
# ---------------------------------------------------------------------------


def _detect_imbalance_boundaries(
    metric: np.ndarray,
    exp_ticks_init: float,
    exp_imbalance_init: float,
    ewm_span: int,
) -> np.ndarray:
    """
    Detect bar boundaries for tick / volume / dollar imbalance bars.

    A bar closes at tick T when (AFML Eq. 1.1):

        |θ_T| ≥ E_0[T] · |E_0[imbalance per tick]|

    Both expectations are updated after each completed bar using EWM over
    the history of observed bar tick counts and |θ_T| values.

    Parameters
    ----------
    metric             : signed metric per tick (b_t, b_t·v_t, or b_t·d_t)
    exp_ticks_init     : initial E_0[T]  — expected ticks per bar
    exp_imbalance_init : initial E_0[|imbalance per tick|]
                         For tick bars: E_0[|2·P[b=1] - 1|] ∈ (0, 1]
    ewm_span           : EWM span (in bars) for updating both expectations

    Returns
    -------
    np.ndarray of int
        Indices (0-based, inclusive) of the last tick in each completed bar.
        Incomplete trailing bars are excluded.
    """
    boundaries: list[int] = []

    theta = 0.0
    bar_start = 0

    # Seed history with initial guesses so EWM is well-defined from bar 1
    tick_count_history: list[float] = [exp_ticks_init]
    imbalance_history: list[float] = [abs(exp_imbalance_init) * exp_ticks_init]

    exp_T = exp_ticks_init
    exp_abs_imb = abs(exp_imbalance_init)   # per-tick imbalance expectation

    for t, m in enumerate(metric):
        theta += m

        threshold = exp_T * exp_abs_imb

        if abs(theta) >= threshold:
            boundaries.append(t)

            # --- Update expectations with EWM over bar history ---
            bar_len = float(t - bar_start + 1)
            tick_count_history.append(bar_len)
            imbalance_history.append(abs(theta))

            exp_T = _ewm_scalar(tick_count_history, ewm_span)
            # Normalise: E[|θ_T|] / E[T] ≈ E[|imbalance per tick|]
            exp_abs_imb = _ewm_scalar(imbalance_history, ewm_span) / max(exp_T, 1.0)

            # Reset accumulator for next bar
            theta = 0.0
            bar_start = t + 1

    return np.array(boundaries, dtype=np.intp)


# ---------------------------------------------------------------------------
# Boundary detection — runs bars
# ---------------------------------------------------------------------------


def _detect_runs_boundaries(
    metric: np.ndarray,
    b: np.ndarray,
    exp_ticks_init: float,
    exp_runs_init: float,
    ewm_span: int,
) -> np.ndarray:
    """
    Detect bar boundaries for tick / volume / dollar runs bars.

    A bar closes at tick T when (AFML Eq. 1.2):

        θ_T = max(Σ_{b=+1} q_t, Σ_{b=-1} q_t) ≥ E_0[T] · max_run_expectation

    Where q_t = |metric_t| (the unsigned magnitude of the per-tick metric).

    The max_run_expectation is updated after each bar using EWM over the
    history of observed θ_T values.

    Parameters
    ----------
    metric         : signed metric per tick; sign encodes direction
    b              : tick directions (+1/-1), used to split buy / sell
    exp_ticks_init : initial E_0[T]
    exp_runs_init  : initial E_0[max run component per tick]
                     For tick bars this is E_0[max(P[b=+1], P[b=-1])] ∈ (0.5, 1]
    ewm_span       : EWM span (in bars) for updating expectations

    Returns
    -------
    np.ndarray of int
        0-based inclusive indices of the last tick in each completed bar.
    """
    boundaries: list[int] = []

    buy_sum = 0.0
    sell_sum = 0.0
    bar_start = 0

    tick_count_history: list[float] = [exp_ticks_init]
    runs_history: list[float] = [abs(exp_runs_init) * exp_ticks_init]

    exp_T = exp_ticks_init
    exp_run = abs(exp_runs_init)

    for t in range(len(metric)):
        q = abs(metric[t])      # Unsigned magnitude
        if b[t] > 0:
            buy_sum += q
        else:
            sell_sum += q

        theta = max(buy_sum, sell_sum)
        threshold = exp_T * exp_run

        if theta >= threshold:
            boundaries.append(t)

            bar_len = float(t - bar_start + 1)
            tick_count_history.append(bar_len)
            runs_history.append(theta)

            exp_T = _ewm_scalar(tick_count_history, ewm_span)
            exp_run = _ewm_scalar(runs_history, ewm_span) / max(exp_T, 1.0)

            buy_sum = 0.0
            sell_sum = 0.0
            bar_start = t + 1

    return np.array(boundaries, dtype=np.intp)


# ---------------------------------------------------------------------------
# OHLC aggregation
# ---------------------------------------------------------------------------


def _aggregate_bars(
    tick_df: pd.DataFrame,
    boundaries: np.ndarray,
    price_col: str,
    tick_num: bool,
) -> pd.DataFrame:
    """
    Aggregate tick data into OHLC bars using pre-detected bar boundaries.

    Each bar spans tick_df.iloc[prev_end : end_idx + 1].  The bar's timestamp
    is the last tick's time + 1 µs, consistent with the convention in make_bars.

    Parameters
    ----------
    tick_df   : tick DataFrame with DatetimeIndex and required columns
    boundaries: 0-based inclusive last-tick indices per bar
    price_col : price column used for OHLC ('mid_price', 'bid', 'ask')
    tick_num  : if True, record the 1-based global tick index at bar formation

    Returns
    -------
    pd.DataFrame
        OHLC bars indexed by bar-close time.  Empty DataFrame if no boundaries.
    """
    if len(boundaries) == 0:
        logger.warning("No bar boundaries detected — returning empty DataFrame.")
        return pd.DataFrame()

    has_volume = "volume" in tick_df.columns
    records: list[dict] = []
    prev_end = 0

    for end_idx in boundaries:
        chunk = tick_df.iloc[prev_end : end_idx + 1]

        if chunk.empty:
            prev_end = end_idx + 1
            continue

        prices = chunk[price_col].to_numpy()

        row: dict = {
            # +1 µs so the bar timestamp is strictly after the last tick,
            # matching the convention used in make_bars for non-time bars
            "time":        chunk.index[-1] + pd.Timedelta(microseconds=1),
            "open":        prices[0],
            "high":        prices.max(),
            "low":         prices.min(),
            "close":       prices[-1],
            "spread":      chunk["spread"].mean(),
            "spread_bps":  chunk["spread_bps"].mean(),
            "tick_volume": len(chunk),
        }

        if has_volume:
            row["volume"] = chunk["volume"].sum()

        if tick_num:
            row["tick_num"] = end_idx + 1      # 1-based global tick index

        records.append(row)
        prev_end = end_idx + 1

    bars = pd.DataFrame(records).set_index("time")
    bars.index.name = "time"
    return bars


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_information_bars(
    tick_df: pd.DataFrame,
    bar_info_type: BarInfoType = "tick_imbalance",
    exp_ticks_init: Union[int, float] = 1_000,
    exp_imbalance_init: float = 0.1,
    ewm_span: int = 20,
    price: str = "mid_price",
    tick_num: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Construct information bars (imbalance or runs) from tick data.

    Implements the framework from:
        López de Prado, M. (2018).
        *Advances in Financial Machine Learning*, Ch. 1.

    Bar Types
    ---------
    **Imbalance bars** close when the cumulative signed imbalance |θ_T| exceeds
    a dynamically updated threshold:

        |θ_T| ≥ E_0[T] · |E_0[imbalance per tick]|

    **Runs bars** close when the dominant directional run exceeds a threshold:

        max(Σ_{b=+1} q_t, Σ_{b=-1} q_t) ≥ E_0[T] · E_0[max run per tick]

    Both thresholds adapt after each completed bar via EWM over bar history.

    The per-tick metric q_t differs by bar type:

    +-----------------------+------------------+
    | bar_info_type         | metric q_t       |
    +=======================+==================+
    | tick_imbalance/runs   | b_t              |
    +-----------------------+------------------+
    | volume_imbalance/runs | b_t · v_t        |
    +-----------------------+------------------+
    | dollar_imbalance/runs | b_t · v_t · p_t  |
    +-----------------------+------------------+

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with DatetimeIndex.  Required columns:

            bid, ask            (always required)
            volume              (required for volume_* and dollar_* bar types)

        Optional pre-computed columns (computed here if absent):
            mid_price, spread, spread_bps

    bar_info_type : str
        One of:
            'tick_imbalance'   | 'tick_runs'
            'volume_imbalance' | 'volume_runs'
            'dollar_imbalance' | 'dollar_runs'

    exp_ticks_init : int or float
        Initial guess for E_0[T] — expected number of ticks per bar.
        Use ``calculate_ticks_per_period`` from the standard bars module
        to derive a data-driven estimate.

    exp_imbalance_init : float
        Initial guess for the expected signed imbalance per tick (imbalance bars)
        or the expected max run component per tick (runs bars).

        Guidance by bar type:
            tick_imbalance / tick_runs:
                ≈ E_0[|2·P[b=+1] - 1|] or E_0[max(P[b=+1], P[b=-1])]
                Typical range: 0.01 – 0.5
            volume / dollar variants:
                Same interpretation but scaled by average tick volume or
                average dollar value; tune empirically.

    ewm_span : int
        EWM span (in completed bars) controlling how quickly thresholds adapt.
        Smaller → faster adaptation; larger → more stable thresholds.
        Typical range: 5 – 50.

    price : str
        Price column used for OHLC construction.
        One of: 'mid_price' (default), 'bid', 'ask'.

    tick_num : bool
        If True, adds a 'tick_num' column containing the 1-based global tick
        index at which each bar closed.

    verbose : bool
        If True, logs bar count, tick count, and DataFrame structure.

    Returns
    -------
    pd.DataFrame
        OHLC bars indexed by bar-close time (last tick + 1 µs) with columns:

            open, high, low, close      OHLC prices
            spread                      mean bid-ask spread per bar
            spread_bps                  mean spread in basis points per bar
            tick_volume                 number of ticks in bar
            volume                      sum of tick volumes (if available)
            tick_num                    global tick index at bar close (if tick_num=True)

    Raises
    ------
    NotImplementedError
        If bar_info_type is not one of the six supported types.
    KeyError
        If 'volume' column is missing for volume_* or dollar_* bar types.
    TypeError
        If a DatetimeIndex cannot be established from tick_df.

    Examples
    --------
    >>> from src.bars.bars import calculate_ticks_per_period
    >>> from src.bars.information_bars import make_information_bars

    >>> # Estimate a sensible starting point
    >>> exp_T = calculate_ticks_per_period(tick_df, timeframe="M5", method="median")

    >>> bars = make_information_bars(
    ...     tick_df,
    ...     bar_info_type="dollar_imbalance",
    ...     exp_ticks_init=exp_T,
    ...     exp_imbalance_init=0.05,
    ...     ewm_span=20,
    ...     verbose=True,
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Validate bar_info_type
    # ------------------------------------------------------------------
    _VALID_BAR_INFO_TYPES: set[str] = {
        "tick_imbalance", "volume_imbalance", "dollar_imbalance",
        "tick_runs",      "volume_runs",      "dollar_runs",
    }
    if bar_info_type not in _VALID_BAR_INFO_TYPES:
        raise NotImplementedError(
            f"bar_info_type must be one of {_VALID_BAR_INFO_TYPES}, "
            f"got '{bar_info_type}'"
        )

    needs_volume = bar_info_type not in ("tick_imbalance", "tick_runs")
    if needs_volume and "volume" not in tick_df.columns:
        raise KeyError(
            f"'volume' column is required for '{bar_info_type}' bars."
        )

    # ------------------------------------------------------------------
    # 2. Prepare tick DataFrame
    # ------------------------------------------------------------------
    tick_df = tick_df.copy(deep=False)

    if not isinstance(tick_df.index, pd.DatetimeIndex):
        try:
            tick_df.set_index("time", inplace=True)
        except KeyError as e:
            raise TypeError("Could not set 'time' as index.") from e

    if not tick_df.index.is_monotonic_increasing:
        tick_df.sort_index(inplace=True)

    # Compute derived price columns only when absent
    if "mid_price" not in tick_df.columns:
        tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2
    if "spread" not in tick_df.columns:
        tick_df["spread"] = tick_df["ask"] - tick_df["bid"]
    if "spread_bps" not in tick_df.columns:
        tick_df["spread_bps"] = (
            tick_df["spread"] / tick_df["mid_price"] * 10_000
        )

    # ------------------------------------------------------------------
    # 3. Tick rule → direction series b_t
    # ------------------------------------------------------------------
    prices = tick_df[price].to_numpy()
    b = _tick_rule(prices)

    # ------------------------------------------------------------------
    # 4. Per-tick metric
    # ------------------------------------------------------------------
    if needs_volume:
        volumes = tick_df["volume"].to_numpy()
        dollar_values = volumes * tick_df["mid_price"].to_numpy()
    else:
        n = len(tick_df)
        volumes = np.ones(n)
        dollar_values = np.ones(n)

    metric = _compute_metric(b, volumes, dollar_values, bar_info_type)

    # ------------------------------------------------------------------
    # 5. Detect bar boundaries
    # ------------------------------------------------------------------
    is_runs_bar = bar_info_type.endswith("runs")

    if is_runs_bar:
        boundaries = _detect_runs_boundaries(
            metric=metric,
            b=b,
            exp_ticks_init=float(exp_ticks_init),
            exp_runs_init=exp_imbalance_init,
            ewm_span=ewm_span,
        )
    else:
        boundaries = _detect_imbalance_boundaries(
            metric=metric,
            exp_ticks_init=float(exp_ticks_init),
            exp_imbalance_init=exp_imbalance_init,
            ewm_span=ewm_span,
        )

    n_bars = len(boundaries)
    n_ticks = len(tick_df)
    avg_ticks = n_ticks / max(n_bars, 1)
    logger.info(
        f"{bar_info_type}: {n_bars:,} bars from {n_ticks:,} ticks "
        f"(avg {avg_ticks:.0f} ticks/bar)"
    )

    # ------------------------------------------------------------------
    # 6. Aggregate ticks → OHLC
    # ------------------------------------------------------------------
    ohlc_df = _aggregate_bars(
        tick_df=tick_df,
        boundaries=boundaries,
        price_col=price,
        tick_num=tick_num,
    )

    if ohlc_df.empty:
        return ohlc_df

    # ------------------------------------------------------------------
    # 7. Post-processing  (mirrors make_bars)
    # ------------------------------------------------------------------
    try:
        ohlc_df = ohlc_df.tz_convert(None)
    except TypeError:
        logger.warning(
            "Tick data lacks timezone information; skipping tz conversion. "
            "Ensure source data is timezone-aware to avoid downstream ambiguity."
        )

    # verbose=False: optimize_dtypes prints to stdout by default — suppress here
    ohlc_df = optimize_dtypes(ohlc_df, verbose=False)

    if verbose:
        logger.info(f"{bar_info_type} bars contain {ohlc_df.shape[0]:,} rows.")
        logger.info(f"Tick data contains {tick_df.shape[0]:,} rows.")
        log_df_info(ohlc_df)

    return ohlc_df
