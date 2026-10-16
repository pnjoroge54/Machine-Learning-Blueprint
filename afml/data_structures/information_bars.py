"""
afml/data_structures/information_bars.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
JIT-compiled boundary detection for all six information bar types.

Imbalance bars: _detect_imbalance_boundaries()
Run bars:       _detect_runs_boundaries()

Both functions use O(1) EWM recurrences (no growing list, no pandas .ewm()).
Aggregation is handled upstream by np.add.reduceat().
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

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
# Tick rule (Numba-accelerated)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _tick_rule_kernel(prices: np.ndarray) -> np.ndarray:
    """
    Assign a direction to every tick using the tick rule.

    Returns an array of float64 values in {+1.0, -1.0}.
    The first tick is always +1.0 (matches CTickRule's default state).
    Unchanged prices carry forward the previous direction.

    The output is stored as float64 (not int8) because it is immediately
    multiplied against floating-point metrics; this avoids a cast instruction
    on every multiply inside the boundary loops below.
    """
    n = len(prices)
    b = np.empty(n, dtype=np.float64)
    b[0] = 1.0
    prev = 1.0
    diffs = np.diff(prices)

    for i in range(1, n):
        diff = diffs[i - 1]
        if diff > 0.0:
            prev = 1.0
        elif diff < 0.0:
            prev = -1.0
        b[i] = prev

    return b


def _tick_rule(prices: pd.Series) -> np.ndarray:
    """Apply the tick rule to produce b_t ∈ {-1, +1}.

    Wraps the Numba kernel with type coercion.
    """
    return _tick_rule_kernel(np.ascontiguousarray(prices, dtype=np.float64))


# ---------------------------------------------------------------------------
# Per-tick metric
# ---------------------------------------------------------------------------


def _compute_metric(
    b: np.ndarray,
    volumes: np.ndarray,
    dollar_values: np.ndarray,
    bar_info_type: BarInfoType,
) -> np.ndarray:
    """Compute signed per-tick metric for θ_T accumulation."""
    if bar_info_type in ("tick_imbalance", "tick_runs"):
        return b
    elif bar_info_type in ("volume_imbalance", "volume_runs"):
        return b * volumes
    elif bar_info_type in ("dollar_imbalance", "dollar_runs"):
        return b * dollar_values
    else:
        raise NotImplementedError(f"Unknown bar_info_type: '{bar_info_type}'")


# ---------------------------------------------------------------------------
# Boundary detection — imbalance bars (Numba JIT)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _detect_imbalance_boundaries(
    metric: np.ndarray,
    exp_ticks_init: float,
    exp_imbalance_init: float,
    ewm_alpha: float,
) -> np.ndarray:
    """
    Detect bar-close indices for tick, volume, or dollar imbalance bars.

    A bar closes at tick t when |theta_T| >= E0[T] * E0[|imbalance/tick|].
    Both expectations are updated with an O(1) EWM recurrence at each close.

    Parameters
    ----------
    metric : signed per-tick metric (b_t, b_t*v_t, or b_t*p_t*v_t)
    exp_ticks_init : initial E0[T]
    exp_imbalance_init : initial |E0[imbalance per tick]| (floored at 1e-10)
    ewm_alpha : EWM decay factor = 2 / (span + 1)

    Returns
    -------
    1-D int64 array of bar-close tick indices.
    """
    n = len(metric)
    boundaries = np.empty(n, dtype=np.int64)
    n_bars = 0

    theta = 0.0
    bar_start = 0

    # Safety floors – match MQL5 MathMax(exp_ticks_init, 1.0) etc.
    ewm_T = max(exp_ticks_init, 1.0)
    ewm_abs_th = max(abs(exp_imbalance_init), 1e-10) * ewm_T
    exp_T = ewm_T
    exp_abs_imb = max(abs(exp_imbalance_init), 1e-10)
    one_minus_a = 1.0 - ewm_alpha

    for t in range(n):
        theta += metric[t]
        threshold = exp_T * exp_abs_imb

        if abs(theta) >= threshold:
            boundaries[n_bars] = t
            n_bars += 1

            bar_len = float(t - bar_start + 1)
            # O(1) EWM: ewm = α*x + (1-α)*ewm
            ewm_T = ewm_alpha * bar_len + one_minus_a * ewm_T
            ewm_abs_th = ewm_alpha * abs(theta) + one_minus_a * ewm_abs_th

            exp_T = ewm_T
            exp_abs_imb = ewm_abs_th / max(exp_T, 1.0)

            theta = 0.0
            bar_start = t + 1

    return boundaries[:n_bars]


# ---------------------------------------------------------------------------
# Boundary detection — run bars (Numba JIT)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _detect_runs_boundaries(
    metric: np.ndarray,
    exp_ticks_init: float,
    exp_runs_buy_init: float,
    exp_runs_sell_init: float,
    ewm_alpha: float,
) -> np.ndarray:
    """
    Detect bar-close indices for tick, volume, or dollar run bars.

    A bar closes at tick t when:
        max(theta_buy, theta_sell) >= E0[T] * max(E0[theta+/T], E0[theta-/T])

    Positive metric values accumulate in theta_buy; negative values
    (routed by their absolute value) accumulate in theta_sell.
    This routing uses >= 0 (not > 0) so that zero-sign ticks — which arise
    only on the very first tick — are handled identically to the MQL5 side.

    Both per-side EWM expectations update independently after each close,
    allowing the threshold to adapt to asymmetric buy/sell flow dynamics.

    Parameters
    ----------
    metric : signed per-tick metric (b_t, b_t*v_t, or b_t*p_t*v_t)
    exp_ticks_init : initial E0[T]
    exp_runs_buy_init : initial E0[theta+/T] (same units as metric)
    exp_runs_sell_init : initial E0[theta-/T] (same units as metric)
    ewm_alpha : EWM decay factor = 2 / (span + 1)

    Returns
    -------
    1-D int64 array of bar-close tick indices.
    """
    n = len(metric)
    boundaries = np.empty(n, dtype=np.int64)
    n_bars = 0

    theta_buy = 0.0
    theta_sell = 0.0
    bar_start = 0

    # Safety floors – identical to MQL5 MathMax checks
    ewm_T = max(exp_ticks_init, 1.0)
    ewm_run_buy = max(exp_runs_buy_init, 1e-10) * ewm_T
    ewm_run_sell = max(exp_runs_sell_init, 1e-10) * ewm_T
    exp_T = ewm_T
    exp_buy = max(exp_runs_buy_init, 1e-10)
    exp_sell = max(exp_runs_sell_init, 1e-10)
    one_minus_a = 1.0 - ewm_alpha

    for t in range(n):
        v = metric[t]
        if v >= 0.0:
            theta_buy += v
        else:
            theta_sell += -v

        threshold = exp_T * max(exp_buy, exp_sell)

        if max(theta_buy, theta_sell) >= threshold:
            boundaries[n_bars] = t
            n_bars += 1

            bar_len = float(t - bar_start + 1)
            # O(1) EWM for both sides
            ewm_T = ewm_alpha * bar_len + one_minus_a * ewm_T
            ewm_run_buy = ewm_alpha * theta_buy + one_minus_a * ewm_run_buy
            ewm_run_sell = ewm_alpha * theta_sell + one_minus_a * ewm_run_sell

            exp_T = ewm_T
            exp_buy = ewm_run_buy / max(exp_T, 1.0)
            exp_sell = ewm_run_sell / max(exp_T, 1.0)

            theta_buy = 0.0
            theta_sell = 0.0
            bar_start = t + 1

    return boundaries[:n_bars]


# ---------------------------------------------------------------------------
# EWM alpha helper
# ---------------------------------------------------------------------------


def _ewm_alpha(span: int) -> float:
    """Convert EWM span to decay factor α = 2 / (span + 1)."""
    return 2.0 / (span + 1)


# ---------------------------------------------------------------------------
# Bar density guard
# ---------------------------------------------------------------------------

_BAR_DENSITY_WARN = 0.10  # warn when more than 10% of ticks close a bar
_BAR_DENSITY_ABORT = 0.50  # abort when more than 50% of ticks close a bar


def _check_bar_density(
    n_bars: int,
    n_ticks: int,
    bar_info_type: str = "",
) -> None:
    """
    Validate the bar-to-tick ratio before downstream aggregation.

    Raises RuntimeError if density >= _BAR_DENSITY_ABORT.
    """
    if n_ticks == 0 or n_bars == 0:
        return

    density = n_bars / n_ticks

    if density >= _BAR_DENSITY_ABORT:
        raise RuntimeError(
            f"{bar_info_type}: {n_bars:,} bars from {n_ticks:,} ticks "
            f"(density={density:.1%}); threshold is effectively zero. "
            f"Increase exp_imbalance_init or exp_ticks_init and retry."
        )

    if density >= _BAR_DENSITY_WARN:
        logger.warning(
            f"{bar_info_type}: unusually high bar density={density:.1%} "
            f"({n_bars:,} bars from {n_ticks:,} ticks); "
            f"check exp_imbalance_init and exp_ticks_init."
        )


# ---------------------------------------------------------------------------
# Vectorized OHLC aggregation
# ---------------------------------------------------------------------------


@njit(cache=True)
def _ohlc_from_boundaries(
    prices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """Extract OHLC and tick count per bar from contiguous price array."""
    n = len(starts)
    bar_open = np.empty(n, dtype=np.float64)
    bar_high = np.empty(n, dtype=np.float64)
    bar_low = np.empty(n, dtype=np.float64)
    bar_close = np.empty(n, dtype=np.float64)
    bar_tick_vol = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = starts[i]
        e = ends[i]
        bar_open[i] = prices[s]
        bar_close[i] = prices[e]

        hi = prices[s]
        lo = prices[s]
        for j in range(s + 1, e + 1):
            hi = max(hi, prices[j])
            lo = min(lo, prices[j])
        bar_high[i] = hi
        bar_low[i] = lo
        bar_tick_vol[i] = float(e - s + 1)

    return bar_open, bar_high, bar_low, bar_close, bar_tick_vol


def _aggregate_bars(
    tick_df: pd.DataFrame,
    boundaries: np.ndarray,
    price_col: str,
    tick_num: bool,
    bar_info_type: str = "",
) -> pd.DataFrame:
    """
    Aggregate ticks into OHLC bars using vectorized numpy operations.
    """
    if len(boundaries) == 0:
        logger.warning("No bar boundaries detected; returning empty DataFrame.")
        return pd.DataFrame()

    n_bars = len(boundaries)
    n_ticks = len(tick_df)

    _check_bar_density(n_bars, n_ticks, bar_info_type)

    prices = tick_df[price_col].to_numpy(dtype=np.float64)
    timestamps = tick_df.index.to_numpy()

    has_volume = "volume" in tick_df.columns
    if has_volume:
        volumes = tick_df["volume"].to_numpy(dtype=np.float64)

    has_spread = "spread" in tick_df.columns
    if has_spread:
        spreads = tick_df["spread"].to_numpy(dtype=np.float64)
        spread_bps = tick_df["spread_bps"].to_numpy(dtype=np.float64)

    starts = np.empty(n_bars, dtype=np.intp)
    starts[0] = 0
    if n_bars > 1:
        starts[1:] = boundaries[:-1] + 1
    ends = boundaries.astype(np.intp)

    (bar_open, bar_high, bar_low, bar_close, bar_tick_vol) = _ohlc_from_boundaries(
        prices, starts, ends
    )

    bar_times = pd.DatetimeIndex(timestamps[ends]) + pd.Timedelta(microseconds=1)

    result = {
        "open": bar_open,
        "high": bar_high,
        "low": bar_low,
        "close": bar_close,
        "tick_volume": bar_tick_vol,
    }

    last_idx = int(ends[-1]) + 1

    if has_volume:
        result["volume"] = np.add.reduceat(volumes[:last_idx], starts)

    if has_spread:
        spread_sum = np.add.reduceat(spreads[:last_idx], starts)
        spread_bps_sum = np.add.reduceat(spread_bps[:last_idx], starts)
        result["spread"] = spread_sum / bar_tick_vol
        result["spread_bps"] = spread_bps_sum / bar_tick_vol

    if tick_num:
        result["tick_num"] = ends + 1

    bars = pd.DataFrame(result, index=bar_times)
    bars.index.name = "time"
    return bars


# ---------------------------------------------------------------------------
# Public entry point (called by make_bars)
# ---------------------------------------------------------------------------


def make_information_bars(
    tick_df: pd.DataFrame,
    bar_info_type: BarInfoType,
    exp_ticks_init: float,
    exp_imbalance_init: float,
    ewm_span: int,
    price: str,
    tick_num: bool,
    exp_runs_buy_init: float | None = None,  # new
    exp_runs_sell_init: float | None = None,  # new
) -> pd.DataFrame:
    """
    Build information (imbalance / runs) OHLC bars.

    Parameters
    ----------
    tick_df            : prepared tick DataFrame (mid_price / spread already added)
    bar_info_type      : one of the six information bar type strings
    exp_ticks_init     : initial E_0[T]
    exp_imbalance_init : initial E_0[imbalance or run per tick] (for runs, fallback if no seeds)
    ewm_span           : EWM span in bars for threshold adaptation
    price              : price column for OHLC
    tick_num           : whether to add tick_num column
    exp_runs_buy_init   : optional initial E_0[theta+/T] for run bars
    exp_runs_sell_init  : optional initial E_0[theta-/T] for run bars

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
        volumes = tick_df["volume"].to_numpy()
        dollar_values = volumes * tick_df["mid_price"].to_numpy()
    else:
        n = len(tick_df)
        volumes = np.ones(n, dtype=np.float64)
        dollar_values = np.ones(n, dtype=np.float64)

    metric = _compute_metric(b, volumes, dollar_values, bar_info_type)
    alpha = _ewm_alpha(ewm_span)

    is_runs = bar_info_type.endswith("runs")

    if is_runs:
        # Use dedicated seeds; fallback to single imbalance init if not provided
        buy_init = exp_runs_buy_init if exp_runs_buy_init is not None else exp_imbalance_init
        sell_init = exp_runs_sell_init if exp_runs_sell_init is not None else exp_imbalance_init

        boundaries = _detect_runs_boundaries(
            metric=np.ascontiguousarray(metric),
            exp_ticks_init=float(exp_ticks_init),
            exp_runs_buy_init=buy_init,
            exp_runs_sell_init=sell_init,
            ewm_alpha=alpha,
        )
    else:
        boundaries = _detect_imbalance_boundaries(
            metric=np.ascontiguousarray(metric),
            exp_ticks_init=float(exp_ticks_init),
            exp_imbalance_init=exp_imbalance_init,
            ewm_alpha=alpha,
        )

    n_bars = len(boundaries)
    n_ticks = len(tick_df)
    logger.info(
        f"{bar_info_type}: {n_bars:,} bars from {n_ticks:,} ticks "
        f"(avg {n_ticks / max(n_bars, 1):.0f} ticks/bar)"
    )

    return _aggregate_bars(
        tick_df=tick_df,
        boundaries=boundaries,
        price_col=price,
        tick_num=tick_num,
        bar_info_type=bar_info_type,
    )
