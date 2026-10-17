"""
Auto-calibration of information bar initial parameters from a target timeframe.

The AFML imbalance/runs threshold

    |θ_T| ≥ E_0[T] · |E_0[imbalance per tick]|

is adaptive via EWM, so initial values only influence the first ``ewm_span``
bars before the data-driven equilibrium takes over. However, the initial
values must be close enough that the first bar closes in finite time and
downstream aggregation does not receive a pathological bar count.

This module derives both initial values from a single user-facing handle:
the target clock-time cadence, expressed as an MT5 timeframe string.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# Lower bound on |E_0[b_t]| to prevent a zero threshold on symmetric tapes.
# Empirically, BTC tick data produces bias values around 0.005–0.02;
# flooring at 0.01 keeps the first-bar threshold meaningful without
# significantly distorting the EWM transient.
_MIN_IMBALANCE_FLOOR = 0.01

# Sample size for empirical imbalance estimation. Large enough to capture
# typical tick dynamics, small enough to keep the estimation cheap.
_CALIBRATION_SAMPLE_SIZE = 500_000

# Safe lower bound for expected per‑tick metric in run‑bar calibration.
# Avoids division‑by‑zero or threshold collapse on one‑sided flow –
# a situation that can occur with very short histories or synthetic tests.
_MIN_RUN_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Imbalance bar calibration
# ---------------------------------------------------------------------------


def _calibrate_information_bar_params(
    tick_df: pd.DataFrame,
    bar_info_type: str,
    target_timeframe: str,
    price_col: str = "mid_price",
    verbose: bool = True,
) -> tuple[int, float]:
    """
    Derive initial E_0[T] and E_0[|imbalance|] for imbalance bars.

    The output is an *anchor* for bar sizing, not a guarantee. Information
    bars sample on information flow, so the realized bar count will
    fluctuate around the target cadence with volatility and volume.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with DatetimeIndex and at least ``price_col`` (and
        ``volume`` for volume/dollar variants).
    bar_info_type : {"tick_imbalance", "volume_imbalance", "dollar_imbalance"}
        The imbalance bar variant.
    target_timeframe : str
        MT5 timeframe string (e.g., 'M5', 'M15', 'H1') expressing the
        target clock-time cadence for bars.
    price_col : str
        Column used to derive the tick direction series.
    verbose : bool
        Log the derived parameters.

    Returns
    -------
    exp_ticks_init : int
        Median number of ticks per ``target_timeframe`` period.
    exp_imbalance_init : float
        Empirical estimate of the expected signed per-tick metric, floored
        at ``_MIN_IMBALANCE_FLOOR`` for tick bars. For volume and dollar
        variants the estimate is multiplied by mean tick volume or dollar
        value respectively.

    Raises
    ------
    ValueError
        If ``bar_info_type`` is not one of the three imbalance variants.
    """
    from .information_bars import _tick_rule
    from .utils import calculate_ticks_per_period

    # E_0[T]: median ticks per target period
    exp_ticks_init = calculate_ticks_per_period(
        tick_df, timeframe=target_timeframe, method="median", verbose=False
    )

    # --- E_0[|imbalance|]: empirical estimate from a sample ---
    n = len(tick_df)
    sample_size = min(_CALIBRATION_SAMPLE_SIZE, n)
    sample = tick_df.iloc[:sample_size]

    prices = sample[price_col].to_numpy(dtype=np.float64)
    b = _tick_rule(prices)  # b_t ∈ {−1, +1}

    emp_bias = abs(b.mean())  # |E[b_t]|

    if bar_info_type == "tick_imbalance":
        exp_imbalance_init = max(emp_bias, _MIN_IMBALANCE_FLOOR)

    elif bar_info_type == "volume_imbalance":
        if "volume" not in sample.columns:
            raise KeyError(f"'volume' column required for '{bar_info_type}' bars.")
        mean_volume = sample["volume"].mean()
        exp_imbalance_init = max(emp_bias, _MIN_IMBALANCE_FLOOR) * mean_volume

    elif bar_info_type == "dollar_imbalance":
        if "volume" not in sample.columns:
            raise KeyError(f"'volume' column required for '{bar_info_type}' bars.")
        mean_dollar = (sample["volume"] * sample[price_col]).mean()
        exp_imbalance_init = max(emp_bias, _MIN_IMBALANCE_FLOOR) * mean_dollar

    else:
        raise ValueError(
            f"Unknown imbalance bar type: '{bar_info_type}'. "
            f"Expected one of 'tick_imbalance', 'volume_imbalance', 'dollar_imbalance'."
        )

    if verbose:
        logger.info(
            f"Auto-calibration for {bar_info_type} at {target_timeframe}: "
            f"exp_ticks_init={exp_ticks_init:,}, "
            f"exp_imbalance_init={exp_imbalance_init:.4g} "
            f"(empirical bias={emp_bias:.4f})"
        )

    return exp_ticks_init, exp_imbalance_init


# ---------------------------------------------------------------------------
# Run bar calibration
# ---------------------------------------------------------------------------


def _calibrate_runs_bar_params(
    tick_df: pd.DataFrame,
    bar_type: str,
    target_timeframe: str,
    price_col: str = "mid_price",
    verbose: bool = True,
) -> tuple[int, float, float]:
    """
    Derive initial E_0[T], E_0[theta+/T], and E_0[theta-/T] for run bars.

    E_0[T] is the median tick count per period at ``target_timeframe``.
    The per‑side expectations are computed by splitting the signed metric
    into buy (positive) and sell (negative) contributions and taking the
    mean of the absolute values. No netting, so the values carry the
    correct units for the chosen bar type (ticks, volume, or dollars).

    Parameters
    ----------
    tick_df : pd.DataFrame
        Cleaned tick data with DatetimeIndex and at least ``price_col``
        (and ``volume`` for volume/dollar variants).
    bar_type : {"tick_runs", "volume_runs", "dollar_runs"}
        The run bar variant.
    target_timeframe : str
        MT5 timeframe string expressing the target cadence.
    price_col : str
        Column used for tick direction and dollar metric.
    verbose : bool
        Log the derived parameters.

    Returns
    -------
    exp_ticks_init : int
        Median tick count per period.
    exp_run_buy_init : float
        Expected buy‑side cumulative metric per bar (theta+/T), floored
        at ``_MIN_RUN_FLOOR``.
    exp_run_sell_init : float
        Expected sell‑side cumulative metric per bar (theta-/T), floored
        at ``_MIN_RUN_FLOOR``.

    Raises
    ------
    ValueError
        If ``bar_type`` is not one of the three run variants.
    """
    from .information_bars import _tick_rule
    from .utils import calculate_ticks_per_period

    # E_0[T]
    exp_ticks_init = calculate_ticks_per_period(
        tick_df, timeframe=target_timeframe, method="median", verbose=False
    )

    # Empirical directional metric
    n = len(tick_df)
    sample_size = min(_CALIBRATION_SAMPLE_SIZE, n)
    sample = tick_df.iloc[:sample_size]

    prices = sample[price_col].to_numpy(dtype=np.float64)
    b = _tick_rule(prices)

    if bar_type == "tick_runs":
        metric = b
    elif bar_type == "volume_runs":
        if "volume" not in sample.columns:
            raise KeyError(f"'volume' column required for '{bar_type}' bars.")
        metric = b * sample["volume"].to_numpy(dtype=np.float64)
    elif bar_type == "dollar_runs":
        if "volume" not in sample.columns:
            raise KeyError(f"'volume' column required for '{bar_type}' bars.")
        metric = (
            b
            * sample[price_col].to_numpy(dtype=np.float64)
            * sample["volume"].to_numpy(dtype=np.float64)
        )
    else:
        raise ValueError(
            f"Unknown run bar type: '{bar_type}'. "
            f"Expected one of 'tick_run', 'volume_run', 'dollar_run'."
        )

    # Split into buy / sell
    buy_mask = metric > 0
    sell_mask = metric < 0

    exp_buy = float(metric[buy_mask].mean()) if buy_mask.any() else _MIN_RUN_FLOOR
    exp_sell = float(np.abs(metric[sell_mask]).mean()) if sell_mask.any() else _MIN_RUN_FLOOR

    # Floors ensure no zero expectations
    exp_buy = max(exp_buy, _MIN_RUN_FLOOR)
    exp_sell = max(exp_sell, _MIN_RUN_FLOOR)

    if verbose:
        logger.info(
            f"Auto-calibration for {bar_type} at {target_timeframe}: "
            f"exp_ticks_init={exp_ticks_init:,}, "
            f"exp_run_buy_init={exp_buy:.4g}, "
            f"exp_run_sell_init={exp_sell:.4g}"
        )

    return exp_ticks_init, exp_buy, exp_sell
