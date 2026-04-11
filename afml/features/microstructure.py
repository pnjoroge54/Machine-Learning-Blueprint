# src/features/microstructure.py

"""
Market Microstructure Features — AFML Chapter 2.

Implements:
    1.  Tick rule (already in information_bars.py — re-exported here)
    2.  Roll model (effective spread proxy)
    3.  High-Low volatility estimator (Corwin & Schultz)
    4.  Kyle's Lambda (price impact)
    5.  Amihud's Lambda (illiquidity)
    6.  Hasbrouck's Lambda (price impact via IV estimator)
    7.  VPIN (Volume-Synchronized Probability of Informed Trading)
    8.  Tick rule–based bar features (per-bar aggregation)

Performance strategy:
    - All inner loops use Numba @njit with cache=True.
    - Embarrassingly parallel loops use prange + parallel=True.
    - Pandas is only used at the outermost layer (construction / indexing).
    - Memory: float32 used throughout; arrays pre-allocated; no intermediate copies.

References:
    López de Prado (2018), AFML Ch. 2
    mlfinlab: https://github.com/hudson-and-thames/mlfinlab
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit, prange

from ..bars.information_bars import _tick_rule


# ---------------------------------------------------------------------------
# Numba kernels — pure array operations
# ---------------------------------------------------------------------------


@njit(cache=True)
def _roll_measure_kernel(
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Roll (1984) effective spread estimator.

    The Roll model estimates the effective spread from the serial
    covariance of price changes:

        spread = 2 * sqrt(max(-cov(Δp_t, Δp_{t-1}), 0))

    where the covariance is computed over a rolling window.

    When the covariance is positive (theoretically impossible under the
    Roll model but common empirically), the result is set to NaN.

    Parameters
    ----------
    close  : 1-D array of close prices
    window : rolling window length

    Returns
    -------
    roll : float32 array, same length as close
        NaN for the first (window) elements.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float32)

    # Pre-compute price differences
    diff = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        diff[i] = close[i + 1] - close[i]

    # Rolling covariance of diff[t] and diff[t-1]
    for i in range(window, n):
        # Slice: diff[i-window : i-1]  (paired lags)
        # We need pairs (diff[t], diff[t-1]) for t in window
        # That is indices [i-window .. i-2] paired with [i-window+1 .. i-1]
        start = i - window
        end   = i - 1          # exclusive upper bound for the leading series

        # Means
        m0, m1 = 0.0, 0.0
        cnt = end - start       # number of pairs = window - 1
        for j in range(cnt):
            m0 += diff[start + j]
            m1 += diff[start + j + 1]
        m0 /= cnt
        m1 /= cnt

        # Covariance
        cov = 0.0
        for j in range(cnt):
            cov += (diff[start + j] - m0) * (diff[start + j + 1] - m1)
        cov /= cnt

        if cov < 0.0:
            out[i] = np.float32(2.0 * np.sqrt(-cov))
        # else: leave NaN (positive covariance → model not applicable)

    return out


@njit(cache=True)
def _roll_impact_kernel(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Roll impact: Roll spread normalised by dollar volume.

        roll_impact = roll_spread / (close * volume)

    Parameters
    ----------
    close  : 1-D close prices
    volume : 1-D tick volumes (same length)
    window : rolling window for Roll spread

    Returns
    -------
    float32 array, same length as close
    """
    roll = _roll_measure_kernel(close, window)
    n    = len(close)
    out  = np.full(n, np.nan, dtype=np.float32)

    for i in range(window, n):
        dv = close[i] * volume[i]
        if dv > 0.0:
            out[i] = np.float32(roll[i] / dv)

    return out


@njit(cache=True)
def _corwin_schultz_kernel(
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corwin & Schultz (2012) high-low spread estimator.

    Derives an annualised volatility and a bid-ask spread estimate from
    daily high / low prices.

    Algorithm (AFML p. 285):

        β = (ln H_t/L_t)^2 + (ln H_{t+1}/L_{t+1})^2        (adjacent-day sum)
        γ = (ln max(H_t, H_{t+1}) / min(L_t, L_{t+1}))^2   (two-day range)

        k_2 = (8/π)^0.5                                      (constant ≈ 1.596)

        α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))

        spread = 2 * (exp(α) - 1) / (1 + exp(α))
        sigma  = √(β/2) / k2                                  (daily volatility)

    Parameters
    ----------
    high : 1-D array of bar high prices
    low  : 1-D array of bar low prices

    Returns
    -------
    spread : float32 array, len(high) — NaN at index 0
    sigma  : float32 array, len(high) — NaN at index 0
    """
    n      = len(high)
    spread = np.full(n, np.nan, dtype=np.float32)
    sigma  = np.full(n, np.nan, dtype=np.float32)

    k2     = np.float64((8.0 / np.pi) ** 0.5)          # ≈ 1.5958
    c      = 3.0 - 2.0 * (2.0 ** 0.5)                  # 3 - 2√2 ≈ 0.1716

    for i in range(1, n):
        hl_t0  = np.log(high[i - 1] / low[i - 1])
        hl_t1  = np.log(high[i]     / low[i])

        beta   = hl_t0 * hl_t0 + hl_t1 * hl_t1

        h2     = max(high[i - 1], high[i])
        l2     = min(low[i - 1],  low[i])
        if l2 <= 0.0:
            continue
        gamma  = (np.log(h2 / l2)) ** 2

        sqrt2b = (2.0 * beta) ** 0.5
        sqrtb  = beta ** 0.5
        alpha  = (sqrt2b - sqrtb) / c - (gamma / c) ** 0.5

        # Negative alpha → spread model breaks down → set NaN
        if np.isnan(alpha) or alpha < 0.0:
            continue

        exp_a        = np.exp(alpha)
        spread[i]    = np.float32(2.0 * (exp_a - 1.0) / (1.0 + exp_a))
        sigma[i]     = np.float32(sqrtb / (k2 * (2.0 ** 0.5)))

    return spread, sigma


@njit(cache=True)
def _kyle_lambda_kernel(
    close: np.ndarray,
    volume: np.ndarray,
    b: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Kyle's Lambda — price impact coefficient (AFML p. 286).

    Estimated via OLS on a rolling window:

        Δp_t = λ * b_t * v_t + ε_t

    where b_t is the tick direction (+1/-1) and v_t is volume.
    λ is the OLS slope (no intercept needed under the model).

    Parameters
    ----------
    close  : 1-D close prices
    volume : 1-D tick volumes
    b      : 1-D tick directions (+1/-1, float)
    window : rolling window length

    Returns
    -------
    float32 array, same length as close
        NaN for first (window) elements.
    """
    n   = len(close)
    out = np.full(n, np.nan, dtype=np.float32)

    for i in range(window, n):
        # Construct x = b * v, y = Δp over the window
        # Use indices [i-window : i]
        sx  = 0.0   # Σ x
        sy  = 0.0   # Σ y
        sxx = 0.0   # Σ x²
        sxy = 0.0   # Σ xy
        cnt = 0

        for j in range(i - window, i):
            dp = close[j + 1] - close[j] if j + 1 < n else 0.0
            x  = b[j] * volume[j]
            sx  += x
            sy  += dp
            sxx += x * x
            sxy += x * dp
            cnt += 1

        denom = cnt * sxx - sx * sx
        if denom != 0.0:
            slope = (cnt * sxy - sx * sy) / denom
            out[i] = np.float32(slope)

    return out


@njit(cache=True)
def _amihud_lambda_kernel(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Amihud's (2002) ILLIQ measure — rolling version.

        ILLIQ_t = mean(|Δln p_t| / dollar_volume_t)    over window

    where dollar_volume_t = close_t * volume_t.

    Parameters
    ----------
    close  : 1-D close prices
    volume : 1-D tick volumes
    window : rolling window length

    Returns
    -------
    float32 array, same length as close
    """
    n   = len(close)
    out = np.full(n, np.nan, dtype=np.float32)

    for i in range(window, n):
        acc = 0.0
        cnt = 0
        for j in range(i - window, i):
            if close[j] <= 0.0:
                continue
            abs_ret = abs(np.log(close[j + 1] / close[j])) if j + 1 < n else 0.0
            dv      = close[j] * volume[j]
            if dv > 0.0:
                acc += abs_ret / dv
                cnt += 1

        if cnt > 0:
            out[i] = np.float32(acc / cnt)

    return out


@njit(cache=True)
def _hasbrouck_lambda_kernel(
    close: np.ndarray,
    volume: np.ndarray,
    b: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Hasbrouck's (2009) Lambda — IV estimator of price impact.

    Regresses Δln p_t on sign(trade) * sqrt(dollar_volume_t):

        Δln p_t = λ * b_t * √(close_t * v_t) + ε_t

    OLS slope estimated over a rolling window (no intercept).

    Parameters
    ----------
    close  : 1-D close prices
    volume : 1-D tick volumes
    b      : 1-D tick directions (+1/-1, float)
    window : rolling window length

    Returns
    -------
    float32 array
    """
    n   = len(close)
    out = np.full(n, np.nan, dtype=np.float32)

    for i in range(window, n):
        sx  = 0.0
        sy  = 0.0
        sxx = 0.0
        sxy = 0.0
        cnt = 0

        for j in range(i - window, i):
            if close[j] <= 0.0 or volume[j] <= 0.0:
                continue
            dlnp = np.log(close[j + 1] / close[j]) if j + 1 < n else 0.0
            x    = b[j] * (close[j] * volume[j]) ** 0.5
            sx   += x
            sy   += dlnp
            sxx  += x * x
            sxy  += x * dlnp
            cnt  += 1

        denom = cnt * sxx - sx * sx
        if denom != 0.0:
            out[i] = np.float32((cnt * sxy - sx * sy) / denom)

    return out


@njit(parallel=True, cache=True)
def _vpin_kernel(
    volume: np.ndarray,
    b: np.ndarray,
    bucket_size: float,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    VPIN — Volume-Synchronized Probability of Informed Trading (AFML p. 289).

    Algorithm:
        1. Partition the tick stream into equal-volume buckets of size V*.
        2. Within each bucket, classify volume as buy (V^B) or sell (V^S)
           using the tick rule: V^B = Σ_{b=+1} v, V^S = Σ_{b=-1} v.
        3. VPIN over a window of n_buckets:

               VPIN = Σ|V^B_τ - V^S_τ| / (n_buckets * V*)

    This kernel fills buckets sequentially (cannot be parallelised at the
    tick level), then computes VPIN in parallel across bucket windows.

    Parameters
    ----------
    volume      : 1-D tick volumes (float64)
    b           : 1-D tick directions (+1/-1)
    bucket_size : V* — target volume per bucket
    n_buckets   : rolling window length (number of buckets) for VPIN

    Returns
    -------
    vpin        : float32 array, length = number of complete buckets
    bucket_ends : int64 array, tick index of the last tick in each bucket
    """
    n = len(volume)

    # --- Pass 1: fill buckets sequentially ---
    # Upper bound on bucket count
    max_buckets = int(np.ceil(volume.sum() / bucket_size)) + 1
    buy_vol     = np.zeros(max_buckets, dtype=np.float64)
    sell_vol    = np.zeros(max_buckets, dtype=np.float64)
    ends        = np.zeros(max_buckets, dtype=np.int64)

    bucket_idx  = 0
    remaining   = bucket_size

    for t in range(n):
        v = volume[t]
        while v > 0.0 and bucket_idx < max_buckets:
            fill   = min(v, remaining)
            if b[t] > 0.0:
                buy_vol[bucket_idx]  += fill
            else:
                sell_vol[bucket_idx] += fill

            v         -= fill
            remaining -= fill

            if remaining <= 0.0:
                ends[bucket_idx] = t
                bucket_idx      += 1
                if bucket_idx < max_buckets:
                    remaining = bucket_size

    n_complete = bucket_idx   # Number of fully filled buckets

    # Trim arrays to completed buckets
    buy_vol  = buy_vol[:n_complete]
    sell_vol = sell_vol[:n_complete]
    ends     = ends[:n_complete]

    # --- Pass 2: rolling VPIN over bucket windows (parallelisable) ---
    n_vpin = max(0, n_complete - n_buckets + 1)
    vpin   = np.full(n_vpin, np.nan, dtype=np.float32)

    for i in prange(n_vpin):
        imbalance = 0.0
        for j in range(i, i + n_buckets):
            imbalance += abs(buy_vol[j] - sell_vol[j])
        vpin[i] = np.float32(imbalance / (n_buckets * bucket_size))

    return vpin, ends[:n_vpin + n_buckets - 1]


@njit(parallel=True, cache=True)
def _bar_features_kernel(
    b: np.ndarray,
    volume: np.ndarray,
    close: np.ndarray,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bar microstructure features in parallel.

    Features computed for each bar [bar_starts[i] : bar_ends[i] + 1]:

        tick_imbalance  : Σ b_t  (signed tick count)
        volume_imbalance: Σ b_t * v_t
        dollar_imbalance: Σ b_t * v_t * p_t
        buy_fraction    : fraction of ticks classified as buys

    Parameters
    ----------
    b          : tick directions (+1/-1)
    volume     : tick volumes
    close      : close prices (used as price proxy)
    bar_starts : 1-D int array of bar start indices (inclusive)
    bar_ends   : 1-D int array of bar end indices (inclusive)

    Returns
    -------
    tick_imb, vol_imb, dollar_imb, buy_frac : float32 arrays, length = n_bars
    """
    n_bars      = len(bar_starts)
    tick_imb    = np.empty(n_bars, dtype=np.float32)
    vol_imb     = np.empty(n_bars, dtype=np.float32)
    dollar_imb  = np.empty(n_bars, dtype=np.float32)
    buy_frac    = np.empty(n_bars, dtype=np.float32)

    for i in prange(n_bars):
        s  = bar_starts[i]
        e  = bar_ends[i] + 1     # Exclusive end

        t_imb = 0.0
        v_imb = 0.0
        d_imb = 0.0
        buys  = 0.0
        cnt   = 0.0

        for k in range(s, e):
            t_imb += b[k]
            v_imb += b[k] * volume[k]
            d_imb += b[k] * volume[k] * close[k]
            if b[k] > 0.0:
                buys += 1.0
            cnt += 1.0

        tick_imb[i]   = np.float32(t_imb)
        vol_imb[i]    = np.float32(v_imb)
        dollar_imb[i] = np.float32(d_imb)
        buy_frac[i]   = np.float32(buys / cnt if cnt > 0.0 else np.nan)

    return tick_imb, vol_imb, dollar_imb, buy_frac


# ---------------------------------------------------------------------------
# Public feature-computation functions
# ---------------------------------------------------------------------------


def roll_measure(
    close: Union[pd.Series, np.ndarray],
    window: int = 20,
) -> pd.Series:
    """
    Roll (1984) effective spread estimator.

    Parameters
    ----------
    close  : close price series or array
    window : rolling window length

    Returns
    -------
    pd.Series of float32, same index as close (if Series)
    """
    arr = _to_float64(close)
    out = _roll_measure_kernel(arr, window)
    return _wrap(out, close, "roll_measure")


def roll_impact(
    close: Union[pd.Series, np.ndarray],
    volume: Union[pd.Series, np.ndarray],
    window: int = 20,
) -> pd.Series:
    """
    Roll spread normalised by dollar volume.

    Parameters
    ----------
    close  : close price series
    volume : tick volume series
    window : rolling window for Roll spread

    Returns
    -------
    pd.Series of float32
    """
    c = _to_float64(close)
    v = _to_float64(volume)
    out = _roll_impact_kernel(c, v, window)
    return _wrap(out, close, "roll_impact")


def corwin_schultz_spread(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """
    Corwin & Schultz (2012) high-low spread and volatility estimator.

    Parameters
    ----------
    high : bar high prices
    low  : bar low prices

    Returns
    -------
    pd.DataFrame with columns ['cs_spread', 'cs_sigma']
    """
    h = _to_float64(high)
    l = _to_float64(low)
    spread, sigma = _corwin_schultz_kernel(h, l)

    idx = high.index if isinstance(high, pd.Series) else np.arange(len(h))
    return pd.DataFrame(
        {"cs_spread": spread, "cs_sigma": sigma},
        index=idx,
        dtype=np.float32,
    )


def kyle_lambda(
    close: Union[pd.Series, np.ndarray],
    volume: Union[pd.Series, np.ndarray],
    b: Union[pd.Series, np.ndarray, None] = None,
    window: int = 20,
) -> pd.Series:
    """
    Kyle's Lambda — price impact coefficient.

    Parameters
    ----------
    close  : close prices
    volume : tick volumes
    b      : tick directions; computed via tick rule if None
    window : rolling OLS window

    Returns
    -------
    pd.Series of float32
    """
    c  = _to_float64(close)
    v  = _to_float64(volume)
    bt = _to_float64(b) if b is not None else _tick_rule(c)
    out = _kyle_lambda_kernel(c, v, bt, window)
    return _wrap(out, close, "kyle_lambda")


def amihud_lambda(
    close: Union[pd.Series, np.ndarray],
    volume: Union[pd.Series, np.ndarray],
    window: int = 20,
) -> pd.Series:
    """
    Amihud's (2002) ILLIQ measure.

    Parameters
    ----------
    close  : close prices
    volume : tick volumes
    window : rolling window

    Returns
    -------
    pd.Series of float32
    """
    c   = _to_float64(close)
    v   = _to_float64(volume)
    out = _amihud_lambda_kernel(c, v, window)
    return _wrap(out, close, "amihud_lambda")


def hasbrouck_lambda(
    close: Union[pd.Series, np.ndarray],
    volume: Union[pd.Series, np.ndarray],
    b: Union[pd.Series, np.ndarray, None] = None,
    window: int = 20,
) -> pd.Series:
    """
    Hasbrouck's (2009) Lambda — IV price impact estimator.

    Parameters
    ----------
    close  : close prices
    volume : tick volumes
    b      : tick directions; computed via tick rule if None
    window : rolling OLS window

    Returns
    -------
    pd.Series of float32
    """
    c   = _to_float64(close)
    v   = _to_float64(volume)
    bt  = _to_float64(b) if b is not None else _tick_rule(c)
    out = _hasbrouck_lambda_kernel(c, v, bt, window)
    return _wrap(out, close, "hasbrouck_lambda")


def vpin(
    volume: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    b: Union[pd.Series, np.ndarray, None] = None,
    bucket_size: Union[float, None] = None,
    n_buckets: int = 50,
) -> pd.Series:
    """
    VPIN — Volume-Synchronized Probability of Informed Trading.

    Parameters
    ----------
    volume      : tick volumes
    close       : close prices (used to derive tick rule when b is None)
    b           : tick directions; computed via tick rule if None
    bucket_size : V* target volume per bucket;
                  defaults to total_volume / (50 * n_buckets)
    n_buckets   : rolling window (number of buckets) for VPIN

    Returns
    -------
    pd.Series of float32, indexed by the end-tick of each VPIN window.
        The index is taken from the original close index when possible.
    """
    v  = _to_float64(volume)
    c  = _to_float64(close)
    bt = _to_float64(b) if b is not None else _tick_rule(c)

    if bucket_size is None:
        # Default: divide total volume into 50 * n_buckets buckets
        bucket_size = float(v.sum() / (50.0 * n_buckets))

    if bucket_size <= 0.0:
        raise ValueError("bucket_size must be positive.")

    vpin_vals, bucket_end_ticks = _vpin_kernel(v, bt, bucket_size, n_buckets)

    # Map tick indices back to the original index
    if isinstance(close, pd.Series):
        idx = close.index[bucket_end_ticks]
    else:
        idx = bucket_end_ticks

    return pd.Series(vpin_vals, index=idx, name="vpin", dtype=np.float32)


def bar_microstructure_features(
    tick_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute per-bar microstructure features from underlying tick data.

    This function re-examines the raw ticks that comprise each OHLC bar and
    computes:

        tick_imbalance   : Σ b_t                 (net signed tick count)
        volume_imbalance : Σ b_t * v_t           (net signed volume)
        dollar_imbalance : Σ b_t * v_t * p_t     (net signed dollar flow)
        buy_fraction     : fraction of buy ticks

    These are the building blocks of imbalance and runs bars, but computed
    here *per existing bar* so they can be used as ML features.

    Requirements
    ------------
    - tick_df must have a DatetimeIndex.
    - ohlc_df must have a DatetimeIndex where each timestamp is the bar-close
      time (last tick + 1 µs), matching the convention in make_bars.
    - tick_df must contain columns: 'bid' or 'mid_price', 'volume'.

    Parameters
    ----------
    tick_df   : raw tick DataFrame
    ohlc_df   : OHLC bar DataFrame produced by make_bars
    price_col : price column in tick_df to use for dollar imbalance

    Returns
    -------
    pd.DataFrame indexed like ohlc_df with four float32 feature columns.
    """
    if "volume" not in tick_df.columns:
        raise KeyError("'volume' column required in tick_df for bar features.")

    price_src = "mid_price" if "mid_price" in tick_df.columns else price_col
    if price_src not in tick_df.columns:
        if "bid" in tick_df.columns and "ask" in tick_df.columns:
            tick_df = tick_df.copy(deep=False)
            tick_df["mid_price"] = (tick_df["bid"] + tick_df["ask"]) / 2
            price_src = "mid_price"
        else:
            raise KeyError(
                f"Could not find a usable price column in tick_df. "
                f"Expected 'mid_price', 'bid'+'ask', or '{price_col}'."
            )

    # --- Map each tick to its bar ---
    # ohlc_df index = bar close time = last_tick_time + 1µs
    # So bar i spans: (ohlc_df.index[i-1], ohlc_df.index[i]]
    # i.e. ticks where tick.time < bar_close and tick.time >= prev_bar_close

    tick_times = tick_df.index
    bar_times  = ohlc_df.index

    # searchsorted maps each tick to its bar index
    bar_membership = np.searchsorted(bar_times, tick_times, side="right")

    # Build bar start/end arrays in tick-index space
    n_bars     = len(ohlc_df)
    bar_starts = np.full(n_bars, -1, dtype=np.int64)
    bar_ends   = np.full(n_bars, -1, dtype=np.int64)

    for bar_i in range(n_bars):
        mask = np.where(bar_membership == bar_i)[0]
        if len(mask):
            bar_starts[bar_i] = mask[0]
            bar_ends[bar_i]   = mask[-1]

    # Filter bars with no ticks (shouldn't happen but be safe)
    valid   = (bar_starts >= 0) & (bar_ends >= 0)
    n_valid = valid.sum()

    if n_valid == 0:
        logger.warning("No tick-to-bar mapping found — check index alignment.")
        return pd.DataFrame(index=ohlc_df.index)

    v_starts = bar_starts[valid].astype(np.int64)
    v_ends   = bar_ends[valid].astype(np.int64)

    # Raw arrays
    close_arr  = tick_df[price_src].to_numpy(dtype=np.float64)
    volume_arr = tick_df["volume"].to_numpy(dtype=np.float64)
    b_arr      = _tick_rule(close_arr)

    tick_imb, vol_imb, dollar_imb, buy_frac = _bar_features_kernel(
        b_arr, volume_arr, close_arr, v_starts, v_ends
    )

    out = pd.DataFrame(
        {
            "tick_imbalance":   tick_imb,
            "volume_imbalance": vol_imb,
            "dollar_imbalance": dollar_imb,
            "buy_fraction":     buy_frac,
        },
        index=ohlc_df.index[valid],
        dtype=np.float32,
    )

    # Reindex to full ohlc_df index (NaN for bars with no ticks)
    return out.reindex(ohlc_df.index)


def compute_all_features(
    ohlc_df: pd.DataFrame,
    tick_df: Union[pd.DataFrame, None] = None,
    window: int = 20,
    n_buckets: int = 50,
    include_bar_features: bool = True,
    include_vpin: bool = True,
) -> pd.DataFrame:
    """
    Compute the full suite of microstructure features for a bar DataFrame.

    This is the primary entry point intended for ML feature pipelines.
    All features are computed from ohlc_df columns; tick_df is only required
    for bar-level imbalance features and VPIN.

    Features computed from ohlc_df
    --------------------------------
    roll_measure        Roll (1984) effective spread
    roll_impact         Roll spread / dollar volume
    cs_spread           Corwin-Schultz bid-ask spread
    cs_sigma            Corwin-Schultz volatility
    kyle_lambda         Kyle price impact
    amihud_lambda       Amihud ILLIQ
    hasbrouck_lambda    Hasbrouck IV price impact

    Features requiring tick_df
    ---------------------------
    tick_imbalance      Σ b_t per bar
    volume_imbalance    Σ b_t·v_t per bar
    dollar_imbalance    Σ b_t·v_t·p_t per bar
    buy_fraction        fraction of buy ticks per bar
    vpin                VPIN over rolling bucket window

    Parameters
    ----------
    ohlc_df              : OHLC bar DataFrame (must have high, low, close, volume)
    tick_df              : raw tick data (optional; enables bar features and VPIN)
    window               : rolling window length for all rolling estimators
    n_buckets            : VPIN bucket window
    include_bar_features : compute per-bar imbalance features (requires tick_df)
    include_vpin         : compute VPIN (requires tick_df)

    Returns
    -------
    pd.DataFrame, same index as ohlc_df, with all computed feature columns.
    """
    _require_columns(ohlc_df, ["close"])
    has_volume = "volume" in ohlc_df.columns
    has_hl     = {"high", "low"}.issubset(ohlc_df.columns)

    close  = ohlc_df["close"]
    volume = ohlc_df["volume"] if has_volume else None

    features: dict[str, pd.Series | pd.DataFrame] = {}

    # --- Roll ---
    features["roll_measure"] = roll_measure(close, window)

    if has_volume:
        features["roll_impact"] = roll_impact(close, volume, window)

    # --- Corwin-Schultz ---
    if has_hl:
        cs = corwin_schultz_spread(ohlc_df["high"], ohlc_df["low"])
        features["cs_spread"] = cs["cs_spread"]
        features["cs_sigma"]  = cs["cs_sigma"]

    # --- Impact measures ---
    if has_volume:
        features["kyle_lambda"]      = kyle_lambda(close, volume, window=window)
        features["amihud_lambda"]    = amihud_lambda(close, volume, window)
        features["hasbrouck_lambda"] = hasbrouck_lambda(close, volume, window=window)

    # --- Bar-level tick features ---
    if include_bar_features and tick_df is not None:
        bar_feats = bar_microstructure_features(tick_df, ohlc_df)
        for col in bar_feats.columns:
            features[col] = bar_feats[col]

    # --- VPIN ---
    if include_vpin and tick_df is not None and "volume" in tick_df.columns:
        vpin_series = vpin(
            volume    = tick_df["volume"],
            close     = tick_df["mid_price"] if "mid_price" in tick_df.columns
                        else (tick_df["bid"] + tick_df["ask"]) / 2,
            n_buckets = n_buckets,
        )
        # VPIN is indexed by tick; forward-fill onto bar index
        features["vpin"] = (
            vpin_series
            .reindex(vpin_series.index.union(ohlc_df.index))
            .ffill()
            .reindex(ohlc_df.index)
            .astype(np.float32)
        )

    result = pd.concat(features, axis=1)
    result.columns = result.columns.get_level_values(-1)   # Drop concat keys
    return result.reindex(ohlc_df.index)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_float64(x: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Extract a contiguous float64 numpy array from a Series or array."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(x, dtype=np.float64)


def _wrap(
    arr: np.ndarray,
    source: Union[pd.Series, np.ndarray],
    name: str,
) -> pd.Series:
    """Wrap a numpy result array in a pd.Series with the source index."""
    idx = source.index if isinstance(source, pd.Series) else np.arange(len(arr))
    return pd.Series(arr, index=idx, name=name, dtype=np.float32)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise KeyError if any required column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
