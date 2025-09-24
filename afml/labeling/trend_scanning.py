"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger
from numba import njit, prange

from ..cache import smart_cacheable
from ..labeling.triple_barrier import (
    drop_labels,
    get_event_weights,
    triple_barrier_labels,
)


@njit(parallel=True, cache=True)
def _window_stats_numba(y, window_length):
    """
    Compute slopes, t-values, and R² for all fixed-length windows.
    This function is optimized for performance using Numba's JIT compilation.

    :param y: (np.ndarray) The input data array.
    :param window_length: (int) The length of the sliding window.
    :return: (tuple) A tuple containing:
        - t_values: (np.ndarray) The t-values for each window.
        - slopes: (np.ndarray) The slopes for each window.
        - r_squared: (np.ndarray) The R² values for each window.
    """
    n = len(y)
    num_windows = n - window_length + 1

    t_values = np.empty(num_windows)
    slopes = np.empty(num_windows)
    r_squared = np.empty(num_windows)

    t = np.arange(window_length)
    mean_t = t.mean()
    Var_t = ((t - mean_t) ** 2).sum()

    for i in prange(num_windows):
        window = y[i : i + window_length]
        mean_y = window.mean()
        sum_y = window.sum()
        sum_y2 = (window**2).sum()

        # Slope estimation
        S_ty = (window * t).sum()
        slope = (S_ty - window_length * mean_t * mean_y) / Var_t
        slopes[i] = slope

        # SSE calculation
        beta0 = mean_y - slope * mean_t
        SSE = sum_y2 - beta0 * sum_y - slope * S_ty

        # R² calculation
        SST = sum_y2 - (sum_y**2) / window_length
        epsilon = 1e-9
        r_squared[i] = max(0.0, 1.0 - SSE / (SST + epsilon)) if SST > epsilon else 0.0

        # t-value calculation
        sigma2 = SSE / (window_length - 2 + epsilon)
        se_slope = np.sqrt(sigma2 / Var_t)
        t_values[i] = slope / (se_slope + epsilon)

    return t_values, slopes, r_squared


@smart_cacheable
def trend_scanning_labels(
    close: pd.Series,
    span: Union[List[int], Tuple[int, int]] = (5, 20),
    volatility_threshold: float = 0.1,
    lookforward: bool = True,
    use_log: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.
    It fits OLS regressions over multiple rolling windows and selects the one with the highest absolute t-value.
    The sign of the t-value indicates trend direction, while its magnitude reflects confidence.
    The method incorporates volatility-based masking to avoid spurious signals in low-volatility regimes.
    This implementation offers a robust, leakage-proof trend-scanning label generator with:
      - Expanding, data-adaptive volatility thresholding
      - Full feature masking (t-value, slope, R²) in low-volatility regimes
      - Boundary protection to avoid look-ahead leaks
      - Support for both look-forward and look-backward scan

    Parameters
    ----------
    close : pd.Series
        Time-indexed raw price series. Must be unique and sorted (monotonic).
    span : list[int] or tuple(int, int), default=(5, 20)
        If list, exact window lengths to scan. If tuple `(min, max)`, uses
        `range(min, max)` as horizons.
    volatility_threshold : float, default=0.1
        Quantile level (0-1) on the expanding rolling std of log-prices. Windows
        below this vol threshold are zero-masked.
    lookforward : bool, default=True
        If True, labels trend on `[t, t+L-1]`; else on `[t-L+1, t]` by reversing.
    use_log : bool, default=True
        Apply log transformation before trend analysis
    verbose : bool, default=False
        Print progress for each horizon.

    Returns
    -------
    pd.DataFrame
        Indexed by the valid subset of `close.index`. Columns:
        - t1 : pd.Timestamp
        End of the event window (lookforward) or start (lookbackward).
        - window : int
        Chosen optimal horizon (argmax |t-value|).
        - slope : float
        Estimated slope over that window.
        - t_value : float
        t-stat for the slope (clipped to ±min(var, 20)).
        - r_squared : float
        Goodness-of-fit (zero if below vol threshold).
        - ret : float
        Hold-period return over the chosen window.
        - bin : int8
        Sign of `t_value` (-1, 0, +1), zero if |t_value|≈0.

    Notes
    -----
    1. Log-transformation stabilizes variance before regression.
    2. Uses a precompiled Numba `_window_stats_numba` for the heavy sliding
       O(N·H) regressions.
    3. Boundary slices ensure no forward-looking data leak into features.
    """
    # Input validation and setup
    close = close.sort_index() if not close.index.is_monotonic_increasing else close.copy()
    hrzns = list(range(*span)) if isinstance(span, tuple) else span
    max_hrzn = max(hrzns)

    if lookforward:
        valid_indices = close.index[:-max_hrzn].to_list()
    else:
        valid_indices = close.index[max_hrzn - 1 :].to_list()

    if not valid_indices:
        return pd.DataFrame(columns=["t1", "window", "slope", "t_value", "rsquared", "ret", "bin"])

    # Log transformation
    if use_log:
        close_processed = close.clip(lower=1e-8).astype(np.float64)
        y = np.log(close_processed).values
    else:
        y = close.values.astype(np.float64)

    N = len(y)

    # Compute volatility threshold
    volatility = pd.Series(y, index=close.index).rolling(max_hrzn, min_periods=1).std().ffill()
    vol_threshold = volatility.expanding().quantile(volatility_threshold).ffill().values

    # Precompute all window stats
    window_stats = np.full((3, N, len(hrzns)), np.nan)
    for k, hrzn in enumerate(hrzns):
        if verbose:
            print(f"Processing horizon {hrzn}", end="\r", flush=True)
        y_window = y if lookforward else y[::-1]
        t_vals, slopes, r_sq = _window_stats_numba(y_window, hrzn)
        if not lookforward:
            t_vals, slopes, r_sq = t_vals[::-1], slopes[::-1], r_sq[::-1]
            start_idx = hrzn - 1
        else:
            start_idx = 0
        n = len(t_vals)
        valid_vol = volatility.iloc[start_idx : start_idx + n].values
        mask = valid_vol > vol_threshold[start_idx : start_idx + n]
        window_stats[0, start_idx : start_idx + n, k] = np.where(mask, t_vals, 0)
        window_stats[1, start_idx : start_idx + n, k] = np.where(mask, slopes, 0)
        window_stats[2, start_idx : start_idx + n, k] = np.where(mask, r_sq, 0)

    # Integer positions for events
    event_idx = close.index.get_indexer(valid_indices)

    # Extract sub-blocks for these events
    t_block = window_stats[0, event_idx, :]  # shape: (E, H)
    s_block = window_stats[1, event_idx, :]
    rsq_block = window_stats[2, event_idx, :]

    # Best horizon per event (argmax of abs t-value)
    best_j = np.nanargmax(np.abs(t_block), axis=1)  # (E,)

    # Gather optimal metrics
    opt_tval = t_block[np.arange(len(event_idx)), best_j]
    opt_slope = s_block[np.arange(len(event_idx)), best_j]
    opt_rsq = rsq_block[np.arange(len(event_idx)), best_j]
    opt_hrzn = np.array(hrzns)[best_j]

    # Compute t1 indices vectorised
    if lookforward:
        t1_idx = np.clip(event_idx + opt_hrzn - 1, 0, N - 1)
    else:
        t1_idx = np.clip(event_idx - opt_hrzn + 1, 0, N - 1)

    # Map to timestamps and returns
    t1_arr = close.index[t1_idx]
    a, b = (event_idx, t1_idx) if lookforward else (t1_idx, event_idx)
    rets = close.iloc[b].array / close.iloc[a].array - 1

    # Filter labels by t-value
    tval_abs = np.abs(opt_tval)
    mask = tval_abs > 1e-6
    bins = np.where(mask, np.sign(opt_tval), 0).astype("int8")

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "t1": t1_arr,
            "window": opt_hrzn,
            "slope": opt_slope,
            "t_value": opt_tval,
            "rsquared": opt_rsq,
            "ret": rets,
            "bin": bins,
        },
        index=pd.Index(valid_indices),
    )

    return df


def plot_trend_labels(close, trend_labels, title="Trend-Scanning Labels", view="bin"):
    """
    Plot the close prices with trend labels.

    param close: Series of close prices.
    param trend_labels: DataFrame with trend labels.
    param title: Title of the plot.
    param view: 'bin' to color by trend bin, 't_value' to color by trend t-value.
    return: None
    """
    plt.figure(figsize=(7.5, 5), dpi=100)
    # plt.plot(close.index, close.values, linewidth=1.5, color="lightgray")
    scatter = plt.scatter(
        trend_labels.index,
        close.loc[trend_labels.index].values,
        c=trend_labels[view].values,
        cmap="coolwarm_r",
        s=40,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.7,
    )

    plt.colorbar(scatter, label=f"trend {view}")
    plt.style.use("dark_background")
    plt.title(title)
    plt.show()
    plt.colorbar(scatter, label=f"trend {view}")
    plt.style.use("dark_background")
    plt.title(title)
    plt.show()
