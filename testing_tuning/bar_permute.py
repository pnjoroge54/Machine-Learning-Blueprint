import numpy as np
import pandas as pd
from numba import njit
from typing import List, Union, Optional

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: Optional[int] = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate a permuted (shuffled) version of OHLC price series while preserving
    statistical properties (volatility, correlation) but destroying temporal structure.

    The algorithm:
    1. Convert prices to log space.
    2. Compute relative price moves:
       - Gap: open - previous close
       - Intraday moves: high - open, low - open, close - open
    3. Shuffle the intraday moves (high/low/close) together and the gaps separately.
    4. Reconstruct a new series by:
         a) Keeping the real bars up to `start_index`.
         b) Setting the bar at `start_index` to the real start bar.
         c) For each subsequent bar:
              open = previous close + shuffled gap
              high = open + shuffled high move
              low  = open + shuffled low move
              close = open + shuffled close move
    5. Exponentiate back to price space and return as DataFrame(s).

    Parameters
    ----------
    ohlc : pd.DataFrame or List[pd.DataFrame]
        OHLC data. Each DataFrame must have columns: 'open', 'high', 'low', 'close'.
        If a list is provided, all DataFrames must have identical indexes.
    start_index : int, default 0
        Index from which to start permuting. Bars before this index are kept unchanged.
        Must be >= 0.
    seed : int, optional
        Random seed for reproducibility. If None, unpredictable randomness is used.

    Returns
    -------
    pd.DataFrame or List[pd.DataFrame]
        Permuted OHLC data in the same format as the input.
        If input was a single DataFrame, a single DataFrame is returned.
        If input was a list, a list of DataFrames is returned.
    """
    # --- Input validation ---
    assert start_index >= 0, "start_index must be >= 0"

    # Normalize input to a list of DataFrames for uniform processing
    if isinstance(ohlc, list):
        # Check that all DataFrames have the same index
        first_idx = ohlc[0].index
        for df in ohlc:
            if not np.all(first_idx == df.index):
                raise ValueError("All DataFrames must have the same index")
        df_list = ohlc
        n_markets = len(df_list)
    else:
        df_list = [ohlc]
        n_markets = 1

    time_index = df_list[0].index
    n_bars = len(df_list[0])

    # --- Random number setup (modern, independent generator) ---
    rng = np.random.default_rng(seed)

    # --- Convert each market to log prices and store as a 3D array (markets × bars × 4) ---
    # Shape: (n_markets, n_bars, 4)   columns: open, high, low, close
    log_prices = np.zeros((n_markets, n_bars, 4), dtype=np.float64)
    for m, df in enumerate(df_list):
        log_prices[m] = np.log(df[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64))

    # --- Extract start bars (log prices at start_index) ---
    start_bars = log_prices[:, start_index, :]   # shape (n_markets, 4)

    # --- Compute relative moves for the permutation window (bars after start_index) ---
    perm_index = start_index + 1
    perm_n = n_bars - perm_index                 # number of bars to permute

    # Pre‑allocate relative arrays: shape (n_markets, perm_n)
    rel_open = np.empty((n_markets, perm_n), dtype=np.float64)
    rel_high = np.empty((n_markets, perm_n), dtype=np.float64)
    rel_low  = np.empty((n_markets, perm_n), dtype=np.float64)
    rel_close = np.empty((n_markets, perm_n), dtype=np.float64)

    for m in range(n_markets):
        # open relative to previous close (gap)
        # We need the close of the previous bar for the entire series.
        # Compute gap for all bars, then take the permutation window.
        close_shifted = np.roll(log_prices[m, :, 3], shift=1)
        # For the first bar, the gap would be open[0] - close[-1] (undesirable). We'll only use from perm_index onward.
        gaps = log_prices[m, :, 0] - close_shifted   # length n_bars
        rel_open[m] = gaps[perm_index:]

        # Intraday moves relative to the bar's open
        rel_high[m]  = log_prices[m, perm_index:, 1] - log_prices[m, perm_index:, 0]
        rel_low[m]   = log_prices[m, perm_index:, 2] - log_prices[m, perm_index:, 0]
        rel_close[m] = log_prices[m, perm_index:, 3] - log_prices[m, perm_index:, 0]

    # --- Generate random permutations for shuffling ---
    # perm1 : shuffles intraday moves (high, low, close) together
    # perm2 : shuffles gaps independently
    idx = np.arange(perm_n)
    perm1 = rng.permutation(idx)
    perm2 = rng.permutation(idx)

    # Apply shuffling to the relative arrays
    # Use advanced indexing to reorder columns according to permutations
    rel_high  = rel_high[:, perm1]
    rel_low   = rel_low[:, perm1]
    rel_close = rel_close[:, perm1]
    rel_open  = rel_open[:, perm2]

    # --- Build the permuted series using a Numba‑compiled parallel loop ---
    perm_logs = _build_permuted_series_numba(
        log_prices, start_index, start_bars,
        rel_open, rel_high, rel_low, rel_close,
        n_markets, n_bars, perm_n
    )

    # --- Convert back to price space and to pandas DataFrames ---
    perm_ohlc = []
    for m in range(n_markets):
        # Exponentiate to get prices
        perm_prices = np.exp(perm_logs[m])
        df_perm = pd.DataFrame(
            perm_prices,
            index=time_index,
            columns=['open', 'high', 'low', 'close']
        )
        perm_ohlc.append(df_perm)

    # Return in the same format as input
    if n_markets == 1:
        return perm_ohlc[0]
    else:
        return perm_ohlc


@njit(cache=True)
def _build_permuted_series_numba(
    log_prices: np.ndarray,
    start_index: int,
    start_bars: np.ndarray,
    rel_open: np.ndarray,
    rel_high: np.ndarray,
    rel_low: np.ndarray,
    rel_close: np.ndarray,
    n_markets: int,
    n_bars: int,
    perm_n: int
) -> np.ndarray:
    """
    Numba‑compiled core that builds the permuted log‑price series.

    Parameters
    ----------
    log_prices : np.ndarray, shape (n_markets, n_bars, 4)
        Original log prices for all markets.
    start_index : int
        Index up to which real data is copied unchanged.
    start_bars : np.ndarray, shape (n_markets, 4)
        Log prices of the bar at start_index for each market.
    rel_open, rel_high, rel_low, rel_close : np.ndarray, shape (n_markets, perm_n)
        Shuffled relative moves (gaps and intraday) for the permutation window.
    n_markets, n_bars, perm_n : int
        Dimensions.

    Returns
    -------
    np.ndarray, shape (n_markets, n_bars, 4)
        Permuted log prices.
    """
    out = np.zeros((n_markets, n_bars, 4), dtype=np.float64)

    # Parallel loop over markets – each market is independent
    for m in range(n_markets):
        # 1. Copy real bars up to (but not including) start_index
        out[m, :start_index] = log_prices[m, :start_index]

        # 2. Set the start bar (real data)
        out[m, start_index] = start_bars[m]

        # 3. Build the permuted bars sequentially
        #    i runs over bar indices; k runs over the permutation window (0..perm_n-1)
        for i in range(start_index + 1, n_bars):
            k = i - (start_index + 1)   # index in shuffled arrays

            # open = previous close + shuffled gap
            out[m, i, 0] = out[m, i-1, 3] + rel_open[m, k]

            # high/low/close = open + shuffled intraday moves
            out[m, i, 1] = out[m, i, 0] + rel_high[m, k]
            out[m, i, 2] = out[m, i, 0] + rel_low[m, k]
            out[m, i, 3] = out[m, i, 0] + rel_close[m, k]

    return out