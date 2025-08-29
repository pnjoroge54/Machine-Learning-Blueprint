"""
Fractional differentiation is a technique to make a time series stationary but also
retain as much memory as possible.  This is done by differencing by a positive real
number. Fractionally differenced series can be used as a feature in machine learning
process.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numba import njit, prange
from statsmodels.tsa.stattools import adfuller

from ..cache import cacheable


@njit
def get_weights(d, size):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 79.

    The helper function generates weights that are used to compute fractionally
    differentiated series. It computes the weights that get used in the computation
    of  fractionally differentiated series. This generates a non-terminating series
    that approaches zero asymptotically. The side effect of this function is that
    it leads to negative drift "caused by an expanding window's added weights"
    (see page 83 AFML)

    When d is real (non-integer) positive number then it preserves memory.

    The book does not discuss what should be expected if d is a negative real
    number. Conceptually (from set theory) negative d leads to set of negative
    number of elements. And that translates into a set whose elements can be
    selected more than once or as many times as one chooses (multisets with
    unbounded multiplicity) - see http://faculty.uml.edu/jpropp/msri-up12.pdf.

    :param d: (float) Differencing amount
    :param size: (int) Length of the series
    :return: (np.ndarray) Weight vector
    """

    # The algorithm below executes the iterative estimation (section 5.4.2, page 78)
    weights = [1.0]  # create an empty list and initialize the first element with 1.
    for k in range(1, size):
        weights_ = -weights[-1] * (d - k + 1) / k  # compute the next weight
        weights.append(weights_)

    # Now, reverse the list, convert into a numpy column vector
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


@njit(parallel=True)
def _frac_diff_numba_core(series_values, weights, skip):
    """
    Numba-optimized core function for fractional differencing.
    This function handles the heavy lifting of applying weights to the series values.

    :param series_values: (np.ndarray) The series values as a NumPy array.
    :param weights: (np.ndarray) The pre-computed weights.
    :param skip: (int) The number of initial calculations to skip.
    :return: (np.ndarray) The differenced series values.
    """
    N = len(series_values)
    output_values = np.empty(N, dtype=np.float64)
    output_values[:] = np.nan  # Initialize with NaN, as per pd.Series dtype='float64' behavior

    for iloc in prange(skip, N):
        output_values[iloc] = np.dot(
            weights[-(iloc + 1) :, :].T, series_values[: iloc + 1].reshape(-1, 1)
        )[0, 0]
    return output_values


@cacheable
def frac_diff(series, d, thres=0.01, use_log=True):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.5, page 82.

    References:
    https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:
    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    This is the expanding window variant of the fracDiff algorithm
    Note 1: For thres-1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarility bounded [0, 1]

    :param series: (pd.DataFrame) A time series that needs to be differenced
    :param d: (float) Differencing amount
    :param thres: (float) Threshold or epsilon
    :param use_log: (bool) If True, apply log transformation before differencing
    :return: (pd.DataFrame) Differenced series
    """
    if isinstance(series, pd.Series):
        series = series.copy().to_frame()

    # Apply log transformation for price series
    if use_log:
        # Ensure no zero or negative values
        series_processed = np.log(series.clip(lower=1e-8))
    else:
        series_processed = series.copy()

    series_processed = series_processed.astype("float64")
    columns = series_processed.columns

    # 1. Compute weights for the longest series
    weights = get_weights(d, series.shape[0])

    # 2. Determine initial calculations to be skipped based on weight-loss threshold
    weights_ = np.cumsum(abs(weights))
    weights_ /= weights_[-1]
    skip = weights_[weights_ > thres].shape[0]

    # 3. Apply weights to values using the Numba-optimized core function
    output_df = {}
    for name in columns:
        # Prepare data for Numba: ensure it's a contiguous NumPy array
        series_f_values = series_processed[[name]].ffill().dropna().values

        # Call the Numba-optimized core function
        output_values_numba = _frac_diff_numba_core(series_f_values, weights, skip)

        # Convert back to Pandas Series, retaining original index
        # We need to ensure the index aligns, and NaNs are handled
        output_series = pd.Series(
            output_values_numba,
            index=series[name].ffill().dropna().index,
            dtype="float64",
        )

        # Merge back with the original series index to handle initial NaNs
        final_output_series = pd.Series(index=series.index, dtype="float64")
        final_output_series.update(output_series)  # Updates matching index values

        output_df[name] = final_output_series

    output_df = pd.concat(output_df, axis=1)
    return output_df


@njit
def get_weights_ffd(d, thres, lim):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 83.

    The helper function generates weights that are used to compute fractionally
    differentiate dseries. It computes the weights that get used in the computation
    of fractionally differentiated series. The series is of fixed width and same
    weights (generated by this function) can be used when creating fractional
    differentiated series.
    This makes the process more efficient. But the side-effect is that the
    fractionally differentiated series is skewed and has excess kurtosis. In
    other words, it is not Gaussian any more.

    The discussion of positive and negative d is similar to that in get_weights
    (see the function get_weights)

    :param d: (float) Differencing amount
    :param thres: (float) Threshold for minimum weight
    :param lim: (int) Maximum length of the weight vector
    :return: (np.ndarray) Weight vector
    """

    weights = [1.0]
    k = 1

    # The algorithm below executes the iterativetive estimation (section 5.4.2, page 78)
    # The output weights array is of the indicated length (specified by lim)
    ctr = 0
    while True:
        # compute the next weight
        weights_ = -weights[-1] * (d - k + 1) / k

        if abs(weights_) < thres:
            break

        weights.append(weights_)
        k += 1
        ctr += 1
        if ctr == lim - 1:  # if we have reached the size limit, exit the loop
            break

    # Now, reverse the list, convert into a numpy column vector
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


@njit(parallel=True)
def _frac_diff_ffd_numba_core(series_values, weights, skip):
    """
    Numba-optimized core function for fractional differencing.
    This function handles the heavy lifting of applying weights to the series values.

    :param series_values: (np.ndarray) The series values as a NumPy array.
    :param weights: (np.ndarray) The pre-computed weights.
    :param skip: (int) The number of initial calculations to skip.
    :return: (np.ndarray) The differenced series values.
    """
    N = len(series_values)
    weights = weights.T
    arr = np.empty(N, dtype=np.float64)

    for i in prange(skip, N):
        arr[i] = np.dot(weights, series_values[i - skip : i + 1])[0, 0]

    return arr[skip:]


@cacheable
def frac_diff_ffd(series, d, thres=1e-5, use_log=True):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.5, page 83.

    References:

    * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    * https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    * https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:

    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarity bounded [0, 1].

    :param series: (pd.DataFrame) A time series that needs to be differenced
    :param d: (float) Differencing amount
    :param thres: (float) Threshold for minimum weight
    :param use_log: (bool) If True, apply log transformation before differencing
    :return: (pd.Series or pd.DataFrame) A Series or DataFrame of Fractionally differenced series
    """
    if isinstance(series, pd.Series):
        series = series.copy().to_frame()

    # Apply log transformation for price series
    if use_log:
        # Ensure no zero or negative values
        series_processed = np.log(series.clip(lower=1e-8))
    else:
        series_processed = series.copy()

    series_processed = series_processed.astype("float64")
    columns = series_processed.columns

    # Compute weights
    weights = get_weights_ffd(d, thres, series_processed.shape[0])
    width = len(weights) - 1

    # Apply weights to values
    df = {}
    for name in columns:
        series_f = series_processed[[name]].ffill().dropna()
        ffd = _frac_diff_ffd_numba_core(series_f.values, weights, width)
        df[name] = pd.Series(ffd, index=series_f.index[width:])

    df = pd.concat(df, axis=1)

    if len(columns) == 1:
        return df.squeeze()

    return df


def adf_data(df1, df2, d=0, out_df=None, alpha=0.05):
    """
    Calculate ADF statistics and correlation between original data series and differenced series.

    :param df1: (pd.Series) Original data
    :param df2: (pd.Series) Fractionally differentiated data
    :param alpha: (float) either 0.01 or 0.05
    :param out_df: (pd.DataFrame) A data frame of ADF statistics
    :return out_df: (pd.DataFrame) A data frame of ADF statistics
    """
    corr = np.corrcoef(df1.loc[df2.index], df2)[0, 1]
    adf = adfuller(df2, maxlag=1, regression="c", autolag=None)
    tc_col = f"{1 - alpha:.0%} conf"
    columns = [
        "adfStat",
        "pVal",
        "lags",
        "nObs",
        "window",
        tc_col,
        "corr",
        "stationary",
    ]
    vals = list(adf[:4]) + [df1.shape[0] - adf[3]] + [adf[4][f"{alpha:.0%}"]] + [corr] + [False]

    if out_df is None or out_df.empty:
        out_df = {d: {k: v for k, v in zip(columns, vals)}}
        out_df = pd.DataFrame.from_dict(out_df, orient="index")
    else:
        out_df.loc[d, columns] = vals

    stationary = (out_df.loc[d, "adfStat"] < out_df.loc[d, tc_col]) and (
        out_df.loc[d, "pVal"] < alpha
    )
    out_df.loc[d, "stationary"] = stationary
    out_df.index.name = "d"

    return out_df


@cacheable
def fracdiff_optimal(
    series, fixed_width=True, alpha=0.05, max_d=1.0, tol=1e-3, use_log=True, verbose=False
):
    """
    Determines the smallest differentiation factor (d) required for stationarity
    in fractional differencing using a binary search approach.

    This function finds the optimal differentiation parameter (d) such that
    the series passes the Augmented Dickey-Fuller (ADF) test for stationarity,
    minimizing the loss of memory in the differenced series.

    Parameters
    ----------
    series : pd.Series
        Time series data to be fractionally differenced.
    fixed_width : bool, default=True
        If True, applies fixed-width fractional differencing. If False,
        uses an expanding window approach.
    alpha : float, default=0.05
        Significance threshold for the ADF test. A smaller alpha requires
        stronger evidence for stationarity.
    max_d : float, default=1.0
        Upper bound for the differentiation factor in the binary search.
    tol : float, default=1e-3
        Tolerance for convergence in the binary search for optimal d.
    use_log : bool, default=True
        If True, apply log transformation before fractional differencing.
        Recommended for price series to handle multiplicative relationships.
        Set to False for return series or other already-processed data.
    verbose : bool, default=False
        If True, prints progress updates during the search.

    Returns
    -------
    out_df : pd.Series or None
        Fractionally differenced series corresponding to the optimal d value.
    d : float or None
        Optimal differentiation factor that achieves stationarity. Returns None
        if stationarity is not detected within the given bounds.
    diff_adf : pd.DataFrame
        Dataframe containing ADF test results for all attempted d values, including
        p-values and correlation coefficients.

    Notes
    -----
    - The function performs a **binary search** to efficiently determine the smallest
      d value that results in a stationary series.
    - Uses **fixed-width** differencing (`frac_diff_ffd`) or **expanding window** differencing
      (`frac_diff`) depending on `fixed_width`.
    - The `use_log` parameter controls preprocessing: True for price series (multiplicative),
      False for return series or other additive data.
    - The Augmented Dickey-Fuller test (`adf_data`) is applied to each differenced series
      to check for stationarity.
    - Ensures a **minimum sample size** (default: 100) for valid ADF test execution.

    Example
    -------
    ```python
    import pandas as pd
    from ..fracdiff import fracdiff_optimal

    # For price series (use log transformation)
    price_series = pd.Series([100, 101, 102, 103, ...])
    diff_series, d, adf_results = fracdiff_optimal(price_series, use_log=True)

    # For return series (skip log transformation)
    return_series = pd.Series([0.01, 0.01, 0.01, ...])
    diff_series, d, adf_results = fracdiff_optimal(return_series, use_log=False)
    ```
    """

    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected `series` to be a pandas Series, but got type {type(series)}")

    low, high = 0.0, max_d
    best_d = None
    diff_adf = None
    out_df = None
    max_len = 0

    # Cache intermediate results to avoid recomputation
    frac_diff_cache = {}
    adf_cache = {}

    def frac_diff_cached(series, d, use_log):
        cache_key = (d, use_log)  # Include use_log in cache key
        return frac_diff_cache.setdefault(
            cache_key,
            (
                frac_diff_ffd(series, d, use_log=use_log)
                if fixed_width
                else frac_diff(series, d, use_log=use_log)
            ),
        )

    def adf_cached(series, diff_series, d, out_df, alpha):
        return adf_cache.setdefault(d, adf_data(series, diff_series, d, out_df, alpha))

    for i, _ in enumerate(range(20), 1):  # Max 20 iterations for 1e-6 precision
        mid = (low + high) / 2

        if verbose:
            msg = f"{i}. Testing d = {mid:.4f}"
            logger.debug(msg)
            if len(msg) > max_len:
                max_len = len(msg)

        diff = frac_diff_cached(series, mid, use_log)  # Pass use_log parameter

        if len(diff) < 100:  # Minimum sample size
            high = mid
            continue

        # Dataframe with ADF stats for each d
        diff_adf = adf_cached(series, diff, mid, diff_adf, alpha)
        p_value = diff_adf.loc[mid, "pVal"]

        if diff_adf.loc[mid, "stationary"] == True:
            out_df = diff.copy()

        if p_value < alpha:
            best_d = mid
            high = mid  # Try smaller d
        else:
            low = mid  # Need larger d

        if high - low < tol:
            break

    d = round(best_d, 4) if best_d is not None else max_d

    if verbose:
        if best_d is None:
            logger.info("No stationary series found.")
            return (None, None, None)

        log_msg = " (log-transformed)" if use_log else ""
        msg = (
            f"d = {d} makes {series.name if series.name else 'series'}{log_msg} stationary for ADF test (α={alpha}). "
            f"Corr(y, y_fracdiff) = {diff_adf.loc[best_d, 'corr']:.4f}."
        )
        logger.info(msg)

    return out_df, d, diff_adf


def plot_min_ffd(adf: pd.DataFrame):
    """
    Create plot of minimum fractional differentiation value needed
    to make Series stationary at 5% confidence with Augmented Dickey-Fuller test.
    """
    adf = adf.sort_index()
    fig = adf[["adfStat", "corr"]].plot(figsize=(12, 6), secondary_y="adfStat")
    plt.axhline(adf["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    return fig


def plot_ffd_vs_data(ffd: pd.Series, data: pd.Series, d: float, name: str = "close"):
    # from matplotlib.ticker import FuncFormatter

    x1 = ffd.index
    x2 = data.index
    y1 = ffd.values
    y2 = data.values
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(x1, y1, "g-")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"FFD_{name}", color="g")

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, "b-")
    ax2.set_ylabel(name, color="b")

    # Format x-axis labels as dates
    # ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: df2['time'].iloc[int(x)].strftime("%Y-%m-%d")))

    corr = np.corrcoef(data.reindex(ffd.index), ffd)[0, 1]
    plt.title(f"Fixed Width Fractional Differentiation (d={d:.4f}, corr={corr:.4f})")

    return fig
