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


@njit(cache=True)
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


@njit(parallel=True, cache=True)
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


@njit(cache=True)
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


@njit(parallel=True, cache=True)
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


def adf_data(df1, df2, d=0, out_df=None, alpha=0.05, autolag="AIC", maxlag=None):
    """
    Calculate ADF statistics and correlation between original data series and differenced series.

    :param df1: (pd.Series) Original data
    :param df2: (pd.Series) Fractionally differentiated data
    :param d: (float) Differencing amount associated with df2. Used as the row
        label when the result is inserted into, or read back from, out_df.
    :param out_df: (pd.DataFrame) A data frame of ADF statistics
    :param alpha: (float) either 0.01 or 0.05
    :param autolag: (str or None) Lag-order selection passed to ``adfuller``.
        ``"AIC"`` is the default. Pass ``None`` with ``maxlag=1`` to reproduce
        the pre-2026 behavior, which under-lags a long-memory series and
        over-rejects the unit-root null.
    :param maxlag: (int or None) Maximum augmenting lags. ``None`` lets
        ``adfuller`` apply the Schwert rule.
    :return out_df: (pd.DataFrame) A data frame of ADF statistics
    """
    corr = np.corrcoef(df1.loc[df2.index], df2)[0, 1]
    adf = adfuller(df2, maxlag=maxlag, regression="c", autolag=autolag)
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


# Fraction of the series the FFD window may consume before the result is
# considered unusable. At thres=1e-5 the window peaks near 4,000 lags around
# d = 0.1, so an uncapped window can swallow an entire series of ordinary length.
DEFAULT_MAX_WINDOW_FRAC = 0.25


def _frac_diff_ffd_core(series_values: np.ndarray, weights: np.ndarray, skip: int) -> np.ndarray:
    """Dot the weight vector along the series. Kept out of the public surface.

    In the production module this is the numba kernel ``_frac_diff_ffd_numba_core``.
    Import that instead when integrating; the pure-NumPy version here exists so
    the refactor can be reviewed and tested without a numba toolchain.

    :param series_values: (np.ndarray) Column vector of (log-)values, shape
        (n_obs, 1).
    :param weights: (np.ndarray) FFD weight vector, shape (width + 1, 1).
    :param skip: (int) Number of leading observations to skip; equal to the
        window width, since the first full-width dot product needs that many
        prior values.
    :return: (np.ndarray) Differenced values, length ``n_obs - skip``.
    """
    n = series_values.shape[0]
    width = weights.shape[0]
    out = np.empty(n - skip, dtype=np.float64)
    flat = weights.ravel()
    for i in range(skip, n):
        out[i - skip] = float(np.dot(flat, series_values[i - width + 1 : i + 1, 0]))
    return out


def frac_diff_ffd(
    series,
    d: float,
    thres: float = 1e-5,
    use_log: bool = True,
    max_window_frac: float | None = None,
    window: int | None = None,
):
    """Fixed-width fractional differentiation (AFML Snippet 5.3).

    :param series: (pd.Series or pd.DataFrame) Input series.
    :param d: (float) Differencing amount.
    :param thres: (float) Weight-magnitude cutoff for the fixed window.
    :param use_log: (bool) Log-transform before differencing. Use for price series.
    :param max_window_frac: (float) Upper bound on the window as a fraction of the
        series length. ``None`` reproduces the historical behavior exactly, in
        which the window may consume the entire series. ``fracdiff_optimal``
        passes ``0.25``.
    :param window: (int) Exact window length in lags, overriding ``thres`` and
        ``max_window_frac``. Used by ``fracdiff_optimal`` to evaluate every
        candidate ``d`` on an identical sample.
    :return: (pd.Series or pd.DataFrame) Differenced series.

    .. note:: **Fix 1.** The single-column return path previously used
       ``df.squeeze()``, which collapses a one-row frame to a NumPy scalar. When
       the FFD window consumed all but one observation the function returned a
       float, and the ``len(diff) < 100`` guard in ``fracdiff_optimal``
       immediately raised ``TypeError: object of type 'numpy.float64' has no
       len()``. The guard existed to catch precisely this case. It is now
       ``df.squeeze(axis=1)``, which always returns a Series.
    """
    if isinstance(series, pd.Series):
        series = series.copy().to_frame()

    if use_log:
        series_processed = np.log(series.clip(lower=1e-8))
    else:
        series_processed = series.copy()

    series_processed = series_processed.astype("float64")
    columns = series_processed.columns
    n_obs = series_processed.shape[0]

    lim = n_obs
    if max_window_frac is not None:
        if not 0.0 < max_window_frac <= 1.0:
            raise ValueError(f"max_window_frac must lie in (0, 1]; got {max_window_frac}")
        lim = max(2, int(n_obs * max_window_frac))

    if window is not None:
        # Explicit common window: force exactly `window` lags for every d so that
        # candidates are comparable. thres=0 disables the magnitude cutoff.
        weights = get_weights_ffd(d, 0.0, int(window) + 1)
    else:
        weights = get_weights_ffd(d, thres, lim)
    width = len(weights) - 1

    remaining = n_obs - width
    if remaining < 2:
        raise ValueError(
            f"FFD window of {width} lags at d={d} leaves {remaining} observation(s) "
            f"from a series of {n_obs}. Raise `thres`, set `max_window_frac`, or "
            f"supply a longer series."
        )

    frames: dict[str, pd.Series] = {}
    for name in columns:
        series_f = series_processed[[name]].ffill().dropna()
        values = _frac_diff_ffd_numba_core(series_f.values, weights, width)
        frames[name] = pd.Series(values, index=series_f.index[width:], name=name)

    df = pd.concat(frames, axis=1)

    # Fix 1: squeeze(axis=1), not squeeze(). The latter returns a scalar when the
    # frame has one row, silently breaking every length-based guard downstream.
    if len(columns) == 1:
        return df.squeeze(axis=1)
    return df


def fracdiff_optimal(
    series: pd.Series,
    fixed_width: bool = True,
    alpha: float = 0.05,
    max_d: float = 1.0,
    tol: float = 1e-3,
    use_log: bool = True,
    thres: float = 1e-5,
    max_window_frac: float | None = DEFAULT_MAX_WINDOW_FRAC,
    min_obs: int = 100,
    adf_autolag: str | None = "AIC",
    adf_maxlag: int | None = None,
    common_window: bool = True,
    confirm_steps: int = 3,
    confirm_gap: float = 0.05,
    verbose: bool = False,
) -> tuple[pd.Series | None, float | None, pd.DataFrame | None]:
    """Smallest differencing amount that renders ``series`` stationary.

    Binary search on ``d`` over ``[0, max_d]``, minimizing memory loss subject to
    passing the Augmented Dickey-Fuller test at level ``alpha``.

    :param series: (pd.Series) Series to difference.
    :param fixed_width: (bool) Fixed-width window (``frac_diff_ffd``) when True,
        expanding window (``frac_diff``) when False.
    :param alpha: (float) ADF significance level.
    :param max_d: (float) Upper bound of the search.
    :param tol: (float) Binary-search convergence tolerance on ``d``.
    :param use_log: (bool) Log-transform before differencing. True for prices,
        False for series that are already returns.
    :param thres: (float) Weight cutoff passed to the fixed-width transform.
    :param max_window_frac: (float) Window cap as a fraction of series length.
    :param min_obs: (int) Fewest post-window observations for a usable ADF test.
    :param adf_autolag: (str or None) Passed through to ``adf_data`` (and hence
        ``adfuller``) as ``autolag``. Default ``"AIC"``. Pass ``None`` together
        with ``adf_maxlag`` to reproduce the pre-2026 fixed-lag behavior.
    :param adf_maxlag: (int or None) Passed through to ``adf_data`` as
        ``maxlag``. Default ``None``, letting ``adfuller`` apply the Schwert
        rule.
    :param common_window: (bool) When ``fixed_width`` is True, evaluate every
        candidate ``d`` over an identical FFD window (same lag count, same
        sample size) instead of the thres-driven window, so ADF power does not
        vary with the parameter being searched. Ignored when ``fixed_width``
        is False.
    :param confirm_steps: (int) Number of points above a candidate ``best_d``
        to probe before accepting the crossing, guarding against a boundary
        rejection that does not hold at nearby larger ``d``. Set to ``0`` to
        disable confirmation.
    :param confirm_gap: (float) Spacing in ``d`` between successive
        confirmation probes above ``best_d``.
    :param verbose: (bool) Emit per-iteration progress. Affects logging only.
    :return: (tuple) ``(differenced_series, d, adf_table)``.

        * Stationary at ``d = 0``: the untransformed series, ``0.0``, and a
          one-row table.
        * Stationary at some ``d > 0``: the differenced series, that ``d``, and
          the full search trace.
        * Never stationary: ``(None, None, adf_table)``. The table is returned
          rather than ``None`` so the caller can inspect how close the search got.

    :raises ValueError: When no candidate ``d`` yielded a usable sample.

    Notes
    -----
    **Fix 2 — ``d = 0`` is now evaluated.** The search opened at ``mid = 0.5`` and
    never tested ``d = 0``. For an already-stationary input every candidate passed,
    ``high`` halved until ``high - low < tol``, and the function returned
    ``d ≈ tol/2`` together with a differenced series. Measured on 10,000
    stationary observations: ``d = 0.001`` and 97 observations discarded to the
    window. The series is now ADF-tested at ``d = 0`` first and returned untouched
    when it passes.

    **Fix 3 — the failure path no longer depends on ``verbose``.** The early
    return for an unsuccessful search was nested inside ``if verbose:``. With the
    default ``verbose=False`` a series for which no stationary ``d`` exists
    returned ``(None, max_d, adf_table)``, so a caller unpacking
    ``series, d, _ = fracdiff_optimal(...)`` received a plausible ``d`` alongside
    a ``None`` series and no error. Reproduced on a trending series with
    ``max_d=0.02``: ``verbose=False`` gave ``(None, 0.02, DataFrame)`` while
    ``verbose=True`` gave ``(None, None, None)``.

    **Fix 4 — the failure check precedes any use of the ADF table.**
    ``diff_adf.index = diff_adf.index.round(4)`` ran before the ``best_d is None``
    test. When every candidate was skipped by the sample guard, ``diff_adf`` was
    still ``None`` and that line raised ``AttributeError``.

    **Fix 5 — one stationarity criterion, not two.** ``best_d`` was assigned on
    ``p_value < alpha`` alone while the returned series was assigned on the
    stricter ``stationary`` flag, which additionally requires the ADF statistic to
    fall below the critical value. A ``d`` satisfying the first but not the second
    produced a returned ``d`` that did not correspond to the returned series. Both
    now key off ``stationary``.

    **Fix 6 — the memo caches now work.** Both used
    ``cache.setdefault(key, expensive_call(...))``. Python evaluates the second
    argument before the lookup, so every call recomputed the transform and the ADF
    test and then discarded the result. Replaced with explicit membership tests.

    **Sample-size guard.** The FFD window length is *unimodal* in ``d``, peaking
    near ``d ≈ 0.1`` at roughly 4,000 lags for ``thres=1e-5``. The previous guard
    responded to an undersized sample with ``high = mid``, steering the search on
    a non-monotone quantity. The window is now bounded by ``max_window_frac`` so
    the condition is largely prevented; when it still occurs the candidate is
    recorded as unusable and the search moves to larger ``d``, which shortens the
    window in the region above the peak where the stationary range lies.

    **Monotonicity assumption.** Binary search presumes that stationarity is
    monotone in ``d``. This holds for the ADF test on ordinary price series but is
    not guaranteed. Inspect the returned trace before trusting a boundary result.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected `series` to be a pandas Series, but got {type(series)}")
    if series.dropna().shape[0] < min_obs:
        raise ValueError(f"series has fewer than min_obs={min_obs} usable observations")

    name = series.name if series.name is not None else "series"
    label = "log-transformed " if use_log else ""

    # -- Common evaluation window --------------------------------------------
    # Every candidate d is transformed with the same number of lags, so every
    # ADF test sees the same sample size over the same calendar span. With a
    # thres-driven window the lag count is unimodal in d, peaking near d = 0.1,
    # so candidates were previously compared across different sample sizes and
    # ADF power varied with the parameter being searched.
    fixed_window: int | None = None
    if fixed_width and common_window:
        n_obs = int(series.dropna().shape[0])
        cap = n_obs if max_window_frac is None else max(2, int(n_obs * max_window_frac))
        probe = np.round(np.arange(0.0, float(max_d) + 1e-9, 0.01), 4)
        fixed_window = max(len(get_weights_ffd(float(dv), thres, cap)) - 1 for dv in probe)
        fixed_window = min(fixed_window, cap - 1, n_obs - min_obs - 1)
        if fixed_window < 1:
            raise ValueError(
                f"series of {n_obs} observations cannot support a common window "
                f"with min_obs={min_obs}. Raise `thres` or lower `max_window_frac`."
            )
        if verbose:
            logger.debug(
                "Common FFD window: {} lags, {} observations per candidate.",
                fixed_window,
                n_obs - fixed_window,
            )

    diff_cache: dict[float, pd.Series] = {}
    adf_table: pd.DataFrame | None = None
    skipped: list = []

    def transform(d_value: float) -> pd.Series:
        """Differenced series at ``d_value``, memoized. Fix 6."""
        if d_value not in diff_cache:
            if fixed_width:
                diff_cache[d_value] = frac_diff_ffd(
                    series,
                    d_value,
                    thres=thres,
                    use_log=use_log,
                    max_window_frac=max_window_frac,
                    window=fixed_window,
                )
            else:
                diff_cache[d_value] = frac_diff(series, d_value, use_log=use_log)
        return diff_cache[d_value]

    def evaluate(d_value: float) -> bool | None:
        """Test ``d_value`` and fold the result into the trace.

        :return: (bool) Stationarity, or ``None`` when the sample was too small.
        """
        nonlocal adf_table
        try:
            diff = transform(d_value)
        except ValueError as exc:
            logger.warning("d = {:.6f} unusable for '{}': {}", d_value, name, exc)
            skipped.append(d_value)
            return None

        if not isinstance(diff, pd.Series) or diff.shape[0] < min_obs:
            n_left = diff.shape[0] if isinstance(diff, pd.Series) else 1
            logger.warning(
                "d = {:.6f} leaves only {} observations for '{}'; below min_obs={}.",
                d_value,
                n_left,
                name,
                min_obs,
            )
            skipped.append(d_value)
            return None

        adf_table = adf_data(
            series, diff, d_value, adf_table, alpha, autolag=adf_autolag, maxlag=adf_maxlag
        )
        return bool(adf_table.loc[d_value, "stationary"])

    # -- Fix 2: evaluate d = 0 before entering the search ---------------------
    zero_stationary = evaluate(0.0)
    if zero_stationary:
        if verbose:
            logger.info(
                "'{}' is already stationary at d = 0 (ADF alpha = {}); returning it untransformed.",
                name,
                alpha,
            )
        # The common window is an evaluation device, not an output format. At
        # d = 0 the transform is the identity, so the full untrimmed series is
        # returned; trimming it would discard observations for no transform.
        identity = np.log(series.clip(lower=1e-8)) if use_log else series.copy()
        return identity.astype("float64"), 0.0, adf_table

    # -- Binary search over (0, max_d], with confirmation ---------------------
    low, high = 0.0, float(max_d)
    best_d: float | None = None
    best_series: pd.Series | None = None

    for iteration in range(1, 21):
        mid = (low + high) / 2.0
        if verbose:
            logger.debug("{}. Testing d = {:.4f}", iteration, mid)

        stationary = evaluate(mid)

        if stationary is None:
            # Unusable sample. Move toward larger d, which shortens the window
            # above the unimodal peak near d = 0.1.
            low = mid
        elif stationary:
            # Fix 5: one criterion drives both the returned d and the series.
            best_d, best_series = mid, transform(mid)
            high = mid
        else:
            low = mid

        if high - low < tol:
            # Confirmation: a crossing must be sustained, not a single lucky
            # rejection. Probe `confirm_steps` points above the candidate; if any
            # fails, the crossing was noise and the search resumes above it.
            if best_d is None or confirm_steps <= 0:
                break
            failed = [
                probe_d
                for k in range(1, confirm_steps + 1)
                if (probe_d := min(float(max_d), best_d + k * confirm_gap)) > best_d
                and evaluate(probe_d) is False
            ]
            if not failed:
                break
            resume = max(failed)
            logger.warning(
                "Crossing at d = {:.4f} was not sustained ({} of {} confirmation "
                "points failed); resuming search above d = {:.4f}.",
                best_d,
                len(failed),
                confirm_steps,
                resume,
            )
            best_d, best_series = None, None
            low, high = resume, float(max_d)

    if adf_table is None:
        raise ValueError(
            f"No candidate d produced a usable sample for '{name}'. "
            f"Skipped d values: {skipped}. Raise `thres` or supply a longer series."
        )

    # -- Fix 3 and Fix 4: failure handled before the table is touched ---------
    if best_d is None:
        logger.warning(
            "No stationary series found for '{}' with d <= {} at alpha = {}.",
            name,
            max_d,
            alpha,
        )
        return None, None, adf_table

    corr = float(adf_table.loc[best_d, "corr"])
    d_reported = round(best_d, 4)
    if verbose:
        logger.info(
            "d = {} makes {}'{}' stationary for ADF test (alpha = {}) with rho = {:.4f}",
            d_reported,
            label,
            name,
            alpha,
            corr,
        )

    adf_table = adf_table.sort_index()
    adf_table.index.name = "d"
    return best_series, d_reported, adf_table


def plot_min_ffd(adf: pd.DataFrame):
    """
    Create plot of minimum fractional differentiation value needed
    to make Series stationary at 5% confidence with Augmented Dickey-Fuller test.

    :param adf: (pd.DataFrame) ADF trace as returned by ``fracdiff_optimal`` or
        ``adf_data``, indexed by d, with columns including adfStat, corr, and
        a confidence-level column such as "95% conf".
    :return: (matplotlib.figure.Figure) adfStat and corr plotted against d,
        with a horizontal line at the mean 95% critical value.
    """
    adf = adf.sort_index()
    fig = adf[["adfStat", "corr"]].plot(figsize=(12, 6), secondary_y="adfStat")
    plt.axhline(adf["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    return fig


def plot_ffd_vs_data(ffd: pd.Series, data: pd.Series, d: float, name: str = "close"):
    """
    Plot a fractionally differentiated series against the original series on
    twin y-axes.

    :param ffd: (pd.Series) Fractionally differentiated series.
    :param data: (pd.Series) Original (undifferenced) series.
    :param d: (float) Differencing amount used to produce ffd. Shown in the title.
    :param name: (str) Label for the original series' y-axis and the title.
    :return: (matplotlib.figure.Figure) ffd and data plotted on separate
        y-axes sharing a common x-axis.
    """
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
