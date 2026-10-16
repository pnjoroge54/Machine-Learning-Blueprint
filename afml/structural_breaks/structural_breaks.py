"""
afml.structural_breaks — optimized standalone implementation.

Covers:
    CUSUM tests
        - Chu-Stinchcombe-White (one-sided and two-sided)
    Explosiveness tests
        - Chow-Type Dickey-Fuller (SDFC)
        - SADF: linear, quadratic, sm_poly_1, sm_poly_2, sm_exp, sm_power
        - QADF: Quantile ADF (§17.4.2.4)
        - CADF: Conditional ADF (§17.4.2.5)

Key optimizations over the original mlfinlab-derived code:
    1. cusum: precompute sigma²_t and convert to NumPy arrays before the inner
       loop — eliminates O(n) Series.loc calls inside the O(n²) scan.
    2. chow: precompute the full lagged arrays once per molecule call; the
       D_t dummy is applied via NumPy masking, not series copy+slice assignment.
    3. get_betas: @njit kernel — called O(T²) times in SADF, so JIT matters.
    4. _lag_df replacement: build the lag matrix with NumPy strides in one pass
       rather than repeated DataFrame.join allocations.
    5. _get_y_x: returns NumPy arrays directly; no Pandas overhead in the
       inner loop.
    6. QADF and CADF added as new functions (§17.4.2.4 and §17.4.2.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# ══════════════════════════════════════════════════════════════════════════════
# Core OLS kernel  (called O(T²) times inside SADF — must be fast)
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _get_betas_numba(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OLS beta and variance of beta.  AFML Snippet 17.4, p. 259.

    :param X: (T, N) design matrix — NumPy float64
    :param y: (T, 1) outcome vector — NumPy float64
    :return: (b_mean (N,1), b_var (N,N))
    """
    xy = X.T @ y
    xx = X.T @ X

    # Attempt inversion; numba does not raise LinAlgError so we check det
    det = np.linalg.det(xx)
    if det == 0.0 or np.isnan(det):
        nan_vec = np.full((X.shape[1], 1), np.nan)
        nan_mat = np.full((X.shape[1], X.shape[1]), np.nan)
        return nan_vec, nan_mat

    xx_inv = np.linalg.inv(xx)
    b_mean = xx_inv @ xy
    err = y - X @ b_mean
    b_var = (err.T @ err)[0, 0] / (X.shape[0] - X.shape[1]) * xx_inv
    return b_mean, b_var


def get_betas(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Public wrapper — accepts both NumPy arrays and Pandas DataFrames.

    :param X: (T, N) design matrix
    :param y: (T, 1) or (T,) outcome
    :return: (b_mean, b_var)
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    return _get_betas_numba(X_arr, y_arr)


# ══════════════════════════════════════════════════════════════════════════════
# Lag matrix (replaces _lag_df  — NumPy strides, one pass)
# ══════════════════════════════════════════════════════════════════════════════


def _build_lag_matrix(
    arr: np.ndarray,
    lags: int | list,
) -> np.ndarray:
    """
    Build a (T, len(lags)) lag matrix from a 1-D array.

    Equivalent to _lag_df but operates entirely in NumPy.

    :param arr: 1-D float64 array of length T
    :param lags: int (1..lags) or list of specific lag values
    :return: (T, len(lags)) array with NaN fill for unavailable positions
    """
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = [int(lg) for lg in lags]

    T = len(arr)
    out = np.full((T, len(lag_list)), np.nan)
    for col, lag in enumerate(lag_list):
        out[lag:, col] = arr[: T - lag]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SADF dataset preparation (returns NumPy — no Pandas in inner loop)
# ══════════════════════════════════════════════════════════════════════════════


def _get_y_x_numpy(
    series: pd.Series,
    model: str,
    lags: int | list,
    add_const: bool,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Prepare design matrix X and response y for SADF estimation.

    Returns NumPy arrays; also returns the valid index for result alignment.

    :param series: log-price series
    :param model: 'linear' | 'quadratic' | 'sm_poly_1' | 'sm_poly_2' |
                  'sm_exp' | 'sm_power'
    :param lags: number of lags or list of specific lags
    :param add_const: whether to add an intercept column
    :return: (X_arr, y_arr, valid_index)
    """
    vals = series.values.astype(np.float64)
    T = len(vals)

    diff_vals = np.diff(vals)  # length T-1

    # Lag matrix on diff_vals
    lag_mat = _build_lag_matrix(diff_vals, lags)  # (T-1, n_lags)

    # Determine the first row that has no NaN (after lagging)
    max_lag = max(lags) if not isinstance(lags, int) else lags
    start = max_lag  # first valid row index in diff_vals

    diff_trimmed = diff_vals[start:]  # (T-1-start,)
    lag_trimmed = lag_mat[start:]  # (T-1-start, n_lags)
    # y_lagged: level value at t-1 aligned to diff_trimmed
    # diff_vals[i] = vals[i+1] - vals[i], so y_lagged = vals[start:T-1]
    y_lagged = vals[start : T - 1]  # (T-1-start,)
    n = len(diff_trimmed)

    # Build valid index
    valid_index = series.index[start + 1 :]  # +1 because diff removes first

    if model in ("linear", "quadratic"):
        # y = diff_vals
        y_arr = diff_trimmed.reshape(-1, 1)
        cols = [y_lagged]
        cols += [lag_trimmed[:, i] for i in range(lag_trimmed.shape[1])]
        if add_const:
            cols.append(np.ones(n))
        cols.append(np.arange(n, dtype=np.float64))  # linear trend
        if model == "quadratic":
            cols.append(np.arange(n, dtype=np.float64) ** 2)  # quad trend
        X_arr = np.column_stack(cols)
        beta_col = 0  # y_lagged is the test coefficient column

    elif model == "sm_poly_1":
        y_arr = vals[start + 1 :].reshape(-1, 1)  # levels
        t_ = np.arange(n, dtype=np.float64)
        cols = [np.ones(n), t_, t_**2]
        X_arr = np.column_stack(cols)
        beta_col = 2  # quad_trend

    elif model == "sm_poly_2":
        y_arr = np.log(vals[start + 1 :]).reshape(-1, 1)  # log levels
        t_ = np.arange(n, dtype=np.float64)
        cols = [np.ones(n), t_, t_**2]
        X_arr = np.column_stack(cols)
        beta_col = 2

    elif model == "sm_exp":
        y_arr = np.log(vals[start + 1 :]).reshape(-1, 1)
        t_ = np.arange(n, dtype=np.float64)
        cols = [np.ones(n), t_]
        X_arr = np.column_stack(cols)
        beta_col = 1  # trend

    elif model == "sm_power":
        y_arr = np.log(vals[start + 1 :]).reshape(-1, 1)
        with np.errstate(divide="ignore"):
            log_t = np.log(np.arange(n, dtype=np.float64))
        cols = [np.ones(n), log_t]
        X_arr = np.column_stack(cols)
        beta_col = 1  # log_trend

    else:
        raise ValueError(f"Unknown model: {model!r}")

    # Move beta_col to column 0 for consistent extraction in inner loop
    if beta_col != 0:
        order = [beta_col] + [i for i in range(X_arr.shape[1]) if i != beta_col]
        X_arr = X_arr[:, order]

    return X_arr, y_arr, valid_index


# ══════════════════════════════════════════════════════════════════════════════
# SADF inner loop
# ══════════════════════════════════════════════════════════════════════════════


def _get_sadf_at_t(
    X: np.ndarray,
    y: np.ndarray,
    min_length: int,
    model: str,
    phi: float,
) -> float:
    """
    SADF inner loop: return max ADF t-stat over all start points.  AFML §17.4.2.6

    :param X: full X up to time t
    :param y: full y up to time t
    :param min_length: minimum window length τ
    :param model: model name (sm_* uses abs value and phi penalisation)
    :param phi: penalisation exponent in [0, 1] for SMT
    :return: bsadf value at t
    """
    n_obs = y.shape[0]
    bsadf = -np.inf
    is_smt = model.startswith("sm")

    for start in range(n_obs - min_length + 1):
        X_ = X[start:]
        y_ = y[start:]
        X_f = X_.astype(np.float64)
        y_f = y_.astype(np.float64)
        if not (np.isfinite(X_f).all() and np.isfinite(y_f).all()):
            continue
        b_mean, b_var = _get_betas_numba(X_f, y_f)
        if np.isnan(b_mean[0, 0]):
            continue
        coef = b_mean[0, 0]
        std = b_var[0, 0] ** 0.5
        if std == 0.0:
            continue
        t_stat = coef / std
        if is_smt:
            sample_len = y_.shape[0]
            t_stat = abs(t_stat) / (sample_len**phi) if phi > 0 else abs(t_stat)
        bsadf = max(bsadf, t_stat)

    return bsadf


def get_sadf(
    series: pd.Series,
    model: str = "linear",
    lags: int | list = 1,
    min_length: int = 20,
    add_const: bool = False,
    phi: float = 0.0,
) -> pd.Series:
    """
    Supremum Augmented Dickey-Fuller statistic.  AFML p. 258-259.

    :param series: log-price pd.Series
    :param model: 'linear' | 'quadratic' | 'sm_poly_1' | 'sm_poly_2' |
                  'sm_exp' | 'sm_power'
    :param lags: int or list
    :param min_length: minimum window τ
    :param add_const: add intercept (for linear/quadratic)
    :param phi: SMT penalisation exponent
    :return: pd.Series of SADF statistics indexed like series
    """
    X, y, idx = _get_y_x_numpy(series, model, lags, add_const)
    molecule = idx[min_length:]
    results = {}

    for loc in range(min_length, len(idx)):
        t_idx = idx[loc]
        X_sub = X[: loc + 1]
        y_sub = y[: loc + 1]
        results[t_idx] = _get_sadf_at_t(X_sub, y_sub, min_length, model, phi)

    return pd.Series(results, name="sadf")


# ══════════════════════════════════════════════════════════════════════════════
# QADF — Quantile ADF  (§17.4.2.4)
# ══════════════════════════════════════════════════════════════════════════════


def get_qadf(
    series: pd.Series,
    model: str = "linear",
    lags: int | list = 1,
    min_length: int = 20,
    add_const: bool = False,
    q: float = 0.95,
    v: float = 0.025,
) -> pd.DataFrame:
    """
    Quantile ADF: reports Q_{t,q} (centrality) and Qdot_{t,q,v} (dispersion)
    of the inner ADF distribution s_t = {ADF_{t0,t}} at each endpoint t.
    AFML §17.4.2.4.

    Note: SADF = Q_{t,1}.

    :param series: log-price pd.Series
    :param model: as per get_sadf
    :param lags: as per get_sadf
    :param min_length: minimum window τ
    :param add_const: as per get_sadf
    :param q: quantile of interest, default 0.95
    :param v: half-width for dispersion band, 0 < v ≤ min(q, 1-q)
    :return: pd.DataFrame with columns 'q_adf' and 'q_dot'
    """
    if not (0 < v <= min(q, 1 - q)):
        raise ValueError("v must satisfy 0 < v ≤ min(q, 1-q)")

    X, y, idx = _get_y_x_numpy(series, model, lags, add_const)
    results = {}

    for loc in range(min_length, len(idx)):
        t_idx = idx[loc]
        X_sub = X[: loc + 1]
        y_sub = y[: loc + 1]
        n_obs = y_sub.shape[0]

        adf_vals = []
        for start in range(n_obs - min_length + 1):
            X_ = X_sub[start:].astype(np.float64)
            y_ = y_sub[start:].astype(np.float64)
            if not (np.isfinite(X_).all() and np.isfinite(y_).all()):
                continue
            b_mean, b_var = _get_betas_numba(X_, y_)
            if np.isnan(b_mean[0, 0]):
                continue
            std = b_var[0, 0] ** 0.5
            if std == 0.0:
                continue
            adf_vals.append(b_mean[0, 0] / std)

        if len(adf_vals) < 2:
            results[t_idx] = {"q_adf": np.nan, "q_dot": np.nan}
            continue

        s_t = np.array(adf_vals)
        q_val = float(np.quantile(s_t, q))
        q_lo = float(np.quantile(s_t, q - v))
        q_hi = float(np.quantile(s_t, q + v))
        results[t_idx] = {"q_adf": q_val, "q_dot": q_hi - q_lo}

    return pd.DataFrame.from_dict(results, orient="index", columns=["q_adf", "q_dot"])


# ══════════════════════════════════════════════════════════════════════════════
# CADF — Conditional ADF  (§17.4.2.5)
# ══════════════════════════════════════════════════════════════════════════════


def get_cadf(
    series: pd.Series,
    model: str = "linear",
    lags: int | list = 1,
    min_length: int = 20,
    add_const: bool = False,
    q: float = 0.95,
) -> pd.DataFrame:
    """
    Conditional ADF: C_{t,q} (conditional mean above quantile) and
    Cdot_{t,q} (conditional std above quantile).  AFML §17.4.2.5.

    :param series: log-price pd.Series
    :param model: as per get_sadf
    :param lags: as per get_sadf
    :param min_length: minimum window τ
    :param add_const: as per get_sadf
    :param q: conditioning quantile threshold, default 0.95
    :return: pd.DataFrame with columns 'c_adf' (C_{t,q}) and 'c_dot' (Ċ_{t,q})
    """
    X, y, idx = _get_y_x_numpy(series, model, lags, add_const)
    results = {}

    for loc in range(min_length, len(idx)):
        t_idx = idx[loc]
        X_sub = X[: loc + 1]
        y_sub = y[: loc + 1]
        n_obs = y_sub.shape[0]

        adf_vals = []
        for start in range(n_obs - min_length + 1):
            X_ = X_sub[start:].astype(np.float64)
            y_ = y_sub[start:].astype(np.float64)
            if not (np.isfinite(X_).all() and np.isfinite(y_).all()):
                continue
            b_mean, b_var = _get_betas_numba(X_, y_)
            if np.isnan(b_mean[0, 0]):
                continue
            std = b_var[0, 0] ** 0.5
            if std == 0.0:
                continue
            adf_vals.append(b_mean[0, 0] / std)

        if len(adf_vals) < 2:
            results[t_idx] = {"c_adf": np.nan, "c_dot": np.nan}
            continue

        s_t = np.array(adf_vals)
        threshold = np.quantile(s_t, q)
        tail = s_t[s_t >= threshold]

        if len(tail) == 0:
            results[t_idx] = {"c_adf": np.nan, "c_dot": np.nan}
            continue

        c_mean = float(tail.mean())
        c_std = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
        results[t_idx] = {"c_adf": c_mean, "c_dot": c_std}

    return pd.DataFrame.from_dict(results, orient="index", columns=["c_adf", "c_dot"])


# ══════════════════════════════════════════════════════════════════════════════
# Chow-Type Dickey-Fuller  (optimized)
# ══════════════════════════════════════════════════════════════════════════════


def get_chow_type_stat(
    series: pd.Series,
    min_length: int = 20,
) -> pd.Series:
    """
    Chow-Type Dickey-Fuller test (SDFC).  AFML p. 251-252.

    Optimization: precomputes diff, lag, and converts to NumPy arrays once;
    the D_t* dummy is applied via integer slicing rather than series assignment.

    :param series: price or log-price pd.Series
    :param min_length: number of observations to trim from each end
    :return: pd.Series of DFC_τ* statistics
    """
    vals = series.values.astype(np.float64)
    T = len(vals)
    diff_vals = np.diff(vals)  # length T-1
    lag_vals = vals[:-1]  # length T-1, aligned to diff_vals
    idx = series.index[1:]  # aligned index

    molecule = series.index[min_length : T - min_length]
    results = {}

    for tau_star in molecule:
        # Integer position of tau_star in the full (length-T) index
        pos = series.index.get_loc(tau_star)  # position in original series
        # In diff-space: diff_vals[i] = vals[i+1]-vals[i], so position in
        # diff_vals corresponds to original index position pos-1 onwards.
        # D_t*(τ*) = 1 for t ≥ τ*T, i.e. diff index >= pos-1
        n_diff = len(diff_vals)
        y = diff_vals
        # x = lag_vals * D_t*(τ*): zero before pos, lag_vals[pos-1:] after
        x_d = np.zeros(n_diff)
        cut = pos - 1  # first index in diff-space where D=1
        if cut < 0 or cut >= n_diff:
            continue
        x_d[cut:] = lag_vals[cut:]

        X_arr = x_d.reshape(-1, 1)
        y_arr = y.reshape(-1, 1)
        b_mean, b_var = _get_betas_numba(X_arr, y_arr)
        if np.isnan(b_mean[0, 0]):
            continue
        std = b_var[0, 0] ** 0.5
        if std == 0.0:
            continue
        results[tau_star] = b_mean[0, 0] / std

    return pd.Series(results, name="dfc")


# ══════════════════════════════════════════════════════════════════════════════
# Chu-Stinchcombe-White CUSUM  (optimized)
# ══════════════════════════════════════════════════════════════════════════════


def get_chu_stinchcombe_white_statistics(
    series: pd.Series,
    test_type: str = "one_sided",
) -> pd.DataFrame:
    """
    Chu-Stinchcombe-White CUSUM test on levels.  AFML p. 251.

    Optimization: sigma²_t is precomputed as a cumulative sum over the
    full series once; the inner loop uses NumPy indexing rather than
    Series.loc per step.

    :param series: log-price pd.Series
    :param test_type: 'one_sided' or 'two_sided'
    :return: pd.DataFrame with columns 'stat' and 'critical_value'
    """
    if test_type not in ("one_sided", "two_sided"):
        raise ValueError("test_type must be 'one_sided' or 'two_sided'")

    vals = series.values.astype(np.float64)
    T = len(vals)
    idx = series.index

    # Precompute σ̂²_t = (t-1)^{-1} Σ_{i=2}^{t} (Δy_i)²
    sq_diff = np.diff(vals) ** 2  # length T-1; sq_diff[i] = (y[i+1]-y[i])²
    cum_sq = np.concatenate(
        [[0.0], np.cumsum(sq_diff)]
    )  # length T; cum_sq[t] = Σ_{i=1}^{t} sq_diff[i-1]
    # sigma_sq[t] for t >= 2: (1/(t-1)) * cum_sq[t-1]
    # Note: cum_sq[t] contains t elements of sq_diff (indices 0..t-1)
    # We want sigma_sq at integer index t (0-based): (t-1)^{-1} * cum_sq[t-1]
    # For t=1 (0-based): not enough info (need t>=2)

    results = {}
    # Molecule: index[2:] in original code
    for t in range(2, T):
        sigma_sq_t = cum_sq[t] / (t - 1)  # σ̂²_t: t diffs, denominator (t-1)
        if sigma_sq_t <= 0:
            continue
        sigma_t = sigma_sq_t**0.5

        max_s = -np.inf
        best_n = None

        # Inner scan: n from 0 to t-1
        for n in range(t):  # n is 0-based index in vals
            if test_type == "one_sided":
                diff = vals[t] - vals[n]
            else:
                diff = abs(vals[t] - vals[n])
            span = t - n
            s_n_t = diff / (sigma_t * span**0.5)
            if s_n_t > max_s:
                max_s = s_n_t
                best_n = n

        crit = np.sqrt(4.6 + np.log(t - best_n)) if best_n is not None else np.nan
        results[idx[t]] = {"stat": max_s, "critical_value": crit}

    return pd.DataFrame.from_dict(results, orient="index")
