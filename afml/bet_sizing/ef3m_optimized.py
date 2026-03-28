"""
An implementation of the Exact Fit of the first 3 Moments (EF3M) algorithm
for fitting a mixture of 2 Gaussian distributions.

Based on: López de Prado, M. & Foreman, M. D. (2014). A mixture of two
Gaussians approach to mathematical portfolio oversight: The EF3M algorithm.
*Quantitative Finance*, 14(5), 913-930.

Key optimizations over the reference implementation:
- ``__slots__`` reduces per-instance memory overhead.
- Frequently used values (``_std_dev``, ``_mu_range``) are pre-computed and
  cached to avoid repeated work across ``single_fit_loop`` calls.
- Numba JIT functions carry ``fastmath=True`` and ``cache=True``: ``fastmath``
  enables reassociation of floating-point operations for throughput;
  ``cache=True`` avoids recompilation on every interpreter start.
- ``iter_4_optimized`` / ``iter_5_optimized`` return a ``(bool, ndarray)``
  tuple instead of an empty list, eliminating the ``tolist()`` round-trip and
  the length check that followed it.
- ``get_moments_fast`` writes into a pre-allocated result array rather than
  constructing a new one on each call, reducing GC pressure inside the hot
  loop.
- ``mp_fit`` uses a context-manager pool and polls progress without a
  busy-wait.
"""

import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import comb
from scipy.stats import gaussian_kde


class M2N:
    """
    M2N -- A Mixture of 2 Normal distributions.

    This class contains the parameters and fitting equations for the EF3M
    algorithm. It fits a five-parameter mixture

        f(x) = p1 * N(mu1, sigma1^2) + (1 - p1) * N(mu2, sigma2^2)

    by matching the first four (variant 1) or all five (variant 2) raw
    moments of the observed distribution to those implied by the mixture.

    Parameters
    ----------
    moments : list of float
        The first five raw moments of the mixture distribution (1 to 5).

    epsilon : float, default=1e-5
        Fitting tolerance. Also controls the spacing of the mu2 search grid:
        smaller values produce a finer grid at the cost of more iterations.

    factor : float, default=5
        Scaling factor lambda applied to the standard deviation when
        constructing the mu2 search grid. Increasing this widens the range.

    n_runs : int, default=1
        Number of independent ``single_fit_loop`` calls dispatched by
        ``mp_fit``. Each run starts from a fresh random p1 draw.

    variant : {1, 2}, default=1
        EF3M variant:
        - ``1`` -- uses the first four moments (equations 22-25 in the paper).
        - ``2`` -- uses all five moments (equations 22-24 and 27-29).

    max_iter : int, default=100_000
        Maximum number of convergence iterations inside ``fit``.

    num_workers : int, default=-1
        Number of CPU cores for ``mp_fit``. ``-1`` uses all available cores.

    Attributes
    ----------
    parameters : ndarray of shape (5,)
        Best-fit parameters ``[mu1, mu2, sigma1, sigma2, p1]`` found so far.

    error : float
        Sum of squared moment residuals for ``parameters``.

    Examples
    --------
    >>> from afml.bet_sizing.ef3m import M2N, centered_moment, raw_moment
    >>>
    >>> # Convert a series of P&L returns to raw moments
    >>> central = [centered_moment(moments_list, i) for i in range(1, 6)]
    >>> raw = raw_moment(central, mean_return)
    >>>
    >>> m2n = M2N(raw, n_runs=10, variant=1)
    >>> df = m2n.mp_fit()
    >>>
    >>> from afml.bet_sizing.ef3m import most_likely_parameters
    >>> params = most_likely_parameters(df)
    """

    __slots__ = [
        "epsilon",
        "factor",
        "n_runs",
        "variant",
        "max_iter",
        "num_workers",
        "moments",
        "new_moments",
        "parameters",
        "error",
        "_std_dev",
        "_mu_range",
    ]

    def __init__(
        self,
        moments,
        epsilon=1e-5,
        factor=5,
        n_runs=1,
        variant=1,
        max_iter=100_000,
        num_workers=-1,
    ):
        self.epsilon     = epsilon
        self.factor      = factor
        self.n_runs      = n_runs
        self.variant     = variant
        self.max_iter    = max_iter
        self.num_workers = num_workers

        self.moments     = np.array(moments, dtype=np.float64)
        self.new_moments = np.zeros(5, dtype=np.float64)
        self.parameters  = np.zeros(5, dtype=np.float64)
        self.error       = float(np.sum(self.moments ** 2))

        # Pre-computed std dev of the mixture used to scale the mu2 grid.
        self._std_dev  = float(centered_moment_fast(self.moments) ** 0.5)
        self._mu_range = None  # Lazy-initialised on first access.

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_mu_range(self):
        """
        Construct and cache the mu2 search grid.

        Reproduces the original grid exactly:
            mu2[i] = moments[0] + i * epsilon * factor * std_dev
            for i = 1, 2, ..., int(1/epsilon) - 1
        """
        if self._mu_range is None:
            step    = self.epsilon * self.factor * self._std_dev
            n_pts   = int(1 / self.epsilon)          # same bound as original
            indices = np.arange(1, n_pts, dtype=np.float64)
            self._mu_range = self.moments[0] + indices * step
        return self._mu_range

    # ------------------------------------------------------------------
    # Core fitting
    # ------------------------------------------------------------------

    def fit(self, mu_2):
        """
        Fit mixture parameters for a given initial mu2 value.

        A random p1 is drawn from Uniform(0, 1) at the start of each call.
        The algorithm then iterates the variant-specific update equations
        until either convergence (|delta p1| < epsilon) or max_iter is
        reached.

        Parameters
        ----------
        mu_2 : float
            Initial estimate for the mean of the second Gaussian component.

        Returns
        -------
        None
            Results are stored in ``self.parameters`` and ``self.error``
            when an improvement is found.
        """
        p_1 = np.random.uniform(0, 1)

        if self.variant == 1:
            fit_func       = iter_4_optimized
            moments_subset = self.moments[:4]
        elif self.variant == 2:
            fit_func       = iter_5_optimized
            moments_subset = self.moments[:5]
        else:
            raise ValueError("variant must be 1 or 2.")

        for _ in range(self.max_iter):
            success, new_params = fit_func(mu_2, p_1, moments_subset)

            if not success:
                return None

            get_moments_fast(new_params, self.new_moments)
            error = calculate_error_fast(
                self.moments, self.new_moments, len(moments_subset)
            )

            if error < self.error:
                self.parameters[:] = new_params
                self.error         = error

            if abs(p_1 - new_params[4]) < self.epsilon:
                break

            p_1  = new_params[4]
            mu_2 = new_params[1]

        return None

    def single_fit_loop(self, epsilon=0):
        """
        Scan the full mu2 grid once, returning the best-fit row.

        Parameters
        ----------
        epsilon : float, optional
            Override the instance epsilon for this run only. Passing ``0``
            (default) keeps the instance value unchanged.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with columns
            ``['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1', 'error']``,
            or an empty DataFrame if no valid fit was found.
        """
        if epsilon != 0:
            self.epsilon   = epsilon
            self._mu_range = None  # Invalidate cached grid.

        self.parameters.fill(0.0)
        self.error = float(np.sum(self.moments ** 2))

        mu_range    = self._get_mu_range()
        err_min     = self.error
        best_params = None

        for mu_2_val in mu_range:
            self.fit(mu_2=float(mu_2_val))
            if self.error < err_min:
                err_min     = self.error
                best_params = self.parameters.copy()

        if best_params is not None:
            return pd.DataFrame(
                {
                    "mu_1":    [best_params[0]],
                    "mu_2":    [best_params[1]],
                    "sigma_1": [best_params[2]],
                    "sigma_2": [best_params[3]],
                    "p_1":     [best_params[4]],
                    "error":   [err_min],
                }
            )

        return pd.DataFrame()

    def mp_fit(self):
        """
        Parallelized multi-run fitting via multiprocessing.Pool.

        Dispatches ``n_runs`` independent ``single_fit_loop`` calls across
        ``num_workers`` CPU cores. Progress is written to stderr after
        polling each 0.5-second interval.

        Returns
        -------
        pd.DataFrame
            Concatenated results from all successful runs. Each row
            corresponds to the best-fit parameters found in one run.
            Returns an empty DataFrame if no run produced a valid fit.
        """
        n_workers = self.num_workers if self.num_workers > 0 else cpu_count()
        epsilons  = [self.epsilon] * self.n_runs
        total     = self.n_runs
        bar_len   = 25

        with Pool(n_workers) as pool:
            async_result = pool.map_async(self.single_fit_loop, epsilons)

            # Poll until done; report progress at each wake-up.
            while not async_result.ready():
                try:
                    remaining = async_result._number_left
                    done      = total - remaining
                    filled    = int(done / total * bar_len) if total > 0 else bar_len
                    bar       = "|" + "#" * filled + " " * (bar_len - filled) + "|"
                    sys.stderr.write(
                        f"\r{bar} {done}/{total} fitting rounds complete."
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                async_result.wait(timeout=0.5)

            # Final completion line.
            sys.stderr.write(
                f"\r{'|' + '#' * bar_len + '|'} {total}/{total} "
                "fitting rounds complete.\n"
            )
            sys.stderr.flush()

            df_list = [df for df in async_result.get() if not df.empty]

        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame()


# ======================================================================
# Numba-accelerated inner functions
# ======================================================================

@njit(fastmath=True, cache=True)
def centered_moment_fast(moments):  # pragma: no cover
    """
    Compute the variance (second central moment) from raw moments.

    Uses the identity Var[X] = E[X^2] - E[X]^2, requiring only the
    first two raw moments.

    Parameters
    ----------
    moments : ndarray
        Raw moments array. Only ``moments[0]`` and ``moments[1]`` are used.

    Returns
    -------
    float
        Variance of the distribution.
    """
    return moments[1] - moments[0] ** 2


@njit(fastmath=True, cache=True)
def get_moments_fast(parameters, result_array):  # pragma: no cover
    """
    Compute the first five raw moments of a 2-Gaussian mixture in place.

    Implements equations (6)-(10) of Lopez de Prado & Foreman (2014).
    Powers are pre-computed to avoid redundant multiplications inside the
    convergence loop.

    Parameters
    ----------
    parameters : ndarray of shape (5,)
        Mixture parameters ``[mu1, mu2, sigma1, sigma2, p1]``.

    result_array : ndarray of shape (5,)
        Pre-allocated output array. Written in place; no return value.
    """
    u_1, u_2, s_1, s_2, p_1 = parameters
    p_2 = 1.0 - p_1

    u_1_2, u_1_3, u_1_4, u_1_5 = u_1**2, u_1**3, u_1**4, u_1**5
    u_2_2, u_2_3, u_2_4, u_2_5 = u_2**2, u_2**3, u_2**4, u_2**5
    s_1_2, s_1_4               = s_1**2, s_1**4
    s_2_2, s_2_4               = s_2**2, s_2**4

    result_array[0] = p_1 * u_1 + p_2 * u_2
    result_array[1] = p_1 * (s_1_2 + u_1_2) + p_2 * (s_2_2 + u_2_2)
    result_array[2] = (
        p_1 * (3 * s_1_2 * u_1 + u_1_3) + p_2 * (3 * s_2_2 * u_2 + u_2_3)
    )
    result_array[3] = p_1 * (3 * s_1_4 + 6 * s_1_2 * u_1_2 + u_1_4) + p_2 * (
        3 * s_2_4 + 6 * s_2_2 * u_2_2 + u_2_4
    )
    result_array[4] = p_1 * (
        15 * s_1_4 * u_1 + 10 * s_1_2 * u_1_3 + u_1_5
    ) + p_2 * (
        15 * s_2_4 * u_2 + 10 * s_2_2 * u_2_3 + u_2_5
    )


@njit(fastmath=True, cache=True)
def calculate_error_fast(moments, new_moments, n_moments):  # pragma: no cover
    """
    Sum of squared moment residuals.

    Parameters
    ----------
    moments : ndarray
        Target raw moments (observed distribution).

    new_moments : ndarray
        Moments implied by the current parameter estimate.

    n_moments : int
        Number of moments to include (4 for variant 1, 5 for variant 2).

    Returns
    -------
    float
        Sum of squared residuals across the first ``n_moments`` moments.
    """
    error = 0.0
    for i in range(n_moments):
        diff   = moments[i] - new_moments[i]
        error += diff * diff
    return error


@njit(fastmath=True, cache=True)
def iter_4_optimized(mu_2, p_1, moments):  # pragma: no cover
    """
    Single iteration of EF3M variant 1 (first four moments).

    Implements equations (22)-(25) of Lopez de Prado & Foreman (2014).
    Returns a (success, parameters) tuple so the caller branches without
    an intermediate list allocation or length check.

    Parameters
    ----------
    mu_2 : float
        Current estimate for mu2.

    p_1 : float
        Current estimate for p1.

    moments : ndarray of shape (4,)
        The first four raw moments of the observed distribution.

    Returns
    -------
    success : bool
        False if any validity check fails (divide-by-zero, negative
        variance, or probability outside [0, 1]).

    parameters : ndarray of shape (5,)
        Updated estimates ``[mu1, mu2, sigma1, sigma2, p1]`` when
        success is True; a zero array otherwise.
    """
    m_1, m_2, m_3, m_4 = moments[0], moments[1], moments[2], moments[3]
    _zero = np.zeros(5, dtype=np.float64)

    # Guard degenerate mixture weights.
    if abs(p_1) < 1e-15 or abs(1.0 - p_1) < 1e-15:
        return False, _zero

    # Eq. (22): mu1
    mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1

    # Eq. (24): sigma2^2
    denom = 3.0 * (1.0 - p_1) * (mu_2 - mu_1)
    if abs(denom) < 1e-15:
        return False, _zero

    sigma_2_sq = (
        m_3
        + 2.0 * p_1 * mu_1**3
        + (p_1 - 1.0) * mu_2**3
        - 3.0 * mu_1 * (m_2 + mu_2**2 * (p_1 - 1.0))
    ) / denom

    if sigma_2_sq < 0.0:
        return False, _zero
    sigma_2 = sigma_2_sq**0.5

    # Eq. (23): sigma1^2
    sigma_1_sq = (
        (m_2 - sigma_2**2 - mu_2**2) / p_1
        + sigma_2**2 + mu_2**2 - mu_1**2
    )
    if sigma_1_sq < 0.0:
        return False, _zero
    sigma_1 = sigma_1_sq**0.5

    # Eq. (25): p1 update
    p_1_deno = (
        3.0 * (sigma_1**4 - sigma_2**4)
        + 6.0 * (sigma_1**2 * mu_1**2 - sigma_2**2 * mu_2**2)
        + mu_1**4 - mu_2**4
    )
    if abs(p_1_deno) < 1e-15:
        return False, _zero

    p_1_new = (
        m_4 - 3.0 * sigma_2**4 - 6.0 * sigma_2**2 * mu_2**2 - mu_2**4
    ) / p_1_deno

    if p_1_new < 0.0 or p_1_new > 1.0:
        return False, _zero

    return True, np.array([mu_1, mu_2, sigma_1, sigma_2, p_1_new], dtype=np.float64)


@njit(fastmath=True, cache=True)
def iter_5_optimized(mu_2, p_1, moments):  # pragma: no cover
    """
    Single iteration of EF3M variant 2 (first five moments).

    Implements equations (22)-(24) and (27)-(29) of Lopez de Prado &
    Foreman (2014).

    Parameters
    ----------
    mu_2 : float
        Current estimate for mu2.

    p_1 : float
        Current estimate for p1.

    moments : ndarray of shape (5,)
        The first five raw moments of the observed distribution.

    Returns
    -------
    success : bool
        False if any validity check fails.

    parameters : ndarray of shape (5,)
        Updated estimates ``[mu1, mu2, sigma1, sigma2, p1]`` when
        success is True; a zero array otherwise.
    """
    m_1, m_2, m_3, m_4, m_5 = (
        moments[0], moments[1], moments[2], moments[3], moments[4]
    )
    _zero = np.zeros(5, dtype=np.float64)

    if abs(p_1) < 1e-15 or abs(1.0 - p_1) < 1e-4:
        return False, _zero

    # Eq. (22): mu1
    mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1

    # Eq. (24): sigma2^2
    denom = 3.0 * (1.0 - p_1) * (mu_2 - mu_1)
    if abs(denom) < 1e-15:
        return False, _zero

    sigma_2_sq = (
        m_3
        + 2.0 * p_1 * mu_1**3
        + (p_1 - 1.0) * mu_2**3
        - 3.0 * mu_1 * (m_2 + mu_2**2 * (p_1 - 1.0))
    ) / denom

    if sigma_2_sq < 0.0:
        return False, _zero
    sigma_2 = sigma_2_sq**0.5

    # Eq. (23): sigma1^2
    sigma_1_sq = (
        (m_2 - sigma_2**2 - mu_2**2) / p_1
        + sigma_2**2 + mu_2**2 - mu_1**2
    )
    if sigma_1_sq < 0.0:
        return False, _zero
    sigma_1 = sigma_1_sq**0.5

    # Eq. (27): mu2 update via a1
    mu_1_terms = 3.0 * sigma_1**4 + 6.0 * sigma_1**2 * mu_1**2 + mu_1**4
    a_1_sq = 6.0 * sigma_2**4 + (m_4 - p_1 * mu_1_terms) / (1.0 - p_1)
    if a_1_sq < 0.0:
        return False, _zero

    a_1      = a_1_sq**0.5
    mu_2_sq  = a_1 - 3.0 * sigma_2**2
    if mu_2_sq < 0.0:
        return False, _zero
    mu_2_new = mu_2_sq**0.5

    # Eqs. (28)-(29): p1 update
    a_2 = 15.0 * sigma_1**4 * mu_1 + 10.0 * sigma_1**2 * mu_1**3 + mu_1**5
    b_2 = (
        15.0 * sigma_2**4 * mu_2_new
        + 10.0 * sigma_2**2 * mu_2_new**3
        + mu_2_new**5
    )

    if abs(a_2 - b_2) < 1e-15:
        return False, _zero

    p_1_new = (m_5 - b_2) / (a_2 - b_2)
    if p_1_new < 0.0 or p_1_new > 1.0:
        return False, _zero

    return True, np.array(
        [mu_1, mu_2_new, sigma_1, sigma_2, p_1_new], dtype=np.float64
    )


# ======================================================================
# Pure-Python utility functions (public API)
# ======================================================================

def centered_moment(moments, order):
    """
    Compute a single central moment of a given order from raw moments.

    Parameters
    ----------
    moments : list of float
        The first ``order`` raw moments (1 to order).

    order : int
        The order of the central moment to compute. Order 2 is handled
        analytically (Var[X] = E[X^2] - E[X]^2) for efficiency; higher
        orders use the general binomial expansion.

    Returns
    -------
    float
        The central moment of the specified order.
    """
    if order == 2:
        return moments[1] - moments[0] ** 2

    moment_c = 0
    for j in range(order + 1):
        combin = int(comb(order, j))
        a_1    = 1 if j == order else moments[order - j - 1]
        moment_c += (-1) ** j * combin * moments[0] ** j * a_1
    return moment_c


def raw_moment(central_moments, dist_mean):
    """
    Convert a list of central moments to raw moments.

    Parameters
    ----------
    central_moments : list of float
        The first n central moments (1 to n).

    dist_mean : float
        The mean of the distribution (first raw moment).

    Returns
    -------
    list of float
        The first n raw moments (1 to n).
    """
    raw_moments  = [dist_mean]
    central_aug  = [1] + list(central_moments)

    for n_i in range(2, len(central_aug)):
        k_range  = np.arange(n_i + 1, dtype=np.float64)
        combs    = np.array([comb(n_i, int(k)) for k in k_range])
        centrals = np.array([central_aug[int(k)] for k in k_range])
        powers   = np.array([dist_mean ** (n_i - int(k)) for k in k_range])
        raw_moments.append(float(np.sum(combs * centrals * powers)))

    return raw_moments


def most_likely_parameters(data, ignore_columns="error", res=10_000):
    """
    Determine the most likely parameter estimates via kernel density estimation.

    For each parameter column in ``data``, a Gaussian KDE is fitted and the
    mode of the density is returned as the point estimate.

    Parameters
    ----------
    data : pd.DataFrame
        Parameter estimates from all fitting runs, as returned by ``mp_fit``.

    ignore_columns : str or list of str, default='error'
        Column name(s) to exclude from the analysis.

    res : int, default=10_000
        Number of evaluation points for the KDE. Reduced automatically to
        ``len(data) * 10`` when the dataset is small, preventing over-smoothing
        on sparse results.

    Returns
    -------
    dict
        Mapping of parameter name to its KDE mode, rounded to five decimal
        places.
    """
    df = data.copy()
    if isinstance(ignore_columns, str):
        ignore_columns = [ignore_columns]

    columns   = [c for c in df.columns if c not in ignore_columns]
    d_results = {}

    for col in columns:
        col_data   = df[col].to_numpy()
        actual_res = min(res, len(col_data) * 10)
        x_range    = np.linspace(col_data.min(), col_data.max(), num=actual_res)
        kde        = gaussian_kde(col_data)
        y_kde      = kde.evaluate(x_range)
        d_results[col] = round(float(x_range[np.argmax(y_kde)]), 5)

    return d_results

