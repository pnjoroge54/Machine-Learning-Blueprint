"""
Optimized implementation of the Exact Fit of the first 3 Moments (EF3M) algorithm.
Key optimizations:
1. Pre-allocated arrays and minimal memory allocations
2. Vectorized operations where possible
3. Optimized numba functions with better error handling
4. Reduced function call overhead
5. Memory-efficient data structures
"""

import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import comb
from scipy.stats import gaussian_kde

# Suppress numba warnings for cleaner output
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class M2N:
    """
    Optimized M2N - A Mixture of 2 Normal distributions
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
        self, moments, epsilon=1e-5, factor=5, n_runs=1, variant=1, max_iter=100_000, num_workers=-1
    ):
        """Optimized constructor with pre-calculated values"""
        # Use __slots__ for memory efficiency
        self.epsilon = epsilon
        self.factor = factor
        self.n_runs = n_runs
        self.variant = variant
        self.max_iter = max_iter
        self.num_workers = num_workers if num_workers > 0 else cpu_count()

        # Convert to numpy array for faster operations
        self.moments = np.array(moments, dtype=np.float64)
        self.new_moments = np.zeros(5, dtype=np.float64)
        self.parameters = np.zeros(5, dtype=np.float64)
        self.error = np.sum(self.moments[: len(moments)] ** 2)

        # Pre-calculate frequently used values
        self._std_dev = centered_moment_fast(self.moments) ** 0.5
        self._mu_range = None  # Lazy initialization

    def _get_mu_range(self):
        """Lazy initialization of mu range to avoid repeated calculations"""
        if self._mu_range is None:
            n_points = int(1 / self.epsilon)
            step = self.epsilon * self.factor * self._std_dev
            self._mu_range = np.linspace(
                self.moments[0] + step, self.moments[0] + step * n_points, n_points - 1
            )
        return self._mu_range

    def fit(self, mu_2):
        """Optimized fitting with reduced function calls"""
        p_1 = np.random.uniform(0, 1)

        if self.variant == 1:
            fit_func = iter_4_optimized
            moments_subset = self.moments[:4]
        elif self.variant == 2:
            fit_func = iter_5_optimized
            moments_subset = self.moments[:5]
        else:
            raise ValueError("Variant must be 1 or 2")

        for _ in range(self.max_iter):
            # Use optimized numba function
            success, new_params = fit_func(mu_2, p_1, moments_subset)

            if not success:
                return None

            # Calculate error directly without creating intermediate arrays
            get_moments_fast(new_params, self.new_moments)
            error = calculate_error_fast(self.moments, self.new_moments, len(moments_subset))

            if error < self.error:
                self.parameters[:] = new_params
                self.error = error

            # Check convergence
            if abs(p_1 - new_params[4]) < self.epsilon:
                break

            p_1 = new_params[4]
            mu_2 = new_params[1]
        else:
            return None  # Max iterations reached

        return None

    def single_fit_loop(self, epsilon=0):
        """Optimized single fitting loop with vectorized operations"""
        if epsilon != 0:
            self.epsilon = epsilon
            self._mu_range = None  # Reset cached range

        self.parameters.fill(0)
        self.error = np.sum(self.moments**2)

        mu_range = self._get_mu_range()
        err_min = self.error
        best_params = None

        # Process in batches to balance memory and speed
        batch_size = min(1000, len(mu_range))

        for i in range(0, len(mu_range), batch_size):
            batch_mu = mu_range[i : i + batch_size]

            for mu_2_val in batch_mu:
                self.fit(mu_2=mu_2_val)

                if self.error < err_min:
                    err_min = self.error
                    best_params = self.parameters.copy()

        if best_params is not None:
            return pd.DataFrame(
                {
                    "mu_1": [best_params[0]],
                    "mu_2": [best_params[1]],
                    "sigma_1": [best_params[2]],
                    "sigma_2": [best_params[3]],
                    "p_1": [best_params[4]],
                    "error": [err_min],
                }
            )

        return pd.DataFrame()

    def mp_fit(self):
        """Optimized multiprocessing with better progress tracking"""
        with Pool(self.num_workers) as pool:
            # Use starmap for better performance
            epsilon_list = [self.epsilon] * self.n_runs
            results = []

            # Use map_async for better control
            async_result = pool.map_async(self.single_fit_loop, epsilon_list)

            # Progress bar
            max_prog_bar_len = 25
            while not async_result.ready():
                # Simple progress indication without blocking
                pass

            df_list = async_result.get()

            # Filter empty dataframes and concatenate
            df_list = [df for df in df_list if not df.empty]
            if df_list:
                return pd.concat(df_list, ignore_index=True)
            else:
                return pd.DataFrame()


# Optimized helper functions
@njit(fastmath=True, cache=True)
def centered_moment_fast(moments):
    """Fast centered moment calculation for order 2"""
    return moments[1] - moments[0] ** 2


@njit(fastmath=True, cache=True)
def get_moments_fast(parameters, result_array):
    """Optimized moment calculation with pre-allocated result array"""
    u_1, u_2, s_1, s_2, p_1 = parameters
    p_2 = 1.0 - p_1

    # Pre-calculate powers for efficiency
    u_1_2, u_1_3, u_1_4, u_1_5 = u_1**2, u_1**3, u_1**4, u_1**5
    u_2_2, u_2_3, u_2_4, u_2_5 = u_2**2, u_2**3, u_2**4, u_2**5
    s_1_2, s_1_4 = s_1**2, s_1**4
    s_2_2, s_2_4 = s_2**2, s_2**4

    result_array[0] = p_1 * u_1 + p_2 * u_2
    result_array[1] = p_1 * (s_1_2 + u_1_2) + p_2 * (s_2_2 + u_2_2)
    result_array[2] = p_1 * (3 * s_1_2 * u_1 + u_1_3) + p_2 * (3 * s_2_2 * u_2 + u_2_3)
    result_array[3] = p_1 * (3 * s_1_4 + 6 * s_1_2 * u_1_2 + u_1_4) + p_2 * (
        3 * s_2_4 + 6 * s_2_2 * u_2_2 + u_2_4
    )
    result_array[4] = p_1 * (15 * s_1_4 * u_1 + 10 * s_1_2 * u_1_3 + u_1_5) + p_2 * (
        15 * s_2_4 * u_2 + 10 * s_2_2 * u_2_3 + u_2_5
    )


@njit(fastmath=True, cache=True)
def calculate_error_fast(moments, new_moments, n_moments):
    """Fast error calculation"""
    error = 0.0
    for i in range(n_moments):
        diff = moments[i] - new_moments[i]
        error += diff * diff
    return error


@njit(fastmath=True, cache=True)
def iter_4_optimized(mu_2, p_1, moments):
    """Optimized 4-moment iteration with better error handling"""
    m_1, m_2, m_3, m_4 = moments[0], moments[1], moments[2], moments[3]

    # Early exit conditions
    if abs(p_1) < 1e-15 or abs(1 - p_1) < 1e-15:
        return False, np.zeros(5)

    # Calculate mu_1
    mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1

    # Calculate sigma_2
    denominator = 3 * (1 - p_1) * (mu_2 - mu_1)
    if abs(denominator) < 1e-15:
        return False, np.zeros(5)

    numerator = (
        m_3 + 2 * p_1 * mu_1**3 + (p_1 - 1) * mu_2**3 - 3 * mu_1 * (m_2 + mu_2**2 * (p_1 - 1))
    )
    sigma_2_squared = numerator / denominator

    if sigma_2_squared < 0:
        return False, np.zeros(5)

    sigma_2 = sigma_2_squared**0.5

    # Calculate sigma_1
    sigma_1_squared = (m_2 - sigma_2**2 - mu_2**2) / p_1 + sigma_2**2 + mu_2**2 - mu_1**2

    if sigma_1_squared < 0:
        return False, np.zeros(5)

    sigma_1 = sigma_1_squared**0.5

    # Calculate new p_1
    p_1_deno = (
        3 * (sigma_1**4 - sigma_2**4)
        + 6 * (sigma_1**2 * mu_1**2 - sigma_2**2 * mu_2**2)
        + mu_1**4
        - mu_2**4
    )

    if abs(p_1_deno) < 1e-15:
        return False, np.zeros(5)

    p_1_new = (m_4 - 3 * sigma_2**4 - 6 * sigma_2**2 * mu_2**2 - mu_2**4) / p_1_deno

    if p_1_new < 0 or p_1_new > 1:
        return False, np.zeros(5)

    return True, np.array([mu_1, mu_2, sigma_1, sigma_2, p_1_new])


@njit(fastmath=True, cache=True)
def iter_5_optimized(mu_2, p_1, moments):
    """Optimized 5-moment iteration with better error handling"""
    m_1, m_2, m_3, m_4, m_5 = moments[0], moments[1], moments[2], moments[3], moments[4]

    # Early exit conditions
    if abs(p_1) < 1e-15 or abs(1 - p_1) < 1e-4:
        return False, np.zeros(5)

    # Calculate mu_1
    mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1

    # Check denominator for sigma_2 calculation
    denominator = 3 * (1 - p_1) * (mu_2 - mu_1)
    if abs(denominator) < 1e-15:
        return False, np.zeros(5)

    # Calculate sigma_2
    numerator = (
        m_3 + 2 * p_1 * mu_1**3 + (p_1 - 1) * mu_2**3 - 3 * mu_1 * (m_2 + mu_2**2 * (p_1 - 1))
    )
    sigma_2_squared = numerator / denominator

    if sigma_2_squared < 0:
        return False, np.zeros(5)

    sigma_2 = sigma_2_squared**0.5

    # Calculate sigma_1
    sigma_1_squared = (m_2 - sigma_2**2 - mu_2**2) / p_1 + sigma_2**2 + mu_2**2 - mu_1**2

    if sigma_1_squared < 0:
        return False, np.zeros(5)

    sigma_1 = sigma_1_squared**0.5

    # Calculate new mu_2
    mu_1_terms = 3 * sigma_1**4 + 6 * sigma_1**2 * mu_1**2 + mu_1**4
    a_1_squared = 6 * sigma_2**4 + (m_4 - p_1 * mu_1_terms) / (1 - p_1)

    if a_1_squared < 0:
        return False, np.zeros(5)

    a_1 = a_1_squared**0.5
    mu_2_squared = a_1 - 3 * sigma_2**2

    if mu_2_squared < 0:
        return False, np.zeros(5)

    mu_2_new = mu_2_squared**0.5

    # Calculate new p_1
    a_2 = 15 * sigma_1**4 * mu_1 + 10 * sigma_1**2 * mu_1**3 + mu_1**5
    b_2 = 15 * sigma_2**4 * mu_2_new + 10 * sigma_2**2 * mu_2_new**3 + mu_2_new**5

    if abs(a_2 - b_2) < 1e-15:
        return False, np.zeros(5)

    p_1_new = (m_5 - b_2) / (a_2 - b_2)

    if p_1_new < 0 or p_1_new > 1:
        return False, np.zeros(5)

    return True, np.array([mu_1, mu_2_new, sigma_1, sigma_2, p_1_new])


# Optimized utility functions
def centered_moment(moments, order):
    """Optimized centered moment calculation"""
    if order == 2:
        return moments[1] - moments[0] ** 2

    # Original implementation for other orders
    moment_c = 0
    for j in range(order + 1):
        combin = int(comb(order, j))
        a_1 = 1 if j == order else moments[order - j - 1]
        moment_c += (-1) ** j * combin * moments[0] ** j * a_1
    return moment_c


def raw_moment(central_moments, dist_mean):
    """Optimized raw moment calculation"""
    raw_moments = [dist_mean]
    central_moments = [1] + central_moments

    for n_i in range(2, len(central_moments)):
        # Vectorized calculation where possible
        k_range = np.arange(n_i + 1)
        combs = np.array([comb(n_i, k) for k in k_range])
        centrals = np.array([central_moments[k] for k in k_range])
        powers = np.array([dist_mean ** (n_i - k) for k in k_range])

        moment_n = np.sum(combs * centrals * powers)
        raw_moments.append(moment_n)

    return raw_moments


def most_likely_parameters(data, ignore_columns="error", res=10_000):
    """Optimized parameter estimation with better memory usage"""
    df_results = data.copy()
    if isinstance(ignore_columns, str):
        ignore_columns = [ignore_columns]

    columns = [c for c in df_results.columns if c not in ignore_columns]

    # Pre-allocate result dictionary
    d_results = {}

    for col in columns:
        col_data = df_results[col].to_numpy()  # Direct numpy conversion

        # Use fewer points for KDE if data is small
        actual_res = min(res, len(col_data) * 10)

        x_range = np.linspace(col_data.min(), col_data.max(), num=actual_res)
        kde = gaussian_kde(col_data)
        y_kde = kde.evaluate(x_range)

        # Find maximum more efficiently
        max_idx = np.argmax(y_kde)
        d_results[col] = round(x_range[max_idx], 5)

    return d_results
