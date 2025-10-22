"""
Optimized implementation of logic regarding sequential bootstrapping from chapter 4.
"""

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict


@njit(cache=True)
def precompute_active_indices_nopython(t0_array, t1_array, price_bars_array):
    """
    Nopython implementation: Map each sample to the bars it influences.

    Args:
        t0_array (np.ndarray): Array of start times for each sample (int64 timestamps).
        t1_array (np.ndarray): Array of end times for each sample (int64 timestamps).
        price_bars_array (np.ndarray): Array of bar timestamps (int64).

    Returns:
        Dict[int64, int64[:]]: A numba typed dictionary mapping sample_id to array of bar indices.
    """
    n_samples = len(t0_array)

    # Create a typed dictionary for numba
    active_indices = Dict.empty(key_type=types.int64, value_type=types.int64[:])

    for sample_id in range(n_samples):
        t0 = t0_array[sample_id]
        t1 = t1_array[sample_id]

        # Find indices where price_bars are within [t0, t1]
        mask = (price_bars_array >= t0) & (price_bars_array <= t1)
        indices = np.where(mask)[0]

        active_indices[sample_id] = indices

    return active_indices


def precompute_active_indices(samples_info_sets, price_bars_index):
    """
    Wrapper function to convert pandas objects to numpy arrays for nopython implementation.

    Args:
        samples_info_sets (pd.Series): Triple barrier events(t1) from labeling.get_events.
            Index: start times (t0), Values: end times (t1)
        price_bars_index (pd.DatetimeIndex or array-like): Bar indices/timestamps.

    Returns:
        dict: Standard Python dictionary mapping sample_id to numpy arrays of indices.
    """
    # Convert to numpy arrays with int64 timestamps
    t0_array = samples_info_sets.index.values.astype("int64")
    t1_array = samples_info_sets.values.astype("int64")

    # Ensure price_bars_index is int64
    if hasattr(price_bars_index, "values"):
        price_bars_array = price_bars_index.values.astype("int64")
    else:
        price_bars_array = np.asarray(price_bars_index, dtype="int64")

    # Call nopython implementation
    typed_dict = precompute_active_indices_nopython(t0_array, t1_array, price_bars_array)

    # Convert numba typed dict to regular Python dict for compatibility
    result = {int(k): v for k, v in typed_dict.items()}

    return result


def seq_bootstrap_optimized(active_indices, s_length=None, random_seed=None):
    """
    Generate sample indices using sequential bootstrap.

    Args:
        active_indices (dict): Dictionary mapping sample identifiers to arrays of bar indices.
        s_length (int): Desired number of samples to generate.
        random_seed (int, RandomState, or None): Seed for random number generation.

    Returns:
        list: A list of generated sample indices.
    """
    # Handle different types of random_seed input
    if random_seed is None:
        random_state = np.random.RandomState()
    elif isinstance(random_seed, np.random.RandomState):
        random_state = random_seed
    else:
        # Convert to integer and create RandomState
        try:
            random_state = np.random.RandomState(int(random_seed))
        except (ValueError, TypeError):
            random_state = np.random.RandomState()

    sample_ids = np.array(list(active_indices.keys()))
    phi = []

    # Determine the maximum bar index
    active_indices_values = list(active_indices.values())
    T = max(max(indices) for indices in active_indices_values) + 1 if active_indices else 0
    concurrency = np.zeros(T, dtype=int)

    s_length = len(active_indices) if s_length is None else s_length

    # Sequential bootstrap sampling loop
    for _ in range(s_length):
        prob = _seq_bootstrap_optimized_loop(active_indices_values, concurrency)
        chosen = random_state.choice(sample_ids, p=prob)  # Use random_state instead of np.random
        phi.append(chosen)
        concurrency[active_indices[chosen]] += 1

    return phi


@njit(cache=True)
def _seq_bootstrap_optimized_loop(active_indices_values, concurrency):
    N = len(active_indices_values)
    av_uniqueness = np.zeros(N)  # Array to store average uniqueness of each sample.

    for i in range(N):
        indices = active_indices_values[i]  # Get influenced bar indices for the sample.
        c = concurrency[indices]  # Retrieve concurrency values for these indices.
        uniqueness = 1 / (c + 1)  # Calculate uniqueness as the inverse of concurrency.
        av_uniqueness[i] = (
            np.mean(uniqueness) if len(uniqueness) > 0 else 0.0
        )  # Compute average uniqueness.
    total = av_uniqueness.sum()  # Sum of uniqueness values across all samples.
    prob = (
        av_uniqueness / total if total > 0 else np.ones(N) / N
    )  # Compute probabilities for sampling.

    return prob
