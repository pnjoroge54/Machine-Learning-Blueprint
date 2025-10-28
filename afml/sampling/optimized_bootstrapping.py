"""
Optimized implementation of logic regarding sequential bootstrapping from chapter 4.
"""

import numpy as np
from numba import njit


def get_active_indices(samples_info_sets, price_bars_index):
    """
    Build an indicator mapping from each sample to the bar indices it influences.

    Args:
        samples_info_sets (pd.Series):
            Triple-barrier events (t1) returned by labeling.get_events.
            Index: start times (t0) as pd.DatetimeIndex.
            Values: end times (t1) as pd.Timestamp (or NaT for open events).
        price_bars_index (pd.DatetimeIndex or array-like):
            Sorted bar timestamps (pd.DatetimeIndex or array-like). Will be converted to
            np.int64 timestamps for internal processing.

    Returns:
        dict:
            Standard Python dictionary mapping sample_id (int) to a numpy.ndarray of
            bar indices (dtype=int64). Example: {0: array([0,1,2], dtype=int64), 1: array([], dtype=int64), ...}
    """
    t0 = samples_info_sets.index
    t1 = samples_info_sets.values
    n = len(samples_info_sets)
    active_indices = {}

    # precompute searchsorted positions to restrict scanning range
    starts = np.searchsorted(price_bars_index, t0, side="left")
    ends = np.searchsorted(price_bars_index, t1, side="right")  # exclusive

    for sample_id in range(n):
        s = starts[sample_id]
        e = ends[sample_id]
        if e > s:
            active_indices[sample_id] = np.arange(s, e, dtype=int)
        else:
            active_indices[sample_id] = np.empty(0, dtype=int)

    return active_indices


def seq_bootstrap_optimized(active_indices, sample_length=None, random_seed=None):
    """
    Generate sample indices using sequential bootstrap.

    Args:
        active_indices (dict): Dictionary mapping sample identifiers to arrays of bar indices.
        sample_length (int): Desired number of samples to generate.
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

    phi = []
    sample_ids = list(active_indices.keys())

    # Determine the maximum bar index
    active_indices_values = list(active_indices.values())
    T = max(indices.max() for indices in active_indices_values) + 1 if active_indices else 0
    concurrency = np.zeros(T, dtype=int)

    sample_length = len(active_indices) if sample_length is None else sample_length

    # Sequential bootstrap sampling loop
    for _ in range(sample_length):
        prob = _seq_bootstrap_optimized_loop(active_indices_values, concurrency)
        chosen = random_state.choice(sample_ids, p=prob)  # Use random_state instead of np.random
        phi.append(chosen)
        concurrency[active_indices[chosen]] += 1

    return phi


@njit(cache=True)
def _seq_bootstrap_optimized_loop(active_indices_values, concurrency):
    N = len(active_indices_values)
    av_uniqueness = np.empty(N)

    for i in range(N):
        indices = active_indices_values[i]
        if len(indices) > 0:
            c = concurrency[indices]
            uniqueness = 1.0 / (c + 1.0)
            av_uniqueness[i] = np.mean(uniqueness)
        else:
            av_uniqueness[i] = 0.0

    total = np.sum(av_uniqueness)
    if total > 0:
        prob = av_uniqueness / total
    else:
        prob = np.ones(N) / N
    return prob
