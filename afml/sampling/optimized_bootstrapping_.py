import numpy as np
from numba import njit


def precompute_active_indices_contiguous(samples_info_sets, price_bars_index, ensure_sorted=True):
    """
    Returns (sample_ids, all_indices, offsets).
    - sample_ids: np.int64 array of original sample ids in the same order as starts/ends
    - all_indices: concatenated bar positions (0..T-1) for all samples
    - offsets: int64 array length n_samples+1; sample i indices are all_indices[offsets[i]:offsets[i+1]]
    """
    bars = np.asarray(price_bars_index)
    if ensure_sorted:
        sort_idx = np.argsort(bars)
        bars_sorted = bars[sort_idx]
    else:
        sort_idx = None
        bars_sorted = bars

    vals = np.asarray(list(samples_info_sets.values))
    if vals.ndim != 2 or vals.shape[1] != 2:
        n = len(samples_info_sets)
        vals = np.empty((n, 2), dtype=bars_sorted.dtype)
        for i, (t0, t1) in enumerate(samples_info_sets.values):
            vals[i, 0] = t0
            vals[i, 1] = t1

    t0s = vals[:, 0]
    t1s = vals[:, 1]

    starts = np.searchsorted(bars_sorted, t0s, side="left").astype(np.int64)
    ends = np.searchsorted(bars_sorted, t1s, side="right").astype(np.int64)
    lengths = (ends - starts).astype(np.int64)

    total = int(lengths.sum())
    all_indices = np.empty(total, dtype=np.int64)
    offsets = np.empty(len(starts) + 1, dtype=np.int64)
    offsets[0] = 0
    pos = 0
    for i in range(len(starts)):
        L = lengths[i]
        if L > 0:
            rng = np.arange(starts[i], starts[i] + L, dtype=np.int64)
            if sort_idx is None:
                all_indices[pos : pos + L] = rng
            else:
                all_indices[pos : pos + L] = sort_idx[rng]
        pos += L
        offsets[i + 1] = pos

    sample_ids = samples_info_sets.index.to_numpy(dtype=np.int64)
    return sample_ids, all_indices, offsets


@njit(cache=True)
def _seq_bootstrap_probabilities(all_indices, offsets, concurrency):
    n = offsets.shape[0] - 1
    av_uniqueness = np.empty(n, dtype=np.float64)

    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        if start >= end:
            av_uniqueness[i] = 0.0
            continue
        s = 0.0
        L = end - start
        for p in range(start, end):
            bar = all_indices[p]
            c = concurrency[bar]
            s += 1.0 / (c + 1.0)
        av_uniqueness[i] = s / L
    total = av_uniqueness.sum()
    if total > 0.0:
        return av_uniqueness / total
    # fallback uniform
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = 1.0 / n
    return out


def seq_bootstrap_from_contiguous(
    sample_ids, all_indices, offsets, s_length=None, random_seed=None
):
    # prepare RNG
    if random_seed is None:
        rnd = np.random.RandomState()
    elif isinstance(random_seed, np.random.RandomState):
        rnd = random_seed
    else:
        try:
            rnd = np.random.RandomState(int(random_seed))
        except Exception:
            rnd = np.random.RandomState()

    n_samples = sample_ids.shape[0]
    # Determine T (max bar position +1). all_indices contains bar positions 0..T-1
    T = int(all_indices.max()) + 1 if all_indices.size else 0
    concurrency = np.zeros(T, dtype=np.int64)

    s_length = n_samples if s_length is None else int(s_length)
    phi = np.empty(s_length, dtype=sample_ids.dtype)

    for k in range(s_length):
        prob = _seq_bootstrap_probabilities(all_indices, offsets, concurrency)
        # rnd.choice over sample indices; cannot directly pass sample_ids to njit, so do in Python
        idx = rnd.choice(np.arange(n_samples), p=prob)
        chosen = sample_ids[idx]
        phi[k] = chosen
        # increment concurrency over chosen sample's bars
        start = offsets[idx]
        end = offsets[idx + 1]
        for p in range(start, end):
            concurrency[int(all_indices[p])] += 1

    return phi.tolist()
