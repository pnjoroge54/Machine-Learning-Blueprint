import numpy as np
from numba import njit


# -------------------------
# Packing helpers
# -------------------------
def pack_active_indices(active_indices):
    """
    Convert dict/list-of-arrays active_indices into flattened arrays and offsets.

    Args:
        active_indices (dict or list): mapping sample_id -> 1D ndarray of bar indices

    Returns:
        flat_indices (ndarray int64): concatenated bar indices for all samples
        offsets (ndarray int64): start index in flat_indices for each sample (len = n+1)
        lengths (ndarray int64): number of indices per sample (len = n)
        sample_ids (list): list of sample ids in the order used to pack data
    """
    # Preserve sample id ordering to allow mapping between chosen index and original id
    if isinstance(active_indices, dict):
        sample_ids = list(active_indices.keys())
        values = [active_indices[sid] for sid in sample_ids]
    else:
        # assume list-like ordered by sample id 0..n-1
        sample_ids = list(range(len(active_indices)))
        values = list(active_indices)

    lengths = np.array([v.size for v in values], dtype=np.int64)
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(lengths)

    total = int(offsets[-1])
    if total == 0:
        flat_indices = np.empty(0, dtype=np.int64)
    else:
        flat_indices = np.empty(total, dtype=np.int64)
        pos = 0
        for v in values:
            l = v.size
            if l:
                flat_indices[pos : pos + l] = v
            pos += l

    return flat_indices, offsets, lengths, sample_ids


# -------------------------
# Numba-accelerated primitives
# -------------------------
@njit
def _compute_scores_flat(flat_indices, offsets, lengths, concurrency, eps):
    """
    Compute an unnormalized score for each sample using flattened indices.

    Score used: score_i = 1 / (1 + mean_concurrency_over_sample_i)

    Args:
        flat_indices (ndarray int64): concatenated indices
        offsets (ndarray int64): start positions (len = n+1)
        lengths (ndarray int64): counts per sample
        concurrency (ndarray int64): current concurrency counts per bar
        eps (float): minimum score floor to avoid zeros

    Returns:
        scores (ndarray float64): unnormalized positive scores per sample
    """
    n = offsets.shape[0] - 1
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = offsets[i]
        e = offsets[i + 1]
        length = lengths[i]

        if length == 0:
            # If a sample covers no bars, give it small positive weight (not zero)
            scores[i] = eps
        else:
            # mean concurrency across bars the sample would cover
            sum_conc = 0.0
            for k in range(s, e):
                bar = flat_indices[k]
                sum_conc += concurrency[bar]
            mean_conc = sum_conc / length
            # inverse penalty: prefer samples with lower average concurrency
            score = 1.0 / (1.0 + mean_conc)
            if score < eps:
                score = eps
            scores[i] = score

    return scores


@njit
def _normalize_to_prob(scores):
    """
    Normalize non-negative scores to a probability vector. If all zero, return uniform.
    """
    n = scores.shape[0]
    total = 0.0
    for i in range(n):
        total += scores[i]

    prob = np.empty(n, dtype=np.float64)
    if total == 0.0:
        # fallback to uniform distribution
        uni = 1.0 / n
        for i in range(n):
            prob[i] = uni
    else:
        for i in range(n):
            prob[i] = scores[i] / total
    return prob


@njit
def _choose_index_from_cdf(prob, u):
    """
    Convert a uniform random number u in [0,1) to an index using the cumulative distribution.

    This avoids calling numpy.choice inside numba and is efficient.
    """
    n = prob.shape[0]
    cum = 0.0
    for i in range(n):
        cum += prob[i]
        if u < cum:
            return i
    # numerical fallback: return last index
    return n - 1


@njit
def _increment_concurrency_flat(flat_indices, offsets, chosen, concurrency):
    """
    Increment concurrency for the bars covered by sample `chosen`.
    """
    s = offsets[chosen]
    e = offsets[chosen + 1]
    for k in range(s, e):
        bar = flat_indices[k]
        concurrency[bar] += 1


# -------------------------
# Fully jitted sampling procedure (no Python loops per-iteration)
# -------------------------
@njit
def _seq_bootstrap_full_jit(flat_indices, offsets, lengths, concurrency, uniforms, eps):
    """
    Njitted sequential bootstrap loop.

    Args:
        flat_indices, offsets, lengths: flattened index layout
        concurrency (ndarray int64): initial concurrency vector (will be mutated)
        uniforms (ndarray float64): pre-drawn uniform random numbers in [0,1), length = sample_length
        eps (float64): small floor for scores

    Returns:
        chosen_indices (ndarray int64): sequence of chosen sample indices (positions in packed order)
    """
    sample_length = uniforms.shape[0]
    n_samples = offsets.shape[0] - 1
    chosen_indices = np.empty(sample_length, dtype=np.int64)

    for it in range(sample_length):
        # compute scores and probabilities given current concurrency
        scores = _compute_scores_flat(flat_indices, offsets, lengths, concurrency, eps)
        prob = _normalize_to_prob(scores)

        # map uniform to a sample index
        u = uniforms[it]
        idx = _choose_index_from_cdf(prob, u)
        chosen_indices[it] = idx

        # update concurrency for selected sample
        _increment_concurrency_flat(flat_indices, offsets, idx, concurrency)

    return chosen_indices


# -------------------------
# Public wrapper: reproducible, packs data, runs njit sampler, maps to original ids
# -------------------------
def seq_bootstrap_optimized_flat(active_indices, sample_length=None, random_seed=None, eps=1e-12):
    """
    End-to-end sequential bootstrap using flattened arrays + Numba.

    Args:
        active_indices (dict or list): mapping sample id -> ndarray of bar indices
        sample_length (int or None): requested number of draws; defaults to number of samples
        random_seed (int, RandomState, or None): seed controlling the pre-drawn uniforms
        eps (float): small floor to avoid degenerate scores

    Returns:
        phi (list): list of chosen original sample ids (length = sample_length)
    """
    # Pack into contiguous arrays and keep mapping from packed index -> original sample id
    flat_indices, offsets, lengths, sample_ids = pack_active_indices(active_indices)
    n_samples = offsets.shape[0] - 1

    if sample_length is None:
        sample_length = n_samples

    # Concurrency vector length: bars are indices into price-bar positions.
    # When there are no bars (flat_indices empty), create an empty concurrency of length 0.
    if flat_indices.size == 0:
        T = 0
    else:
        # max bar index + 1 (bars are zero-based indices)
        T = int(flat_indices.max()) + 1

    concurrency = np.zeros(T, dtype=np.int64)

    # Prepare reproducible uniforms. Accept either integer seed or RandomState.
    if random_seed is None:
        rng = np.random.RandomState()
    elif isinstance(random_seed, np.random.RandomState):
        rng = random_seed
    else:
        try:
            rng = np.random.RandomState(int(random_seed))
        except (ValueError, TypeError):
            rng = np.random.RandomState()

    # Pre-draw uniforms in Python and pass them into njit function (numba cannot accept RandomState)
    uniforms = rng.random_sample(sample_length).astype(np.float64)

    # Run njit loop (this mutates concurrency but we don't need concurrency afterwards)
    chosen_packed = _seq_bootstrap_full_jit(
        flat_indices, offsets, lengths, concurrency, uniforms, eps
    )

    # Map packed indices back to original sample ids
    phi = [sample_ids[int(i)] for i in chosen_packed.tolist()]

    return phi


# -------------------------
# Example usage (for quick manual test)
# -------------------------
if __name__ == "__main__":
    # small synthetic test: 4 samples with overlapping bar ranges
    # sample 0 covers bars [0,1], sample 1 covers [1,2], sample 2 covers [2,3], sample 3 covers []
    active = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([1, 2], dtype=np.int64),
        2: np.array([2, 3], dtype=np.int64),
        3: np.empty(0, dtype=np.int64),
    }

    phi = seq_bootstrap_optimized_flat(active, sample_length=6, random_seed=42)
    print("phi:", phi)
