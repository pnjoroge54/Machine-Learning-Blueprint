"""
Monte Carlo simulatio illustrating the effectiveness of standard vs. sequential bootstrap.
"""

from os import cpu_count
from pprint import pprint

import numpy as np
import pandas as pd

from ..sampling.bootstrapping import (
    get_active_indices,
    get_ind_mat_average_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
)
from ..util.multiprocess import process_jobs, process_jobs_

NUM_CPU = cpu_count()


def get_rnd_t1(num_obs, num_bars, max_h):
    """
    Create random t1 Series
    """

    t1 = {}
    for _ in range(num_obs):
        ix = np.random.randint(0, num_bars)
        val = ix + np.random.randint(1, max_h)
        t1[ix] = val
    return pd.Series(t1).sort_index()


def aux_mc(num_obs, num_bars, max_h):
    # Parallelized auxiliary function
    t1 = get_rnd_t1(num_obs, num_bars, max_h)
    bar_ix = range(t1.max() + 1)
    ind_mat = get_ind_matrix(t1, bar_ix)
    phi = np.random.choice(ind_mat.columns, size=ind_mat.shape[1])
    std_uniq = get_ind_mat_average_uniqueness(ind_mat[phi]).mean()
    active_indices = get_active_indices(t1, bar_ix)
    phi = seq_bootstrap(active_indices)
    seq_uniq = get_ind_mat_average_uniqueness(ind_mat[phi]).mean()
    return {"std_uniq": std_uniq, "seq_uniq": seq_uniq}


def main_mc(num_obs=10, num_bars=100, max_h=5, num_iters=1e6, num_threads=NUM_CPU):
    # Monte Carlo experiments
    jobs = []
    for _ in range(int(num_iters)):
        job = {"func": aux_mc, "num_obs": num_obs, "num_bars": num_bars, "max_h": max_h}
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    pprint(pd.DataFrame(out).describe())
    return out
