"""
Contains the logic regarding the sequential bootstrapping from chapter 4, as well as the concurrent labels.
"""

from .bootstrapping import (
    get_ind_mat_average_uniqueness,
    get_ind_mat_label_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
)
from .concurrent import (
    _get_average_uniqueness,
    get_av_uniqueness_from_triple_barrier,
    num_concurrent_events,
)
from .optimized_concurrent import (
    _compute_concurrent_events_numba,
    _compute_uniqueness_numba,
    _get_average_uniqueness_optimized,
    get_av_uniqueness_from_triple_barrier_optimized,
    get_num_conc_events_optimized,
)
from .sb_bagging import SequentiallyBootstrappedBaggingClassifier, SequentiallyBootstrappedBaggingRegressor
