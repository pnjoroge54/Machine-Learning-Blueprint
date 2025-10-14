"""
Contains the logic regarding the sequential bootstrapping from chapter 4, as well as the concurrent labels.
"""

from .bootstrapping import (
    get_ind_mat_average_uniqueness,
    get_ind_mat_label_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
)
from .concurrent import get_av_uniqueness_from_triple_barrier, num_concurrent_events
from .optimized_bootstrapping import precompute_active_indices, seq_bootstrap_optimized
from .optimized_concurrent import (
    get_av_uniqueness_from_triple_barrier_optimized,
    get_num_conc_events_optimized,
)
from .optimized_sb_bagging import (
    SequentiallyBootstrappedBaggingClassifier,
    SequentiallyBootstrappedBaggingRegressor,
)
