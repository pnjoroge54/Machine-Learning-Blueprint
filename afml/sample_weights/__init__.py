"""
Contains the code for implementing sample weights.
"""

from .attribution import get_weights_by_return, get_weights_by_time_decay
from .optimized_attribution import (
    _apply_time_decay_numba,
    _apply_weight_by_return_optimized,
    _compute_return_weights_numba,
    get_weights_by_return_optimized,
    get_weights_by_time_decay_optimized,
)
