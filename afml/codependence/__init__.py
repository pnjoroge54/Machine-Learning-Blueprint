"""
Varoius codependence measure: mutual info, distance correlations, variation of information
"""

from .codependence_matrix import get_dependence_matrix, get_distance_matrix
from .correlation import (
    absolute_angular_distance,
    angular_distance,
    distance_correlation,
    squared_angular_distance,
)
from .gnpr_distance import gnpr_distance, gpr_distance, spearmans_rho
from .information import (
    get_mutual_info,
    get_optimal_number_of_bins,
    variation_of_information_score,
)
