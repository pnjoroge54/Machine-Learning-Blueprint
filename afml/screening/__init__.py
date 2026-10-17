"""
afml.screening
==============

Pre-model diagnostics: does a feature carry information, and is the evidence
that it does distinguishable from what the estimator produces on noise?

Three gates, in this order: stationarity, entropy, mutual information. See
:mod:`afml.screening.pipeline` for why the order is fixed.

Design invariant: no screening function returns a bare float. Statistics are
bound to the null that calibrates them via
:class:`~afml.screening.report.ScreeningResult`.
"""

from .entropy_features import (
    bin_counts,
    entropy_screen,
    indicator_entropy,
    modal_occupancy,
)
from .multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    effective_number_of_tests,
    holm_bonferroni,
    westfall_young_maxt,
)
from .mutual_information import (
    mi_screen,
    miller_madow_bias,
    normalized_mutual_info,
)
from .permutation import (
    SCHEMES,
    bar_permutation,
    build_null,
    circular_shift_indices,
    effective_sample_size,
    naive_shuffle_indices,
    permutation_pvalue,
    stationary_bootstrap_indices,
    suggested_block_length,
)
from .pipeline import ScreeningReport, screen_features
from .report import ScreeningResult, results_to_frame
from .stationarity import (
    VERDICTS,
    assess_stationarity,
    rolling_stationarity,
    stationarity_gate,
)

__all__ = [
    "SCHEMES",
    "VERDICTS",
    "ScreeningReport",
    "ScreeningResult",
    "assess_stationarity",
    "bar_permutation",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bin_counts",
    "bonferroni",
    "build_null",
    "circular_shift_indices",
    "effective_number_of_tests",
    "effective_sample_size",
    "entropy_screen",
    "holm_bonferroni",
    "indicator_entropy",
    "mi_screen",
    "miller_madow_bias",
    "modal_occupancy",
    "naive_shuffle_indices",
    "normalized_mutual_info",
    "permutation_pvalue",
    "results_to_frame",
    "rolling_stationarity",
    "screen_features",
    "stationarity_gate",
    "stationary_bootstrap_indices",
    "suggested_block_length",
    "westfall_young_maxt",
]
