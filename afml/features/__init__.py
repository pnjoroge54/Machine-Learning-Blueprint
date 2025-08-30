"""
Functions derived from Chapter 5: Fractional Differentiation.
Functions that use price and time to create features.
"""

from .fracdiff import (
    frac_diff,
    frac_diff_ffd,
    fracdiff_optimal,
    get_weights,
    get_weights_ffd,
    plot_min_ffd,
)
from .fractals import (
    calculate_basic_fractals,
    calculate_enhanced_fractals,
    calculate_fractal_levels,
    calculate_fractal_trend_features,
    comprehensive_fractal_analysis,
    generate_fractal_signals,
    get_fractal_features,
)
from .moving_averages import calculate_ma_differences
from .returns import (
    get_lagged_returns,
    get_period_autocorr,
    get_period_returns,
    get_return_dist_features,
    rolling_autocorr_numba,
)
from .time import (
    encode_cyclical_features,
    get_time_features,
    trading_session_encoded_features,
)
from .volatility_regime import identify_structural_breaks, plot_structural_breaks
