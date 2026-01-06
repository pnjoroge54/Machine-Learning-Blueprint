"""
Implements general backtest statistics
"""

from .backtests import CampbellBacktesting
from .meta_labeling_analysis import (
    analyze_signal_quality,
    calculate_risk_adjusted_metrics,
    evaluate_meta_labeling_performance,
    get_validation_metrics,
)
from .meta_labeling_analysis_reports import (
    compare_strategies,
    generate_complete_meta_labeling_report,
    generate_summary_report,
    plot_strategy_comparison,
)
from .perfomance_statistics import (
    all_bets_concentration,
    average_holding_period,
    bets_concentration,
    deflated_sharpe_ratio,
    drawdown_and_time_under_water,
    information_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    timing_of_flattening_and_flips,
)
from .performance_analysis import (
    calculate_performance_metrics,
    get_annualization_factors,
    get_positions_from_events,
)

__all__ = [
    "analyze_signal_quality",
    "CampbellBacktesting",
    "calculate_performance_metrics",
    "get_annualization_factors",
    "get_positions_from_events",
    "all_bets_concentration",
    "average_holding_period",
    "bets_concentration",
    "deflated_sharpe_ratio",
    "drawdown_and_time_under_water",
    "information_ratio",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "timing_of_flattening_and_flips",
    "evaluate_meta_labeling_performance",
    "compare_strategies",
    "calculate_risk_adjusted_metrics",
    "analyze_signal_quality",
    "plot_strategy_comparison",
    "generate_complete_meta_labeling_report",
    "generate_summary_report",
    "get_validation_metrics",
]
