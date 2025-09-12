"""
Implements general backtest statistics
"""

from .backtester import backtest_model, backtest_with_labeling
from .backtests import CampbellBacktesting
from .log_analyser import load_model_logs
from .performance_analysis import (
    calculate_performance_metrics,
    get_annualization_factors,
    get_trades,
    run_meta_labeling_analysis,
)
from .primary_model_experiments import (
    DataSetupHook,
    ModelTrainingHook,
    PerformanceLoggingHook,
    PredictionHook,
    get_train_test_split,
    load_my_data,
    train_my_model,
)
from .reporting import (
    compare_pr_curves,
    compare_roc_curves,
    compare_roc_pr_curves,
    create_classification_report_image,
    meta_labelling_classification_report_images,
    meta_labelling_classification_report_tables,
    meta_labelling_reports,
)
from .research_framework import ExperimentHook, ExperimentRunner, ResearchExperiment
from .statistics import (
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
