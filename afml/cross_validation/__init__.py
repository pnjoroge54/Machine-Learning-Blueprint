"""
Functions derived from Chapter 7: Cross Validation
"""

from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .anchored_walkforward import vin_v_anchored_walkforward
from .combinatorial import (
    CombinatorialPurgedCV,
    CPCVAnalyzer,
    fill_average_active_sides,
    fill_sides_numba,
)
from .cross_validation import (
    PurgedKFold,
    PurgedSplit,
    PurgedWalkForwardCV,
    analyze_cross_val_scores,
    ml_cross_val_score,
    ml_get_train_times,
)
from .hyper_fit import clf_hyper_fit, clf_hyper_fit_cached
from .hyper_fit_analysis import generate_complete_hyperparameter_report
from .nested_cv import (
    DataPartition,
    UnifiedValidationCalibrator,
    inner_cv_search,
    partition_data,
)
from .optuna_hyper_fit import (
    FinancialModelSuggester,
    check_for_overfitting,
    optimize_trading_model,
    print_best_trial,
    save_intermediate_results,
)
from .pbo import compute_pbo
from .process_permutation import (
    ProcessPermutationResult,
    assert_cache_isolation,
    permute_development_process,
    quantile_ci,
    spurious_performance_curve,
)
from .scoring import probability_weighted_accuracy
from .selection_overfit import (
    CriterionVarianceReport,
    EffectiveTrialCount,
    HPOEarlyStopping,
    OffsetPurgedKFold,
    aggregate_fold_params,
    criterion_variance_audit,
    effective_trial_count,
    expected_max_sharpe,
    one_standard_error_selection,
    selection_optimism_curve,
    top_m_ensemble,
)
from .trial_tracker import StrategyTrialTracker

scoring_methods = {
    "accuracy": accuracy_score,
    "pwa": probability_weighted_accuracy,
    "neg_log_loss": log_loss,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

__all__ = [
    "CPCVAnalyzer",
    "CombinatorialPurgedCV",
    "CriterionVarianceReport",
    "DataPartition",
    "EffectiveTrialCount",
    "FinancialModelSuggester",
    "HPOEarlyStopping",
    "OffsetPurgedKFold",
    "ProcessPermutationResult",
    "PurgedKFold",
    "PurgedSplit",
    "PurgedWalkForwardCV",
    "StrategyTrialTracker",
    "UnifiedValidationCalibrator",
    "aggregate_fold_params",
    "analyze_cross_val_scores",
    "assert_cache_isolation",
    "check_for_overfitting",
    "clf_hyper_fit",
    "clf_hyper_fit_cached",
    "compute_pbo",
    "criterion_variance_audit",
    "effective_trial_count",
    "expected_max_sharpe",
    "fill_average_active_sides",
    "fill_sides_numba",
    "generate_complete_hyperparameter_report",
    "inner_cv_search",
    "ml_cross_val_score",
    "ml_get_train_times",
    "one_standard_error_selection",
    "optimize_trading_model",
    "partition_data",
    "permute_development_process",
    "quantile_ci",
    "print_best_trial",
    "probability_weighted_accuracy",
    "save_intermediate_results",
    "scoring_methods",
    "selection_optimism_curve",
    "spurious_performance_curve",
    "top_m_ensemble",
    "vin_v_anchored_walkforward",
]
