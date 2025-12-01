"""
Functions derived from Chapter 7: Cross Validation
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .combinatorial import CombinatorialPurgedKFold
from .cross_validation import (
    PurgedKFold,
    PurgedSplit,
    analyze_cross_val_scores,
    analyze_cross_val_scores_calibrated,
    ml_cross_val_score,
    ml_get_train_times,
)
from .hyperfit import (
    MyPipeline,
    clf_hyper_fit,
    clf_hyper_fit_auto_cache,
    param_grid_size,
)
from .scoring import probability_weighted_accuracy

scoring_methods = {
    "accuracy": accuracy_score,
    "pwa": probability_weighted_accuracy,
    "neg_log_loss": log_loss,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

__all__ = [
    "ml_get_train_times",
    "ml_cross_val_score",
    "analyze_cross_val_scores",
    "analyze_cross_val_scores_calibrated",
    "PurgedKFold",
    "PurgedSplit",
    "probability_weighted_accuracy",
    "MyPipeline",
    "clf_hyper_fit",
    "CombinatorialPurgedKFold",
    "param_grid_size",
    "scoring_methods",
    "clf_hyper_fit_auto_cache",
]
