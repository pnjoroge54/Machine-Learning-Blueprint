"""
Functions derived from Chapter 7: Cross Validation
"""

from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)

from .combinatorial import CombinatorialPurgedCV, CPCVAnalyzer, fill_sides_numba, fill_average_active_sides
from .cross_validation import (PurgedKFold, PurgedSplit, PurgedWalkForwardCV,
                               analyze_cross_val_scores, ml_cross_val_score,
                               ml_get_train_times)
from .hyper_fit import clf_hyper_fit, clf_hyper_fit_cached
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
    "fill_sides_numba",
    "fill_average_active_sides",
    "ml_get_train_times",
    "ml_cross_val_score",
    "analyze_cross_val_scores",
    "PurgedKFold",
    "PurgedSplit",
    "PurgedWalkForwardCV",
    "probability_weighted_accuracy",
    "clf_hyper_fit",
    "clf_hyper_fit_cached",
    "CombinatorialPurgedCV",
    "CPCVAnalyzer",
    "scoring_methods",
]
