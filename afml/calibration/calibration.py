"""
Probability Calibration Toolkit for Financial Machine Learning

This module provides comprehensive tools for calibrating classifier probabilities
and evaluating calibration quality, with special considerations for financial
time series data.

Key Features:
- Calibration metrics (Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE))
- Reliability diagrams with confidence intervals
- Multiple calibration methods (Platt scaling, isotonic regression)
- Integration with purged cross-validation
- Bootstrap confidence intervals for calibration curves

The module is designed to work seamlessly with the combinatorial purged
cross-validation framework from Chapter 12 of Advances in Financial Machine Learning.

Examples:
    >>> # Plot reliability diagram
    >>> plot_reliability_with_ci(y, oof_probs, n_bins=10, title="Pre-Calibration")

Reference:
    López de Prado, M. (2018) Advances in Financial Machine Learning, Chapter 7.
    Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.

Probability calibration utilities for financial time series.

Provides CalibratorCV (OOF isotonic regression with PurgedKFold)
and a comprehensive cross-validation analysis that directly compares
raw vs calibrated performance on PWA and Brier score.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array, check_is_fitted

from ..cross_validation.cross_validation import PurgedKFold
from ..cross_validation.scoring import probability_weighted_accuracy
from ..ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier


def _find_seq_bagging(estimator):
    """
    Locate a SequentiallyBootstrappedBaggingClassifier inside ``estimator``.

    The classifier may be passed directly or nested as a step inside a
    (My)Pipeline.  Returns the SB classifier instance if found, otherwise
    ``None``.  Used by CalibratorCV and analyze_calibrated_cross_val_scores
    to re-inject the per-fold ``samples_info_sets`` so the sequential
    bootstrap sampler is actually exercised during cross-validated refits
    (a plain ``clone`` discards the fitted sampler state).
    """
    if isinstance(estimator, SequentiallyBootstrappedBaggingClassifier):
        return estimator
    if hasattr(estimator, "steps"):  # sklearn Pipeline / MyPipeline
        for _, step in estimator.steps:
            if isinstance(step, SequentiallyBootstrappedBaggingClassifier):
                return step
    return None


def _resolve_n_jobs(n_jobs):
    """Map an ``n_jobs`` value to the joblib convention, treating ``None`` as 1.

    Returning 1 for ``None`` means the fold loop runs serially by default, so
    enabling Lever-A fold parallelism is strictly opt-in and never changes the
    numeric result of an existing call — only its wall-clock time.
    """
    return 1 if n_jobs is None else n_jobs


def _force_inner_single_threaded(estimator):
    """Pin every nested ``n_jobs`` parameter on ``estimator`` to 1.

    When the cross-validation fold loop is parallelised (Lever A), the inner
    estimator must not also spawn its own worker pool — nested parallelism
    oversubscribes the CPU and is typically slower than either level alone.
    This walks the (deep) sklearn parameter tree and sets any ``n_jobs`` /
    ``*__n_jobs`` entry to 1 so the outer ``Parallel`` owns all the cores.
    Estimators that expose no ``n_jobs`` are left untouched.
    """
    try:
        params = estimator.get_params(deep=True)
    except (AttributeError, TypeError):
        return
    to_set = {k: 1 for k in params if k == "n_jobs" or k.endswith("__n_jobs")}
    if to_set:
        estimator.set_params(**to_set)


def _fit_oof_fold(estimator, X_train, y_train, sw_train, X_test, seq_t1_train, force_single):
    """Fit one OOF fold and return the positive-class probabilities for the test rows.

    Extracted from ``CalibratorCV.fit`` Phase 1 so the fold loop can be driven
    by ``joblib.Parallel``. ``estimator`` is cloned here (each call is fully
    independent), the per-fold sequential-bootstrap sampler state is injected
    when present, and the inner estimator is pinned to a single thread when the
    fold loop itself is running in parallel.
    """
    fold_clf = clone(estimator)
    if force_single:
        _force_inner_single_threaded(fold_clf)
    if seq_t1_train is not None:
        fold_seq = _find_seq_bagging(fold_clf)
        fold_seq.samples_info_sets = seq_t1_train
        fold_seq.active_indices_ = None  # force recompute for this fold
    try:
        fold_clf.fit(X_train, y_train, sample_weight=sw_train)
    except TypeError:
        fold_clf.fit(X_train, y_train)
    return fold_clf.predict_proba(X_test)[:, 1]


# ---- Calibration Methods ----


def fit_platt_scaling(
    y_calib: np.ndarray,
    scores_calib: np.ndarray,
    sample_weight: np.ndarray | None = None,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Fit Platt scaling calibration mapping.

    Platt scaling fits a logistic regression to map raw scores to calibrated probabilities.

    Args:
        y_calib: Calibration set true labels
        scores_calib: Calibration set predicted scores or probabilities
        C: Inverse of regularization strength
        solver: Optimization algorithm
        max_iter: Maximum number of iterations

    Returns:
        Fitted LogisticRegression calibrator
    """
    X_calib = np.asarray(scores_calib).reshape(-1, 1)
    platt = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    platt.fit(X_calib, y_calib, sample_weight=sample_weight)
    return platt


class CalibratorCV(BaseEstimator, ClassifierMixin):
    """
    Probability calibrator using cross-validated out-of-fold (OOF) predictions
    generated by PurgedKFold, with a choice of calibration method.

    Standard sklearn calibration methods (``CalibratedClassifierCV``) use either
    a random held-out split or stratified k-fold to produce the predictions on
    which the calibration map is fitted. Both approaches violate temporal ordering
    and the embargo requirement that prevents label information from leaking across
    the purge boundary. This class corrects that by generating OOF predictions
    through ``PurgedKFold``, which purges overlapping labels and applies a
    configurable embargo before each test fold.

    The fitting procedure has three phases:

    1. **OOF generation** — the base estimator is cloned and refitted on each
       fold's training set. Predictions on the corresponding held-out test set
       are collected into a full-length OOF probability array.
    2. **Calibration map fitting** — either ``IsotonicRegression`` (non-parametric,
       requires ≥ ~200 OOF observations) or Platt scaling (logistic regression,
       two parameters, suitable for small calibration sets) is fitted on the OOF
       probabilities and true labels, weighted by ``sample_weight`` when provided.
    3. **Full-data refit** — the base estimator is cloned again and refitted on
       the complete training dataset so that ``predict_proba()`` at inference time
       uses a model with maximum historical coverage while applying the calibration
       map learned from the OOF predictions.

    Parameters
    ----------
    estimator : estimator object
        An unfitted sklearn-compatible classifier that implements ``predict_proba``.
        The estimator is cloned at each fold to prevent state leakage between folds.
        A ``Pipeline`` ending in a classifier is fully supported.

    cv : PurgedKFold instance, required
        Cross-validation strategy. A pre-configured ``PurgedKFold`` instance must
        be supplied explicitly. This requirement is intentional: there is no
        financially sensible default because ``PurgedKFold`` requires a ``t1``
        Series (label end times) specific to each dataset. Omitting ``cv`` raises
        ``NotImplementedError`` at ``fit()`` time with an explanatory message.
        Using ``StratifiedKFold`` or any non-purging splitter will silently leak
        label information across fold boundaries and produce optimistically biased
        calibration.

    method : {'isotonic', 'platt'}, default='isotonic'
        Calibration method to apply in Phase 2.

        ``'isotonic'`` — fits an ``IsotonicRegression`` with
        ``out_of_bounds='clip'``. Non-parametric; can represent any monotone
        non-decreasing map from raw probability to calibrated probability. Requires
        approximately 200 or more valid OOF observations for the step function to
        be well-estimated. This is the correct default for most financial ML
        classifiers, where the miscalibration structure is not assumed to be
        logistic.

        ``'platt'`` — fits a one-dimensional logistic regression (Platt scaling)
        mapping raw probabilities to calibrated ones. Only two parameters; stable
        on small calibration sets. Prefer this when the effective OOF sample size
        after purging and embargo is below ~200 observations, or when the base
        estimator is known to produce roughly monotone but linearly scaled
        probabilities (e.g. SVM scores).

    n_jobs : int or None, default=None
        Number of parallel jobs for the Phase-1 OOF fold loop (Lever A).
        ``None`` (or ``1``) runs the folds serially, exactly reproducing the
        previous behaviour. Any other value (e.g. ``-1`` for all cores) fits the
        folds concurrently via ``joblib.Parallel``. When fold-level parallelism
        is active, the inner estimator is automatically pinned to a single
        thread (its own ``n_jobs`` is forced to 1) to prevent CPU
        oversubscription from nested parallelism. The numeric result is
        identical regardless of ``n_jobs``; only wall-clock time changes.

    Attributes
    ----------
    estimator_ : estimator object
        The base estimator refitted on the full training dataset (Phase 3).
        Used by ``predict_proba`` and ``predict`` at inference time.

    calibrator_ : IsotonicRegression or LogisticRegression
        The fitted calibration map (Phase 2). For ``method='isotonic'`` this is
        an ``IsotonicRegression`` instance; for ``method='platt'`` it is a
        ``LogisticRegression`` instance. Either can be called directly on an
        externally generated raw probability array:

        >>> raw = clf.estimator_.predict_proba(X_new)[:, 1]
        >>> calibrated = clf.calibrator_.predict(raw)          # isotonic
        >>> calibrated = clf.calibrator_.predict_proba(        # platt
        ...     raw.reshape(-1, 1))[:, 1]

    oof_probs_ : ndarray of shape (n_samples,)
        Raw (uncalibrated) OOF probability predictions from Phase 1. Entries
        corresponding to observations that were never used as a test sample
        (e.g. observations purged or embargoed in every fold) are ``np.nan``.
        Use this attribute to compute pre-calibration diagnostics:

        >>> valid = ~np.isnan(clf.oof_probs_)
        >>> raw_brier = np.mean((clf.oof_probs_[valid] - y.values[valid]) ** 2)

    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``. For binary classification this is
        ``array([0, 1])`` or ``array([-1, 1])`` depending on the label space.

    n_features_in_ : int
        Number of features seen during ``fit``. Conforms to the sklearn estimator
        API convention used by ``check_is_fitted`` and pipeline compatibility checks.

    Examples
    --------
    Isotonic calibration (default):

    >>> from afml.calibration.calibration import CalibratorCV
    >>> from afml.cross_validation.cross_validation import PurgedKFold
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> cv = PurgedKFold(n_splits=5, t1=events['t1'], pct_embargo=0.01)
    >>> clf = CalibratorCV(
    ...     estimator=RandomForestClassifier(n_estimators=100),
    ...     cv=cv,
    ...     method='isotonic',
    ... )
    >>> clf.fit(X_train, y_train, sample_weight=events['tW'].to_numpy())
    CalibratorCV(...)
    >>> probs = clf.predict_proba(X_new)[:, 1]

    Platt scaling for small calibration sets:

    >>> clf_platt = CalibratorCV(
    ...     estimator=RandomForestClassifier(n_estimators=100),
    ...     cv=cv,
    ...     method='platt',
    ... )
    >>> clf_platt.fit(X_train, y_train, sample_weight=events['tW'].to_numpy())

    OOF diagnostics (pre-calibration):

    >>> valid = ~np.isnan(clf.oof_probs_)
    >>> raw_brier = np.mean((clf.oof_probs_[valid] - y_train.values[valid]) ** 2)

    Notes
    -----
    **Scoring convention** — the default ``score()`` method returns
    **Probability-Weighted Accuracy (PWA)**, not plain accuracy. PWA rewards
    confident correct predictions and penalizes confident incorrect ones, making
    it the appropriate default for a calibrator whose primary purpose is producing
    well-formed probabilities. Use ``brier_score()`` for the classical calibration
    quality metric (mean squared probability error, lower is better).

    **Method selection** — for most financial ML classifiers (random forest,
    gradient boosting), isotonic regression is the correct choice. These models
    produce probability outputs that are monotone but not logistically shaped, so
    Platt scaling's linearity assumption is frequently violated. Use Platt scaling
    when either (a) OOF sample size is small (fewer than ~200 observations after
    purging and embargo), or (b) the model is an SVM or similar that outputs
    decision scores rather than structured probabilities.

    **Sample weights** are passed to both the base estimator's ``fit()`` call
    (if it accepts ``sample_weight``) and to the calibration map's ``fit()``.
    AFML-style uniqueness weights (``events['tW']``) are appropriate for both,
    ensuring the calibration map is fitted with the same information weighting as
    the model itself.

    **ONNX export** — ``CalibratorCV`` has no ONNX operator mapping. When
    exporting to ONNX for deployment in MetaTrader 5, use ``estimator_`` (the
    inner sklearn model) as the ONNX source and apply the calibration map
    (``calibrator_.predict()`` for isotonic, ``calibrator_.predict_proba()`` for
    Platt) as a separate post-processing step on the ONNX model's raw output.

    **Re-estimation** — the calibration map reflects the miscalibration structure
    of the model on the training data distribution. After a significant regime
    shift or after retraining the base estimator on a substantially different
    dataset, call ``fit()`` again to update the map.

    References
    ----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. Chapter 10 (bet sizing) and Chapter 4 (sample weights).

    Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
    accurate multiclass probability estimates. *KDD '02*.

    Platt, J. (1999). Probabilistic outputs for support vector machines and
    comparisons to regularized likelihood methods. *Advances in Large Margin
    Classifiers*, 10(3), 61–74.

    See Also
    --------
    fit_platt_scaling : Two-parameter logistic calibration for small samples.
    calibration_report : Tabular summary of Brier score, ECE, and MCE
        before and after calibration.
    plot_reliability_with_ci : Reliability diagram with bootstrap confidence
        intervals.
    """

    _VALID_METHODS = frozenset({"isotonic", "platt"})

    def __init__(self, estimator, cv=None, method="isotonic", n_jobs=None):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """
        Fit the calibrator on training data.

        Executes the three-phase fitting procedure:
        1. Generate OOF probability predictions via PurgedKFold.
        2. Fit the calibration map (isotonic or Platt) on OOF probabilities,
           true labels, and sample weights.
        3. Refit the base estimator on the complete dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Validated via ``check_array`` before use.

        y : array-like of shape (n_samples,)
            Target labels. For binary classification, labels must be
            ``{0, 1}`` or ``{-1, 1}``.

        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. AFML-style uniqueness (``tW``) or time-decay
            weights are appropriate here. When provided, weights are passed
            to the base estimator's ``fit()`` (if it accepts ``sample_weight``)
            and to the calibration map's ``fit()``.
            If ``None``, all observations are weighted equally.

        Returns
        -------
        self : CalibratorCV
            Fitted calibrator.

        Raises
        ------
        NotImplementedError
            If ``cv`` was not supplied at construction time.

        ValueError
            If ``method`` is not ``'isotonic'`` or ``'platt'``.

        ValueError
            If fewer than 10 valid OOF predictions are produced after
            purging and embargo.
        """
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(self._VALID_METHODS)}, got '{self.method}'."
            )

        X_, y_ = X, y
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = check_array(X, ensure_min_samples=2)
        y = check_array(y, ensure_min_samples=2, ensure_2d=False, dtype="int").ravel()

        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        if self.cv is None:
            raise NotImplementedError(
                "Please explicitly pass a PurgedKFold (or compatible) instance to cv. "
                "There is no financially sensible default because PurgedKFold requires "
                "a t1 Series (label end times) specific to each dataset."
            )
        self.cv_ = self.cv

        # ── Sequential-bootstrap awareness ─────────────────────────────────
        # If the base estimator is (or wraps) a SequentiallyBootstrappedBagging
        # classifier, a plain clone() drops the fitted sampler state and the
        # refit silently falls back to standard (uniform) bootstrapping.  To
        # exercise the real sequential sampler on every refit we re-inject the
        # appropriately row-sliced samples_info_sets before each fit.  The
        # series is taken from the estimator itself (it is a constructor param
        # preserved through clone) and must be aligned to the rows of X.
        seq_step = _find_seq_bagging(self.estimator)
        if seq_step is not None:
            seq_t1 = getattr(seq_step, "samples_info_sets", None)
            if seq_t1 is None:
                raise ValueError(
                    "The estimator contains a SequentiallyBootstrappedBaggingClassifier "
                    "but its samples_info_sets is not set.  Set samples_info_sets (the "
                    "t1 label-span Series, aligned to the rows of X) on the classifier "
                    "before fitting CalibratorCV so the sequential bootstrap can be "
                    "reproduced per fold."
                )
            if len(seq_t1) != n_samples:
                raise ValueError(
                    f"samples_info_sets length ({len(seq_t1)}) does not match the number "
                    f"of training rows ({n_samples}); it must be aligned to X."
                )
        else:
            seq_t1 = None

        # ── Phase 1: OOF predictions (Lever A: optional fold parallelism) ──
        oof_probs = np.full(n_samples, np.nan)

        n_jobs = _resolve_n_jobs(self.n_jobs)
        force_single = n_jobs != 1  # pin inner estimator only when folds run in parallel
        splits = list(self.cv_.split(X_, y_))

        fold_probs = Parallel(n_jobs=n_jobs)(
            delayed(_fit_oof_fold)(
                self.estimator,
                X[train_idx],
                y[train_idx],
                sample_weight[train_idx],
                X[test_idx],
                seq_t1.iloc[train_idx] if seq_t1 is not None else None,
                force_single,
            )
            for train_idx, test_idx in splits
        )

        for (_, test_idx), probs in zip(splits, fold_probs):
            oof_probs[test_idx] = probs

        # ── Validate coverage ─────────────────────────────────────────────
        valid_mask = ~np.isnan(oof_probs)
        if valid_mask.sum() < 10:
            raise ValueError(
                "Too few valid OOF predictions after purging and embargo "
                f"({valid_mask.sum()} observations). "
                "Consider reducing pct_embargo or increasing n_samples."
            )

        # ── Phase 2: Fit calibration map ──────────────────────────────────
        if self.method == "isotonic":
            # self.calibrator_ = MyPipeline(["iso", IsotonicRegression(out_of_bounds="clip", increasing=True)])
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", increasing=True)
            self.calibrator_.fit(
                oof_probs[valid_mask],
                y[valid_mask],
                sample_weight=sample_weight[valid_mask],
            )
        else:  # 'platt'
            self.calibrator_ = fit_platt_scaling(
                y_calib=y[valid_mask],
                scores_calib=oof_probs[valid_mask],
                sample_weight=sample_weight[valid_mask],
            )

        # ── Phase 3: Refit base estimator on full data ────────────────────
        self.estimator_ = clone(self.estimator)
        if seq_t1 is not None:
            full_seq = _find_seq_bagging(self.estimator_)
            full_seq.samples_info_sets = seq_t1
            full_seq.active_indices_ = None
        try:
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.estimator_.fit(X, y)

        self.oof_probs_ = oof_probs
        return self

    def predict_proba(self, X):
        """
        Return calibrated class probability estimates.

        For ``method='isotonic'``, applies:
        ``X → estimator_.predict_proba → calibrator_.predict(raw) → [1-p, p]``

        For ``method='platt'``, applies:
        ``X → estimator_.predict_proba → calibrator_.predict_proba(raw) → [1-p, p]``

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Calibrated probability estimates. Column 0 is the probability of
            the negative class; column 1 is the probability of the positive
            class. Values are clipped to [0, 1].

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        check_is_fitted(self, ["calibrator_", "estimator_"])
        X = check_array(X)
        raw_probs = self.estimator_.predict_proba(X)[:, 1]

        if self.method == "isotonic":
            calibrated = np.clip(self.calibrator_.predict(raw_probs), 0.0, 1.0)
        else:  # 'platt'
            calibrated = self.calibrator_.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        """
        Predict class labels.

        Assigns label 1 when the calibrated positive-class probability is
        at least 0.5, and label 0 otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (integer dtype).

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y, sample_weight=None):
        """
        Return the Probability-Weighted Accuracy (PWA) — higher is better.

        PWA rewards confident correct predictions and penalizes confident
        incorrect ones, making it the appropriate default score for a calibrator
        whose purpose is producing well-formed probabilities. Plain accuracy
        discards the probability estimates entirely; PWA evaluates them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        y : array-like of shape (n_samples,)
            True class labels.

        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Return-attribution weights (``events['w']``)
            are appropriate here, following the dual-weight convention from
            the AFML pipeline.

        Returns
        -------
        pwa : float
            Probability-Weighted Accuracy in [0, 1].

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.

        See Also
        --------
        brier_score : Classical calibration quality metric (lower is better).
        """
        check_is_fitted(self, ["calibrator_", "estimator_"])
        proba = self.predict_proba(X)
        return probability_weighted_accuracy(
            y_true=y,
            y_pred=proba,
            sample_weight=sample_weight,
            labels=self.classes_,
        )

    def brier_score(self, X, y, sample_weight=None):
        """
        Return the Brier score — lower is better.

        The Brier score is the mean squared error between predicted probabilities
        and true binary outcomes. It is the canonical metric for evaluating
        probability calibration quality: a perfectly calibrated model achieves a
        Brier score equal to the base rate variance of the label distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        y : array-like of shape (n_samples,)
            True class labels (binary, ``{0, 1}`` or ``{-1, 1}``).

        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Return-attribution weights (``events['w']``)
            are appropriate here, consistent with the dual-weight convention
            from the AFML pipeline.

        Returns
        -------
        brier : float
            Brier score in [0, 1]. Lower values indicate better calibration.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.

        See Also
        --------
        score : Probability-Weighted Accuracy (higher is better).
        calibration_report : Tabular comparison of pre- and post-calibration
            Brier score, ECE, and MCE.
        """
        check_is_fitted(self, ["calibrator_", "estimator_"])
        proba = self.predict_proba(X)[:, 1]
        y = check_array(y, ensure_2d=False, dtype="int").ravel()
        return brier_score_loss(y, proba, sample_weight=sample_weight)


def _analyze_calibrated_fold(
    base_estimator,
    X,
    y,
    train,
    test,
    sample_weight_train,
    sample_weight_score,
    seq_bootstrap,
    full_t1,
    cal_n_splits,
    cal_pct_embargo,
    force_single,
    method="isotonic",
):
    """Run one outer CV fold of ``analyze_calibrated_cross_val_scores``.

    Fits both the raw ``base_estimator`` and a ``CalibratorCV`` on the train
    fold, scores them on the test fold, and returns the per-fold metrics and
    confusion matrices. Extracted from the main loop so the folds can be driven
    by ``joblib.Parallel`` (Lever A). When the fold loop runs in parallel,
    ``force_single`` pins the inner estimators to a single thread to avoid
    nested-parallel CPU oversubscription.

    Returns
    -------
    raw_scores : dict mapping each score key to its float value for this fold.
    cal_scores : dict mapping each score key to its calibrated value.
    cm_raw : np.ndarray confusion matrix for the raw estimator.
    cm_cal : np.ndarray confusion matrix for the calibrated estimator.
    """
    raw_scores = {}
    cal_scores = {}

    # ====================== RAW ESTIMATOR ======================
    raw_clf = clone(base_estimator)
    if force_single:
        _force_inner_single_threaded(raw_clf)
    if seq_bootstrap:
        raw_seq = _find_seq_bagging(raw_clf)
        raw_seq.samples_info_sets = full_t1.iloc[train]
        raw_seq.active_indices_ = None
        raw_seq.oob_score = False

    raw_fit = raw_clf.fit(
        X=X.iloc[train, :].to_numpy(),
        y=y.iloc[train].to_numpy(),
        sample_weight=sample_weight_train.iloc[train].to_numpy(),
    )

    prob_raw = raw_fit.predict_proba(X.iloc[test, :].to_numpy())
    pred_raw = (prob_raw[:, 1] > 0.5).astype(int)

    params_raw = dict(
        y_true=y.iloc[test],
        y_pred=pred_raw,
        labels=raw_clf.classes_,
        sample_weight=sample_weight_score.iloc[test].to_numpy(),
    )

    # Raw metrics (same logic as analyze_cross_val_scores)
    for k, scoring in zip(
        ["accuracy", "pwa", "neg_log_loss", "brier", "precision", "recall", "f1"],
        [
            accuracy_score,
            probability_weighted_accuracy,
            log_loss,
            brier_score_loss,
            precision_score,
            recall_score,
            f1_score,
        ],
    ):
        if scoring in (probability_weighted_accuracy, log_loss):
            params_raw["y_pred"] = prob_raw
            score = scoring(**params_raw)
            if k == "neg_log_loss":
                score *= -1
        elif scoring == brier_score_loss:
            # brier_score_loss takes the positive-class probability as its
            # second positional arg (named y_prob/y_proba across sklearn
            # versions); pass positionally to stay version-agnostic.
            score = brier_score_loss(
                params_raw["y_true"],
                prob_raw[:, 1],
                sample_weight=params_raw["sample_weight"],
            )
        else:
            params_raw["y_pred"] = pred_raw
            try:
                score = scoring(**params_raw)
            except Exception:
                del params_raw["labels"]
                score = scoring(**params_raw)
                params_raw["labels"] = raw_clf.classes_
        raw_scores[k] = score

    cm_raw = confusion_matrix(**params_raw).round(2)

    # ====================== CALIBRATED (CalibratorCV) ======================
    # The inner CalibratorCV runs its own PurgedKFold over the *train* fold,
    # so it needs a splitter whose t1 is aligned to that subset (the passed
    # calibrator_cv is aligned to the full dataset and would mis-index).  We
    # also pass the train fold as a DataFrame so the inner PurgedKFold's
    # index check succeeds, and slice the sequential sampler's info sets to
    # the same subset.
    X_train_df = X.iloc[train, :]
    fold_t1 = full_t1.iloc[train]
    fold_cal_cv = PurgedKFold(n_splits=cal_n_splits, t1=fold_t1, pct_embargo=cal_pct_embargo)

    inner_est = clone(base_estimator)
    if force_single:
        _force_inner_single_threaded(inner_est)
    if seq_bootstrap:
        inner_seq = _find_seq_bagging(inner_est)
        inner_seq.samples_info_sets = fold_t1
        inner_seq.active_indices_ = None
        inner_seq.oob_score = False

    calibrator = CalibratorCV(estimator=inner_est, cv=fold_cal_cv, method=method)
    calibrator.fit(
        X=X_train_df,
        y=y.iloc[train].to_numpy(),
        sample_weight=sample_weight_train.iloc[train].to_numpy(),
    )

    prob_cal = calibrator.predict_proba(X.iloc[test, :].to_numpy())
    pred_cal = calibrator.predict(X.iloc[test, :].to_numpy())

    # Use the calibrator's built-in methods for PWA and Brier
    cal_scores["pwa"] = calibrator.score(
        X=X.iloc[test, :].to_numpy(),
        y=y.iloc[test].to_numpy(),
        sample_weight=sample_weight_score.iloc[test].to_numpy(),
    )
    cal_scores["brier"] = calibrator.brier_score(
        X=X.iloc[test, :].to_numpy(),
        y=y.iloc[test].to_numpy(),
        sample_weight=sample_weight_score.iloc[test].to_numpy(),
    )

    # Other metrics for calibrated
    params_cal = dict(
        y_true=y.iloc[test],
        y_pred=pred_cal,
        labels=calibrator.classes_,
        sample_weight=sample_weight_score.iloc[test].to_numpy(),
    )

    for k, scoring in zip(
        ["accuracy", "neg_log_loss", "precision", "recall", "f1"],
        [accuracy_score, log_loss, precision_score, recall_score, f1_score],
    ):
        if scoring == log_loss:
            params_cal["y_pred"] = prob_cal
            score = scoring(**params_cal)
            score *= -1
        else:
            params_cal["y_pred"] = pred_cal
            try:
                score = scoring(**params_cal)
            except Exception:
                del params_cal["labels"]
                score = scoring(**params_cal)
                params_cal["labels"] = calibrator.classes_
        cal_scores[k] = score

    cm_cal = confusion_matrix(**params_cal).round(2)

    return raw_scores, cal_scores, cm_raw, cm_cal


def analyze_calibrated_cross_val_scores(
    base_estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: pd.Series | None = None,
    sample_weight_score: pd.Series | None = None,
    calibrator_cv: BaseCrossValidator | None = None,
    n_jobs: int | None = None,
    method: str = "isotonic",
):
    """
    Comprehensive cross-validation analysis that compares:
      - the raw (unadjusted) base_estimator
      - the calibrated CalibratorCV

    Returns PWA and Brier score for BOTH, plus the usual metrics from
    analyze_cross_val_scores. Uses the exact same style / return format.

    Parameters
    ----------
    base_estimator : unfitted sklearn-compatible classifier
    calibrator_cv : PurgedKFold (or compatible). If None, it is auto-created
                    from cv_gen (recommended to use the same PurgedKFold).
    method : {"isotonic", "platt"}, default="isotonic"
        Calibration map used by the per-fold CalibratorCV. Should match the
        method of the deployed calibrator so this diagnostic reflects what is
        actually shipped.
    n_jobs : int or None, default=None
        Number of parallel jobs for the outer cross-validation fold loop
        (Lever A). ``None`` (or ``1``) evaluates folds serially, reproducing the
        previous behaviour exactly. Any other value (e.g. ``-1`` for all cores)
        runs the folds concurrently via ``joblib.Parallel``; the inner raw and
        calibrated estimators are then pinned to a single thread to avoid
        nested-parallel CPU oversubscription. Results are identical regardless
        of ``n_jobs`` — only wall-clock time changes.

    Returns
    -------
    ret_scores : dict of np.ndarray (per-fold scores, keys like "raw_pwa", "cal_pwa", ...)
    scores_df : pd.DataFrame with mean / std
    confusion_matrix_breakdown : list of dicts (raw + calibrated per fold)
    """
    if calibrator_cv is None:
        if hasattr(cv_gen, "t1") and isinstance(cv_gen.t1, pd.Series):
            t1_series = cv_gen.t1
            pct = getattr(cv_gen, "pct_embargo", 0.01)
            n_splits_cal = getattr(cv_gen, "n_splits", 5)
            calibrator_cv = PurgedKFold(n_splits=n_splits_cal, t1=t1_series, pct_embargo=pct)
        else:
            raise ValueError(
                "calibrator_cv must be provided (PurgedKFold with same t1/pct_embargo as cv_gen)"
            )

    # Score keys (raw + calibrated)
    score_keys = ["accuracy", "pwa", "neg_log_loss", "brier", "precision", "recall", "f1"]
    ret_scores = {f"raw_{k}": np.zeros(cv_gen.n_splits) for k in score_keys}
    ret_scores.update({f"cal_{k}": np.zeros(cv_gen.n_splits) for k in score_keys})

    cms_raw = []
    cms_cal = []

    # Default weights
    if sample_weight_train is None:
        sample_weight_train = pd.Series(np.ones((X.shape[0],)), index=y.index)
    if sample_weight_score is None:
        sample_weight_score = pd.Series(np.ones((X.shape[0],)), index=y.index)

    # Sequential bootstrap handling.  Detect a SequentiallyBootstrappedBagging
    # classifier whether it is passed directly or nested inside a Pipeline, and
    # reconstruct the full label-span Series (aligned to the rows of X) from the
    # outer splitter so it can be sliced per fold.
    seq_step = _find_seq_bagging(base_estimator)
    seq_bootstrap = seq_step is not None
    # Reconstruct the full label-span Series, preserving pandas dtype/timezone so
    # PurgedKFold's index-equality check against X.index holds.  PurgedKFold
    # stores t1 as a pandas array (t1) and its index separately (t1_index).
    full_t1 = pd.Series(cv_gen.t1, index=pd.Index(cv_gen.t1_index, copy=False))
    cal_n_splits = getattr(calibrator_cv, "n_splits", cv_gen.n_splits)
    cal_pct_embargo = getattr(calibrator_cv, "pct_embargo", 0.01)

    # ── Lever A: optionally evaluate the outer CV folds in parallel ────────
    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    force_single = resolved_n_jobs != 1  # pin inner estimators only when folds run in parallel
    splits = list(cv_gen.split(X=X, y=y))

    fold_results = Parallel(n_jobs=resolved_n_jobs)(
        delayed(_analyze_calibrated_fold)(
            base_estimator,
            X,
            y,
            train,
            test,
            sample_weight_train,
            sample_weight_score,
            seq_bootstrap,
            full_t1,
            cal_n_splits,
            cal_pct_embargo,
            force_single,
            method,
        )
        for train, test in splits
    )

    for i, (raw_scores, cal_scores, cm_raw, cm_cal) in enumerate(fold_results):
        for k, v in raw_scores.items():
            ret_scores[f"raw_{k}"][i] = v
        for k, v in cal_scores.items():
            ret_scores[f"cal_{k}"][i] = v
        cms_raw.append(cm_raw)
        cms_cal.append(cm_cal)

    # Mean / std DataFrame (same format as analyze_cross_val_scores)
    scores_df = pd.DataFrame.from_dict(
        {
            scoring: {"mean": scores.mean(), "std": scores.std()}
            for scoring, scores in ret_scores.items()
        },
        orient="index",
    )

    # Confusion matrix breakdown (both raw and calibrated)
    confusion_matrix_breakdown = []
    for i, cm in enumerate(cms_raw, 1):
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "raw", "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            )
        else:
            confusion_matrix_breakdown.append({"fold": i, "type": "raw", "confusion_matrix": cm})

    for i, cm in enumerate(cms_cal, 1):
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "calibrated", "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            )
        else:
            confusion_matrix_breakdown.append(
                {"fold": i, "type": "calibrated", "confusion_matrix": cm}
            )

    return ret_scores, scores_df, confusion_matrix_breakdown


# ---- Core Calibration Metrics ----


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """
    Compute Brier score for probability predictions.

    The Brier score is the mean squared error between predicted probabilities
    and actual outcomes. Lower scores indicate better calibrated probabilities.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class

    Returns:
        Brier score (lower is better)

    Example:
        >>> y_true = np.array([0, 1, 0, 1])
        >>> p_pred = np.array([0.1, 0.9, 0.2, 0.8])
        >>> brier_score(y_true, p_pred)
        0.025
    """
    y_true = np.asarray(y_true).ravel()
    p_pred = np.asarray(p_pred).ravel()
    return float(brier_score_loss(y_true, p_pred))


def expected_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted probabilities
    and observed frequencies across probability bins.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        Expected Calibration Error (lower is better)

    Note:
        For financial applications, quantile bins are often preferred as they
        handle imbalanced probability distributions better.

    Example:
        >>> y_true = np.random.randint(0, 2, 1000)
        >>> p_pred = np.clip(y_true + np.random.normal(0, 0.2, 1000), 0, 1)
        >>> ece = expected_calibration_error(y_true, p_pred, n_bins=10)
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if y_true.shape != p_pred.shape:
        raise ValueError("y_true and p_pred must have the same shape")

    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bins[1:-1], right=True)
    ece_val = 0.0
    total_samples = len(p_pred)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = np.sum(mask)

        if bin_size == 0:
            continue

        # Average predicted probability in bin (confidence)
        mean_confidence = np.mean(p_pred[mask])
        # Observed frequency of positive class (accuracy)
        mean_accuracy = np.mean(y_true[mask])

        ece_val += (bin_size / total_samples) * abs(mean_accuracy - mean_confidence)

    return float(ece_val)


def maximum_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the worst-case absolute difference between predicted probabilities
    and observed frequencies across all probability bins.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        Maximum Calibration Error (lower is better)

    Example:
        >>> mce = maximum_calibration_error(y_true, p_pred, n_bins=10)
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if y_true.shape != p_pred.shape:
        raise ValueError("y_true and p_pred must have the same shape")

    if strategy == "quantile":
        bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bins[1:-1], right=True)
    max_error = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx

        if not np.any(mask):
            continue

        mean_confidence = np.mean(p_pred[mask])
        mean_accuracy = np.mean(y_true[mask])
        bin_error = abs(mean_accuracy - mean_confidence)
        max_error = max(max_error, bin_error)

    return float(max_error)


# ---- Reliability Analysis and Visualization ----


def compute_reliability(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """
    Compute reliability curve data for calibration assessment.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins to use for probability discretization
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        DataFrame with columns:
        - bin: Bin index
        - count: Number of samples in bin
        - pred_mean: Mean predicted probability in bin
        - true_frac: Observed frequency of positive class in bin
        - bin_center: Center point of the bin
        - bin_lower: Lower bound of bin
        - bin_upper: Upper bound of bin

    Example:
        >>> reliability_df = compute_reliability(y_true, p_pred, n_bins=10)
        >>> print(reliability_df[['bin_center', 'pred_mean', 'true_frac']])
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if strategy == "quantile":
        bin_edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
        bin_edges[0], bin_edges[-1] = 0.0, 1.0  # Ensure bounds
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(p_pred, bin_edges[1:-1], right=True)
    results = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask)
        bin_lower = bin_edges[bin_idx]
        bin_upper = bin_edges[bin_idx + 1]
        bin_center = (bin_lower + bin_upper) / 2

        if bin_count == 0:
            results.append(
                {
                    "bin": bin_idx,
                    "count": 0,
                    "pred_mean": np.nan,
                    "true_frac": np.nan,
                    "bin_center": bin_center,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                }
            )
        else:
            results.append(
                {
                    "bin": bin_idx,
                    "count": bin_count,
                    "pred_mean": float(np.mean(p_pred[mask])),
                    "true_frac": float(np.mean(y_true[mask])),
                    "bin_center": bin_center,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                }
            )

    return pd.DataFrame(results)


def plot_reliability(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: plt.Axes | None = None,
    show_perfect: bool = True,
    draw_hist: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Axes:
    """
    Plot reliability diagram (calibration curve).

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        strategy: "uniform" or "quantile" binning
        ax: Matplotlib axes to plot on (creates new if None)
        show_perfect: Whether to show perfect calibration line
        draw_hist: Whether to draw probability distribution histogram
        title: Plot title
        figsize: Figure size when creating new axes

    Returns:
        Matplotlib axes object

    Example:
        >>> ax = plot_reliability(y_true, p_pred, title="Model Calibration")
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    valid_mask = df["count"] > 0

    # Plot calibration curve
    ax.plot(
        df.loc[valid_mask, "pred_mean"],
        df.loc[valid_mask, "true_frac"],
        marker="o",
        markersize=6,
        linewidth=2,
        color="blue",
        label="Calibration Curve",
    )

    if show_perfect:
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect Calibration")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title, fontsize=14)

    # Add probability distribution histogram
    if draw_hist:
        hist_ax = ax.inset_axes([0.15, -0.25, 0.7, 0.15])
        hist_ax.hist(
            p_pred,
            bins=20,
            range=(0, 1),
            color="lightblue",
            edgecolor="black",
            alpha=0.7,
        )
        hist_ax.set_xlabel("Predicted Probability Distribution", fontsize=10)
        hist_ax.set_ylabel("Count", fontsize=10)
        hist_ax.set_xlim(0, 1)

    return ax


# ---- Bootstrap Confidence Intervals ----


def bootstrap_reliability_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    strategy: str = "uniform",
    random_state: int | None = None,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for reliability curve.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        n_bootstraps: Number of bootstrap samples
        strategy: "uniform" or "quantile" binning
        random_state: Random seed for reproducibility
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (base_dataframe, lower_bounds, upper_bounds)

    Example:
        >>> df, lower, upper = bootstrap_reliability_ci(y_true, p_pred)
    """
    rng = np.random.default_rng(seed=random_state)
    n_samples = len(y_true)
    base_df = compute_reliability(y_true, p_pred, n_bins=n_bins, strategy=strategy)

    # Bootstrap samples of true fractions
    boot_true_fracs = np.full((n_bootstraps, n_bins), np.nan)

    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true[indices]
        p_boot = p_pred[indices]

        df_boot = compute_reliability(y_boot, p_boot, n_bins=n_bins, strategy=strategy)
        boot_true_fracs[i, :] = df_boot["true_frac"].values

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower = np.nanpercentile(boot_true_fracs, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(boot_true_fracs, 100 * (1 - alpha / 2), axis=0)

    return base_df, lower, upper


def plot_reliability_with_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    strategy: str = "uniform",
    ax: plt.Axes | None = None,
    random_state: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Axes:
    """
    Plot reliability diagram with bootstrap confidence intervals.

    Args:
        y_true: Array of true binary labels (0 or 1)
        p_pred: Array of predicted probabilities for positive class
        n_bins: Number of bins for discretization
        n_bootstraps: Number of bootstrap samples
        strategy: "uniform" or "quantile" binning
        ax: Matplotlib axes to plot on
        random_state: Random seed for reproducibility
        title: Plot title
        figsize: Figure size when creating new axes

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    base_df, lower, upper = bootstrap_reliability_ci(
        y_true,
        p_pred,
        n_bins=n_bins,
        n_bootstraps=n_bootstraps,
        strategy=strategy,
        random_state=random_state,
    )

    valid_mask = base_df["count"] > 0

    # Plot calibration curve with confidence intervals
    ax.plot(
        base_df.loc[valid_mask, "pred_mean"],
        base_df.loc[valid_mask, "true_frac"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="Calibration Curve",
    )

    ax.fill_between(
        base_df.loc[valid_mask, "pred_mean"],
        lower[valid_mask],
        upper[valid_mask],
        color="blue",
        alpha=0.2,
        label="95% CI",
    )

    ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect Calibration")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title, fontsize=14)

    return ax


# ---- Comprehensive Calibration Report ----


def calibration_report(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    p_calibrated: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generate comprehensive calibration assessment report.

    Args:
        y_true: True labels
        p_pred: Original predicted probabilities
        p_calibrated: Calibrated probabilities (optional)
        n_bins: Number of bins for analysis

    Returns:
        DataFrame with calibration metrics. Includes Brier/ECE/MCE, the Brier
        Skill Score (``*_bss``) relative to a climatology baseline, and ROC-AUC
        (``*_auc``) for both the original and calibrated probabilities.
    """
    metrics = {}

    y_true = np.asarray(y_true)

    # Climatology baseline for the Brier Skill Score: a no-skill model that
    # always predicts the observed event rate. Its Brier score is p_bar*(1-p_bar).
    # BSS = 1 - brier/brier_ref has a meaningful zero (0 = no better than the
    # base rate), so unlike raw Brier it is safe to report as a percentage.
    p_bar = float(np.mean(y_true))
    brier_ref = p_bar * (1.0 - p_bar)

    # ROC-AUC measures ranking/discrimination. Monotonic calibrators (Platt,
    # isotonic) preserve ranking, so a near-zero raw-vs-calibrated AUC change is
    # the signal that any accuracy/precision/recall shift is a 0.5-threshold
    # artifact, not a change in discrimination. Undefined for a single class.
    def _safe_auc(p):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, p))

    # Original probabilities
    metrics["original_brier"] = brier_score(y_true, p_pred)
    metrics["original_ece"] = expected_calibration_error(y_true, p_pred, n_bins)
    metrics["original_mce"] = maximum_calibration_error(y_true, p_pred, n_bins)
    metrics["original_bss"] = (
        1.0 - metrics["original_brier"] / brier_ref if brier_ref > 0 else float("nan")
    )
    metrics["original_auc"] = _safe_auc(p_pred)

    # Calibrated probabilities (if provided)
    if p_calibrated is not None:
        metrics["calibrated_brier"] = brier_score(y_true, p_calibrated)
        metrics["calibrated_ece"] = expected_calibration_error(y_true, p_calibrated, n_bins)
        metrics["calibrated_mce"] = maximum_calibration_error(y_true, p_calibrated, n_bins)
        metrics["calibrated_bss"] = (
            1.0 - metrics["calibrated_brier"] / brier_ref if brier_ref > 0 else float("nan")
        )
        metrics["calibrated_auc"] = _safe_auc(p_calibrated)
        metrics["brier_improvement"] = metrics["original_brier"] - metrics["calibrated_brier"]
        metrics["ece_improvement"] = metrics["original_ece"] - metrics["calibrated_ece"]

    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


# ===================================================================
# Public API definition
# ===================================================================
__all__ = [
    "CalibratorCV",
    "analyze_calibrated_cross_val_scores",
    "brier_score",
    "calibration_report",
    "compute_reliability",
    "expected_calibration_error",
    "fit_platt_scaling",
    "maximum_calibration_error",
    "plot_reliability",
    "plot_reliability_with_ci",
]
