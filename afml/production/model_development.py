"""
model_development.py
--------------------
Production model development pipeline with optional Optuna HPO integration.

This module serves as the central orchestrator for the Machine Learning Blueprint,
integrating high-performance data processing, advanced labeling techniques,
and rigorous cross-validation protocols designed for non-IID financial data.

The pipeline now supports two training paths controlled by
`model_params['use_optuna']`:

    False (default): clf_hyper_fit_cached via GridSearchCV / RandomizedSearchCV.
    True:            optimize_trading_model via Optuna + HyperbandPruner.

When use_optuna=True the following changes apply:
  - Weight computation for HPO is handled internally by _WeightedEstimator;
    get_optimal_sample_weight is still run for meta-features and reporting.
  - train_model dispatches to _train_model_optuna.
  - self.study is populated with the completed Optuna study for visualization.

When calibrate=True (run() parameter):
  - CalibratorCV is fitted after train_model(), wrapping best_model.
  - The calibrator uses PurgedKFold with the same n_splits and pct_embargo
    as the HPO step to prevent temporal leakage.
  - self.calibrator_ is populated with the fitted CalibratorCV.
  - At inference time, best_model.predict_proba() returns calibrated
    probabilities directly — no additional step required.
  - ONNX export unwraps the calibrator and exports the inner estimator only.
    Apply calibrator_.calibrator_.predict() as a post-processing step on
    the ONNX model's raw probabilities in the deployed environment.

Architecture Overview:
This pipeline represents a production-grade implementation of the "Advances in
Financial Machine Learning" (AFML) framework. It orchestrates the complex
interaction between labeling, sample weighting, and cross-validation while
maintaining rigorous data integrity through time-aware caching.

Key AFML Methodologies Implemented:
----------------------------------
1. Triple-Barrier Method (TBM):
   Moves beyond fixed-horizon labeling by utilizing dynamic profit-taking,
   stop-loss, and time-exhaustion barriers. This captures the path-dependency
   essential for realistic trading strategy modeling.

2. Sample Weighting & Time Decay:
   Addresses the issue of overlapping outcomes in financial time series.
   The pipeline searches for optimal weights using:
     - Uniqueness (tW): Weights inverse to the concurrency of labels.
     - Return (w): Weights based on the absolute magnitude of the price move.
     - Time Decay: Both linear and exponential decay to prioritize recent data.

3. Purged & Embargoed Cross-Validation:
   Prevents information leakage by removing training observations that overlap
   with the test set (purging) and adding a buffer following the test set
   (embargo) to account for serial correlation.

4. Meta-Feature Engineering:
   The pipeline calculates rolling performance metrics (Accuracy, Precision,
   Recall, F1) using Numba-accelerated functions. These "self-referential"
   metrics are fed back into the model, allowing it to adapt to changing
   market regimes and its own recent performance.

Pipeline Workflow:
-----------------
1. Data Loading: Fetches tick data and constructs specialized bars.
2. Feature Engineering: Generates primary indicators and time-based features.
3. Label Generation: Applies the Triple-Barrier Method to define 'bin' targets.
4. Weight Optimization: Evaluates multiple weighting schemes to find the best
   fit for the current market environment.
5. Meta-Feature Integration: Joins rolling performance metrics to the feature set.
6. Training/HPO: Executes either Scikit-learn or Optuna-based hyperparameter
   optimization with Purged-KFold validation.
7. Calibration (optional): Wraps best_model in CalibratorCV to correct
   systematic overconfidence before position sizing.
8. Reporting: Generates HTML summaries and hyperparameter importance reports.

Meta-Labeling Flow:
------------------
Primary models (is_primary=True or auto-detected) can hand off to a secondary
pipeline via prepare_meta_labeling_inputs(), which returns the events DataFrame
annotated with the primary model's predicted side column. Pass the result
directly as the `events` argument when constructing the secondary pipeline.

When strategy=None the pipeline uses every bar as a potential entry point with
no directional side signal, producing a symmetric triple-barrier label space
(bin ∈ {-1, 0, 1}). This is the correct default for a purely ML-driven pipeline
with no pre-defined entry logic.
"""

import inspect
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from loguru import logger
from numba import njit, prange
from pathlib import Path
from pprint import pformat
from scipy.stats import uniform
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ..cache import cacheable, get_cache_monitor, log_data_access, print_contamination_report
from ..cross_validation.cross_validation import PurgedKFold, ml_cross_val_score
from ..data_structures.bars import calculate_ticks_per_period, make_bars
from ..ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from ..features.trading_session import get_time_features
from ..labeling.triple_barrier import add_vertical_barrier, get_event_weights, triple_barrier_labels
from ..mt5.tick_data_loader import tick_data_loader as loader
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized
from ..strategies.signal_processing import get_entries
from ..strategies.trading_strategies import BaseStrategy
from ..util.misc import date_conversion, value_counts_data
from ..util.pipelines import make_custom_pipeline, set_pipeline_params, MyPipeline


# ============================================================================
# Cached data helpers
# ============================================================================

@cacheable(time_aware=True)
def get_bar_size(tick_df, bar_size):
    """
    Compute tick-based bar size.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with bid/ask prices.
    bar_size : str
        Bar size specification (e.g., 'M1', 'M5').

    Returns
    -------
    int
        Number of ticks per period.
    """
    return calculate_ticks_per_period(tick_df, bar_size)


@cacheable(time_aware=True)
def load_and_prepare_training_data(
    symbol, start_date, end_date, account_name, bar_type, bar_size, price, path=None
):
    """
    Load tick data and construct bars for training.

    Parameters
    ----------
    symbol : str
        Trading instrument symbol.
    start_date : str
        Training start date ('YYYY-MM-DD').
    end_date : str
        Training end date ('YYYY-MM-DD').
    account_name : str
        MT5 account identifier.
    bar_type : str
        Type of bar ('tick', 'volume', 'time').
    bar_size : int or str
        Bar size. If 'tick' and str, converted via `get_bar_size`.
    price : str
        Price type ('bid', 'ask', 'mid_price', 'bid_ask').

    Returns
    -------
    pd.DataFrame
        Constructed bars indexed by timestamp.

    Notes
    -----
    - Logs data access for contamination tracking.
    - Cached for reproducibility.
    - Prevents data leakage via time-aware caching.
    """
    if path is not None:
        loader.path = path

    tick_df = loader.get_tick_data(symbol, start_date, end_date, account_name)

    if bar_type == "tick" and isinstance(bar_size, str):
        bar_size = get_bar_size(tick_df, bar_size)

    data = make_bars(tick_df, bar_type, bar_size, price)
    log_data_access(
        dataset_name=f"{symbol}_{bar_type}_{bar_size}_{price}".lower(),
        start_date=data.index[0],
        end_date=data.index[-1],
        purpose="train",
        data_shape=data.shape,
    )
    return data


@cacheable(time_aware=True)
def create_feature_engineering_pipeline(
    data: pd.DataFrame, feature_config: Dict, data_config: Dict
) -> pd.DataFrame:
    """
    Compute engineered features with caching.

    Parameters
    ----------
    data : pd.DataFrame
        Input bar data.
    feature_config : dict
        Feature configuration.
        Expected keys:
        - func : callable
            Function that computes features from a DataFrame.
        - params : dict
            Parameters passed to `func`.
    data_config:
        Data configuration.
        Expected keys:
        - bar_size : str
            Bar size using MT5 naming conventions, e.g., M1, H1, D1.
        - bar_type : str
            Bar type should be one of "time", "tick", "volume", "dollar"

    Returns
    -------
    pd.DataFrame
        Feature matrix.

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    func = feature_config["func"]
    features = func(data, **feature_config["params"])
    time_feat = get_time_features(
        data, timeframe=data_config["bar_size"], bar_type=data_config["bar_type"]
    )
    return features.join(time_feat, how="left")


@cacheable(time_aware=True)
def generate_events_triple_barrier(
    data: pd.DataFrame,
    strategy: Optional[BaseStrategy],
    target_config: dict,
    profit_target: float = 1,
    stop_loss: float = 1,
    max_holding_period: Dict[str, int] = dict(days=1),
    min_ret: float = 0.0,
    vertical_barrier_zero: bool = True,
    filter_as_series: bool = True,
) -> pd.DataFrame:
    """
    Generate trading events using the triple-barrier method.

    Parameters
    ----------
    data : pd.DataFrame
        Price bars with 'close' column.
    strategy : BaseStrategy or None
        Strategy instance implementing `generate_signals()`. When None, every
        bar in `data` is treated as a potential entry point and no directional
        side signal is applied, producing a symmetric label space
        (bin ∈ {-1, 0, 1}).
    target_config : dict
        Volatility target configuration.
        - func: Volatility target function
        - params: Function parameters
    profit_target : float, default=1
        Profit-taking threshold multiplier.
    stop_loss : float, default=1
        Stop-loss threshold multiplier.
    max_holding_period : dict, default={'days': 1}
        Maximum holding period for vertical barrier.
    min_ret : float, default=0.0
        Minimum return threshold.
    vertical_barrier_zero : bool, default=True
        Set label to zero if vertical barrier is reached.
    filter_as_series : bool, default=True
        Pass volatility threshold as series instead of scalar.

    Returns
    -------
    pd.DataFrame
        Event labels with columns:
        - 'bin' : {-1, 0, 1} for primary; {0, 1} for secondary (meta-label)
        - 't1'  : vertical barrier timestamps
        - 'w'   : sample weights
        - 'tW'  : uniqueness weights
        - 'side': directional signal (only present for secondary models)

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    data_dict = dict(
        open=data["open"], high=data["high"], low=data["low"],
        close=data["close"], df=data, data=data, prices=data,
    )
    close = data["close"]
    target_func   = target_config["func"]
    target_params = target_config["params"].copy()

    sig = inspect.signature(target_func)
    for key in sig.parameters.keys():
        if key not in target_params:
            target_params[key] = data_dict.get(key)

    try:
        target = target_func(**target_params)
    except Exception as e:
        print(e)

    # ── Entry generation ──────────────────────────────────────────────────────
    if strategy is None:
        # No pre-defined signal: use every bar as a candidate entry.
        # side=None → symmetric triple-barrier (bin ∈ {-1, 0, 1}).
        side     = None
        t_events = close.index
    else:
        if filter_as_series is None:
            filter_threshold = None    
        else:
            filter_threshold = target if filter_as_series else target.mean()           
        side, t_events = get_entries(strategy, data, filter_threshold)

    vb = add_vertical_barrier(t_events, close, **max_holding_period)
    events = triple_barrier_labels(
        close, target, t_events,
        vertical_barrier_times=vb,
        side_prediction=side,
        pt_sl=[profit_target, stop_loss],
        min_ret=min_ret,
        min_pct=0.05,
        vertical_barrier_zero=vertical_barrier_zero,
        drop=True,
        verbose=False,
    )
    return get_event_weights(events, close)


# ============================================================================
# Pipeline Components
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _rolling_metrics_numba(y_true, y_pred, weights, window):
    """
    Numba-accelerated rolling metrics calculation.
    Optimized for high-frequency financial data performance.
    """
    n = len(y_true)
    accuracy  = np.full(n, np.nan)
    precision = np.full(n, np.nan)
    recall    = np.full(n, np.nan)
    f1        = np.full(n, np.nan)

    for i in prange(window - 1, n):
        start = i - window + 1
        tp = fp = tn = fn = 0.0
        for j in range(start, i + 1):
            if y_true[j] == 1 and y_pred[j] == 1:
                tp += weights[j]
            elif y_true[j] == 0 and y_pred[j] == 1:
                fp += weights[j]
            elif y_true[j] == 0 and y_pred[j] == 0:
                tn += weights[j]
            elif y_true[j] == 1 and y_pred[j] == 0:
                fn += weights[j]

        total = tp + fp + tn + fn
        if total > 0:
            accuracy[i] = (tp + tn) / total
        denom_prec = tp + fp
        if denom_prec > 0:
            precision[i] = tp / denom_prec
        denom_rec = tp + fn
        if denom_rec > 0:
            recall[i] = tp / denom_rec
        if not np.isnan(precision[i]) and not np.isnan(recall[i]):
            denom_f1 = precision[i] + recall[i]
            if denom_f1 > 0:
                f1[i] = 2 * (precision[i] * recall[i]) / denom_f1

    return accuracy, precision, recall, f1


@cacheable(time_aware=True)
def calculate_rolling_metrics(events, sample_weight, window_sizes=[20, 50]):
    """
    Generates self-referential 'Meta-Features' for the model.

    By observing its own recent performance metrics as input features, the
    secondary model can learn to size down or avoid trades during periods
    where the primary model's recent accuracy or F1-score is declining.
    """
    y_true  = events["bin"].to_numpy(np.int8)
    y_pred  = np.ones(len(y_true), dtype=np.int8)
    weights = sample_weight.to_numpy(np.float32)
    metrics = pd.DataFrame(index=events.index)

    for window in window_sizes:
        if window > len(y_true):
            continue
        accuracy, precision, recall, f1 = _rolling_metrics_numba(
            y_true, y_pred, weights, window
        )
        metrics[f"rolling_accuracy_{window}"]  = accuracy
        metrics[f"rolling_precision_{window}"] = precision
        metrics[f"rolling_recall_{window}"]    = recall
        metrics[f"rolling_f1_{window}"]        = f1

    return metrics.dropna()


@cacheable(time_aware=True)
def best_weighting_scheme(
    classifier,
    X,
    y,
    cv_gen,
    scoring,
    sample_weight_train,
    sample_weight_score,
    scheme=None,
    best_score=0,
    best_scheme=None,
    cv_results=pd.DataFrame(),
):
    scores = ml_cross_val_score(
        classifier,
        X,
        y,
        cv_gen,
        sample_weight_train=sample_weight_train,
        sample_weight_score=sample_weight_score,
        scoring=scoring,
    )
    score = scores.mean()
    cv_results[scheme] = scores

    if not np.isinf(score) and score > best_score:
        best_score = score
        best_scheme = scheme

    return best_score, best_scheme, cv_results


def get_optimal_sample_weight(
    data_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    features: pd.DataFrame,
    n_splits: int = 5,
    linear: bool = None,
    decay_factors: Union[list, np.ndarray] = np.arange(0.1, 1, 0.1),
) -> pd.Series:
    """
    Search-based optimization for sample weighting schemes.

    Financial Rationale:
    Financial observations are rarely IID. This function conducts a systematic
    search to find the weighting scheme that yields the highest cross-validated
    performance, effectively 'de-noising' the dataset.

    Evaluated Schemes:
    1. Uniqueness (tW): Weights samples based on how little they overlap with
       other concurrent labels.
    2. Return (w): Weights samples by the absolute log-return of the outcome.
    3. Time-Decay: Applies linear or exponential decay to prioritize recent
       market structure over distant history.

    Parameters
    ----------
    data_index: pd.DatetimeIndex
        Price data index.
    events : pd.DataFrame
        Event labels with uniqueness weights.
    features: pd.DataFrame
        Training features.
    n_splits : int, optional
        Number of cross-validation splits (default: 5).
    linear : bool, optional
        Default is None, which searches both linear and exponential time-decay.
        If True, use linear time-decay, if False, exponential.
    decay_factors: Union[list, np.ndarray]
        Time-decay factors to apply to best sample weight.

    Returns
    -------
    weights : pd.Series
        Computed sample weights.
    cv_results : dict
        Cross-validation results.
    """
    valid_index = features.dropna().index.intersection(events.index)
    cont = events.loc[valid_index]
    X    = features.loc[valid_index]
    y    = cont["bin"]

    classifier = RandomForestClassifier(
        criterion="entropy",
        class_weight="balanced_subsample",
        n_estimators=100,
        max_depth=4,
        min_weight_fraction_leaf=0.05,
        max_samples=cont["tW"].mean().round(2),
        random_state=42,
    )

    cv_gen = PurgedKFold(n_splits=n_splits, t1=cont["t1"], pct_embargo=0.02)

    weights = {
        "return":     cont["w"],
        "unweighted": pd.Series(1.0, index=cont.index),
        "uniqueness": cont["tW"],
    }

    best_score, best_scheme = 0, None
    cv_results = pd.DataFrame()
    scoring    = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"
    
    pbar1 = tqdm(weights.items(), desc="Analyzing weighting schemes", total=len(weights), mininterval=0.5)
    for i, (scheme, weight) in enumerate(pbar1, 1):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight, weights["return"],
            scheme, best_score, best_scheme, cv_results,
        )
        pbar1.set_postfix({"scheme": scheme, "scoring": scoring, "score": f"{cv_results[scheme].mean():.4f}"})
        if i == len(weights):
            pbar1.set_postfix({
                "best": best_scheme,
                "score": f"{best_score:.4f}"
            })
            
    best_weight  = weights[best_scheme]
    linear_search = [1, 0] if linear is None else ([1] if linear else [0])

    time_decay_weights = {}
    for decay in decay_factors:
        for lin in linear_search:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=cont,
                close_index=data_index,
                last_weight=decay,
                linear=lin,
                av_uniqueness=cont["tW"],
            )
            scheme = f"{best_scheme}_{'linear' if lin else 'exp'}_{decay}"
            time_decay_weights[scheme] = best_weight * decay_vec

    pbar2 = tqdm(time_decay_weights.items(), desc=f"Analyzing time-decay", total=len(time_decay_weights), mininterval=0.5)
    for i, (scheme, weight) in enumerate(pbar2, 1):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight, weights["return"],
            scheme, best_score, best_scheme, cv_results,
        )
        pbar2.set_postfix({"scheme": scheme, "scoring": scoring, "score": f"{cv_results[scheme].mean():.4f}"})
        if i == len(time_decay_weights):
            pbar2.set_postfix({
                "best": best_scheme,
                "score": f"{best_score:.4f}"
            })
        
    weights.update(time_decay_weights)
    cv_results_dict = {
        "best_score":  float(best_score),
        "cv_results":  cv_results,
        "scoring":     scoring,
        "best_scheme": best_scheme,
    }
    return weights[best_scheme], cv_results_dict


# ============================================================================
# ModelDevelopmentPipeline
# ============================================================================

class ModelDevelopmentPipeline:
    """
    The central production controller for Model Training and HPO.

    This class encapsulates the state of the model development lifecycle,
    ensuring that hyperparameters, feature names, and evaluation metrics
    are kept in sync with the physical artifacts saved to disk.

    Core Responsibilities:
    - Pre-processing: Removes constant and duplicate features to reduce
      model variance and training time.
    - Backend Switching: Transparently toggles between Scikit-learn (Grid)
      and Optuna (Bayesian) optimization based on 'use_optuna' config.
    - Artifact Management: Automatically organizes models, parquet data,
      and HTML reports into a versioned directory structure. The model
      filename is prefixed with the bagging wrapper type (sbag/bag/plain)
      so artifacts are immediately distinguishable in the file system.
    - Calibration (optional): Wraps the trained model in CalibratorCV
      to correct systematic overconfidence before downstream bet sizing.
    - Analysis: Triggers feature importance calculation and automated
      contamination reports after every successful run.

    Meta-Labeling:
    - Primary models (is_primary=True or auto-detected) can hand off to a
      secondary pipeline via prepare_meta_labeling_inputs().
    - strategy=None instructs the pipeline to use every bar as a candidate
      entry point with no directional signal (symmetric barriers).
    """

    def __init__(
        self,
        data_config: dict,
        feature_config: dict,
        target_config: dict,
        label_config: dict,
        model_params: dict,
        strategy: Optional[BaseStrategy] = None,
        is_primary: Optional[bool] = None,
        base_dir: str = "Models",
    ):
        """
        Initialize the pipeline with configuration parameters.

        Parameters
        ----------
        data_config : dict
            Bar construction configuration.
            - symbol : str
            - start_date : str  ('YYYY-MM-DD')
            - end_date : str    ('YYYY-MM-DD')
            - account_name : str
            - bar_type : str    ('tick', 'volume', 'time')
            - bar_size : str    (e.g. 'M1', 'M5')
            - price : str       ('bid', 'ask', 'mid_price', 'bid_ask')
            - path : Union[str, Path] = None
        strategy : BaseStrategy or None, optional
            Signal-generating strategy.  When None every bar is treated as a
            potential entry point and no directional side signal is applied,
            producing a symmetric label space (bin ∈ {-1, 0, 1}).  Pass a
            concrete strategy only when entry timing or direction comes from a
            rule-based signal (e.g. moving-average crossover for meta-labeling).
        feature_config : dict
            - func: Feature engineering function
            - params: Function parameters
        target_config : dict
            - func: Volatility target function
            - params: Function parameters
        label_config : dict
            - profit_target : float
            - stop_loss : float
            - max_holding_period : dict
            - min_ret : float
            - vertical_barrier_zero : bool
            - filter_as_series : bool
        model_params : dict
            - pipe_clf : BaseEstimator or Pipeline or MyPipeline
            - param_grid : dict or list of dicts
            - n_splits : int, default=5
            - bagging_n_estimators : int, default=0
            - bagging_sequential : bool, default=False
            - bagging_max_samples : float or int, default=1.0
            - bagging_max_features : float or int, default=1.0
            - rnd_search_iter : int, default=0
            - n_jobs : int, default=-1
            - pct_embargo : float, default=0.02
            - random_state : int or None, default=None
            - use_optuna : bool, default=False
            - n_trials : int  (Optuna only)
            - timeout : int   (Optuna only)
            - pruner_type : str, default='hyperband'
            - verbose : int, default=0
        is_primary : bool or None, optional
            Explicit role override.
            - True  → treat as primary model regardless of events content.
            - False → treat as secondary (meta-labeling) model.
            - None  → auto-detect: primary if 'side' not in events.columns
                       (default, preserves original behaviour).
        base_dir : str
            Root directory for saved artifacts.
        """
        from .file_manager import ModelFileManager
        
        self.data_config    = data_config
        self.symbol         = data_config["symbol"]
        self.train_start    = data_config["start_date"]
        self.train_end      = data_config["end_date"]
        self.strategy       = strategy
        self.feature_config = feature_config
        self.label_config   = label_config
        self.target_config  = target_config
        self.account_name   = data_config.get("account_name", "default")
        self.pipeline_version = "4.4"
        self.model_params   = model_params

        # Explicit is_primary override; None means auto-detect in generate_labels.
        self._is_primary_override = is_primary

        self.config = data_config.copy()
        del self.config["path"] # Not relevant for hashing
        self.config["training_start"] = self.config.pop("start_date")
        self.config["training_end"]   = self.config.pop("end_date")
        self.config["strategy"]       = (
            strategy.get_strategy_name() if strategy is not None else "ml_driven"
        )
        self.config["feature_func"]   = feature_config["func"].__name__
        self.config["feature_params"] = feature_config["params"]
        self.config["target_func"]    = target_config["func"].__name__
        self.config["target_params"]  = target_config["params"]
        self.config.update(label_config)

        # ── State ─────────────────────────────────────────────────────────────
        self.bar_data              = None
        self.features              = None
        self.events                = None
        self.sample_weight         = None
        self.best_weighting_scheme = None
        self.meta_features         = None
        self.preprocessed_features = None
        self.preprocessor          = None
        self.best_model            = None
        self.calibrator_           = None
        self.cv_results            = None
        self.weight_cv_results     = None
        self.feature_importance    = None
        self.metrics               = None
        self.study                 = None
        self.is_primary            = None   # resolved in generate_labels()
        self.display               = None
        self.calibrate             = None

        # ── Model type: bagging wrapper prefix + base estimator token ─────────
        # The wrapper prefix is computed here — it depends only on model_params
        # and is therefore stable for the lifetime of this instance. It is
        # prepended to the base estimator token so the model filename in the
        # file system immediately identifies the ensemble strategy used.
        #
        # Examples:  sbag_rf  |  bag_rf  |  plain_rf
        #
        if isinstance(model_params["pipe_clf"], Pipeline):
            _base_model = model_params["pipe_clf"].steps[-1][1]
        else:
            _base_model = model_params["pipe_clf"]

        self.model_type = f"{self._bagging_wrapper}_{get_model_type(_base_model)}"

        self.file_manager = ModelFileManager(base_dir)
        self.file_paths   = self.file_manager.setup_model_directory(
            self.config, self.model_type
        )

        self.decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]

        self.completed_steps = {
            "data_loading":        False,
            "feature_engineering": False,
            "label_generation":    False,
            "weight_computation":  False,
            "meta_features":       False,
            "model_training":      False,
            "calibration":         False,
            "analysis":            False,
        }

        self.log_file = self.file_paths["logs"] / "pipeline.log"
        self._setup_logging()
        self.n_splits = model_params["n_splits"]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _bagging_wrapper(self) -> str:
        """
        Canonical three-state label for the bagging wrapper in use.

        Computed purely from model_params so it is available from __init__
        onward.  Used as a prefix on model_type (and therefore on the saved
        model filename) so artifacts are immediately distinguishable without
        opening them.

        Returns
        -------
        str
            'sbag'  — SequentiallyBootstrappedBaggingClassifier
            'bag'   — standard BaggingClassifier
            'plain' — no bagging; base estimator used directly
        """
        sequential   = self.model_params.get("bagging_sequential", False)
        n_estimators = self.model_params.get("bagging_n_estimators", 0)
        if sequential and n_estimators > 0:
            return "sbag"
        if n_estimators > 0:
            return "bag"
        return "plain"

    # ── Logging ───────────────────────────────────────────────────────────────

    def _setup_logging(self):
        # Remove default handler and add our custom one 
        logger.remove()
        logger.add(
            self.log_file, level="INFO",
            format="{time} | {name} | {level} | {message}", rotation="10 MB",
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss:ms}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        self.logger = logger.bind(context=self.__class__.__name__)
       
    # ── run() ─────────────────────────────────────────────────────────────────

    def run(
        self,
        generate_reports: bool = True,
        cache_reports:    bool = False,
        save:             bool = True,
        export_onnx:      bool = False,
        calibrate:        bool = False,
        display:          bool = False,
        continue_study:    bool = False,
        verbose:          bool = True,
    ) -> Tuple:
        """
        Run the complete model development pipeline with integrated reporting.

        Parameters
        ----------
        generate_reports : bool, optional
            Generate analysis reports (default: True).
        cache_reports : bool, optional
            Display cache performance reports (default: False).
        save : bool, optional
            Save model and artifacts (default: True).
        export_onnx : bool, optional
            Export model to ONNX format (default: False).
        calibrate : bool, optional
            Fit CalibratorCV on OOF predictions after training.
            When True, self.best_model is replaced with the fitted calibrator,
            so all downstream calls to best_model.predict_proba() return
            calibrated probabilities.  Default: False.
        display : bool, optional
            Display the hyperparameter analysis report inline in a Jupyter
            notebook using IPython.display.  Default: False.
        verbose : bool, optional
            Print progress information (default: True).

        Returns
        -------
        tuple
            (best_model, feature_columns, metrics, config)
        """
        time0      = time.time()
        use_optuna = self.model_params.get("use_optuna", False)

        if verbose:
            print("\n" + "=" * 70)
            print(
                f"PRODUCTION MODEL DEVELOPMENT PIPELINE "
                f"(Backend: {'Optuna' if use_optuna else 'sklearn'})"
            )
            print("=" * 70)

        try:
            self.load_training_data()
            self.engineer_features()
            self.generate_labels()
            self.compute_sample_weights()
            self.add_meta_features()
            self.preprocess_features()
            self.train_model(continue_study)

            # ── Calibration (optional) ────────────────────────────────────────
            self.calibrate = calibrate
            if calibrate:
                self.calibrate_model()

            self.analyze_features()
            self._compile_metrics()

            if generate_reports:
                self._generate_analysis_reports(display=display)
            if cache_reports:
                self._display_cache_reports()
            if (save or export_onnx) and self.best_model is not None:
                self.export_onnx = export_onnx
                try:
                    self._save_all_artifacts()
                except Exception as e:
                    logger.error(f"Failed to save artifacts: {e}")
                    raise

            if verbose:
                elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace('0 days ', '')
                print(f"\n✓ Completed in {elapsed}")

            return self.best_model, self._get_feature_names(), self.metrics, self.config

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def load_training_data(self):
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        self.features = create_feature_engineering_pipeline(
            self.bar_data, self.feature_config, self.data_config
        )
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        """
        Apply the triple-barrier method and resolve the model's role.

        Role resolution order:
        1. If ``_is_primary_override`` was set at construction, use it.
        2. Otherwise auto-detect: primary when 'side' is absent from events.
        """
        self.events = generate_events_triple_barrier(
            self.bar_data, self.strategy, self.target_config, **self.label_config
        )

        if self._is_primary_override is not None:
            self.is_primary = self._is_primary_override
        else:
            self.is_primary = "side" not in self.events.columns

        self.config["model_role"] = "primary" if self.is_primary else "secondary"
        logger.info(
            f"Model role: {self.config['model_role']} | "
            f"Label space: {np.unique(self.events['bin']).tolist()}"
        )
        logger.info(f"Average uniqueness: ({self.events['tW'].mean():.4f})")
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
            self.bar_data.index, self.events, self.features,
            self.n_splits, None, self.decay_factors,
        )
        self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
        logger.info(f"best_weighting_scheme: {self.best_weighting_scheme}")
        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        """
        Append rolling performance meta-features for secondary models.

        Primary models skip this step — they have no prior predictions to
        reference.  Secondary (meta-labeling) models receive rolling accuracy,
        precision, recall, and F1 metrics derived from the primary model's
        label outcomes, allowing the secondary to adapt to changing regimes.
        """
        if self.is_primary:
            self.meta_features = pd.DataFrame(index=self.events.index)
            logger.info("Primary model — rolling meta-features skipped.")
        else:
            self.meta_features = calculate_rolling_metrics(
                self.events, self.sample_weight
            )
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        if self.meta_features.empty:
            combined = self.features.dropna()
        else:
            combined = self.features.join(self.meta_features, how="inner").dropna()

        self.preprocessor = Pipeline([
            ("dcf", DropConstantFeatures()),
            ("ddf", DropDuplicateFeatures()),
        ])
        self.preprocessed_features = self.preprocessor.fit_transform(combined)
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self, continue_study):
        """
        Dispatch to the appropriate HPO backend.

        The Optuna study optimises the base estimator's hyperparameters only.
        The bagging wrapper (if any) is applied post-HPO using the tuned base
        estimator.  Because bagging is not part of the optimisation loop the
        wrapper type does not affect the study name or config hash — a single
        study's trials are valid priors regardless of which wrapper is
        subsequently applied.

        Post-dispatch (both paths):
            The fitted preprocessor (DropConstant + DropDuplicate) is prepended
            to best_model so that sklearn inference is fully self-contained.
            NOTE: for ONNX export the preprocessor step is stripped before
            conversion.  Apply self.preprocessor.transform() as a standalone
            step before passing data to the deployed ONNX model.
        """
        self.model_params["pipe_clf"] = make_custom_pipeline(self.model_params["pipe_clf"])
        pipe = clone(self.model_params["pipe_clf"])

        bagging_n_estimators = self.model_params.get("bagging_n_estimators", 0)
        if bagging_n_estimators > 0:
            if self.model_params.get("bagging_max_samples") is None:
                av_uniqueness = float(self.events["tW"].mean().round(2))
                self.model_params["bagging_max_samples"] = av_uniqueness
                logger.info(
                    f"bagging_max_samples set to average uniqueness ({av_uniqueness:.4f})"
                )

        self.model_params["pipe_clf"] = pipe

        if self.model_params.get("use_optuna", False):
            self._train_model_optuna(continue_study)
        else:
            self._train_model_sklearn()

        self.best_model = Pipeline([
            ("preprocessor", self.preprocessor),
            *self.best_model.steps,
        ])
        self.best_model = set_pipeline_params(self.best_model, n_jobs=-1)
        self.completed_steps["model_training"] = True

    def calibrate_model(self) -> None:
        """
        Fit CalibratorCV wrapping best_model on the full training data.

        The calibrator uses PurgedKFold with the same n_splits and pct_embargo
        as the HPO step, ensuring that the OOF predictions used to fit the
        isotonic map are generated without temporal leakage.

        After this method returns:
        - self.best_model is replaced with the fitted CalibratorCV so all
          downstream calls to best_model.predict_proba() return calibrated
          probabilities.
        - self.calibrator_ holds the CalibratorCV instance for post-hoc
          diagnostics (reliability diagram, Brier score, oof_probs_).

        Notes
        -----
        ONNX export: CalibratorCV is not ONNX-compatible.  When export_onnx=True,
        _save_all_artifacts() unwraps the calibrator and exports the inner
        estimator (self.calibrator_.estimator_).
        """
        from ..calibration.calibration import CalibratorCV
        
        X             = self.preprocessed_features.loc[self.events.index]
        y             = self.events["bin"]
        sample_weight = self.sample_weight.loc[self.events.index]
        pct_embargo   = self.model_params.get("pct_embargo", 0.01)

        cv = PurgedKFold(
            n_splits=self.n_splits,
            t1=self.events["t1"],
            pct_embargo=pct_embargo,
        )

        self.calibrator_ = CalibratorCV(
            estimator=make_custom_pipeline(self.best_model),
            cv=cv,
        )
        self.calibrator_.fit(X, y, sample_weight=sample_weight)

        oof_brier = float(np.mean(
            (self.calibrator_.oof_probs_[~np.isnan(self.calibrator_.oof_probs_)]
             - y.values[~np.isnan(self.calibrator_.oof_probs_)]) ** 2
        ))
        logger.info(f"CalibratorCV fitted.  OOF Brier score: {oof_brier:.4f}")

        self.best_model = self.calibrator_
        self.completed_steps["calibration"] = True

    # ── Meta-labeling handoff ─────────────────────────────────────────────────

    def prepare_meta_labeling_inputs(self) -> pd.DataFrame:
        """
        Produce the events DataFrame annotated with the primary model's
        predicted side, ready for consumption by a secondary pipeline.

        The secondary pipeline detects 'side' in events.columns and switches
        to binary meta-labeling (bin ∈ {0, 1}).  The side signal is derived
        from the primary model's calibrated (or raw) probabilities:

            side = +1  where P(positive class) >= 0.5
            side = -1  otherwise

        Usage
        -----
        >>> meta_events = primary_pipeline.prepare_meta_labeling_inputs()
        >>> secondary = ModelDevelopmentPipeline(
        ...     data_config    = data_config,
        ...     feature_config = feature_config,
        ...     target_config  = target_config,
        ...     label_config   = {**label_config, "events": meta_events},
        ...     model_params   = secondary_model_params,
        ...     is_primary     = False,
        ... )

        Returns
        -------
        pd.DataFrame
            Copy of self.events with an additional 'side' column
            (values ∈ {-1, +1}).

        Raises
        ------
        RuntimeError
            If the pipeline has not completed training, or if called on a
            secondary model.
        """
        if self.best_model is None:
            raise RuntimeError(
                "Pipeline must complete training before calling "
                "prepare_meta_labeling_inputs()."
            )
        if not self.is_primary:
            raise RuntimeError(
                "prepare_meta_labeling_inputs() is only valid for primary models."
            )

        X = self.preprocessed_features.loc[self.events.index]
        proba = self.best_model.predict_proba(X)

        # The last column is always P(positive class) for both binary and
        # ternary label spaces under sklearn's sorted-classes convention.
        side = np.where(proba[:, -1] >= 0.5, 1, -1)

        meta_events = self.events.copy()
        meta_events["side"] = side
        return meta_events

    # ── Private training backends ─────────────────────────────────────────────

    def _train_model_sklearn(self):
        from ..cross_validation.hyper_fit import clf_hyper_fit_cached
        
        bagging_sequential = self.model_params.get("bagging_sequential", False)
        bagging_n          = self.model_params.get("bagging_n_estimators", 0)
        sample_weight_train = self.sample_weight.loc[self.events.index]
        sample_weight_score = self.events["w"].loc[sample_weight_train.index]

        included = inspect.signature(clf_hyper_fit_cached).parameters.keys()
        params   = {k: v for k, v in self.model_params.items() if k in included}

        if bagging_sequential and bagging_n > 0:
            params["bagging_n_estimators"] = 0
            tuned_pipeline, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],
                **params,
                sample_weight_train=sample_weight_train,
                sample_weight_score=sample_weight_score,
            )
            self.best_model = self._apply_sequential_bagging(
                self.preprocessed_features, self.events["bin"],
                tuned_pipeline, sample_weight=sample_weight_train,
            )
        else:
            self.best_model, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],
                **params,
                sample_weight_train=sample_weight_train,
                sample_weight_score=sample_weight_score,
            )

    def _train_model_optuna(self, continue_study):
        from ..cross_validation.optuna_hyper_fit import (
            optimize_trading_model,
            optuna_to_cv_results,
            print_best_trial,
            check_for_overfitting,
        )
        
        X, y = self.preprocessed_features, self.events["bin"]
        base_clf = self.model_params["pipe_clf"].steps[-1][1]
        metric = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"

        included = inspect.signature(optimize_trading_model).parameters.keys()
        opt_params = {"metric": metric, "continue_study": continue_study}
        for k, v in self.model_params.items():
            if k == "param_grid":
                opt_params["param_distributions"] = v
            elif k in included:
                opt_params[k] = v

        # ── Study name ────────────────────────────────────────────────────────
        # Tokens that change the Optuna optimization surface:
        #   strategy  — determines entry set and label distribution
        #   symbol    — different instruments have different dynamics
        #   bar_type/size — sampling frequency changes autocorrelation structure
        #   role      — primary (ternary) vs secondary (binary) label space
        #   config hash — catches all remaining surface dimensions (CV protocol,
        #                 search space shape, barrier params, target function)
        #
        # The bagging wrapper is intentionally omitted: HPO optimises the base
        # estimator only; bagging is applied post-study.  A study's trials are
        # valid priors regardless of which wrapper is subsequently applied.
        _role = "pri" if self.is_primary else "sec"
        _config_hash = self._get_study_config_hash(metric=metric)

        opt_params["study_name"] = (
            f"{self.config['strategy']}"
            f"_{self.symbol}"
            f"_{self.data_config.get('bar_type', 'unk')}"
            f"{self.data_config.get('bar_size', 'unk')}"
            f"_{_role}"
            f"_s{_config_hash}"
        )

        db_path: Path = self.file_paths["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        opt_params["db_path"] = f"sqlite:///{db_path.resolve()}"
        opt_params["reports_path"] = self.file_paths["reports"] / "trials"
        callbacks = [check_for_overfitting, print_best_trial]

        try:
            from .dashboard import launch_optuna_dashboard
            launch_optuna_dashboard(storage=opt_params["db_path"], timeout=60)
        except Exception as e:
            logger.error(e)

        self.study, cv_results_df = optimize_trading_model(
            classifier=base_clf, X=X, y=y, events=self.events,
            data_index=self.bar_data.index,
            refit=True,
            callbacks=callbacks,
            **opt_params,
        )

        logger.info(
            f"Optuna complete.\nBest score: {self.study.best_value:.4f}"
            f"\nBest params: {pformat(self.study.best_params)}"
        )

        best_estimator = make_custom_pipeline(self.study.best_estimator_.base_estimator)
        bagging_sequential = self.model_params.get("bagging_sequential", False)
        bagging_n_estimators = self.model_params.get("bagging_n_estimators", 0)
        bagging_max_samples  = self.model_params.get("bagging_max_samples", 1.0)
        bagging_max_features = self.model_params.get("bagging_max_features", 1.0)
        n_jobs = self.model_params.get("n_jobs", -1)
        random_state = self.model_params.get("random_state", None)

        if bagging_sequential and bagging_n_estimators > 0:
            self.best_model = self._apply_sequential_bagging(
                X, y, best_estimator,
                sample_weight=self.study.best_estimator_.sample_weight_,
            )
        elif bagging_n_estimators > 0:
            time0 = time.time()
            base_est = set_pipeline_params(best_estimator, n_jobs=1)
            bag = BaggingClassifier(
                estimator=MyPipeline(base_est.steps),
                n_estimators=int(bagging_n_estimators),
                max_samples=bagging_max_samples,
                max_features=bagging_max_features,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            bag.fit(X, y, sample_weight=self.study.best_estimator_.sample_weight_)
            self.best_model = Pipeline([("bag", bag)])
            elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace('0 days ', '')
            logger.info(f"\n✓ BaggingClassifier fitted in {elapsed}")
        else:
            self.best_model = best_estimator

        pruner_type = self.model_params.get("pruner_type", "hyperband")
        self.cv_results = {
            "best_params":        self.study.best_params,
            "best_score":         self.study.best_value,
            "cv_results":         cv_results_df,
            "scoring":            metric,
            "search_method":      "optuna",
            "pruner_type":        pruner_type,
            "n_trials_completed": len([t for t in self.study.trials if t.state.name == "COMPLETE"]),
            "n_trials_pruned":    len([t for t in self.study.trials if t.state.name == "PRUNED"]),
        }

    # ── Config hash ───────────────────────────────────────────────────────────

    def _get_study_config_hash(self, metric: str = "") -> str:
        """
        Return an 8-character SHA-256 prefix uniquely identifying the
        combination of parameters that determines the Optuna study's
        optimisation surface.

        Resuming an existing study is only correct when the optimisation
        surface is identical to the one those trials explored.  Every
        dimension that, if changed, should produce a fresh study is included.

        Covered dimensions
        ------------------
        model
            Base classifier type and constructor parameters.
        search
            Sorted list of search-space keys.  Adding or removing a
            hyperparameter changes the dimensionality of the space; existing
            trials become incomplete records and corrupt the TPE surrogate.
        cv
            n_splits and pct_embargo.
        metric
            'f1' (binary) or 'neg_log_loss' (ternary).
        role
            'primary' vs 'secondary'.
        label
            All label_config fields.
        target
            Volatility target function name and parameters.

        Note: bagging configuration is intentionally excluded.  HPO optimises
        the base estimator only; the wrapper applied afterwards does not change
        the optimisation surface.

        Parameters
        ----------
        metric : str
            Scoring metric passed to optimize_trading_model.

        Returns
        -------
        str
            8-character lowercase hex string (SHA-256 prefix).
        """
        import hashlib

        # ── 1. Base classifier ────────────────────────────────────────────────
        pipe = self.model_params["pipe_clf"]
        base_clf = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

        # ── 2. Search space shape ─────────────────────────────────────────────
        param_grid = self.model_params.get("param_grid", {})
        search_space = sorted(param_grid.keys())

        # ── 3. CV protocol ────────────────────────────────────────────────────
        cv_config = {
            "n_splits": self.model_params.get("n_splits", 5),
            "pct_embargo": self.model_params.get("pct_embargo", 0.02),
        }

        # ── 4. Combine and hash ───────────────────────────────────────────────
        combined = {
            "model": type(base_clf).__name__,
            "search": search_space,
            "cv": cv_config,
            "metric": metric,
            "role": "primary" if self.is_primary else "secondary",
            "config": self.config,
        }

        digest = hashlib.sha256(
            json.dumps(combined, sort_keys=True).encode()
        ).hexdigest()

        return digest[:8]

    # ── Bagging helpers ───────────────────────────────────────────────────────

    def _apply_sequential_bagging(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tuned_pipeline,
        sample_weight: pd.Series = None,
    ) -> Pipeline:
        """
        Wrap a tuned base pipeline in SequentiallyBootstrappedBaggingClassifier.

        After fitting, the trained estimators are transferred to a standard
        BaggingClassifier shell so that inference (predict / predict_proba) is
        available without requiring the events index at deployment time.
        """
        time0 = time.time()
        logger.info("\nSequential bootstrap is being fitted...")
        
        bagging_n = self.model_params.get("bagging_n_estimators", 0)
        bagging_samples = self.model_params.get("bagging_max_samples", 1.0)
        bagging_feats = self.model_params.get("bagging_max_features", 1.0)
        random_state = self.model_params.get("random_state", 1)

        base_est = set_pipeline_params(tuned_pipeline, n_jobs=1)

        bag = apply_seq_bootstrap(
            X=X,
            y=y,
            estimator=MyPipeline(base_est.steps),
            n_estimators=int(bagging_n),
            max_samples=bagging_samples,
            max_features=bagging_feats,
            samples_info_sets=self.events["t1"],
            price_bars_index=self.bar_data.index,
            random_state=random_state,
            sample_weight=sample_weight
        )

        # Transfer fitted estimators to a standard BaggingClassifier for
        # deployment — SequentiallyBootstrappedBaggingClassifier requires
        # the price bar index at fit time but not at inference time.
        standard_bag = BaggingClassifier(
            estimator=MyPipeline(base_est.steps),
            n_estimators=len(bag.estimators_),
            max_samples=1.0,
            max_features=bag.max_features,
            bootstrap=bag.bootstrap,
            bootstrap_features=bag.bootstrap_features,
            random_state=random_state,
            n_jobs=bag.n_jobs,
        )
        standard_bag.estimators_ = bag.estimators_
        standard_bag.estimators_features_ = bag.estimators_features_
        standard_bag.classes_ = bag.classes_
        standard_bag.n_classes_ = bag.n_classes_
        standard_bag.n_features_in_ = bag.n_features_in_

        for attr in ("oob_score_", "oob_decision_function_", "oob_prediction_"):
            if hasattr(bag, attr):
                setattr(standard_bag, attr, getattr(bag, attr))
        
        elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace('0 days ', '')
        logger.info(f"\n✓ Sequential bootstrap fitted in {elapsed}")

        return Pipeline([("seq_bag", standard_bag)])

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze_features(self):
        from .weighted_estimator import _WeightedEstimator

        clf = self.best_model
        if self.calibrate:
            clf = clf.estimator_

        if hasattr(clf, "steps"):
            clf = clf.steps[-1][1]

        feat_names = self._get_feature_names()

        if isinstance(clf, (SequentiallyBootstrappedBaggingClassifier, BaggingClassifier)):
            importances = np.mean([
                est.steps[-1][1].feature_importances_
                for est in clf.estimators_
            ], axis=0)
        elif isinstance(clf, _WeightedEstimator):
            try:
                importances = clf.base_estimator.feature_importances_
            except Exception as e:
                importances = np.zeros(len(feat_names))
                logger.error(e)
        else:
            try:
                importances = clf.feature_importances_
            except Exception as e:
                importances = np.zeros(len(feat_names))
                logger.error(e)

        self.feature_importance = pd.DataFrame({
            "feature":    feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        self.metrics = {
            "cv_results":            self.cv_results,
            "feature_importance":    self.feature_importance,
            "training_samples":      len(self.bar_data),
            "feature_count":         len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme,
            "average_uniqueness":    self.events["tW"].mean(),
            "completed_steps":       self.completed_steps,
            "calibrated":            self.calibrator_ is not None,
            "bagging_wrapper":       self._bagging_wrapper,
            "model_role":            self.config.get("model_role"),
        }

    def _get_feature_names(self):
        if self.preprocessed_features is None:
            return []
        return self.preprocessed_features.columns.tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _convert_mypipeline_for_onnx(pipeline: Pipeline) -> None:
        """
        Recursively replace every MyPipeline instance inside a fitted sklearn
        Pipeline with a standard sklearn Pipeline in-place.
        """
        for i, (name, step) in enumerate(pipeline.steps):
            if isinstance(step, MyPipeline):
                pipeline.steps[i] = (name, Pipeline(step.steps))
            elif isinstance(step, (BaggingClassifier, SequentiallyBootstrappedBaggingClassifier)):
                if isinstance(step.estimator, MyPipeline):
                    step.estimator = Pipeline(step.estimator.steps)
                if hasattr(step, "estimators_"):
                    step.estimators_ = [
                        Pipeline(e.steps) if isinstance(e, MyPipeline) else e
                        for e in step.estimators_
                    ]

    def _save_all_artifacts(self):
        metadata = {
            "strategy":         self.config["strategy"],
            "feature_names":    self._get_feature_names(),
            "use_optuna":       self.model_params.get("use_optuna", False),
            "pipeline_version": self.pipeline_version,
            "calibrated":       self.calibrator_ is not None,
            "bagging_wrapper":  self._bagging_wrapper,
            "model_role":       self.config.get("model_role"),
        }
        self.file_manager.save_model(self.best_model, metadata)

        if self.strategy is not None:
            self.file_manager.save_object(self.strategy, "strategy")

        self.file_manager.save_dataframe(self.preprocessed_features, "features")
        self.file_manager.save_dataframe(self.events, "events")

        if self.sample_weight is not None:
            self.file_manager.save_dataframe(
                self.sample_weight.to_frame("weights"), "weights"
            )

        self.file_manager.save_object(self.metrics, "metrics")

        if self.feature_config is not None:
            self.file_manager.save_object(self.feature_config, "feature_config")

        feature_names = self._get_feature_names()
        if feature_names:
            self.file_manager.save_object(feature_names, "feature_names")

        if self.preprocessor is not None:
            self.file_manager.save_object(self.preprocessor, "preprocessor")

        if self.target_config is not None:
            self.file_manager.save_object(self.target_config, "target_config")

        if self.calibrator_ is not None:
            self.file_manager.save_object(self.calibrator_.calibrator_, "calibrator")

        if self.export_onnx:
            if self.calibrator_ is not None:
                onnx_source = Pipeline(self.calibrator_.estimator_.steps[1:])
            else:
                onnx_source = Pipeline(self.best_model.steps[1:])

            self._convert_mypipeline_for_onnx(onnx_source)
            self.file_manager.save_model_as_onnx(
                onnx_source, self._get_feature_names(), metadata
            )

        logger.info(f"Saved all artifacts to {self.file_paths['base_dir']}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _generate_analysis_reports(self, display: bool = False):
        """
        Generates hyperparameter analysis, importance plots, and HTML summary.

        Any column in cv_results["cv_results"] that is 100% NaN is dropped
        before calling generate_complete_hyperparameter_report to prevent the
        matplotlib "autodetected range of [nan, nan]" error when param_grid
        contains heterogeneous search spaces.
        """
        from ..cross_validation.hyper_fit_analysis import generate_complete_hyperparameter_report
        
        try:
            if self.cv_results and "cv_results" in self.cv_results:
                cv_df = pd.DataFrame(self.cv_results["cv_results"])
                nan_cols = cv_df.columns[cv_df.isna().all()].tolist()
                if nan_cols:
                    cv_df = cv_df.dropna(axis=1, how="all")
                    logger.info(
                        f"Hyperparameter report: dropped {len(nan_cols)} fully-NaN "
                        f"column(s): {nan_cols}"
                    )
                generate_complete_hyperparameter_report(
                    cv_results=cv_df,
                    strategy_config=self.config,
                    output_dir=self.file_paths["reports"],
                    display_in_notebook=display,
                )

            self._generate_training_summary_html()
            if self.study is not None:
                self._generate_optuna_report()

        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    def _generate_optuna_report(self):
        """
        Save a self-contained HTML report with Optuna visualisation plots.
        """
        import optuna.visualization as vis
        from plotly.io import to_html
        import matplotlib.pyplot as plt

        report_path = self.file_paths["reports"] / "optuna_study_report.html"
        study       = self.study

        n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
        n_pruned   = len([t for t in study.trials if t.state.name == "PRUNED"])

        plot_specs = [
            (
                "Optimization History",
                "Objective value per trial. Dashed line shows the running best.",
                lambda: vis.plot_optimization_history(study),
            ),
            (
                "Fold Scores per Trial (Intermediate Values)",
                "Each line is one trial. Trials that end early were pruned.",
                lambda: vis.plot_intermediate_values(study),
            ),
            (
                "Hyperparameter Importances",
                "fANOVA estimate of each parameter's contribution to score variance.",
                lambda: vis.plot_param_importances(study),
            ),
            (
                "Parallel Coordinates",
                "One line per completed trial, coloured by score.",
                lambda: vis.plot_parallel_coordinate(study),
            ),
        ]

        html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Optuna Study Report — {study.study_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a;
               color: #f1f5f9; padding: 40px; line-height: 1.6; }}
        h1   {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
        h2   {{ color: #7dd3fc; margin-top: 48px; }}
        p.caption {{ color: #64748b; font-size: 0.88rem; margin-top: 4px; }}
        .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 32px; }}
        .meta span {{ color: #22c55e; font-weight: 600; }}
        .plot {{ margin-bottom: 56px; }}
    </style>
</head>
<body>
<h1>Optuna Study Report</h1>
<p class="meta">
    Study: <b>{study.study_name}</b><br>
    Completed: <span>{n_complete}</span> trials &nbsp;|&nbsp;
    Pruned: <span style="color:#f97316">{n_pruned}</span> trials &nbsp;|&nbsp;
    Best score: <span>{study.best_value:.4f}</span>
</p>"""]

        for title, caption, plot_fn in plot_specs:
            try:
                fig  = plot_fn()
                html = to_html(fig, full_html=False, include_plotlyjs="cdn")
                html_parts.append(
                    f'<div class="plot"><h2>{title}</h2>'
                    f'<p class="caption">{caption}</p>{html}</div>'
                )
            except Exception as e:
                logger.warning(f"Optuna plot '{title}' failed: {e}")

        try:
            from ..cross_validation.optuna_hyper_fit import plot_model_vs_baseline
            original_backend = plt.get_backend()
            plt.switch_backend("agg")
            with plt.ioff():
                plot_model_vs_baseline(study, self.events["bin"], self.events)
                baseline_path = self.file_paths["reports"] / "optuna_baseline_comparison.png"
                plt.savefig(
                    baseline_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f172a", edgecolor="none",
                )
                plt.close("all")
            plt.switch_backend(original_backend)
            logger.info(f"Baseline comparison plot saved: {baseline_path}")
        except Exception as e:
            logger.warning(f"Baseline plot failed: {e}")

        html_parts.append("</body></html>")

        report_path.write_text("\n".join(html_parts), encoding="utf-8")
        logger.info(f"Optuna study report saved: {report_path}")

    def _generate_training_summary_html(self):
        """Constructs a comprehensive HTML training report."""
        try:
            report_path = self.file_paths["reports"] / "training_summary.html"

            best_score    = self.cv_results.get("best_score", 0)
            search_method = "Optuna" if self.study is not None else "Scikit-Learn"
            calibrated    = self.calibrator_ is not None

            html_content = f"""
            <html>
            <head>
                <title>Training Report - {self.symbol}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background-color: #0f172a; color: #f1f5f9; padding: 40px;
                            line-height: 1.6; }}
                    .container {{ max-width: 900px; margin: auto; }}
                    h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
                    .card {{ background-color: #1e293b; border-radius: 12px; padding: 24px;
                             margin-bottom: 24px; border: 1px solid #334155; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #334155; }}
                    th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase;
                          font-size: 0.8rem; }}
                    .metric {{ font-size: 1.5rem; font-weight: 700; color: #22c55e; }}
                    .label {{ color: #94a3b8; font-size: 0.9rem; }}
                    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
                              font-size: 0.78rem; font-weight: 600; }}
                    .badge-on  {{ background: #14532d; color: #4ade80; }}
                    .badge-off {{ background: #1e293b; color: #64748b; border: 1px solid #334155; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Training Summary: {self.symbol}</h1>
                    <p class="label">Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

                    <div class="card">
                        <h2>Performance Snapshot</h2>
                        <table>
                            <tr>
                                <td>
                                    <span class="label">
                                        Primary Metric ({self.cv_results.get('scoring', 'F1')})
                                    </span><br>
                                    <span class="metric">{best_score:.4f}</span>
                                </td>
                                <td>
                                    <span class="label">Backend</span><br>
                                    <strong>{search_method}</strong>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Training Samples</span><br>
                                    <strong>{len(self.events)}</strong>
                                </td>
                                <td>
                                    <span class="label">Average Uniqueness</span><br>
                                    <strong>{self.events['tW'].mean():.4f}</strong>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Model Role</span><br>
                                    <strong>{self.config.get('model_role', 'N/A').capitalize()}</strong>
                                </td>
                                <td>
                                    <span class="label">Bagging Wrapper</span><br>
                                    <strong>{self._bagging_wrapper}</strong>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Calibrated</span><br>
                                    <span class="badge {'badge-on' if calibrated else 'badge-off'}">
                                        {'CalibratorCV' if calibrated else 'No calibration'}
                                    </span>
                                </td>
                                <td></td>
                            </tr>
                        </table>
                    </div>

                    <div class="card">
                        <h2>Weighting Logic</h2>
                        <p><strong>Selected Scheme:</strong>
                           {self.best_weighting_scheme or "Standard/Time-Decay"}</p>
                        <p class="label">
                            Weights were optimized via Purged-KFold to minimize
                            serial correlation leakage.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            report_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Generated HTML summary report: {report_path}")

        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")

    def _display_cache_reports(self):
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE REPORT")
        print("=" * 70)
        monitor = get_cache_monitor()
        if monitor:
            monitor.print_report()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def check_contamination(self):
        print("\n" + "=" * 70)
        print("DATA CONTAMINATION CHECK")
        print("=" * 70)
        print_contamination_report()

    def get_data_summary(self) -> pd.DataFrame:
        summary_data = []
        components = [
            ("bar_data",              self.bar_data),
            ("features",              self.features),
            ("preprocessed_features", self.preprocessed_features),
            ("events",                self.events),
            ("meta_features",         self.meta_features),
            ("sample_weight",         self.sample_weight),
        ]
        for name, data in components:
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    shape, dtype, columns = data.shape, "DataFrame", f"{len(data.columns)} cols"
                elif isinstance(data, pd.Series):
                    shape, dtype, columns = (len(data),), "Series", "N/A"
                else:
                    shape, dtype, columns = "N/A", type(data).__name__, "N/A"
                summary_data.append({
                    "Component": name,
                    "Type":      dtype,
                    "Rows":      shape[0] if isinstance(shape, tuple) else shape,
                    "Columns":   (
                        shape[1]
                        if isinstance(shape, tuple) and len(shape) > 1
                        else columns
                    ),
                    "Memory (MB)": (
                        data.memory_usage(deep=True).sum() / (1024 ** 2)
                        if hasattr(data, "memory_usage") else "N/A"
                    ),
                })
        return pd.DataFrame(summary_data)


# ============================================================================
# Helpers
# ============================================================================

def get_model_type(model) -> str:
    types = {
        "RandomForestClassifier":                    "rf",
        "SequentiallyBootstrappedBaggingClassifier": "seq_rf",
        "DecisionTreeClassifier":                    "dt",
    }
    name = type(model).__name__
    return types.get(name, name.replace("Classifier", "").lower())


def is_tree(estimator) -> bool:
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))


@cacheable(time_aware=True)
def apply_seq_bootstrap(
    X,
    y,
    estimator,
    n_estimators,
    max_samples,
    max_features,
    samples_info_sets,
    price_bars_index,
    random_state,
    sample_weight
):
    bag = SequentiallyBootstrappedBaggingClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            samples_info_sets=samples_info_sets,
            price_bars_index=price_bars_index,
            random_state=random_state,
    )
    
    if sample_weight is not None:
        bag.fit(X, y, sample_weight=sample_weight)
    else:
        bag.fit(X, y)

    return bag
