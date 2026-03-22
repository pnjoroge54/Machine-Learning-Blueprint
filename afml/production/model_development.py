"""
model_development.py
--------------------
Production model development pipeline with optional Optuna HPO integration.

This module serves as the central orchestrator for the Machine Learning Blueprint,
integrating high-performance data processing, advanced labeling techniques,
and rigorous cross-validation protocols designed for non-IID financial data.

The pipeline now supports two training paths controlled by
`model_params['use_optuna']`:

    False (default): clf_hyper_fit via GridSearchCV / RandomizedSearchCV.
    True:            optimize_trading_model via Optuna + HyperbandPruner.

When use_optuna=True the following changes apply:
  - Weight computation for HPO is handled internally by _WeightedEstimator;
    get_optimal_sample_weight is still run for meta-features and reporting.
  - train_model dispatches to _train_model_optuna.
  - self.study is populated with the completed Optuna study for visualization.

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

Performance Optimizations:
-------------------------
- Numba JIT Compilation: High-frequency rolling metrics and weight attributions
  are offloaded to parallelized C-level code for maximum throughput.
- Time-Aware Caching: Utilizes a custom hashing mechanism to ensure that
  feature engineering and data loading are only re-computed when the
  underlying data or parameters change, effectively eliminating look-ahead bias.
- Memory Management: Efficient use of Parquet for artifact storage and
  DataFrame pruning via constant and duplicate feature removal.

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
7. Reporting: Generates HTML summaries and hyperparameter importance reports.
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
from scipy.stats import uniform
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ..cache import cacheable, get_cache_monitor, log_data_access, print_contamination_report
from ..cross_validation.hyper_fit import clf_hyper_fit_cached
from ..cross_validation.cross_validation import PurgedKFold, ml_cross_val_score
from ..cross_validation.hyper_fit_analysis import generate_complete_hyperparameter_report
from ..cross_validation.optuna_hyper_fit import (
    optimize_trading_model,
    optuna_to_cv_results,
    print_best_trial,
    check_for_overfitting,
)
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
from .file_manager import ModelFileManager

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
    strategy: BaseStrategy,
    target_config: dict,
    profit_target: float = 1,
    stop_loss: float = 1,
    max_holding_period: Dict[str, int] = dict(days=1),
    min_ret: float = 0.0,
    vertical_barrier_zero: bool = True,
    filter_as_series: bool = True,
    on_crossover: bool = True,
) -> pd.DataFrame:
    """
    Generate trading events using the triple-barrier method.

    Parameters
    ----------
    data : pd.DataFrame
        Price bars with 'close' column.
    strategy : BaseStrategy
        Strategy instance implementing `generate_signals()`.
    target_config : int
        Lookback window for volatility estimation.
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
    on_crossover : bool, default=True
        Whether strategy expects crossover for signal
    Returns
    -------
    pd.DataFrame
        Event labels with columns:
        - 'bin' : {-1, 0, 1} classification
        - 't1'  : vertical barrier timestamps
        - 'w'   : sample weights
        - 'tW'  : uniqueness weights

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    data_dict = dict(
        open=data["open"], high=data["high"], low=data["low"],
        close=data["close"], df=data, data=data, prices=data,
    )
    close = data["close"]
    target_func = target_config["func"]
    target_params = target_config["params"].copy()

    sig = inspect.signature(target_func)
    for key in sig.parameters.keys():
        if key not in target_params:
            target_params[key] = data_dict[key]

    try:
        target = target_func(**target_params)
    except Exception as e:
        print(e)

    if strategy.get_objective() == "mean_reversion":
        filter_threshold = target if filter_as_series else target.mean()
    else:
        filter_threshold = None

    side, t_events = get_entries(strategy, data, filter_threshold, on_crossover)
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
    accuracy = np.full(n, np.nan)
    precision = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    f1 = np.full(n, np.nan)

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
    primary model can learn to 'size down' or avoid trades during periods
    where its recent accuracy or F1-score is declining (Meta-Labeling concept).
    """

    y_true = events["bin"].to_numpy(np.int8)
    y_pred = np.ones(len(y_true), dtype=np.int8)
    weights = sample_weight.to_numpy(np.float32)
    metrics = pd.DataFrame(index=events.index)

    for window in window_sizes:
        if window > len(y_true):
            continue
        accuracy, precision, recall, f1 = _rolling_metrics_numba(y_true, y_pred, weights, window)
        metrics[f"rolling_accuracy_{window}"] = accuracy
        metrics[f"rolling_precision_{window}"] = precision
        metrics[f"rolling_recall_{window}"] = recall
        metrics[f"rolling_f1_{window}"] = f1

    return metrics.dropna()


def best_weighting_scheme(
    classifier,
    X,
    y,
    cv_gen,
    scoring,
    sample_weight,
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
        sample_weight_train=sample_weight,
        sample_weight_score=sample_weight,
        scoring=scoring,
    )
    score = scores.mean()
    cv_results[scheme] = scores

    if not np.isinf(score) and score > best_score:
        best_score = score
        best_scheme = scheme

    return best_score, best_scheme, cv_results


@cacheable(time_aware=True)
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
        Training features
    n_splits : int, optional
        Number of cross-validation splits (default: 5).
    linear : bool, optional
        Default is None, which seraches both linear and exponential time-decay.
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

    valid_index = features.index.intersection(events.index)
    cont = events.loc[valid_index]
    X = features.loc[valid_index]
    y = cont["bin"]

    classifier = RandomForestClassifier(
        criterion="entropy",
        class_weight="balanced_subsample",
        max_samples=cont["tW"].mean(),
        max_depth=4,
        min_weight_fraction_leaf=0.05,
        n_jobs=-1
    )

    cv_gen = PurgedKFold(n_splits=n_splits, t1=cont["t1"], pct_embargo=0.01)

    weights = {
        "return": cont["w"],
        "unweighted": pd.Series(1.0, index=cont.index),
        "uniqueness": cont["tW"],
    }

    best_score, best_scheme = 0, None
    cv_results = pd.DataFrame()
    scoring = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"

    for scheme, weight in tqdm(weights.items(), desc="Analyzing weighting schemes", total=len(weights)):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight, scheme, best_score, best_scheme, cv_results
        )

    best_weight = weights[best_scheme]
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

    for scheme, weight in tqdm(time_decay_weights.items(), desc=f"Analyzing time-decay for {best_scheme}"):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight, scheme, best_score, best_scheme, cv_results
        )

    weights.update(time_decay_weights)
    cv_results_dict = {
        "best_score": best_score,
        "cv_results": cv_results,
        "scoring": scoring,
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
      and HTML reports into a versioned directory structure.
    - Analysis: Triggers feature importance calculation and automated
      contamination reports after every successful run.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data_config: dict,
        feature_config: dict,
        target_config: dict,
        label_config: dict,
        model_params: dict,
        base_dir: str = "Models",
    ):
        """
        Initialize the pipeline with configuration parameters.

        Parameters
        ----------
        data_config : dict
            Bar construction configuration.
            - symbol : str
                Trading instrument symbol.
            - start_date : str
                Training start date ('YYYY-MM-DD').
            - end_date : str
                Training end date ('YYYY-MM-DD').
            - account_name : str
                Name of trading account
            - bar_type : str
                Type of bar ('tick', 'volume', 'time').
            - bar_size : str
                Bar size specification (e.g., 'M1', 'M5').
            - price : str
                Price type ('bid', 'ask', 'mid_price', 'bid_ask').
            - path : Union[str, Path] = None
                Path to data folder. If None uses default, Path.home() / "tick_data_parquet"
        strategy : BaseStrategy
            Signal generating strategy.
        feature_config : dict
            Feature engineering configuration.
            - func: Feature engineering function
            - params: Function parameters
        target_config : dict
            Volatility target configuration.
            - func: Volatility target function
            - params: Function parameters
        label_config : dict
            Triple-barrier labeling configuration.
            - profit_target : float
            - stop_loss : float
            - max_holding_period : int
            - min_ret : float
            - vertical_barrier_zero : bool
            - filter_as_series : bool
        model_params : dict
            Model training configuration.
                - pipe_clf : BaseEstimator or sklearn.pipeline.Pipeline or MyPipeline
                    A BaseEstimator or Pipeline containing preprocessing and classification steps.
                - param_grid : dict or list of dicts
                    Hyperparameter grid for search. Keys should include pipeline step
                    names as prefixes (e.g., 'classifier__max_depth').
                - n_splits : int, default=5
                    Number of folds for purged k-fold cross-validation.
                - bagging_n_estimators : int, default=0
                    Number of base estimators in bagging ensemble. If 0, no bagging
                    is applied and the best single estimator is returned. If > 0,
                    returns a BaggingClassifier fitted on the full dataset.
                - bagging_max_samples : float or int, default=1.0
                    For bagging: fraction (if float in (0, 1]) or number (if int) of
                    samples to draw for each base estimator.
                - bagging_max_features : float or int, default=1.0
                    For bagging: fraction (if float in (0, 1]) or number (if int) of
                    features to draw for each base estimator.
                - rnd_search_iter : int, default=0
                    If 0, uses GridSearchCV (exhaustive search). If > 0, uses
                    RandomizedSearchCV with this many iterations.
                - n_jobs : int, default=-1
                    Number of parallel jobs. -1 uses all available cores.
                - pct_embargo : float, default=0.02
                    Percentage of samples to embargo in test folds to prevent leakage
                    from serially correlated labels. Range: [0, 1).
                - random_state : int, RandomState instance or None, default=None
                    Random state for reproducibility.
                - use_optuna : bool, default=False
                    Toggles Bayesian HPO (True) vs. Grid/Random Search (False).
                - n_trials : (int, if Optuna=True)
                    Number of Optuna trials.
                - timeout : (int, if Optuna=True)
                    Time limit in seconds for HPO.
                - pruner_type : (str, if Optuna=True, default="hyperband")
                    Model used for pruning. Options are "hyperband" and "median".
                    If any other value, SuccessiveHalvingPruner is used.
                - verbose : int, default=0
                    Controls verbosity of output.
        base_dir: str
            Path to save pipeline data
        """

        self.data_config = data_config
        self.symbol = data_config["symbol"]
        self.train_start = data_config["start_date"]
        self.train_end = data_config["end_date"]
        self.strategy = strategy
        self.feature_config = feature_config
        self.label_config = label_config
        self.target_config = target_config
        self.account_name = data_config.get("account_name", "default")
        self.pipeline_version = "4.0"
        self.model_params = model_params

        self.config = data_config.copy()
        self.config["training_start"] = self.config.pop("start_date")
        self.config["training_end"] = self.config.pop("end_date")
        self.config["strategy"] = strategy.get_strategy_name()
        self.config["feature_func"] = feature_config["func"].__name__
        self.config["feature_params"] = feature_config["params"]
        self.config["target_func"] = target_config["func"].__name__
        self.config["target_params"] = target_config["params"]
        self.config.update(label_config)

        self.bar_data = None
        self.features = None
        self.events = None
        self.sample_weight = None
        self.best_weighting_scheme = None
        self.meta_features = None
        self.preprocessed_features = None
        self.preprocessor = None          # fitted DropConstant+DropDuplicate, prepended to best_model
        self.best_model = None
        self.cv_results = None
        self.weight_cv_results = None
        self.feature_importance = None
        self.metrics = None
        self.study = None
        self.is_primary = None            # True = primary model ({-1,0,1}), False = secondary ({0,1})

        if isinstance(model_params["pipe_clf"], Pipeline):
            model = model_params["pipe_clf"].steps[-1][1]
        else:
            model = model_params["pipe_clf"]

        self.model_type = get_model_type(model)
        self.file_manager = ModelFileManager(base_dir)
        self.file_paths = self.file_manager.setup_model_directory(self.config, self.model_type)

        self.decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]

        self.completed_steps = {
            "data_loading": False, "feature_engineering": False, "label_generation": False,
            "weight_computation": False, "meta_features": False, "model_training": False, "analysis": False,
        }

        self.log_file = self.file_paths["logs"] / "pipeline.log"
        self._setup_logging()
        self.n_splits = model_params["n_splits"]

    def _setup_logging(self):
        logger.remove()
        logger.add(self.log_file, level="INFO", format="{time} | {name} | {level} | {message}", rotation="10 MB")
        logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)
        self.logger = logger.bind(context=self.__class__.__name__)

    def run(
        self,
        generate_reports: bool = True,
        cache_reports: bool = False,
        save: bool = True,
        export_onnx: bool = False,
        verbose: bool = True,
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
        export_onxx : bool, optional
            Export model to ONNX format (default: False).
        verbose : bool, optional
            Print progress information (default: True).

        Returns
        -------
        tuple
            (best_model, features_columns, metrics, config)
        """
        time0 = time.time()
        self.export_onnx = export_onnx
        use_optuna = self.model_params.get("use_optuna", False)

        if verbose:
            print("\n" + "=" * 70)
            print(f"PRODUCTION MODEL DEVELOPMENT PIPELINE (Backend: {'Optuna' if use_optuna else 'sklearn'})")
            print("=" * 70)

        try:
            self.load_training_data()
            self.engineer_features()
            self.generate_labels()
            self.compute_sample_weights()
            self.add_meta_features()
            self.preprocess_features()
            self.train_model()
            self.analyze_features()
            self._compile_metrics()

            if generate_reports:
                self._generate_analysis_reports()
            if cache_reports:
                self._display_cache_reports()
            if save and self.best_model is not None:
                self._save_all_artifacts()

            if verbose:
                print(f"\n✓ Completed in {pd.Timedelta(seconds=time.time()-time0).round('1s')}")
            return self.best_model, self._get_feature_names(), self.metrics, self.config

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def load_training_data(self):
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        self.features = create_feature_engineering_pipeline(self.bar_data, self.feature_config, self.data_config)
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        self.events = generate_events_triple_barrier(self.bar_data, self.strategy, self.target_config, **self.label_config)
        # 'side' is present only when side_prediction was passed (secondary / meta-labeling model).
        # For primary models (triple-barrier without side, trend-scanning), side is absent.
        # This flag gates meta-feature computation and artifact naming.
        self.is_primary = 'side' not in self.events.columns
        self.config['model_role'] = 'primary' if self.is_primary else 'secondary'
        logger.info(f"Model role: {self.config['model_role']} | Label space: {sorted(self.events['bin'].unique())}")
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        if self.file_paths["weights"].exists():
            self.sample_weight = pd.read_parquet(self.file_paths["weights"])
        else:
            self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
                self.bar_data.index, self.events, self.features, self.n_splits, None, self.decay_factors
            )
            self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
            if self.sample_weight is not None:
                self.file_manager.save_dataframe(self.sample_weight.to_frame("weight"), "weights")
        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        if self.is_primary:
            # No prior model exists whose rolling performance we can track.
            # Meta-features are a self-referential concept that only makes sense
            # for a secondary model evaluating a primary model's signal quality.
            self.meta_features = pd.DataFrame(index=self.events.index)
            logger.info("Primary model detected — rolling meta-features skipped.")
        else:
            self.meta_features = calculate_rolling_metrics(self.events, self.sample_weight)
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        # Join meta-features only when they exist (secondary model path).
        # For primary models meta_features is an empty DataFrame, so we skip the join.
        if self.meta_features.empty:
            combined = self.features.dropna()
        else:
            combined = self.features.join(self.meta_features, how='inner').dropna()

        # Store the fitted preprocessor so it can be prepended to best_model after training.
        # This makes the column-dropping a permanent part of inference — the saved model
        # always produces predictions on exactly the same column set it was trained on,
        # regardless of what columns appear in new data.
        self.preprocessor = Pipeline([
            ("dcf", DropConstantFeatures()),
            ("ddf", DropDuplicateFeatures()),
        ])
        self.preprocessed_features = self.preprocessor.fit_transform(combined)
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self):
        """
        Dispatch to the appropriate HPO backend.

        When use_optuna=True:
            - optimize_trading_model is called with the base estimator and events.
            - FinancialModelSuggester wraps it in _WeightedEstimator per trial.
            - weight_scheme, weight_decay, and weight_linear are jointly optimized.
            - Return-attribution weights (events['w']) always used for scoring.
            - self.study is populated for post-study visualization.

        When use_optuna=False:
            - clf_hyper_fit is called with the pre-computed sample_weight.
            - Behavior is identical to the original pipeline.

        In both paths, when bagging_sequential=True:
            - HPO tunes the base classifier without any bagging wrapper.
            - SequentiallyBootstrappedBaggingClassifier is applied post-HPO using
              the tuned base estimator, events['t1'], and bar_data.index.

        Post-dispatch (both paths):
            - The fitted preprocessor (DropConstant + DropDuplicate) is prepended
              to best_model so that inference is fully self-contained.
        """
        self.model_params['pipe_clf'] = make_custom_pipeline(self.model_params['pipe_clf'])
        pipe = clone(self.model_params['pipe_clf'])

        bagging_n_estimators = self.model_params.get('bagging_n_estimators', 0)

        if bagging_n_estimators > 0:
            # Auto-resolve bagging_max_samples to average uniqueness when left as None.
            if self.model_params.get('bagging_max_samples') is None:
                av_uniqueness = self.events['tW'].mean().round(2)
                self.model_params['bagging_max_samples'] = av_uniqueness
                logger.info(f"bagging_max_samples set to average uniqueness ({av_uniqueness:.4f})")
        elif hasattr(pipe.steps[-1][-1], "max_samples"):
            # max_samples is a bootstrap parameter specific to RandomForestClassifier.
            # DecisionTreeClassifier has no bootstrap step and no max_samples parameter.
            av_uniqueness = self.events['tW'].mean().round(2)
            pipe = set_pipeline_params(pipe, max_samples=av_uniqueness)
            if self.model_params.get("use_optuna", False) or self.model_params.get("rnd_search_iter", 0) > 0:
                self.model_params["param_grid"]["max_samples"] = uniform(av_uniqueness, 1 - av_uniqueness)

        self.model_params['pipe_clf'] = pipe

        if self.model_params.get('use_optuna', False):
            self._train_model_optuna()
        else:
            self._train_model_sklearn()

        # Prepend the fitted preprocessor so the saved model is fully self-contained.
        # best_model.predict(raw_features) now applies DropConstant + DropDuplicate
        # (fitted on training data) before reaching the classifier, at inference time.
        self.best_model = Pipeline([
            ("preprocessor", self.preprocessor),
            *self.best_model.steps,
        ])
        self.best_model = set_pipeline_params(self.best_model, n_jobs=-1)
        self.completed_steps['model_training'] = True

    def _train_model_sklearn(self):
        bagging_sequential = self.model_params.get('bagging_sequential', False)
        bagging_n = self.model_params.get('bagging_n_estimators', 0)
        sample_weight = self.sample_weight.loc[self.events.index]  # meta_features change indexing
        sample_weight *= sample_weight.shape[0] / sample_weight.sum()

        # Keys that belong to the Optuna path or are handled outside clf_hyper_fit
        included = inspect.signature(clf_hyper_fit_cached).parameters.keys()
        params = {k: v for k, v in self.model_params.items() if k in included}        

        if bagging_sequential and bagging_n > 0:
            # Tune the base classifier first with no bagging, then apply sequential
            # bootstrapping post-HPO. Exclude bagging params so clf_hyper_fit returns
            # the plain tuned pipeline rather than a standard BaggingClassifier.
            params["bagging_n_estimators"] = 0
            tuned_pipeline, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],
                **params,
                sample_weight=sample_weight
            )
            self.best_model = self._apply_sequential_bagging(
                self.preprocessed_features, self.events["bin"],
                tuned_pipeline, sample_weight=sample_weight,
            )
        else:
            self.best_model, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],                 
                **params,
                sample_weight=sample_weight
            )

    def _train_model_optuna(self):
        X, y = self.preprocessed_features, self.events["bin"]
        base_clf = self.model_params['pipe_clf'].steps[-1][1]
        metric = 'f1' if set(y.unique()) == {0, 1} else 'neg_log_loss'
        
        included = inspect.signature(optimize_trading_model).parameters.keys()
        opt_params = {"metric": metric}
        for k, v in self.model_params.items():
            if k == "param_grid":
                opt_params["param_distributions"] = v
            elif k in included:
                opt_params[k] = v

        # study_name encodes every experiment dimension 
        # The config_hash is the last path segment of base_dir — a deterministic
        # MD5 of the full config dict. The same experiment always maps to the same
        # study, so a crashed run resumes exactly where it left off.
        config_hash = self.file_paths["base_dir"].name
        study_config_hash = self._get_study_config_hash()
        
        opt_params["study_name"] = (
            f"{self.strategy.get_strategy_name()}"
            f"_{self.symbol}"
            f"_{self.data_config.get('bar_type', 'unk')}"
            f"_{self.data_config.get('bar_size', 'unk')}"
            f"_{config_hash}"
            f"_s{study_config_hash}"
        )      
        
        # Build the SQLite URI with an absolute path so it is correct regardless
        # of the working directory at runtime.
        db_path: Path = self.file_paths["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        opt_params["db_path"] = f"sqlite:///{db_path.resolve()}"
        opt_params["reports_path"] = self.file_paths["reports"]
        callbacks = [check_for_overfitting, print_best_trial]

        try:
            from .dashboard import launch_optuna_dashboard
            launch_optuna_dashboard(storage=opt_params["db_path"], timeout=90)
        except Exception as e:
            logger.error(e)

        # ── Run the study (refit=True: handled internally) ───────────────────
        self.study, cv_results_df = optimize_trading_model(
            classifier=base_clf, X=X, y=y, events=self.events,
            data_index=self.bar_data.index,
            refit=True,
            callbacks=callbacks,
            **opt_params,
        )

        logger.info(
            f"Optuna complete. Best score: {self.study.best_value:.4f}  "
            f"Best params: {self.study.best_params}"
        )
        # Unwrap: best_estimator_ is a _WeightedEstimator. Extract the tuned base
        # classifier and re-wrap in MyPipeline for downstream compatibility.
        best_estimator = make_custom_pipeline(self.study.best_estimator_.base_estimator)

        bagging_sequential = self.model_params.get('bagging_sequential', False)
        bagging_n_estimators = self.model_params.get('bagging_n_estimators', 0)
        bagging_max_samples = self.model_params.get('bagging_max_samples', 1.0)
        bagging_max_features = self.model_params.get('bagging_max_features', 1.0)
        n_jobs = self.model_params.get('n_jobs', -1)
        random_state = self.model_params.get('random_state', None)
        
        if bagging_sequential and bagging_n_estimators > 0:
            # Sequential bootstrapping with the winning trial's weights.
            # Weights are passed explicitly because SequentiallyBootstrappedBaggingClassifier
            # uses numpy integer indexing for bootstrap samples, which strips the pandas
            # index that _WeightedEstimator relies on for alignment.
            self.best_model = self._apply_sequential_bagging(
                X, y, best_estimator,
                sample_weight=self.study.best_estimator_.sample_weight_,
            )
        elif bagging_n_estimators > 0:
            base_est = set_pipeline_params(best_estimator, n_jobs=1)
            bag = BaggingClassifier(
                estimator=MyPipeline(base_est.steps),
                n_estimators=int(bagging_n_estimators),
                max_samples=bagging_max_samples,
                max_features=bagging_max_features,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            # Single fit with the optimal weights from the winning trial.
            bag.fit(X, y, sample_weight=self.study.best_estimator_.sample_weight_)
            self.best_model = Pipeline([('bag', bag)])
        else:
            self.best_model = best_estimator  # MyPipeline([("clf", RF)])

        pruner_type = self.model_params.get("pruner_type", "hyperband")
        self.cv_results = {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'cv_results': cv_results_df,
            'scoring': metric,
            'search_method': 'optuna',
            'pruner_type': pruner_type,
            'n_trials_completed': len([t for t in self.study.trials
                                        if t.state.name == 'COMPLETE']),
            'n_trials_pruned': len([t for t in self.study.trials
                                     if t.state.name == 'PRUNED']),
        }

    def _get_study_config_hash(self) -> str:
        """
        Produce a short deterministic hash covering everything that defines
        the structure of an Optuna study:
    
          - Bagging configuration (changes ensemble architecture)
          - Base model type and fixed parameters (changes what is being tuned)
          - param_grid keys (changes the search space dimensionality)
    
        Any change to these produces a new study name, preventing Optuna from
        loading trials whose parameter space is incompatible with the current run.
    
        Note: av_uniqueness resolution (bagging_max_samples=None) happens before
        this is called, so the hash reflects the actual value used, not None.
        A shift in av_uniqueness therefore correctly starts a new study, since
        the bootstrap sample size is part of the experiment definition.
        """
        import hashlib, json
    
        # ── Bagging config ───────────────────────────────────────────────────────
        bagging_config = {
            "sequential":   self.model_params.get("bagging_sequential", False),
            "n_estimators": self.model_params.get("bagging_n_estimators", 0),
            "max_samples":  self.model_params.get("bagging_max_samples", 1.0),
            "max_features": self.model_params.get("bagging_max_features", 1.0),
        }
    
        # ── Base model ───────────────────────────────────────────────────────────
        # Extract the classifier from the pipeline. At this point pipe_clf has
        # already been passed through make_custom_pipeline() in train_model(),
        # so steps[-1][1] is the actual estimator being tuned.
        pipe = self.model_params["pipe_clf"]
        base_clf = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    
        model_config = {
            "type": type(base_clf).__name__,
            # Fixed params only — deep=False avoids hashing nested estimators
            # that may have non-deterministic repr (e.g. warm_start state).
            # We only care about the model's structural identity, not trained state.
            "params": {
                k: str(v)
                for k, v in base_clf.get_params(deep=False).items()
            },
        }
    
        # ── Search space structure ───────────────────────────────────────────────
        # Hash the keys only, not the distribution objects themselves.
        # Changing which hyperparameters are searched changes the trial structure
        # and makes old trials incompatible. Changing distribution bounds does not
        # change structure, so we deliberately exclude the values here —
        # Optuna handles bound changes gracefully within the same study.
        param_grid = self.model_params.get("param_grid", {})
        search_space = sorted(param_grid.keys())
    
        combined = {
            "bagging": bagging_config,
            "model":   model_config,
            "search":  search_space,
        }
    
        return hashlib.md5(
            json.dumps(combined, sort_keys=True).encode()
        ).hexdigest()[:6] 
        
    def _apply_sequential_bagging(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tuned_pipeline,
        sample_weight: pd.Series = None,
    ) -> Pipeline:
        """
        Wrap a tuned base pipeline in SequentiallyBootstrappedBaggingClassifier.

        Called by both training paths when bagging_sequential=True.

        Sequential bootstrapping (López de Prado, Chapter 4) draws bootstrap
        samples proportionally to average label uniqueness rather than uniformly.
        This reduces the redundancy that arises when overlapping triple-barrier
        labels are resampled together, producing a less correlated ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Full training feature matrix.
        y : pd.Series
            Full training labels.
        tuned_pipeline : MyPipeline
            The HPO-optimised base classifier wrapped in MyPipeline.
        sample_weight : pd.Series, optional
            Sample weights to pass to fit(). For the sklearn path this is
            self.sample_weight; for the Optuna path it is
            study.best_estimator_.sample_weight_ (the winning trial's weights).

        Returns
        -------
        Pipeline
            sklearn Pipeline whose single step is the fitted
            SequentiallyBootstrappedBaggingClassifier.
        """
        bagging_n = self.model_params.get('bagging_n_estimators', 0)
        bagging_samples = self.model_params.get('bagging_max_samples', 1.0)
        bagging_feats = self.model_params.get('bagging_max_features', 1.0)
        random_state = self.model_params.get('random_state', 1)

        # n_jobs=1 on the base estimator avoids nested parallelism;
        # the outer SequentiallyBootstrappedBaggingClassifier parallelises across bags.
        base_est = set_pipeline_params(tuned_pipeline, n_jobs=1)

        seq_bag = SequentiallyBootstrappedBaggingClassifier(
            base_estimator=MyPipeline(base_est.steps),
            n_estimators=int(bagging_n),
            max_samples=bagging_samples,
            max_features=bagging_feats,
            samples_info_sets=self.events['t1'],
            price_bars_index=self.bar_data.index,
            random_state=random_state,
        )

        if sample_weight is not None:
            seq_bag.fit(X, y, sample_weight=sample_weight)
        else:
            seq_bag.fit(X, y)

        return Pipeline([('seq_bag', seq_bag)])

    def analyze_features(self):
        from .weighted_estimator import _WeightedEstimator

        # best_model is now preprocessor → clf/bag. The last step is the classifier.
        clf = self.best_model.steps[-1][1]
        feat_names = self._get_feature_names()

        if isinstance(clf, SequentiallyBootstrappedBaggingClassifier):
            # Each bag is a MyPipeline([("clf", RF)]). Average importances across bags.
            importances = np.mean([
                est.steps[-1][1].feature_importances_
                for est in clf.estimators_
            ], axis=0)
        elif isinstance(clf, BaggingClassifier):
            # Standard bagging path. Each bag is a MyPipeline([("clf", RF)]).
            importances = np.mean([
                est.steps[-1][1].feature_importances_
                for est in clf.estimators_
            ], axis=0)
        elif isinstance(clf, _WeightedEstimator):
            # Optuna no-bagging path (edge case if preprocessor prepend is removed).
            importances = clf.base_estimator.feature_importances_
        else:
            importances = clf.feature_importances_

        self.feature_importance = pd.DataFrame({
            "feature": feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        self.metrics = {
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
            "training_samples": len(self.bar_data),
            "feature_count": len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme,
            "average_uniqueness": self.events["tW"].mean(),
            "completed_steps": self.completed_steps,
        }

    def _get_feature_names(self):
        # preprocessed_features already has constant and duplicate columns removed.
        # Since the preprocessor is now part of best_model, this is the authoritative
        # column list for both training and inference.
        if self.preprocessed_features is None:
            return []
        return self.preprocessed_features.columns.tolist()

    def _save_all_artifacts(self):
        metadata = {
            "strategy": self.strategy,
            "feature_names": self._get_feature_names(),
            "use_optuna": self.model_params.get("use_optuna", False),
            "pipeline_version": self.pipeline_version
        }
        self.file_manager.save_model(self.best_model, metadata)
        self.file_manager.save_dataframe(self.preprocessed_features, "features")
        self.file_manager.save_dataframe(self.events, "events")
        if self.sample_weight is not None:
            self.file_manager.save_dataframe(self.sample_weight.to_frame("weight"), "weights")
        self.file_manager.save_object(self.metrics, "metrics")

    def _generate_analysis_reports(self):
        """Generates hyperparameter analysis, importance plots, and HTML summary."""
        try:
            if self.cv_results and "cv_results" in self.cv_results:
                cv_df = pd.DataFrame(self.cv_results["cv_results"])
                generate_complete_hyperparameter_report(
                    cv_results=cv_df, strategy_config=self.config, output_dir=self.file_paths["reports"]
                )
            self._generate_training_summary_html()
            if self.study is not None:
                self._generate_optuna_report()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    def _generate_optuna_report(self):
        """
        Save a self-contained HTML report with Optuna visualisation plots.

        Four plots are included:
          1. Optimization history — objective value vs trial number, showing
             convergence and the improvement trajectory over the study.
          2. Intermediate values — fold-by-fold scores for each trial, making
             the pruner's decisions visible: pruned trials end early.
          3. Hyperparameter importances — fANOVA-based ranking of which parameters
             drove score variance. Parameters near the top deserve tighter search
             bounds on a subsequent study.
          4. Parallel coordinates — one line per completed trial, coloured by
             score. Crossing lines reveal parameter interactions; parallel lines
             indicate an insensitive parameter.

        Additionally, the custom baseline comparison plot from optuna_hyper_fit.py
        is saved as a PNG alongside the HTML report.

        The report is saved to file_paths['reports'] / 'optuna_study_report.html'.
        """
        import optuna.visualization as vis
        from plotly.io import to_html

        report_path = self.file_paths['reports'] / 'optuna_study_report.html'
        study = self.study

        n_complete = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        n_pruned   = len([t for t in study.trials if t.state.name == 'PRUNED'])

        plot_specs = [
            ('Optimization History',
             'Objective value per trial. Dashed line shows the running best.',
             lambda: vis.plot_optimization_history(study)),
            ('Fold Scores per Trial (Intermediate Values)',
             'Each line is one trial. Trials that end early were pruned by HyperbandPruner.',
             lambda: vis.plot_intermediate_values(study)),
            ('Hyperparameter Importances',
             'fANOVA estimate of each parameter\'s contribution to score variance.',
             lambda: vis.plot_param_importances(study)),
            ('Parallel Coordinates',
             'One line per completed trial, coloured by score. '
             'Converging lines on the right indicate the high-scoring region.',
             lambda: vis.plot_parallel_coordinate(study)),
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
                fig = plot_fn()
                fig.update_layout(
                    paper_bgcolor='#1e293b',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                )
                plot_html = to_html(fig, include_plotlyjs='cdn', full_html=False)
                html_parts.append(
                    f'<div class="plot">'
                    f'<h2>{title}</h2>'
                    f'<p class="caption">{caption}</p>'
                    f'{plot_html}'
                    f'</div>'
                )
            except Exception as e:
                logger.warning(f"Optuna plot '{title}' failed: {e}")

        html_parts.append('</body></html>')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        logger.info(f"Optuna study report saved: {report_path}")

        # Baseline comparison plot (matplotlib) saved as a companion PNG
        try:
            from ..cross_validation.optuna_hyper_fit import plot_model_vs_baseline
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plot_model_vs_baseline(study, self.events['bin'], self.events)
            baseline_path = self.file_paths['reports'] / 'optuna_baseline_comparison.png'
            plt.savefig(baseline_path, dpi=150, bbox_inches='tight',
                        facecolor='#0f172a', edgecolor='none')
            plt.close()
            logger.info(f"Baseline comparison plot saved: {baseline_path}")
        except Exception as e:
            logger.warning(f"Baseline plot failed: {e}")

    def _generate_training_summary_html(self):
        """Constructs a comprehensive HTML training report."""
        try:
            report_path = self.file_paths["reports"] / "training_summary.html"

            # Use .get() with defaults to prevent KeyErrors during dict extraction
            best_score = self.cv_results.get("best_score", 0)
            # Detect backend based on presence of study attribute
            search_method = "Optuna" if self.study is not None else "Scikit-Learn"

            html_content = f"""
            <html>
            <head>
                <title>Training Report - {self.symbol}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f1f5f9; padding: 40px; line-height: 1.6; }}
                    .container {{ max-width: 900px; margin: auto; }}
                    h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
                    .card {{ background-color: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #334155; }}
                    th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }}
                    .metric {{ font-size: 1.5rem; font-weight: 700; color: #22c55e; }}
                    .label {{ color: #94a3b8; font-size: 0.9rem; }}
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
                                <td><span class="label">Primary Metric ({self.cv_results.get('scoring', 'F1')})</span><br><span class="metric">{best_score:.4f}</span></td>
                                <td><span class="label">Backend</span><br><strong>{search_method}</strong></td>
                            </tr>
                            <tr>
                                <td><span class="label">Training Samples</span><br><strong>{len(self.events)}</strong></td>
                                <td><span class="label">Average Uniqueness</span><br><strong>{self.events['tW'].mean():.4f}</strong></td>
                            </tr>
                        </table>
                    </div>

                    <div class="card">
                        <h2>Weighting Logic</h2>
                        <p><strong>Selected Scheme:</strong> {self.best_weighting_scheme or "Standard/Time-Decay"}</p>
                        <p class="label">Weights were optimized via Purged-KFold to minimize serial correlation leakage.</p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(report_path, "w") as f:
                f.write(html_content)
            logger.info(f"Generated HTML summary report: {report_path}")

        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")

    def check_contamination(self):
        print("\n" + "=" * 70)
        print("DATA CONTAMINATION CHECK")
        print("=" * 70)
        print_contamination_report()

    def get_data_summary(self) -> pd.DataFrame:
        summary_data = []
        components = [
            ("bar_data", self.bar_data),
            ("features", self.features),
            ("preprocessed_features", self.preprocessed_features),
            ("events", self.events),
            ("meta_features", self.meta_features),
            ("sample_weight", self.sample_weight),
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
                    "Type": dtype,
                    "Rows": shape[0] if isinstance(shape, tuple) else shape,
                    "Columns": shape[1] if isinstance(shape, tuple) and len(shape) > 1 else columns,
                    "Memory (MB)": (
                        data.memory_usage(deep=True).sum() / (1024**2)
                        if hasattr(data, "memory_usage") else "N/A"
                    ),
                })
        return pd.DataFrame(summary_data)


def get_model_type(model):
    types = {
        "RandomForestClassifier": "rf",
        "SequentiallyBootstrappedBaggingClassifier": "seq_rf",
        }
    name = type(model).__name__
    return types.get(name, name.replace("Classifier", ""))


def is_tree(estimator):
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))
