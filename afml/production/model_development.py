"""
model_development.py
--------------------
Production model development pipeline with optional Optuna HPO integration.

The pipeline now supports two training paths controlled by
`model_params['use_optuna']`:

    False (default): clf_hyper_fit via GridSearchCV / RandomizedSearchCV
    True:            optimize_trading_model via Optuna + HyperbandPruner

When use_optuna=True the following changes apply:
  - Weight computation for HPO is handled internally by _WeightedEstimator;
    get_optimal_sample_weight is still run for meta-features and reporting.
  - train_model dispatches to _train_model_optuna.
  - self.study is populated with the completed Optuna study for visualization.
  - Refit uses FinancialModelSuggester.apply_from_params (deterministic).
"""

import inspect
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ..cache import cacheable, cv_cacheable, get_cache_monitor, log_data_access, print_contamination_report
from ..cross_validation import PurgedKFold, clf_hyper_fit
from ..cross_validation.cross_validation import ml_cross_val_score
from ..cross_validation.hyper_fit_analysis import generate_complete_hyperparameter_report
from ..cross_validation.optuna_hyper_fit import (
    FinancialModelSuggester,
    optimize_trading_model,
    optuna_to_cv_results,
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
from ..util.pipelines import make_custom_pipeline, set_pipeline_params
from .utils import ModelFileManager



# ============================================================================
# Cached data helpers (unchanged from original)
# ============================================================================

@cacheable()
def get_bar_size(tick_df, bar_size):
    return calculate_ticks_per_period(tick_df, bar_size)


@cacheable(time_aware=True)
def load_and_prepare_training_data(
    symbol, start_date, end_date, account_name, bar_type, bar_size, price, path=None
):
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


def load_ticks(symbol, start_date, end_date, account_name, path=None):
    if path is not None:
        loader.path = path
    return loader.get_tick_data(symbol, start_date, end_date, account_name)


@cacheable(time_aware=True)
def create_feature_engineering_pipeline(
    data: pd.DataFrame, feature_config: Dict, data_config: Dict
) -> pd.DataFrame:
    func = feature_config["func"]
    features = func(data, **feature_config["params"])
    time_feat = get_time_features(
        data, timeframe=data_config["bar_size"], bar_type=data_config["bar_type"]
    )
    return features.join(time_feat, how="left")


@cacheable()
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
# _WeightedEstimator (unchanged)
# ============================================================================

class _WeightedEstimator(BaseEstimator, ClassifierMixin):
    """Static class for weighted estimators - essential for caching."""

    def __init__(
        self,
        base_estimator,
        events,
        data_index,
        scheme="unweighted",
        decay=1.0,
        linear=True,
        **params,
    ):
        self.base_estimator = base_estimator
        self.base_estimator.set_params(**params)
        self.scheme = scheme
        self.decay = decay
        self.linear = linear
        self.events = events
        self.data_index = data_index

    def fit(self, X, y):
        if self.scheme == "uniqueness":
            weights = self.events["tW"]
        elif self.scheme == "return":
            weights = self.events["w"]
        else:
            weights = pd.Series(np.ones(len(y)), index=y.index)

        valid = X.index.intersection(y.index)
        X, y = X.loc[valid], y.loc[valid]

        if self.decay != 1.0:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=self.events,
                close_index=self.data_index,
                last_weight=self.decay,
                linear=self.linear,
                av_uniqueness=self.events["tW"],
            )
            w *= decay_vec

        self.sample_weight_ = weights
        self.base_estimator.fit(X, y, sample_weight=w.loc[valid])
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = {
            "scheme": self.scheme,
            "decay": self.decay,
            "linear": self.linear,
            "base_estimator": self.base_estimator,
            "events": self.events,
            "data_index": self.data_index,
        }
        if deep:
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f"base_{k}": v for k, v in base_params.items()})
        return params

    def set_params(self, **params):
        base_params = {}
        for key in list(params.keys()):
            if key.startswith("base_"):
                base_params[key[5:]] = params.pop(key)

        for key in ["scheme", "decay", "linear", "base_estimator", "events", "data_index"]:
            if key in params:
                setattr(self, key, params.pop(key))

        if base_params:
            self.base_estimator.set_params(**base_params)
        return self


def weighted_estimator(base_estimator, events, data_index):
    return _WeightedEstimator(base_estimator=base_estimator, events=events, data_index=data_index)


# ============================================================================
# Weight helpers (unchanged)
# ============================================================================

def best_weighting_scheme(
    classifier, X, y, cv_gen, scoring, sample_weight,
    scheme=None, best_score=0, best_scheme=None, cv_results=pd.DataFrame(),
):
    scores = ml_cross_val_score(
        classifier, X, y, cv_gen,
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


@cv_cacheable
def get_optimal_sample_weight(
    data_index, events, features, cv_splits=5, linear=None,
    decay_factors=[0.001, 0.1, 0.25, 0.5, 0.75, 0.9],
):
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
        n_jobs=-1,
    )
    cv_gen = PurgedKFold(n_splits=cv_splits, t1=cont["t1"], pct_embargo=0.01)

    weights = {
        "return": cont["w"],
        "unweighted": pd.Series(1.0, index=cont.index),
        "uniqueness": cont["tW"],
    }

    best_score = 0
    best_scheme = None
    cv_results = pd.DataFrame()
    scoring = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"

    for scheme, weight in tqdm(weights.items(), desc="Analyzing weighting schemes"):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight,
            scheme, best_score, best_scheme, cv_results,
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

    for scheme, weight in tqdm(
        time_decay_weights.items(),
        desc=f"Analyzing time-decay weighting for {best_scheme}",
    ):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier), X, y, cv_gen, scoring, weight,
            scheme, best_score, best_scheme, cv_results,
        )

    weights.update(time_decay_weights)
    best_weight = weights[best_scheme]

    cv_results = {
        "best_score": best_score,
        "cv_results": cv_results,
        "scoring": scoring,
        "best_scheme": best_scheme,
    }
    return best_weight, cv_results


# ============================================================================
# Numba rolling metrics (unchanged)
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _rolling_metrics_numba(y_true, y_pred, weights, window):
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


@cacheable()
def calculate_rolling_metrics(events, sample_weight, window_sizes=[20, 50]):
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


# ============================================================================
# Helpers
# ============================================================================

def is_tree(estimator):
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))


def get_model_type(model):
    model_type = {
        "RandomForestClassifier": "rf",
        "SequentiallyBootstrappedBaggingClassifier": "seq_rf",
    }
    return model_type[type(model).__name__]


def train_model_with_cv(
    features, events, sample_weight, pipe_clf, param_grid,
    cv_splits=5, bagging_n_estimators=0, bagging_max_samples=1.0,
    bagging_max_features=1.0, rnd_search_iter=0, n_jobs=-1,
    pct_embargo=0.02, random_state=None, verbose=0,
):
    valid_index = features.index.intersection(events.index)
    cont = events.loc[valid_index]
    X = features.loc[valid_index]
    y = cont["bin"]
    t1 = cont["t1"]
    w = sample_weight.loc[valid_index]

    return clf_hyper_fit(
        features=X, labels=y, t1=t1, pipe_clf=pipe_clf,
        param_grid=param_grid, cv=cv_splits,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_max_features=bagging_max_features,
        rnd_search_iter=rnd_search_iter, n_jobs=n_jobs,
        pct_embargo=pct_embargo, random_state=random_state,
        verbose=verbose, sample_weight=w,
    )


# ============================================================================
# ModelDevelopmentPipeline — unified pipeline with Optuna integration
# ============================================================================

class ModelDevelopmentPipeline:
    """
    Production model development pipeline.

    Supports two HPO backends controlled by model_params['use_optuna']:

        False (default)
            Uses clf_hyper_fit (GridSearchCV / RandomizedSearchCV).
            Sample weights are passed directly to the scorer and estimator.

        True
            Uses optimize_trading_model (Optuna + HyperbandPruner).
            _WeightedEstimator handles training weight computation internally.
            Return-attribution weights (events['w']) are always used for scoring.
            self.study is populated for post-study visualization.
            FinancialModelSuggester.apply_from_params is used for the final refit.

    In both paths get_optimal_sample_weight still runs (Step 4) to populate
    self.sample_weight for meta-features and reporting.

    Parameters
    ----------
    model_params : dict
        Standard training configuration plus:
        - use_optuna : bool, default=False
            Switch to Optuna HPO backend.
        - n_trials : int, default=100
            Optuna trial budget (only used when use_optuna=True).
        - optuna_timeout : int, default=3600
            Optuna wall-clock timeout in seconds.
        - pruner_type : str, default='hyperband'
            'hyperband' or 'median' (TradingModelPruner).
        - study_name : str, optional
            Optuna study name for SQLite persistence.
        - db_path : str, optional
            SQLite database path (no .db extension).
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

        # Intermediate results
        self.bar_data = None
        self.features = None
        self.events = None
        self.sample_weight = None
        self.best_weighting_scheme = None
        self.weighting_schemes = None
        self.meta_features = None
        self.preprocessed_features = None
        self.best_model = None
        self.cv_results = None
        self.weight_cv_results = None
        self.feature_importance = None
        self.metrics = None
        self.training_metadata = None

        # Optuna-specific results — populated when use_optuna=True
        self.study = None

        if isinstance(model_params["pipe_clf"], Pipeline):
            model = model_params["pipe_clf"].steps[-1][1]
        else:
            model = model_params["pipe_clf"]

        self.model_type = get_model_type(model)

        self.file_manager = ModelFileManager(base_dir)
        self.file_paths = self.file_manager.setup_model_directory(self.config, self.model_type)

        self.linear = None
        self.decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]

        self.completed_steps = {
            "data_loading": False,
            "feature_engineering": False,
            "label_generation": False,
            "weight_computation": False,
            "meta_features": False,
            "model_training": False,
            "analysis": False,
        }

        self.log_file = self.file_paths["logs"] / "pipeline.log"
        self._setup_logging()
        self.cv_splits = model_params["cv_splits"]

    def _setup_logging(self):
        logger.remove()
        logger.add(
            self.log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss:SS} | {name} | {level} | {message}",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss:SS}</green> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{level}</level> | "
                "<yellow>{message}</yellow>"
            ),
            colorize=True,
        )
        self.logger = logger.bind(context=self.__class__.__name__)
        self.logger.info(f"Starting pipeline for {self.symbol}")
        self.logger.info(f"Training period: {self.train_start} to {self.train_end}")
        self.logger.info(f"Output directory: {self.file_paths['base_dir']}")

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self,
        generate_reports: bool = True,
        cache_reports: bool = False,
        save: bool = True,
        export_onnx: bool = False,
        verbose: bool = True,
    ) -> Tuple:
        time0 = time.time()
        self.export_onnx = export_onnx

        use_optuna = self.model_params.get("use_optuna", False)
        backend = "Optuna (HyperbandPruner + TPE)" if use_optuna else "sklearn CV"

        if verbose:
            print("\n" + "=" * 70)
            print("PRODUCTION MODEL DEVELOPMENT PIPELINE")
            print(f"HPO backend: {backend}")
            print("=" * 70)
            print(pd.Series(self.config).to_string(), "\n")

        try:
            if verbose:
                print("\n[Step 1/7] Loading training data...")
            self.load_training_data()

            if verbose:
                print("\n[Step 2/7] Computing features...")
            self.engineer_features()
            if verbose:
                print(f"✓ Generated {len(self.features.columns)} features")

            if verbose:
                print("\n[Step 3/7] Generating events...")
            self.generate_labels()

            if verbose:
                print("\n[Step 4/7] Computing sample weights...")
            self.compute_sample_weights()

            if verbose:
                print("\n[Step 5/7] Computing rolling meta-label features...")
            self.add_meta_features()
            self.preprocess_features()

            if verbose:
                print("\n[Step 6/7] Training model...")
            self.train_model()

            if verbose:
                print("\n[Step 7/7] Analyzing feature importance...")
            self.analyze_features()

            if verbose:
                print("\nTop 10 Features:")
                print(self.feature_importance.head(10).to_string(index=False), "\n")

            self._compile_metrics()

            if generate_reports:
                if verbose:
                    print("\n[Generating Reports] Creating analysis reports...")
                self._generate_analysis_reports()

            if cache_reports:
                self._display_cache_reports()

            if save and self.best_model is not None:
                if verbose:
                    print("\n[Saving] Writing artifacts to disk...")
                self._save_all_artifacts()
                if verbose:
                    print(f"✓ Saved to {self.file_paths['base_dir']}")

            pipeline_duration = time.time() - time0
            if verbose:
                duration_str = str(
                    pd.Timedelta(seconds=pipeline_duration).round("1s")
                ).replace("0 days ", "")
                print(f"\n✓ Pipeline completed in {duration_str}")
                print("=" * 70, "\n")

            return (
                self.best_model,
                self._get_feature_names(),
                self.metrics,
                self.config,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def load_training_data(self):
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        if self.data_config == "tick":
            self.config["tick_bar_size"] = self.bar_data["tick_volume"].iloc[0]
            self.file_manager.save_config(self.config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        self.features = create_feature_engineering_pipeline(
            self.bar_data, self.feature_config, self.data_config
        )
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        self.events = generate_events_triple_barrier(
            self.bar_data, self.strategy, self.target_config, **self.label_config
        )
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        """
        Compute optimal sample weights.

        When use_optuna=True, these weights are NOT passed to HPO — _WeightedEstimator
        handles training weights internally. They are still computed here for
        meta-feature generation and reporting, and cached to avoid re-computation.
        """
        if self.file_paths["weights"].exists():
            self.sample_weight = pd.read_parquet(self.file_paths["weights"])
        else:
            self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
                self.bar_data.index, self.events, self.features,
                self.cv_splits, self.linear, self.decay_factors,
            )
            self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
            if self.sample_weight is not None:
                self.file_manager.save_dataframe(
                    self.sample_weight.to_frame("weight"), "weights"
                )
        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        self.meta_features = calculate_rolling_metrics(self.events, self.sample_weight)
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        enhanced_features = self.features.join(self.meta_features, how="inner").dropna()
        preprocessor = Pipeline([
            ("dcf", DropConstantFeatures()),
            ("ddf", DropDuplicateFeatures()),
        ])
        self.preprocessed_features = preprocessor.fit_transform(enhanced_features)
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self):
        """
        Dispatch to the appropriate HPO backend.

        When use_optuna=True:
            - optimize_trading_model is called with the base estimator and events.
            - _WeightedEstimator jointly optimizes weight_scheme, weight_decay,
              and weight_linear alongside the model hyperparameters.
            - Return-attribution weights (events['w']) are always used for scoring
              regardless of the sampled weight_scheme.
            - FinancialModelSuggester.apply_from_params performs the final refit.
            - self.study is populated for post-study visualization.

        When use_optuna=False:
            - clf_hyper_fit is called with the pre-computed sample_weight.
            - Behaviour is identical to the original pipeline.
        """
        self.model_params["pipe_clf"] = make_custom_pipeline(self.model_params["pipe_clf"])
        pipe = clone(self.model_params["pipe_clf"])

        if is_tree(pipe.steps[-1][-1]):
            av_uniqueness = self.events["tW"].mean()
            pipe = set_pipeline_params(pipe, max_samples=av_uniqueness)

        if isinstance(pipe.steps[-1][-1], SequentiallyBootstrappedBaggingClassifier):
            pipe = set_pipeline_params(
                pipe,
                samples_info_sets=self.events["t1"],
                price_bars_index=self.bar_data.index,
            )

        self.model_params["pipe_clf"] = pipe

        if self.model_params.get("use_optuna", False):
            self._train_model_optuna(pipe)
        else:
            self._train_model_sklearn(pipe)

        self.best_model = set_pipeline_params(self.best_model, n_jobs=-1)
        self.completed_steps["model_training"] = True

    def _train_model_sklearn(self, pipe):
        """
        HPO via clf_hyper_fit (GridSearchCV / RandomizedSearchCV).
        Sample weights flow directly into both fit and scoring.
        """
        self.best_model, self.cv_results = train_model_with_cv(
            self.preprocessed_features,
            self.events,
            self.sample_weight,
            **{
                k: v for k, v in self.model_params.items()
                if k not in ("use_optuna", "n_trials", "optuna_timeout",
                             "pruner_type", "study_name", "db_path")
            },
        )
        logger.info(
            f"sklearn CV complete. Best score: {self.cv_results['best_score']:.4f}"
        )

    def _train_model_optuna(self, pipe):
        """
        HPO via optimize_trading_model (Optuna + HyperbandPruner).

        Key differences from the sklearn path:
          - The base estimator passed to Optuna is the raw (unwrapped) classifier,
            not a _WeightedEstimator. FinancialModelSuggester wraps it internally.
          - weight_scheme, weight_decay, and weight_linear are sampled jointly with
            the model hyperparameters inside each trial.
          - Scoring always uses events['w'] (return-attribution) regardless of
            which training scheme was sampled for that trial.
          - After the study, FinancialModelSuggester.apply_from_params reconstructs
            the best configuration deterministically and fits it on the full dataset.
        """
        X = self.preprocessed_features
        y = self.events["bin"]

        # Extract the raw classifier from the pipeline for _WeightedEstimator
        base_clf = pipe.steps[-1][1]

        n_trials = self.model_params.get("n_trials", 100)
        timeout = self.model_params.get("optuna_timeout", 3600)
        pruner_type = self.model_params.get("pruner_type", "hyperband")
        study_name = self.model_params.get("study_name", None)
        db_path = self.model_params.get("db_path", None)
        param_distributions = self.model_params.get("param_grid", {})
        random_state = self.model_params.get("random_state", 42)

        self.study, cv_results_df = optimize_trading_model(
            classifier=base_clf,
            X=X,
            y=y,
            events=self.events,
            data_index=self.bar_data.index,
            param_distributions=param_distributions,
            n_trials=n_trials,
            timeout=timeout,
            n_splits=self.cv_splits,
            pruner_type=pruner_type,
            metric="f1" if set(y.unique()) == {0, 1} else "neg_log_loss",
            study_name=study_name,
            db_path=db_path,
            random_state=random_state,
            refit=False,  # We handle refit here for full control
        )

        best_score = self.study.best_value
        best_params = self.study.best_params

        logger.info(
            f"Optuna study complete. "
            f"Best score: {best_score:.4f}  "
            f"Best params: {best_params}"
        )

        # Deterministic refit on full dataset using the best parameters
        best_estimator = FinancialModelSuggester.apply_from_params(
            params=best_params,
            base_model=base_clf,
            events=self.events,
            data_index=self.bar_data.index,
        )
        best_estimator.fit(X, y)

        # Wrap in Pipeline for interface compatibility with the rest of the pipeline
        from sklearn.pipeline import Pipeline as _Pipeline
        self.best_model = _Pipeline([("clf", best_estimator)])

        self.cv_results = {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": cv_results_df,
            "scoring": "f1" if set(y.unique()) == {0, 1} else "neg_log_loss",
            "search_method": "optuna",
            "pruner_type": pruner_type,
            "n_trials_completed": len([
                t for t in self.study.trials
                if t.state.name == "COMPLETE"
            ]),
            "n_trials_pruned": len([
                t for t in self.study.trials
                if t.state.name == "PRUNED"
            ]),
        }

    def analyze_features(self):
        features_columns = (
            self.best_model[:-1].get_feature_names_out()
            if self.best_model is not None and len(self.best_model) > 1
            else self.preprocessed_features.columns.to_list()
        )
        clf = self.best_model.steps[-1][1]
        # _WeightedEstimator wraps the base estimator
        if hasattr(clf, "base_estimator"):
            clf = clf.base_estimator

        self.feature_importance = pd.DataFrame({
            "feature": features_columns,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        self.metrics = {
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
            "training_samples": len(self.bar_data),
            "feature_count": len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme,
            "label_distribution": value_counts_data(self.events["bin"]),
            "average_uniqueness": self.events["tW"].mean(),
            "sample_weight_stats": (
                self.sample_weight.describe().to_dict()
                if self.sample_weight is not None else None
            ),
            "events_count": len(self.events),
            "features_shape": self.preprocessed_features.shape,
            "completed_steps": self.completed_steps,
        }

    def _get_feature_names(self):
        if self.best_model is None:
            return []
        if len(self.best_model) > 1:
            return self.best_model[:-1].get_feature_names_out().tolist()
        return self.preprocessed_features.columns.tolist()

    def _generate_analysis_reports(self):
        try:
            if self.cv_results and "cv_results" in self.cv_results:
                cv_results_df = (
                    self.cv_results["cv_results"]
                    if isinstance(self.cv_results["cv_results"], pd.DataFrame)
                    else pd.DataFrame(self.cv_results["cv_results"])
                )
                report_path = self.file_paths["reports"] / "hyperparameter_analysis_report.md"
                generate_complete_hyperparameter_report(
                    cv_results=cv_results_df,
                    strategy_config=self.config,
                    output_dir=self.file_paths["reports"],
                    filename=report_path.name,
                    target_metric="mean_test_score",
                )
                logger.info(f"Generated hyperparameter report: {report_path}")

            if self.feature_importance is not None:
                import matplotlib.pyplot as plt
                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(12, 8))
                top_features = self.feature_importance.head(20)
                ax.barh(range(len(top_features)), top_features["importance"][::-1])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features["feature"][::-1])
                ax.set_xlabel("Importance")
                ax.set_title(f"Top 20 Feature Importance - {self.symbol}")
                plt.tight_layout()
                plot_path = self.file_paths["plots"] / "feature_importance.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Generated feature importance plot: {plot_path}")

        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    def _save_all_artifacts(self):
        try:
            metadata = {
                "strategy": self.strategy,
                "feature_config": self.feature_config,
                "label_config": self.label_config,
                "target_config": self.target_config,
                "feature_names": self._get_feature_names(),
                "feature_count": len(self._get_feature_names()),
                "training_samples": len(self.events),
                "best_weighting_scheme": self.best_weighting_scheme,
                "pipeline_version": self.pipeline_version,
                "use_optuna": self.model_params.get("use_optuna", False),
                "created_by": "AFML Production Pipeline v4",
            }
            self.file_manager.save_model(self.best_model, metadata)

            if self.features is not None:
                self.file_manager.save_dataframe(self.preprocessed_features, "features")
            if self.events is not None:
                self.file_manager.save_dataframe(self.events, "events")
            if self.sample_weight is not None:
                self.file_manager.save_dataframe(
                    self.sample_weight.to_frame("weight"), "weights"
                )
            if self.metrics:
                self.file_manager.save_object(self.metrics, "metrics")
            if self.export_onnx and self.best_model is not None:
                self.file_manager.save_model_as_onxx(
                    self.best_model, self._get_feature_names(), metadata
                )
            logger.info(f"Saved all artifacts to {self.file_paths['base_dir']}")

        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            raise

    def _display_cache_reports(self):
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE REPORT")
        print("=" * 70)
        monitor = get_cache_monitor()
        monitor.print_health_report()
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
