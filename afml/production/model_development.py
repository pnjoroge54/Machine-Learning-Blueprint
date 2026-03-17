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

Architecture Overview:
This pipeline represents a production-grade implementation of the "Advances in 
Financial Machine Learning" (AFML) framework. It orchestrates the complex 
interaction between labeling, sample weighting, and cross-validation while 
maintaining rigorous data integrity through time-aware caching.
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
# Cached data helpers
# ============================================================================

@cacheable()
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
    """
    Generate trading events using the triple-barrier method.
    Superior to standard fixed-horizon labeling by accounting for price 
    targets and stop-losses.
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
            weights *= decay_vec

        self.sample_weight_ = weights
        self.base_estimator.fit(X, y, sample_weight=weights.loc[valid])
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


@cacheable()
def calculate_rolling_metrics(events, sample_weight, window_sizes=[20, 50]):
    """
    Calculate rolling performance metrics with Numba acceleration.
    Enables meta-labeling by allowing the model to 'know' its own recent accuracy.
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


@cv_cacheable
def get_optimal_sample_weight(
    data_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    features: pd.DataFrame,
    cv_splits: int = 5,
    linear: bool = None,
    decay_factors: Union[list, np.ndarray] = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9],
) -> pd.Series:
    """
    Compute best sample weight with time decay.
    Searches for best weighting scheme (return, uniqueness, or time-decay).
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

    cv_gen = PurgedKFold(n_splits=cv_splits, t1=cont["t1"], pct_embargo=0.01)

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
    Encapsulates the entire production model development pipeline.

    Supports two HPO backends:
        False (default): Uses clf_hyper_fit (GridSearchCV / RandomizedSearchCV).
        True: Uses optimize_trading_model (Optuna + HyperbandPruner).

    Key Features:
    - Time-Aware Caching: Prevents look-ahead bias and redundant computation.
    - Purged Cross-Validation: Avoids leakage from serially correlated labels.
    - Meta-Feature Integration: Self-referential performance metrics.
    - Artifact Management: Unified saving of models, weights, and metrics.
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

        self.bar_data = None
        self.features = None
        self.events = None
        self.sample_weight = None
        self.best_weighting_scheme = None
        self.meta_features = None
        self.preprocessed_features = None
        self.best_model = None
        self.cv_results = None
        self.weight_cv_results = None
        self.feature_importance = None
        self.metrics = None
        self.study = None 

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
        self.cv_splits = model_params["cv_splits"]

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

            if generate_reports: self._generate_analysis_reports()
            if cache_reports: self._display_cache_reports()
            if save and self.best_model is not None: self._save_all_artifacts()

            if verbose: print(f"\n✓ Completed in {pd.Timedelta(seconds=time.time()-time0).round('1s')}")
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
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        if self.file_paths["weights"].exists():
            self.sample_weight = pd.read_parquet(self.file_paths["weights"])
        else:
            self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
                self.bar_data.index, self.events, self.features, self.cv_splits, None, self.decay_factors
            )
            self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
            if self.sample_weight is not None:
                self.file_manager.save_dataframe(self.sample_weight.to_frame("weight"), "weights")
        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        self.meta_features = calculate_rolling_metrics(self.events, self.sample_weight)
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        enhanced = self.features.join(self.meta_features, how="inner").dropna()
        preprocessor = Pipeline([("dcf", DropConstantFeatures()), ("ddf", DropDuplicateFeatures())])
        self.preprocessed_features = preprocessor.fit_transform(enhanced)
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self):
        self.model_params["pipe_clf"] = make_custom_pipeline(self.model_params["pipe_clf"])
        pipe = clone(self.model_params["pipe_clf"])

        if is_tree(pipe.steps[-1][-1]):
            pipe = set_pipeline_params(pipe, max_samples=self.events["tW"].mean())

        if isinstance(pipe.steps[-1][-1], SequentiallyBootstrappedBaggingClassifier):
            pipe = set_pipeline_params(pipe, samples_info_sets=self.events["t1"], price_bars_index=self.bar_data.index)

        self.model_params["pipe_clf"] = pipe
        if self.model_params.get("use_optuna", False):
            self._train_model_optuna(pipe)
        else:
            self._train_model_sklearn(pipe)

        self.best_model = set_pipeline_params(self.best_model, n_jobs=-1)
        self.completed_steps["model_training"] = True

    def _train_model_sklearn(self, pipe):
        params = {k: v for k, v in self.model_params.items() if k not in ("use_optuna", "n_trials", "optuna_timeout")}
        self.best_model, self.cv_results = clf_hyper_fit(
            features=self.preprocessed_features, labels=self.events["bin"], t1=self.events["t1"],
            pipe_clf=pipe, sample_weight=self.sample_weight, **params
        )

    def _train_model_optuna(self, pipe):
        X, y = self.preprocessed_features, self.events["bin"]
        base_clf = pipe.steps[-1][1]
        opt_params = {k: v for k, v in self.model_params.items() if k in ("n_trials", "optuna_timeout", "pruner_type", "study_name", "db_path", "param_grid")}
        
        self.study, cv_results_df = optimize_trading_model(
            classifier=base_clf, X=X, y=y, events=self.events, data_index=self.bar_data.index, 
            n_splits=self.cv_splits, **opt_params, refit=False
        )

        best_est = FinancialModelSuggester.apply_from_params(self.study.best_params, base_clf, self.events, self.bar_data.index)
        best_est.fit(X, y)
        self.best_model = Pipeline([("clf", best_est)])
        self.cv_results = {"best_params": self.study.best_params, "best_score": self.study.best_value, "cv_results": cv_results_df}

    def analyze_features(self):
        feat_names = self._get_feature_names()
        self.feature_importance = pd.DataFrame({
            "feature": feat_names,
            "importance": self.best_model.steps[-1][1].feature_importances_,
        }).sort_values("importance", ascending=False)
        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        self.metrics = {
            "cv_results": self.cv_results, "feature_importance": self.feature_importance,
            "training_samples": len(self.bar_data), "feature_count": len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme, "average_uniqueness": self.events["tW"].mean(),
            "completed_steps": self.completed_steps,
        }

    def _get_feature_names(self):
        if self.best_model is None: return []
        if len(self.best_model) > 1: return self.best_model[:-1].get_feature_names_out().tolist()
        return self.preprocessed_features.columns.tolist()

    def _save_all_artifacts(self):
        metadata = {
            "strategy": self.strategy, "feature_names": self._get_feature_names(),
            "use_optuna": self.model_params.get("use_optuna", False), "pipeline_version": self.pipeline_version
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
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

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
    types = {"RandomForestClassifier": "rf", "SequentiallyBootstrappedBaggingClassifier": "seq_rf"}
    return types.get(type(model).__name__, "model")

def is_tree(estimator):
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))
