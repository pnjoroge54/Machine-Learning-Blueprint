from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..cache import (
    cacheable,
    get_cache_monitor,
    log_data_access,
    print_contamination_report,
)

# from ..cache.unified_cacheable import cached
from ..data_structures.bars import calculate_ticks_per_period, make_bars
from ..labeling.triple_barrier import (
    add_vertical_barrier,
    get_event_weights,
    triple_barrier_labels,
)
from ..mt5.load_data import load_tick_data, save_data_to_parquet
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized
from ..strategies.signal_processing import get_entries
from ..strategies.signals import BaseStrategy
from ..util.misc import value_counts_data
from ..util.volatility import get_daily_vol


class TickDataLoader:
    """
    Loader for tick-level bid/ask data with local caching.

    Notes
    -----
    - Uses in-memory cache keyed by (symbol, start_date, end_date, account_name).
    - Falls back to MT5 fetch if parquet data is missing.
    """

    def __init__(self):
        self._cache = {}

    def get_tick_data(self, symbol, start_date, end_date, account_name):
        """
        Retrieve tick-level bid/ask data with local caching.

        Parameters
        ----------
        symbol : str
            Trading instrument symbol (e.g., 'EURUSD').
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        account_name : str
            MT5 account identifier for data retrieval.

        Returns
        -------
        pd.DataFrame
            Tick data with columns ['bid', 'ask'] indexed by timestamp.

        Notes
        -----
        - Typical performance: ~0.5s for cached retrieval.
        """
        key = (symbol, start_date, end_date, account_name)
        if key in self._cache:
            return self._cache[key]

        tick_params = dict(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            account_name=account_name,
            columns=["bid", "ask"],
            verbose=False,
        )
        df = load_tick_data(**tick_params)
        if df.empty:
            logger.info("Data not found on drive, fetching from MT5...")
            save_data_to_parquet(symbol, start_date, end_date, account_name)
            df = load_tick_data(**tick_params)

        self._cache[key] = df
        return df


loader = TickDataLoader()


@cacheable()
def get_bar_size(tick_df, bar_size):
    """
    Compute tick-based bar size.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with bid/ask prices.
    bar_size : str
        Bar size specification (e.g., '1min', '5min').

    Returns
    -------
    int
        Number of ticks per period.
    """
    return calculate_ticks_per_period(tick_df, bar_size)


@cacheable(time_aware=True)
def load_and_prepare_training_data(
    symbol, start_date, end_date, account_name, bar_type, bar_size, price
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
        Price type ('bid', 'ask', 'mid').

    Returns
    -------
    pd.DataFrame
        Constructed bars indexed by timestamp.

    Notes
    -----
    - Logs data access for contamination tracking.
    - Cached for reproducibility.
    """

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


@cacheable()
def create_feature_engineering_pipeline(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute engineered features with caching.

    Parameters
    ----------
    data : pd.DataFrame
        Input bar data.
    config : dict
        Feature configuration.
        Expected keys:
        - func : callable
            Function that computes features from a DataFrame.
        - params : dict
            Parameters passed to `func`.

    Returns
    -------
    pd.DataFrame
        Feature matrix.
    """
    func = config["func"]
    features = func(data, **config["params"])
    return features


@cacheable()
def generate_events_triple_barrier(
    data: pd.DataFrame,
    target_lookback: int,
    strategy: BaseStrategy,
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
    target_lookback : int
        Lookback window for volatility estimation.
    strategy : BaseStrategy
        Strategy instance implementing `generate_signals()`.
    profit_target : float, default=1
        Profit-taking threshold multiplier.
    stop_loss : float, default=1
        Stop-loss threshold multiplier.
    max_holding_period : dict, default={'days': 1}
        Maximum holding period for vertical barrier.
    min_ret : float, default=0.0
        Minimum return threshold.
    vertical_barrier_zero : bool, default=True
        Allow zero-length vertical barriers.
    filter_as_series : bool, default=True
        Pass volatility threshold as series instead of scalar.

    Returns
    -------
    pd.DataFrame
        Event labels with columns:
        - 'bin' : {-1, 0, 1} classification
        - 't1'  : vertical barrier timestamps
        - 'w'   : sample weights
        - 'tW'  : uniqueness weights
    """
    # Compute barriers
    close = data["close"]
    target = get_daily_vol(close, target_lookback)
    filter_threshold = target if filter_as_series else target.mean()
    side, t_events = get_entries(strategy, data, filter_threshold)
    vb = add_vertical_barrier(t_events, close, **max_holding_period)
    events = triple_barrier_labels(
        close,
        target,
        t_events,
        vertical_barrier_times=vb,
        side_prediction=side,
        pt_sl=[profit_target, stop_loss],
        min_ret=min_ret,
        min_pct=0.05,
        vertical_barrier_zero=vertical_barrier_zero,
        drop=True,
        verbose=False,
    )
    events = get_event_weights(events, close)
    return events


@cacheable()
def compute_sample_weights_time_decay(
    events: pd.DataFrame,
    close: pd.Series,
    attribution: str = None,
    decay_factor: float = 0.95,
    linear: bool = True,
) -> pd.Series:
    """
    Compute sample weights with time decay.

    Parameters
    ----------
    events : pd.DataFrame
        Event labels with uniqueness weights.
    close : pd.Series
        Close price series.
    attribution : str, optional
        Attribution mode ('return', 'uniqueness', or None).
    decay_factor : float, default=0.95
        Decay factor for time weighting.
    linear : bool, default=True
        Use linear decay instead of exponential.

    Returns
    -------
    pd.Series
        Time-decay sample weights.

    Notes
    -----
    - First run: ~5s; cached: ~0.1s (≈50x speedup).
    """
    weights = get_weights_by_time_decay_optimized(
        events,
        close.index,
        last_weight=decay_factor,
        linear=linear,
        av_uniqueness=events["tW"],
        verbose=False,
    )
    if attribution == "return":
        return weights * events["w"]
    elif attribution == "uniqueness":
        return weights * events["tW"]
    else:
        return weights


def train_model_with_cv(
    features: pd.DataFrame,
    events: pd.DataFrame,
    sample_weights: np.ndarray,
    pipe_clf: Pipeline,
    param_grid: Dict,
    cv_splits: int = 5,
    bagging_n_estimators: int = 0,
    bagging_max_samples: float = 1.0,
    bagging_max_features: float = 1.0,
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    pct_embargo: float = 0.01,
    random_state: int = None,
    verbose: bool = False,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train model with cross-validation using cached hyperparameter search.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    events : pd.DataFrame
        Event labels.
    sample_weights : np.ndarray
        Sample weights aligned with events.
    pipe_clf : sklearn.Pipeline
        Pipeline including classifier.
    param_grid : dict
        Hyperparameter grid for search.
    cv_splits : int, default=5
        Number of CV splits.
    bagging_n_estimators : int, default=0
        Number of bagging estimators.
    bagging_max_samples : float, default=1.0
        Max samples for bagging.
    bagging_max_features : float, default=1.0
        Max features for bagging.
    rnd_search_iter : int, default=0
        Randomized search iterations.
    n_jobs : int, default=-1
        Parallel jobs.
    pct_embargo : float, default=0.01
        Embargo percentage for purging CV splits.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbosity flag.

    Returns
    -------
    best_model : RandomForestClassifier
        Trained best model.
    cv_results : dict
        Cross-validation results.

    Notes
    -----
    - First run: ~300s; cached: ~2s (≈150x speedup).
    - Prevents data leakage via time-aware caching.
    """
    from ..cross_validation.hyperfit import clf_hyper_fit_auto_cache

    train_idx = features.dropna().index.intersection(events.index)
    X = features.loc[train_idx]
    y = events.loc[train_idx, "bin"]
    t1 = events.loc[train_idx, "t1"]
    w = sample_weights.loc[train_idx]

    # Set max_samples to average uniqueness to prevent overfitting
    if isinstance(pipe_clf.steps[-1][1], RandomForestClassifier):
        av_uniqueness = events.loc[train_idx, "tW"].mean()
        pipe_clf.steps[-1][1].set_params(max_samples=av_uniqueness)

    best_model, cv_results = clf_hyper_fit_auto_cache(
        X,
        y,
        t1,
        pipe_clf,
        param_grid,
        cv_splits,
        bagging_n_estimators,
        bagging_max_samples,
        bagging_max_features,
        rnd_search_iter,
        n_jobs,
        pct_embargo,
        random_state,
        verbose,
        sample_weight=w,
    )

    return best_model, cv_results


def develop_production_model(
    symbol: str,
    train_start: str,
    train_end: str,
    data_config: Dict,
    feature_config: Dict,
    label_config: Dict,
    model_params: Dict,
    sample_weight_params: Dict,
    reports: bool = False,
) -> Tuple[RandomForestClassifier, List[str], Dict]:
    """
    End-to-end production model development pipeline with aggressive caching.

    Parameters
    ----------
    symbol : str
        Trading instrument symbol.
    train_start : str
        Training start date ('YYYY-MM-DD').
    train_end : str
        Training end date ('YYYY-MM-DD').
    data_config : dict
        Bar construction config. Keys:
        - bar_type : str ('tick', 'volume', 'time')
        - bar_size : int or str
        - price : str ('bid', 'ask', 'mid')
    feature_config : dict
        Feature engineering config. Keys:
        - func : callable
        - params : dict
    label_config : dict
        Triple-barrier labeling config. Keys:
        - target_lookback : int
        - strategy : BaseStrategy
        - profit_target : float
        - stop_loss : float
        - max_holding_period : dict
        - min_ret : float
        - vertical_barrier_zero : bool
        - filter_as_series : bool
    model_params : dict
        Configuration for model training and cross-validation.
        Expected keys:
        - pipe_clf : sklearn.Pipeline
            Pipeline including classifier (e.g., RandomForestClassifier).
        - param_grid : dict
            Hyperparameter grid for search.
        - cv_splits : int, optional (default=5)
            Number of CV splits.
        - bagging_n_estimators : int, optional (default=0)
            Number of bagging estimators.
        - bagging_max_samples : float, optional (default=1.0)
            Max samples for bagging.
        - bagging_max_features : float, optional (default=1.0)
            Max features for bagging.
        - rnd_search_iter : int, optional (default=0)
            Randomized search iterations.
        - n_jobs : int, optional (default=-1)
            Parallel jobs.
        - pct_embargo : float, optional (default=0.01)
            Embargo percentage for purging CV splits.
        - random_state : int, optional
            Random seed.
        - verbose : bool, optional (default=False)
            Verbosity flag.
    sample_weight_params : dict
        Configuration for sample weighting.
        Expected keys:
        - attribution : str, optional
            Attribution mode ('return', 'uniqueness', or None).
        - decay_factor : float, optional (default=0.95)
            Decay factor for time weighting.
        - linear : bool, optional (default=True)
            Use linear decay instead of exponential.
    Returns
    -------
    model : RandomForestClassifier
        Trained best model.
    features : list of str
        Names of engineered features.
    metrics : dict
        Dictionary containing:
        - 'cv_results' : cross-validation results
        - 'feature_importance' : ranked feature importance DataFrame
        - 'training_samples' : number of samples used
        - 'feature_count' : number of features generated

    Notes
    -----
    Config dictionaries must include the following keys:

    - data_config : {'bar_type', 'bar_size', 'price'}
    - feature_config : {'func', 'params'}
    - label_config : {'target_lookback', 'strategy', 'profit_target', 'stop_loss', ...}
    - model_params : {'pipe_clf', 'param_grid', 'cv_splits', ...}
    - sample_weight_params : {'attribution', 'decay_factor', 'linear'}

    See individual function docstrings for detailed argument descriptions.
    """

    print("\n" + "=" * 70)
    print("PRODUCTION MODEL DEVELOPMENT PIPELINE")
    print("=" * 70)

    # Step 1: Load data (tracked for contamination)
    print("\n[Step 1/6] Loading training data...")
    bars = load_and_prepare_training_data(symbol, train_start, train_end, **data_config)
    print(f"✓ Loaded {len(bars):,} samples from {train_start} to {train_end}")

    # Step 2: Feature engineering (cached - 98.2% hit rate)
    print("\n[Step 2/6] Computing features...")
    features = create_feature_engineering_pipeline(bars, feature_config)
    print(f"✓ Generated {len(features.columns)} features")

    # Step 3: Label generation (cached - 95.7% hit rate)
    print("\n[Step 3/6] Generating events...")
    events = generate_events_triple_barrier(bars, **label_config)
    print(f"✓ Generated events: \n{value_counts_data(events['bin'])}")

    # Step 4: Sample weights (cached)
    print("\n[Step 4/6] Computing sample weights...")
    sample_weights = compute_sample_weights_time_decay(events, bars.close, **sample_weight_params)
    print(f"✓ Computed time-decay weights")

    # Step 5: Model training with CV (cached)
    print("\n[Step 5/6] Training model with cross-validation...")
    best_model, cv_results = train_model_with_cv(features, events, sample_weights, **model_params)
    print(f"✓ Best CV score: {cv_results['best_score']:.4f}")
    print(f"✓ Best params: {cv_results['best_params']}")

    # Step 6: Feature importance analysis
    print("\n[Step 6/6] Analyzing feature importance...")
    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": best_model.named_steps["clf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))

    if reports:
        # Cache performance report
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE REPORT")
        print("=" * 70)
        monitor = get_cache_monitor()
        monitor.print_health_report()

        # Data contamination check
        print("\n" + "=" * 70)
        print("DATA CONTAMINATION CHECK")
        print("=" * 70)
        print_contamination_report()

    metrics = {
        "cv_results": cv_results,
        "feature_importance": feature_importance,
        "training_samples": len(bars),
        "feature_count": len(features.columns),
    }

    return best_model, features.columns.tolist(), metrics
