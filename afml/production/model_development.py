import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import pickle
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

from ..cache import cacheable, get_cache_monitor, log_data_access, print_contamination_report
from ..cross_validation import PurgedKFold, clf_hyper_fit
from ..cross_validation.cross_validation import ml_cross_val_score
from ..cross_validation.hyper_fit_analysis import generate_complete_hyperparameter_report
from ..data_structures.bars import calculate_ticks_per_period, make_bars
from ..ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from ..features.trading_session import get_time_features
from ..labeling.triple_barrier import add_vertical_barrier, get_event_weights, triple_barrier_labels
from ..mt5.load_data import load_tick_data, save_data_to_parquet
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized
from ..strategies.signal_processing import get_entries
from ..strategies.trading_strategies import BaseStrategy
from ..util.misc import date_conversion, value_counts_data
from ..util.pipelines import make_custom_pipeline, set_pipeline_params
from .utils import ModelFileManager


class TickDataLoader:
    """
    Loader for tick-level bid/ask data with intelligent local caching.

    Features:
    1. Smart caching that checks if requested date range is within cached ranges
    2. Handles partial overlaps by reusing available cached data
    3. Memory management with cache size limits
    4. Cache statistics tracking

    Notes
    -----
    - Typical performance: ~0.5s for cached retrieval
    - Memory usage: ~100MB per 1M ticks
    """

    def __init__(self, max_cache_size_mb: int = 5000, max_cached_symbols: int = 20, path: Union[str, Path] = None):
        """
        Initialize the tick data loader.

        Parameters
        ----------
        max_cache_size_mb : int, optional
            Maximum cache size in MB (default: 5000MB)
        max_cached_symbols : int, optional
            Maximum number of symbols to keep in cache (default: 20)
        path : Union[str, Path]
            Path to folder that contains tick data
        """
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (symbol, account_name) -> DataFrame
        self._cache_metadata: Dict[Tuple[str, str], Dict] = {}  # (symbol, account_name) -> metadata
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cached_symbols = max_cached_symbols
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "partial_hits": 0,
            "total_loaded": 0,
        }
        self.path = path

    def get_tick_data(
        self, symbol: str, start_date: str, end_date: str, account_name: str
    ) -> pd.DataFrame:
        """
        Retrieve tick-level bid/ask data with intelligent caching.

        Parameters
        ----------
        symbol : str
            Trading instrument symbol (e.g., 'EURUSD')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        account_name : str
            MT5 account identifier for data retrieval

        Returns
        -------
        pd.DataFrame
            Tick data with columns ['bid', 'ask'] indexed by timestamp

        Notes
        -----
        - Checks if cached data fully covers requested date range
        - If partial coverage exists, loads only missing data
        - Merges cached and newly loaded data seamlessly
        """
        cache_key = (symbol, account_name)
        start_dt, end_dt = date_conversion(start_date, end_date)

        # Check if we have cached data for this symbol/account
        if cache_key in self._cache:
            cached_df = self._cache[cache_key]
            metadata = self._cache_metadata[cache_key]
            cached_start, cached_end = date_conversion(metadata["start_date"], metadata["end_date"])

            # Check if cached data fully covers requested range
            if cached_start <= start_dt and cached_end >= end_dt:
                self.cache_stats["hits"] += 1
                logger.debug(f"Cache hit for {symbol} {start_date} to {end_date}")

                # Return subset of cached data
                mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                return cached_df[mask].copy()

            # Check if there's partial overlap
            if cached_end >= start_dt and cached_start <= end_dt:
                self.cache_stats["partial_hits"] += 1
                logger.debug(f"Partial cache hit for {symbol}")
                return self._load_with_partial_cache(
                    symbol, start_date, end_date, account_name, cache_key
                )

        # No cache hit, load all data
        self.cache_stats["misses"] += 1
        logger.debug(f"Cache miss for {symbol} {start_date} to {end_date}")
        return self._load_and_cache_data(symbol, start_date, end_date, account_name, cache_key)

    def _load_with_partial_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        account_name: str,
        cache_key: Tuple[str, str],
    ) -> pd.DataFrame:
        """
        Load data when we have partial cache coverage.

        Strategy:
        1. Identify what parts of the requested range are already cached
        2. Load only the missing date ranges
        3. Merge cached and new data
        4. Update cache with extended range
        """
        cached_df = self._cache[cache_key]
        cached_start = self._cache_metadata[cache_key]["start_date"]
        cached_end = self._cache_metadata[cache_key]["end_date"]

        start_dt, end_dt = date_conversion(start_date, end_date)

        # Determine what we need to load
        load_ranges = []

        # Check if we need data before cached range
        if start_dt < cached_start:
            load_ranges.append(
                (start_date, (cached_start - timedelta(days=1)).strftime("%Y-%m-%d"))
            )

        # Check if we need data after cached range
        if end_dt > cached_end:
            load_ranges.append(((cached_end + timedelta(days=1)).strftime("%Y-%m-%d"), end_date))

        # Load missing data ranges
        new_data = []
        for load_start, load_end in load_ranges:
            logger.info(f"Loading additional data for {symbol}: {load_start} to {load_end}")
            df_part = self._load_data(symbol, load_start, load_end, account_name)
            if not df_part.empty:
                new_data.append(df_part)

        # Combine all data
        if new_data:
            all_new_data = pd.concat(new_data) if len(new_data) > 1 else new_data[0]
            combined_data = pd.concat([cached_df, all_new_data])
            combined_data = combined_data.sort_index()

            # Update cache with extended range
            new_start = min(start_dt, cached_start)
            new_end = max(end_dt, cached_end)
            self._cache[cache_key] = combined_data
            self._cache_metadata[cache_key] = {
                "start_date": new_start,
                "end_date": new_end,
                "last_accessed": datetime.now(),
                "size_mb": combined_data.memory_usage(deep=True).sum() / (1024**2),
            }

            # Clean cache if needed
            self._clean_cache()

            # Return requested subset
            mask = (combined_data.index >= start_dt) & (combined_data.index <= end_dt)
            return combined_data[mask].copy()
        else:
            # Shouldn't happen, but return cached subset
            mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
            return cached_df[mask].copy()

    def _load_and_cache_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        account_name: str,
        cache_key: Tuple[str, str],
    ) -> pd.DataFrame:
        """
        Load data from source and cache it.
        """
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        df = self._load_data(symbol, start_date, end_date, account_name)
        start_date, end_date = date_conversion(start_date, end_date)

        if not df.empty:
            # Cache the data
            self._cache[cache_key] = df
            self._cache_metadata[cache_key] = {
                "start_date": start_date,
                "end_date": end_date,
                "last_accessed": datetime.now(),
                "size_mb": df.memory_usage(deep=True).sum() / (1024**2),
            }

            # Clean cache if needed
            self._clean_cache()

            self.cache_stats["total_loaded"] += 1

        return df

    def _load_data(
        self, symbol: str, start_date: str, end_date: str, account_name: str
    ) -> pd.DataFrame:
        """
        Load data from parquet file or MT5.
        """
        tick_params = dict(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            account_name=account_name,
            path=self.path,
            columns=["bid", "ask"],
            verbose=False,
        )

        df = load_tick_data(**tick_params)

        if df.empty:
            logger.info("Data not found on drive, fetching from MT5...")
            save_data_to_parquet(symbol, start_date, end_date, account_name)
            df = load_tick_data(**tick_params)

        return df

    def _clean_cache(self):
        """
        Clean cache based on size and LRU policy.
        """
        # Check if we have too many symbols
        if len(self._cache) > self.max_cached_symbols:
            # Remove least recently used
            lru_items = sorted(self._cache_metadata.items(), key=lambda x: x[1]["last_accessed"])

            for key, _ in lru_items[: len(self._cache) - self.max_cached_symbols]:
                del self._cache[key]
                del self._cache_metadata[key]
                logger.debug(f"Removed {key} from cache (LRU policy)")

        # Check total cache size
        total_size = sum(meta["size_mb"] for meta in self._cache_metadata.values())

        if total_size > self.max_cache_size_mb:
            # Remove largest items until under limit
            items_by_size = sorted(
                self._cache_metadata.items(),
                key=lambda x: x[1]["size_mb"],
                reverse=True,
            )

            removed_size = 0
            for key, meta in items_by_size:
                if total_size - removed_size <= self.max_cache_size_mb:
                    break

                removed_size += meta["size_mb"]
                del self._cache[key]
                del self._cache_metadata[key]
                logger.debug(f"Removed {key} from cache (size: {meta['size_mb']:.2f}MB)")

    def clear_cache(self, symbol: Optional[str] = None, account_name: Optional[str] = None):
        """
        Clear cache for specific symbol/account or all cache.

        Parameters
        ----------
        symbol : str, optional
            Symbol to clear cache for
        account_name : str, optional
            Account name to clear cache for
        """
        if symbol is None and account_name is None:
            self._cache.clear()
            self._cache_metadata.clear()
            logger.info("Cleared all cache")
        else:
            keys_to_remove = []
            for key in self._cache.keys():
                sym, acc = key
                if (symbol is None or sym == symbol) and (
                    account_name is None or acc == account_name
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_metadata[key]

            logger.info(f"Cleared cache for {len(keys_to_remove)} items")

    def get_cache_info(self) -> Dict:
        """
        Get cache statistics and information.

        Returns
        -------
        Dict
            Cache information including:
            - total_cached_symbols: Number of symbols in cache
            - total_cache_size_mb: Total cache size in MB
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate percentage
            - cached_symbols: List of cached symbols with date ranges
        """
        total_size = sum(meta["size_mb"] for meta in self._cache_metadata.values())
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        cached_symbols_info = []
        for (symbol, account), meta in self._cache_metadata.items():
            cached_symbols_info.append(
                {
                    "symbol": symbol,
                    "account": account,
                    "date_range": f"{meta['start_date'].date()} to {meta['end_date'].date()}",
                    "size_mb": meta["size_mb"],
                    "last_accessed": meta["last_accessed"],
                }
            )

        return {
            "total_cached_symbols": len(self._cache),
            "total_cache_size_mb": total_size,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "partial_hits": self.cache_stats["partial_hits"],
            "hit_rate": hit_rate,
            "cached_symbols": cached_symbols_info,
        }

    def preload_data(self, symbols: List[str], start_date: str, end_date: str, account_name: str):
        """
        Preload data for multiple symbols into cache.

        Parameters
        ----------
        symbols : List[str]
            List of symbols to preload
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        account_name : str
            MT5 account identifier
        """
        logger.info(f"Preloading data for {len(symbols)} symbols")
        for symbol in symbols:
            try:
                self.get_tick_data(symbol, start_date, end_date, account_name)
                logger.debug(f"Preloaded {symbol}")
            except Exception as e:
                logger.warning(f"Failed to preload {symbol}: {e}")


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
    return features.join(time_feat, how="left").dropna()


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
    # Compute barriers
    close = data["close"]
    target_func = target_config["func"]
    target_params = target_config["params"]
    target = target_func(close=close, **target_params)

    if strategy.get_objective() == "mean_reversion":
        filter_threshold = target if filter_as_series else target.mean()
    else:
        filter_threshold = None

    side, t_events = get_entries(strategy, data, filter_threshold, on_crossover)
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
        X, y, w = X.loc[valid], y.loc[valid], weights.loc[valid]

        # Apply decay factor
        if self.decay != 1.0:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=self.events,
                close_index=self.data_index,
                last_weight=self.decay,
                linear=self.linear,
                av_uniqueness=self.events.loc[X.index, "tW"],
            )
            w *= decay_vec

        self.base_estimator.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def get_params(self, deep=True):
        # Include ALL parameters for consistent hashing
        params = {
            "scheme": self.scheme,
            "decay": self.decay,
            "linear": self.linear,
            "base_estimator": self.base_estimator,
            "events": self.events,
            "data_index": self.data_index,
        }

        if deep:
            # Get nested params from base estimator
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f"base_{k}": v for k, v in base_params.items()})

        return params

    def set_params(self, **params):
        # Handle base estimator params
        base_params = {}
        for key in list(params.keys()):
            if key.startswith("base_"):
                base_params[key[5:]] = params.pop(key)

        # Set our params
        for key in ["scheme", "decay", "linear", "base_estimator", "events", "data_index"]:
            if key in params:
                setattr(self, key, params.pop(key))

        # Set base estimator params
        if base_params:
            self.base_estimator.set_params(**base_params)

        return self


def weighted_estimator(base_estimator, events, data_index):
    """Factory function that returns an instance of the static class."""
    return _WeightedEstimator(base_estimator=base_estimator, events=events, data_index=data_index)


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

    Parameters
    ----------
    data_index: pd.DatetimeIndex
        Price data index.
    events : pd.DataFrame
        Event labels with uniqueness weights.
    features: pd.DataFrame
        Training features
    cv_splits : int, optional
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

    cv_gen = PurgedKFold(n_splits=cv_splits, t1=cont["t1"], pct_embargo=0.01)

    # Find best weighting scheme
    weights = {
        "return": cont["w"],
        "unweighted": pd.Series(1.0, index=cont.index),
        "uniqueness": cont["tW"],
    }
    
    best_score = 0
    best_scheme = None

    cv_results = pd.DataFrame()
    scoring = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"

    for scheme, weight in tqdm(weights.items(), desc="Analyzing weighting schemes", total=len(weights)):
        best_score, best_scheme, cv_results = best_weighting_scheme(
            classifier, X, y, cv_gen, scoring, weight, scheme, best_score, best_scheme, cv_results
        )
    
    best_weight = weights[best_scheme]
    
    # Apply time-decay to best weighting scheme
    if linear is None:
        linear_search = [1, 0]
    elif linear:
        linear_search = [1]
    else:
        linear_search = [0]
    
    time_decay_weights = {}

    for decay in decay_factors:
        for linear in linear_search:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=cont,
                close_index=data_index,
                last_weight=decay,
                linear=linear,
                av_uniqueness=cont["tW"],
            )
            scheme = f"{best_scheme}_{'linear' if linear else 'exp'}_{decay}"
            time_decay_weights[scheme] = best_weight * decay_vec
    
    decay_loop = tqdm(time_decay_weights.items(), desc=f"Analyzing time-decay weighting for {best_scheme}", total=len(time_decay_weights))
    for scheme, weight in decay_loop:
        best_score, best_scheme, cv_results = best_weighting_scheme(
            classifier, X, y, cv_gen, scoring, weight, scheme, best_score, best_scheme, cv_results
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


@njit(parallel=True, fastmath=True, cache=True)
def _rolling_metrics_numba(y_true, y_pred, weights, window):
    """Numba-accelerated rolling metrics calculation."""
    n = len(y_true)
    accuracy = np.full(n, np.nan)
    precision = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    f1 = np.full(n, np.nan)

    for i in prange(window - 1, n):
        start = i - window + 1
        tp = fp = tn = fn = 0.0

        # Inner loop for window
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

    Returns: DataFrame of rolling metrics
    """
    y_true = events["bin"].to_numpy().astype(np.int8)
    y_pred = np.ones(len(y_true), dtype=np.int8)  # All predictions are 1
    weights = sample_weight.to_numpy().astype(np.float32)

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


def is_tree(estimator):
    "Checks if classfication model is tree based"
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))


def train_model_with_cv(
    features: pd.DataFrame,
    events: pd.DataFrame,
    sample_weight: np.ndarray,
    pipe_clf: Union[ClassifierMixin, Pipeline],
    param_grid: Dict,
    cv_splits: int = 5,
    bagging_n_estimators: int = 0,
    bagging_max_samples: float = 1.0,
    bagging_max_features: float = 1.0,
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    pct_embargo: float = 0.02,
    random_state: int = None,
    verbose: int = 0,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train model with cross-validation using cached hyperparameter search.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    events : pd.DataFrame
        Event labels.
    sample_weight : np.ndarray
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
    pct_embargo : float, default=0.02
        Embargo percentage for purging CV splits.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity flag.

    Returns
    -------
    best_model : RandomForestClassifier
        Trained best model.
    cv_results : dict
        Cross-validation results.

    """
    valid_index = features.index.intersection(events.index)
    cont = events.loc[valid_index]
    X = features.loc[valid_index]
    y = cont["bin"]
    t1 = cont["t1"]
    w = sample_weight.loc[valid_index]

    best_model, cv_results = clf_hyper_fit(
        features=X,
        labels=y,
        t1=t1,
        pipe_clf=pipe_clf,
        param_grid=param_grid,
        cv=cv_splits,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_max_features=bagging_max_features,
        rnd_search_iter=rnd_search_iter,
        n_jobs=n_jobs,
        pct_embargo=pct_embargo,
        random_state=random_state,
        verbose=verbose,
        sample_weight=w,
    )

    return best_model, cv_results


def get_model_type(model):
    model_type = {
        "RandomForestClassifier": "rf",
        "SequentiallyBootstrappedBaggingClassifier": "seq_rf",
    }
    
    return model_type[type(model).__name__]


class ModelDevelopmentPipeline:
    """
    Encapsulates the entire production model development pipeline,
    storing all intermediate data and results as attributes for analysis.
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
                - cv : int, default=5
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
                - verbose : int, default=0
                    Controls verbosity of output.
        base_dir: str
            Path to save pipeline data
        """
        # Configuration
        self.data_config = data_config
        self.symbol = data_config["symbol"]
        self.train_start = data_config["start_date"]
        self.train_end = data_config["end_date"]
        self.strategy = strategy
        self.feature_config = feature_config
        self.label_config = label_config
        self.target_config = target_config
        self.account_name = data_config.get("account_name", "default")
        self.pipeline_version = "3.0"
        self.model_params = model_params
        
        # Build complete config
        self.config = data_config.copy()
        self.config["training_start"] = self.config.pop("start_date")
        self.config["training_end"] = self.config.pop("end_date")
        self.config["strategy"] = strategy.get_strategy_name()
        self.config["feature_func"] = feature_config["func"].__name__
        self.config["feature_params"] = feature_config["params"]
        self.config["target_func"] = target_config["func"].__name__
        self.config["target_params"] = target_config["params"]
        self.config.update(label_config)

        self.label_config["target_config"] = target_config

        # Storage for intermediate results
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

        if isinstance(model_params["pipe_clf"], Pipeline):
            model = model_params["pipe_clf"].steps[-1][1]
        else:
            model = model_params["pipe_clf"]
            
        self.model_type = get_model_type(model)
        
        # Initialize file management and logging
        self.file_manager = ModelFileManager(base_dir)
        self.file_paths = self.file_manager.setup_model_directory(self.config, self.model_type)

        
        # Sample weight settings that can be set after initialization
        self.linear = None
        self.decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]

        # Status tracking
        self.completed_steps = {
            "data_loading": False,
            "feature_engineering": False,
            "label_generation": False,
            "weight_computation": False,
            "meta_features": False,
            "model_training": False,
            "analysis": False,
        }

        # Log file setup
        self.log_file = self.file_paths["logs"] / "pipeline.log"
        self._setup_logging()

        self.cv_splits = model_params["cv_splits"]

    def _setup_logging(self):
        """Set up logging to file using loguru with colors in console."""

        logger.remove()

        # File sink (no colors, structured logs)
        logger.add(
            self.log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss:SS} | {name} | {level} | {message}",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
        )

        # Console sink (colors enabled automatically)
        logger.add(
            # sys.stdout,
            lambda msg: tqdm.write(msg, end=""),
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss:SS}</green> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{level}</level> | "
            "<yellow>{message}</yellow>",
            colorize=True,
        )

        self.logger = logger.bind(context=self.__class__.__name__)

        self.logger.info(f"Starting pipeline for {self.symbol}")
        self.logger.info(f"Training period: {self.train_start} to {self.train_end}")
        self.logger.info(f"Output directory: {self.file_paths['base_dir']}")

    def run(
        self,
        generate_reports: bool = True,
        cache_reports: bool = False,
        save: bool = True,
        export_onxx: bool = False,
        verbose: bool = True,
    ) -> Tuple[RandomForestClassifier, List[str], Dict]:
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
        self.export_onxx = export_onxx

        if verbose:
            print("\n" + "=" * 70)
            print("PRODUCTION MODEL DEVELOPMENT PIPELINE")
            print("=" * 70)
            print("\nConfiguration")
            print("-" * 50)
            print(pd.Series(self.config).to_string(), "\n")

        
        if self.file_paths["model"].exists():
            print("\nLoading trained model and artifacts...")

            self.best_model = joblib.load(self.file_paths["model"])
            self.metrics = pickle.load(self.file_paths["metrics"])
            self.preprocessed_features = pd.read_parquet(self.file_paths["features"])
            self.events = pd.read_parquet(self.file_paths["events"])
            onnx_path = str(self.file_paths["model"]).replace("joblib", "onnx")
            
            if self.export_onxx and not Path(onnx_path).exists():
                metadata = {
                    "strategy": self.strategy,
                    "feature_config": self.feature_config,
                    "label_config": self.label_config,
                    "feature_names": self._get_feature_names(),
                    "feature_count": len(self._get_feature_names()),
                    "training_samples": self.metrics["events_count"],
                    "best_weighting_scheme": self.metrics["best_weighting_scheme"],
                    "pipeline_version": self.pipeline_version,
                    "created_by": "AFML Production Pipeline",
                }
                self.file_manager.save_model_as_onxx(
                    self.best_model, self._get_feature_names(), metadata
                )
                
            return (
                self.best_model,
                self._get_feature_names(),
                self.metrics,
                self.config,
            )

        try:
            # Step 1: Load data
            if verbose:
                print("\n[Step 1/7] Loading training data...")

            self.load_training_data()

            # Step 2: Feature engineering
            if verbose:
                print("\n[Step 2/7] Computing features...")
            self.engineer_features()
            if verbose:
                print(f"✓ Generated {len(self.features.columns)} features")

            # Step 3: Label generation
            if verbose:
                print("\n[Step 3/7] Generating events...")

            self.generate_labels()

            # Step 4: Sample weights
            if verbose:
                print("\n[Step 4/7] Computing sample weights...")

            self.compute_sample_weights()

            # Step 5: Rolling meta-label features
            if verbose:
                print("\n\n[Step 5/7] Computing rolling meta-label features...")

            self.add_meta_features()
            self.preprocess_features()

            # Step 6: Model training
            if verbose:
                print("\n[Step 6/7] Training model with cross-validation...")

            self.train_model()

            # Step 7: Feature importance analysis
            if verbose:
                print("\n[Step 7/7] Analyzing feature importance...")

            self.analyze_features()

            if verbose:
                print("\nTop 10 Features:")
                print(self.feature_importance.head(10).to_string(index=False), "\n")

            # Compile metrics
            self._compile_metrics()

            # Generate reports if requested
            if generate_reports:
                if verbose:
                    print("\n[Generating Reports] Creating analysis reports...")

                self._generate_analysis_reports()

            # Cache reports (optional)
            if cache_reports:
                self._display_cache_reports()

            # Save artifacts
            if save and self.best_model is not None:
                if verbose:
                    print("\n[Saving] Writing artifacts to disk...")
                self._save_all_artifacts()
                if verbose:
                    print(f"✓ Saved to {self.file_paths['base_dir']}")

            # Log pipeline completion
            pipeline_duration = time.time() - time0

            if verbose:
                duration_str = pd.Timedelta(seconds=pipeline_duration).round("1s")
                duration_str = str(duration_str).replace("0 days ", "")
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

    def _generate_analysis_reports(self):
        """Generate comprehensive analysis reports."""
        try:
            # 1. Generate hyperparameter analysis report
            if self.cv_results and "cv_results" in self.cv_results:
                cv_results_df = pd.DataFrame(self.cv_results["cv_results"])

                report_path = self.file_paths["reports"] / "hyperparameter_analysis_report.md"

                generate_complete_hyperparameter_report(
                    cv_results=cv_results_df,
                    strategy_config=self.config,
                    output_dir=self.file_paths["reports"],
                    filename=report_path.name,
                    target_metric="mean_test_score",
                )

                logger.info(f"Generated hyperparameter report: {report_path}")

            # 2. Generate feature importance plot
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

            # 3. Generate training summary HTML
            self._generate_training_summary_html()

        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    def _save_all_artifacts(self):
        """Save all pipeline artifacts using ModelFileManager."""
        try:
            # Save model with metadata
            metadata = metadata = {
                "strategy": self.strategy,
                "feature_config": self.feature_config,
                "label_config": self.label_config,
                "feature_names": self._get_feature_names(),
                "feature_count": len(self._get_feature_names()),
                "training_samples": len(self.events),
                "best_weighting_scheme": self.best_weighting_scheme,
                "pipeline_version": self.pipeline_version,
                "created_by": "AFML Production Pipeline",
            }
            self.file_manager.save_model(self.best_model, metadata)

            if self.features is not None:
                self.file_manager.save_dataframe(self.preprocessed_features, "features")

            if self.events is not None:
                self.file_manager.save_dataframe(self.events, "events")

            if self.sample_weight is not None:
                self.file_manager.save_dataframe(self.sample_weight.to_frame("weight"), "weights")

            # Save metrics
            if self.metrics:
                self.file_manager.save_object(self.metrics, "metrics")

            if self.export_onxx and self.best_model is not None:
                self.file_manager.save_model_as_onxx(
                    self.best_model, self._get_feature_names(), metadata
                )

            logger.info(f"Saved all artifacts to {self.file_paths['base_dir']}")

        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            raise

    def load_training_data(self):
        """Step 1: Load tick data and construct bars."""
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        if self.data_config == "tick":
            self.config["tick_bar_size"] = self.bar_data["tick_volume"].iloc[0]
            self.file_manager.save_config(self.config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        """Step 2: Feature engineering."""
        self.features = create_feature_engineering_pipeline(
            self.bar_data, self.feature_config, self.data_config
        )
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        """Step 3: Generate triple-barrier labels."""
        self.events = generate_events_triple_barrier(
            self.bar_data, self.strategy, **self.label_config
        )
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        """Step 4: Compute optimal sample weights."""
        if self.file_paths["weights"].exists():
            self.sample_weight = pd.read_parquet(self.file_paths["weights"])   
        else:
            self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
                self.bar_data.index, self.events, self.features, self.cv_splits, self.linear, self.decay_factors
            )
            self.best_weighting_scheme = self.weight_cv_results["best_scheme"]     
            if self.sample_weight is not None: 
                # Save rather than cache as computation is expensive and cache misses are not infrequent
                self.file_manager.save_dataframe(self.sample_weight.to_frame("weight"), "weights")

        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        """Step 5: Add rolling performance metrics as features."""
        self.meta_features = calculate_rolling_metrics(self.events, self.sample_weight)
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        """Step 5b: Preprocess features (drop constant/duplicate)."""
        # Join meta-features
        enhanced_features = self.features.join(self.meta_features, how="inner").dropna()

        # Apply preprocessing
        preprocessor = Pipeline(
            [
                ("dcf", DropConstantFeatures()),
                ("ddf", DropDuplicateFeatures()),
            ]
        )
        self.preprocessed_features = preprocessor.fit_transform(enhanced_features)

        # Align events with preprocessed features
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self):
        """Step 6: Train model with cross-validation."""

        # Configure pipeline
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

        # Train model
        self.best_model, self.cv_results = train_model_with_cv(
            self.preprocessed_features,
            self.events,
            self.sample_weight,
            **self.model_params,
        )

        # Set n_jobs for production use
        self.best_model = set_pipeline_params(self.best_model, n_jobs=-1)
        
        self.completed_steps["model_training"] = True

    

    def analyze_features(self):
        """Step 7: Analyze feature importance."""
        features_columns = (
            self.best_model[:-1].get_feature_names_out()
            if self.best_model is not None and len(self.best_model) > 1
            else self.preprocessed_features.columns.to_list()
        )

        self.feature_importance = pd.DataFrame(
            {
                "feature": features_columns,
                "importance": self.best_model.steps[-1][1].feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        """Compile all metrics into a single dictionary."""
        self.metrics = {
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
            "training_samples": len(self.bar_data),
            "feature_count": len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme,
            "label_distribution": value_counts_data(self.events["bin"]),
            "average_uniqueness": self.events["tW"].mean(),
            "sample_weight_stats": (
                self.sample_weight.describe().to_dict() if self.sample_weight is not None else None
            ),
            "events_count": len(self.events),
            "features_shape": self.preprocessed_features.shape,
            "completed_steps": self.completed_steps,
        }

    def _get_feature_names(self):
        """Get feature names from the trained model."""
        if self.best_model is None:
            return []

        if len(self.best_model) > 1:
            return self.best_model[:-1].get_feature_names_out().tolist()
        else:
            return self.preprocessed_features.columns.tolist()

    def _display_cache_reports(self):
        """Display cache performance and contamination reports."""
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
        """Get a summary of all stored data."""
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
                    shape = data.shape
                    dtype = "DataFrame"
                    columns = f"{len(data.columns)} cols"
                elif isinstance(data, pd.Series):
                    shape = (len(data),)
                    dtype = "Series"
                    columns = "N/A"
                else:
                    shape = "N/A"
                    dtype = type(data).__name__
                    columns = "N/A"

                summary_data.append(
                    {
                        "Component": name,
                        "Type": dtype,
                        "Rows": shape[0] if isinstance(shape, tuple) else shape,
                        "Columns": (
                            shape[1] if isinstance(shape, tuple) and len(shape) > 1 else columns
                        ),
                        "Memory (MB)": (
                            data.memory_usage(deep=True).sum() / (1024**2)
                            if hasattr(data, "memory_usage")
                            else "N/A"
                        ),
                    }
                )

        return pd.DataFrame(summary_data)

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        return {
            "model_performance": self.cv_results,
            "feature_analysis": self.feature_importance.to_dict(orient="records"),
            "data_statistics": {
                "training_samples": len(self.bar_data),
                "feature_count": len(self._get_feature_names()),
                "event_distribution": dict(value_counts_data(self.events["bin"])),
                "average_uniqueness": float(self.events["tW"].mean()),
            },
            "weighting_scheme": self.best_weighting_scheme,
        }

    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        if self.feature_importance is None:
            raise ValueError("Feature importance not computed. Run the pipeline first.")

        import matplotlib.pyplot as plt

        top_features = self.feature_importance.head(top_n)

        plt.style.use("dark_background")
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features["importance"][::-1])
        plt.yticks(range(len(top_features)), top_features["feature"][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importance - {self.symbol}")
        plt.tight_layout()
        plt.show()

    def _generate_training_summary_html(self):
        """Generate comprehensive HTML training summary."""
        try:
            # Safely get counts
            def safe_len(obj):
                """Safely get length of an object."""
                if obj is None:
                    return 0
                if isinstance(obj, (list, tuple)):
                    return len(obj)
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    return len(obj)
                if isinstance(obj, np.ndarray):
                    return obj.shape[0] if len(obj.shape) > 0 else 0
                try:
                    return len(obj)
                except Exception:
                    return 0

            # Get safe values
            n_bar_data = safe_len(self.bar_data)
            n_events = safe_len(self.events)
            n_features = len(self._get_feature_names()) if self.best_model else 0
            cv_score = self.cv_results.get("best_score", 0.0) if self.cv_results else 0.0

            # Safe config access
            strategy_name = self.config.get("strategy", "Unknown")

            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Training Summary - {self.symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }}
        .header {{
            background-color: #2d2d30;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: #252526;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #3e3e42;
        }}
        th {{
            background-color: #2d2d30;
            font-weight: bold;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            color: #808080;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #4ec9b0;
        }}
        h1, h2 {{ color: #4ec9b0; }}
        .success {{ color: #4ec9b0; }}
        .warning {{ color: #ce9178; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Training Summary: {self.symbol}</h1>
        <p><strong>Strategy:</strong> {strategy_name}</p>
        <p><strong>Training Period:</strong> {self.train_start} to {self.train_end}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>Key Metrics</h2>
        <div class="metric">
            <div class="metric-label">CV Score</div>
            <div class="metric-value">{cv_score:.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Features</div>
            <div class="metric-value">{n_features}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">{n_bar_data:,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Events</div>
            <div class="metric-value">{n_events:,}</div>
        </div>
    </div>

    <div class="section">
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""

            # Safely iterate config
            for key, value in self.config.items():
                # Handle complex types
                if isinstance(value, (dict, list, tuple)):
                    value_str = str(value)[:100]  # Truncate long values
                else:
                    value_str = str(value)
                html_content += f"            <tr><td>{key}</td><td>{value_str}</td></tr>\n"

            html_content += """
        </table>
    </div>
"""

            # Best model parameters (if available)
            if self.cv_results and "best_params" in self.cv_results:
                html_content += """
    <div class="section">
        <h2>Best Model Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""
                for key, value in self.cv_results["best_params"].items():
                    value_str = str(value)[:100]  # Truncate long values
                    html_content += f"            <tr><td>{key}</td><td>{value_str}</td></tr>\n"

                html_content += """
        </table>
    </div>
"""

            # Feature importance (if available)
            if self.feature_importance is not None and len(self.feature_importance) > 0:
                html_content += """
    <div class="section">
        <h2>Top 10 Features</h2>
        <table>
            <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
"""
                for i, row in self.feature_importance.head(10).iterrows():
                    html_content += f"""            <tr>
                <td>{i + 1}</td>
                <td>{row["feature"]}</td>
                <td>{row["importance"]:.6f}</td>
            </tr>
"""
                html_content += """
        </table>
    </div>
"""

            # Label distribution (if events available)
            if self.events is not None and len(self.events) > 0:
                html_content += """
    <div class="section">
        <h2>Label Distribution</h2>
        <table>
            <tr><th>Label</th><th>Count</th><th>Percentage</th></tr>
"""
                try:
                    label_counts = self.events["bin"].value_counts()
                    total_labels = label_counts.values.sum()
                    for label, count in label_counts.sort_index().items():
                        pct = (count / total_labels) if total_labels > 0 else 0
                        html_content += f"""            <tr>
                <td>{label}</td>
                <td>{count:,}</td>
                <td>{pct:.1%}</td>
            </tr>
"""
                except Exception as e:
                    logger.debug(f"Could not generate label distribution: {e}")
                    html_content += """            <tr>
                <td colspan="3">Label distribution unavailable</td>
            </tr>
"""

                html_content += """
        </table>
    </div>
"""

            # Sample weighting (if available)
            if self.best_weighting_scheme and self.events is not None:
                html_content += f"""
    <div class="section">
        <h2>Sample Weighting</h2>
        <p><strong>Best Scheme:</strong> <span class="success">{self.best_weighting_scheme}</span></p>
"""
                try:
                    avg_uniqueness = self.events["tW"].mean() if "tW" in self.events.columns else 0
                    html_content += f"        <p><strong>Average Uniqueness:</strong> {avg_uniqueness:.4f}</p>\n"
                except Exception as e:
                    logger.error(f"Could not compute average uniqueness: {e}")

                if self.weight_cv_results:
                    weight_score = self.weight_cv_results.get("best_score", 0)
                    html_content += (
                        f"        <p><strong>Weight CV Score:</strong> {weight_score:.4f}</p>\n"
                    )

                html_content += """
    </div>
"""

            # Pipeline steps
            html_content += """
    <div class="section">
        <h2>Pipeline Steps</h2>
        <table>
            <tr><th>Step</th><th>Status</th></tr>
"""

            for step, completed in self.completed_steps.items():
                status = (
                    '<span class="success">✅ Completed</span>'
                    if completed
                    else '<span class="warning">❌ Not Run</span>'
                )
                step_name = step.replace("_", " ").title()
                html_content += f"            <tr><td>{step_name}</td><td>{status}</td></tr>\n"

            html_content += """
        </table>
    </div>
</body>
</html>
"""

            html_path = self.file_paths["reports"] / "training_summary.html"
            html_path.write_text(html_content)

            logger.info(f"Generated training summary: {html_path}")

        except Exception as e:
            logger.warning(f"HTML summary generation failed: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
