from datetime import datetime, timedelta
from pprint import pformat, pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit, prange
from scipy.stats import uniform
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from afml.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from afml.features.time import get_time_features

from ..cache import (
    cacheable,
    get_cache_monitor,
    log_data_access,
    print_contamination_report,
)
from ..cross_validation import (
    MyPipeline,
    PurgedKFold,
    clf_hyper_fit_cached,
    ml_cross_val_score,
)
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


# Define a sink that uses tqdm.write
def tqdm_sink(msg):
    tqdm.write(msg, end="")  # end="" avoids double newlines


# Remove default handler and add our custom one to prevent logger interfering with tqdm
logger.remove()
logger.add(
    tqdm_sink,
    format="<green>{time:YYYY-MM-DD HH:mm:ss:ms}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    colorize=True,
)


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

    def __init__(self, max_cache_size_mb: int = 500, max_cached_symbols: int = 20):
        """
        Initialize the tick data loader.

        Parameters
        ----------
        max_cache_size_mb : int, optional
            Maximum cache size in MB (default: 500MB)
        max_cached_symbols : int, optional
            Maximum number of symbols to keep in cache (default: 20)
        """
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (symbol, account_name) -> DataFrame
        self._cache_metadata: Dict[Tuple[str, str], Dict] = {}  # (symbol, account_name) -> metadata
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cached_symbols = max_cached_symbols
        self.cache_stats = {"hits": 0, "misses": 0, "partial_hits": 0, "total_loaded": 0}

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
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Check if we have cached data for this symbol/account
        if cache_key in self._cache:
            cached_df = self._cache[cache_key]
            metadata = self._cache_metadata[cache_key]
            cached_start = metadata["start_date"]
            cached_end = metadata["end_date"]

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

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

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

        if not df.empty:
            # Cache the data
            self._cache[cache_key] = df
            self._cache_metadata[cache_key] = {
                "start_date": pd.to_datetime(start_date),
                "end_date": pd.to_datetime(end_date),
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
                self._cache_metadata.items(), key=lambda x: x[1]["size_mb"], reverse=True
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


# def generate_features_with_dual_price(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
#     """
#     Generate features using both bid and ask prices separately.
#     Returns features for analysis and conservative estimates for trading.
#     """
#     # For signal generation (use mid)
#     df["mid_price"] = (df["bid"] + df["ask"]) / 2
#     mid_features = calculate_features(df, price_col="mid_price")

#     # For conservative trading estimates (use worse case)
#     # Assume you buy at ask, sell at bid
#     df["conservative_long"] = df["ask"]  # Entry for longs
#     df["conservative_short"] = df["bid"]  # Entry for shorts

#     # For bid-ask spread analysis
#     df["spread"] = df["ask"] - df["bid"]
#     df["spread_bps"] = df["spread"] / df["mid_price"] * 10000

#     return {
#         "mid_features": mid_features,  # For signal generation
#         "bid_features": calculate_features(df, "bid"),
#         "ask_features": calculate_features(df, "ask"),
#         "spread_features": df[["spread", "spread_bps"]],
#     }


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
    - Prevents data leakage via time-aware caching.
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
    return features.join(time_feat)


@cacheable(time_aware=True)
def generate_events_triple_barrier(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    target_lookback: int,
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
    strategy : BaseStrategy
        Strategy instance implementing `generate_signals()`.
    target_lookback : int
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
def get_best_weighting_scheme(
    classifier, X, y, cv_gen, sample_weight, scheme, best_scheme, best_score
):
    cv_scores = ml_cross_val_score(
        classifier,
        X,
        y,
        cv_gen,
        sample_weight_train=sample_weight,
        sample_weight_score=sample_weight,
        scoring="f1",
    )
    score = cv_scores.mean()
    best_score = max(score, best_score)
    if best_scheme is None or score == best_score:
        best_scheme = scheme
    return best_scheme, best_score


def get_best_sample_weight(
    data: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Compute best sample weight with time decay.

    Parameters
    ----------
    data: pd.DataFrame
        Price data.
    events : pd.DataFrame
        Event labels with uniqueness weights.
    features: pd.DataFrame
        Training features

    Returns
    -------
    pd.Series
        Sample weights.

    """
    valid_index = features.index.intersection(events.index)
    cont = events.loc[valid_index]
    weighting_schemes = {
        "unweighted": pd.Series(1.0, index=valid_index),
        "uniqueness": cont["tW"],
        "return": cont["w"],
    }
    X = features.loc[valid_index]
    y = cont["bin"]
    cv_gen = PurgedKFold(n_splits=5, t1=cont["t1"], pct_embargo=0.01)
    classifier = RandomForestClassifier(
        criterion="entropy",
        class_weight="balanced_subsample",
        max_samples=cont["tW"].mean().round(2),
        max_depth=6,
        min_weight_fraction_leaf=0.05,
        n_jobs=-1,
        random_state=7,
    )

    best_scheme = None
    best_score = 0
    for scheme, weight in tqdm(
        weighting_schemes.items(),
        desc=f"Computing best weighting scheme from {list(weighting_schemes.keys())}",
        total=len(weighting_schemes),
    ):
        best_scheme, best_score = get_best_weighting_scheme(
            clone(classifier), X, y, cv_gen, weight, scheme, best_scheme, best_score
        )

    decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]
    best_weighting_scheme = best_scheme
    best_weight = weighting_schemes[best_scheme]
    for time_decay in tqdm(
        reversed(decay_factors),
        total=len(decay_factors),
        position=0,
    ):
        for linear in tqdm((0, 1), total=2, position=1):
            decay_method = "linear" if linear else "exponential"
            scheme = f"{best_weighting_scheme}_{decay_method}_decay_{time_decay}"
            logger.debug(scheme)
            decay_w = get_weights_by_time_decay_optimized(
                triple_barrier_events=events.loc[valid_index],
                close_index=data.index,
                last_weight=time_decay,
                linear=linear,
                av_uniqueness=cont["tW"],
            )
            weight = best_weight * decay_w
            weighting_schemes[scheme] = weight
            best_scheme, best_score = get_best_weighting_scheme(
                clone(classifier), X, y, cv_gen, weight, scheme, best_scheme, best_score
            )

    logger.info(f"\nBest Weighting Scheme: {' '.join(best_scheme.split('_')).title()}")

    return weighting_schemes[best_scheme], best_scheme, weighting_schemes


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


@cacheable(time_aware=True)
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

    """
    valid_index = features.index.intersection(events.index)
    cont = events.loc[valid_index]
    X = features.loc[valid_index]
    y = cont["bin"]
    t1 = cont["t1"]
    w = sample_weight.loc[valid_index]

    # Set max_samples distribution in param_grid from average uniqueness to 1
    if isinstance(pipe_clf.named_steps["clf"], SequentiallyBootstrappedBaggingClassifier):
        av_uniqueness = cont["tW"].mean()
        if rnd_search_iter > 0:
            param_grid["clf__max_samples"] = uniform(loc=av_uniqueness, scale=1 - av_uniqueness)
        else:
            param_grid["clf__max_samples"] = np.linspace(av_uniqueness, 1, 3).tolist()

    best_model, cv_results = clf_hyper_fit_cached(
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


def make_custom_pipeline(pipe_clf):
    if not isinstance(pipe_clf, Pipeline):
        return MyPipeline([("clf", pipe_clf)])
    else:
        return pipe_clf


def develop_production_model(
    symbol: str,
    train_start: str,
    train_end: str,
    strategy: BaseStrategy,
    data_config: Dict,
    feature_config: Dict,
    label_config: Dict,
    model_params: Dict,
    cache_reports: bool = False,
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
    strategy: BaseStrategy
        signal generating strategy
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
            Pipeline including classifier named as "clf", e.g., Pipeline([("clf", RandomForestClassifier)]).
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
        - cache_reports : bool, optional (default=False)
            Verbosity flag.

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

    See individual function docstrings for detailed argument descriptions.
    """

    print("\n" + "=" * 70)
    print("PRODUCTION MODEL DEVELOPMENT PIPELINE")
    print("=" * 70)

    print("\nConfiguration")
    print("-" * 50)
    config = {"strategy": strategy.get_strategy_name(), "symbol": symbol}
    config.update(data_config)
    config.update(label_config)
    print(pd.Series(config).to_string())

    # Step 1: Load data (tracked for contamination)
    print("\n[Step 1/7] Loading training data...")
    data = load_and_prepare_training_data(symbol, train_start, train_end, **data_config)
    print(f"✓ Loaded {len(data):,} samples from {train_start} to {train_end}")

    # Step 2: Feature engineering  (cached)
    print("\n[Step 2/7] Computing features...")
    features = create_feature_engineering_pipeline(data, feature_config, data_config)
    print(f"✓ Generated {len(features.columns)} features")

    # Step 3: Label generation (cached)
    print("\n[Step 3/7] Generating events...")
    events = generate_events_triple_barrier(data, strategy, **label_config)
    print(f"✓ Generated events: \n{value_counts_data(events['bin'])}")
    print(f"\nAverage Uniqueness: {events['tW'].mean():.4f}")

    # Step 4: Sample weights (cached)
    print("\n[Step 4/7] Computing sample weights...")
    sample_weight, best_weighting_scheme, weighting_schemes = get_best_sample_weight(
        data, events, features
    )

    # Step 5: Rolling meta-label features (cached)
    print("\n[Step 5/7] Computing rolling meta-label features...")
    meta_features = calculate_rolling_metrics(events, sample_weight)
    features = features.join(meta_features, how="inner")
    events = events.reindex(features.index)  # Align indices
    print(f"✓ Computed rolling meta-label features")

    # Step 6: Model training with CV (cached)
    print("\n[Step 6/7] Training model with cross-validation...")

    # Set max_samples to average uniqueness for tree-based classifiers
    pipe = model_params["pipe_clf"]
    if is_tree(pipe):
        av_uniqueness = events["tW"].mean().round(4)
        pipe.set_params(max_samples=av_uniqueness)
    model_params["pipe_clf"] = make_custom_pipeline(pipe)

    best_model, cv_results = train_model_with_cv(features, events, sample_weight, **model_params)
    best_model.named_steps["clf"].set_params(n_jobs=-1)
    print(f"✓ Best CV score: {cv_results['best_score']:.4f}")
    print(f"✓ Best params: {cv_results['best_params']}")

    # Step 6: Feature importance analysis
    print("\n[Step 7/7] Analyzing feature importance...")
    features_columns = (
        best_model[:-1].get_feature_names_out()
        if len(best_model) > 1
        else features.columns.tolist()
    )
    feature_importance = pd.DataFrame(
        {
            "feature": features_columns,
            "importance": best_model.named_steps["clf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))

    if cache_reports:
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
        "training_samples": len(data),
        "feature_count": len(features_columns),
        "best_weighting_scheme": best_weighting_scheme,
        "weighting_schemes": weighting_schemes,
    }

    return best_model, features_columns, metrics, config
