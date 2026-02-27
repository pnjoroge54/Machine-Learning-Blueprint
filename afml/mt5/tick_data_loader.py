from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from pathlib import Path

from ..util.misc import date_conversion, value_counts_data



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
        from .load_data import load_tick_data, save_data_to_parquet
      
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


tick_data_loader = TickDataLoader()
