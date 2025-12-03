"""
Optimized Microstructural Features Generator
===========================================
Vectorized implementation for market microstructure feature calculation.
Uses Numba for critical path optimization and avoids Python loops.
"""

from typing import Union, Optional, Dict, List
import numpy as np
import pandas as pd
from numba import njit, prange
import warnings

# Local imports
from ..util.misc import crop_data_frame_in_batches
from .encoding import encode_array, encode_tick_rule_array
from .entropy import (
    get_konto_entropy,
    get_lempel_ziv_entropy,
    get_plug_in_entropy,
    get_shannon_entropy,
)
from .misc import get_avg_tick_size, vwap
from .second_generation import (
    get_trades_based_amihud_lambda,
    get_trades_based_hasbrouck_lambda,
    get_trades_based_kyle_lambda,
)


@njit(cache=True, fastmath=True)
def _process_ticks_numba(prices: np.ndarray, volumes: np.ndarray, bar_indices: np.ndarray) -> tuple:
    """
    Numba-optimized tick processing for a single batch.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values (float64)
    volumes : np.ndarray
        Array of volume values (float64)
    bar_indices : np.ndarray
        Array of cumulative tick counts where bars end (int64)

    Returns
    -------
    tuple
        Arrays of features for each bar:
        - trade_sizes: (n_bars, max_bar_ticks)
        - tick_rules: (n_bars, max_bar_ticks)
        - dollar_sizes: (n_bars, max_bar_ticks)
        - log_rets: (n_bars, max_bar_ticks)
        - price_diffs: (n_bars, max_bar_ticks)
        - ticks_per_bar: (n_bars,)
    """
    n_ticks = len(prices)
    n_bars = len(bar_indices)

    # Pre-allocate arrays
    max_bar_ticks = (
        np.max(np.diff(np.concatenate(([0], bar_indices)))) if n_bars > 1 else bar_indices[0]
    )

    # Features storage
    all_trade_sizes = np.zeros((n_bars, max_bar_ticks), dtype=np.float64)
    all_tick_rules = np.zeros((n_bars, max_bar_ticks), dtype=np.int8)
    all_dollar_sizes = np.zeros((n_bars, max_bar_ticks), dtype=np.float64)
    all_log_rets = np.zeros((n_bars, max_bar_ticks), dtype=np.float64)
    all_price_diffs = np.zeros((n_bars, max_bar_ticks), dtype=np.float64)

    # Count of ticks per bar
    ticks_per_bar = np.zeros(n_bars, dtype=np.int32)

    # Processing state
    prev_price = prices[0]
    prev_tick_rule = 0
    bar_idx = 0
    tick_in_bar = 0

    for i in range(n_ticks):
        price = prices[i]
        volume = volumes[i]

        # Calculate derivatives
        price_diff = price - prev_price if i > 0 else 0.0
        log_ret = np.log(price / prev_price) if i > 0 and prev_price > 0 else 0.0

        # Tick rule
        if price_diff != 0:
            signed_tick = 1 if price_diff > 0 else -1
            prev_tick_rule = signed_tick
        else:
            signed_tick = prev_tick_rule

        # Store in current bar
        if tick_in_bar < max_bar_ticks:
            all_trade_sizes[bar_idx, tick_in_bar] = volume
            all_tick_rules[bar_idx, tick_in_bar] = signed_tick
            all_dollar_sizes[bar_idx, tick_in_bar] = price * volume
            all_log_rets[bar_idx, tick_in_bar] = log_ret
            all_price_diffs[bar_idx, tick_in_bar] = price_diff
            ticks_per_bar[bar_idx] += 1

        tick_in_bar += 1
        prev_price = price

        # Check if we reached bar index
        if i + 1 >= bar_indices[bar_idx]:
            bar_idx += 1
            tick_in_bar = 0
            if bar_idx >= n_bars:
                break

    return (
        all_trade_sizes,
        all_tick_rules,
        all_dollar_sizes,
        all_log_rets,
        all_price_diffs,
        ticks_per_bar,
    )


@njit(cache=True, parallel=True, fastmath=True)
def _compute_basic_features_numba(
    trade_sizes: np.ndarray,
    tick_rules: np.ndarray,
    dollar_sizes: np.ndarray,
    ticks_per_bar: np.ndarray,
) -> tuple:
    """
    Compute basic features (avg tick size, tick rule sum, VWAP) in parallel.

    Parameters
    ----------
    trade_sizes : np.ndarray
        Array of trade sizes for each bar (n_bars, max_ticks)
    tick_rules : np.ndarray
        Array of tick rules for each bar (n_bars, max_ticks)
    dollar_sizes : np.ndarray
        Array of dollar volumes for each bar (n_bars, max_ticks)
    ticks_per_bar : np.ndarray
        Number of valid ticks in each bar (n_bars,)

    Returns
    -------
    tuple
        - avg_tick_sizes: Array of average tick sizes per bar
        - tick_rule_sums: Array of tick rule sums per bar
        - vwap_values: Array of VWAP per bar
    """
    n_bars = len(ticks_per_bar)

    avg_tick_sizes = np.zeros(n_bars, dtype=np.float64)
    tick_rule_sums = np.zeros(n_bars, dtype=np.float64)
    vwap_values = np.zeros(n_bars, dtype=np.float64)

    for i in prange(n_bars):
        n_ticks = ticks_per_bar[i]
        if n_ticks == 0:
            continue

        # Get valid data for this bar
        bar_trade_sizes = trade_sizes[i, :n_ticks]
        bar_tick_rules = tick_rules[i, :n_ticks]
        bar_dollar_sizes = dollar_sizes[i, :n_ticks]

        # Avg tick size
        avg_tick_sizes[i] = np.mean(bar_trade_sizes)

        # Tick rule sum
        tick_rule_sums[i] = np.sum(bar_tick_rules)

        # VWAP
        total_dollar = np.sum(bar_dollar_sizes)
        total_volume = np.sum(bar_trade_sizes)
        if total_volume > 0:
            vwap_values[i] = total_dollar / total_volume

    return avg_tick_sizes, tick_rule_sums, vwap_values


class OptimizedMicrostructuralFeaturesGenerator:
    """
    Optimized version of MicrostructuralFeaturesGenerator with:
    - Numba-accelerated core loops
    - Batch vectorization
    - Memory-efficient processing
    - Parallel computation

    Parameters
    ----------
    trades_input : Union[str, pd.DataFrame]
        Source of tick data. Can be:
        - String path to CSV file with columns: date_time, price, volume
        - Pandas DataFrame with same columns

    tick_num_series : pd.Series
        Series of cumulative tick counts where bars are formed.

    batch_size : int, default=2_000_000
        Number of rows to process per batch.

    volume_encoding : Optional[Dict], default=None
        Encoding scheme for trade sizes. If provided, enables volume entropy features.

    pct_encoding : Optional[Dict], default=None
        Encoding scheme for log returns. If provided, enables return entropy features.

    entropy_types : List[str], default=None
        List of entropy types to compute. Options: ['shannon', 'lempel_ziv', 'plug_in', 'konto']
        If None, computes all types.

    Attributes
    ----------
    generator_object : iterator
        Batch iterator for tick data

    tick_num_array : np.ndarray
        Array of bar tick indices

    current_bar_idx : int
        Current bar index being processed

    columns : List[str]
        Column names for output DataFrame

    Methods
    -------
    get_features(verbose=True, to_csv=False, output_path=None, max_batches=None)
        Generate microstructural features from tick data.

    _process_batch_vectorized(batch)
        Process a batch of tick data using vectorized operations.

    _compute_entropy_features(bar_data, tick_rules, trade_sizes, log_rets)
        Compute entropy features for a bar.

    _assert_csv(test_batch)
        Validate CSV format.
    """

    def __init__(
        self,
        trades_input: Union[str, pd.DataFrame],
        tick_num_series: pd.Series,
        batch_size: int = 2_000_000,
        volume_encoding: Optional[Dict] = None,
        pct_encoding: Optional[Dict] = None,
        entropy_types: Optional[List[str]] = None,
    ):
        # Initialize data source
        if isinstance(trades_input, str):
            self.generator_object = pd.read_csv(trades_input, chunksize=batch_size, parse_dates=[0])
            first_row = pd.read_csv(trades_input, nrows=1)
            self._assert_csv(first_row)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = crop_data_frame_in_batches(trades_input, batch_size)
        else:
            raise ValueError("trades_input must be string path or DataFrame")

        # Convert tick series to numpy for faster access
        self.tick_num_array = tick_num_series.values.astype(np.int64)
        self.current_bar_idx = 0

        # Encoding settings
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding

        # Entropy configuration
        self.entropy_types = entropy_types or ["shannon", "lempel_ziv", "plug_in", "konto"]
        valid_entropy_types = ["shannon", "lempel_ziv", "plug_in", "konto"]
        for et in self.entropy_types:
            if et not in valid_entropy_types:
                raise ValueError(
                    f"Invalid entropy type: {et}. Must be one of {valid_entropy_types}"
                )

        # Cache for entropy results
        self._entropy_cache = {}

        # Define columns
        self._define_columns()

    def _define_columns(self):
        """Predefine column names for faster DataFrame creation."""
        base_cols = [
            "date_time",
            "avg_tick_size",
            "tick_rule_sum",
            "vwap",
            "kyle_lambda",
            "kyle_lambda_t_value",
            "amihud_lambda",
            "amihud_lambda_t_value",
            "hasbrouck_lambda",
            "hasbrouck_lambda_t_value",
        ]

        # Add entropy columns for selected types
        self.columns = base_cols.copy()
        for en_type in self.entropy_types:
            self.columns.append(f"tick_rule_entropy_{en_type}")

        if self.volume_encoding is not None:
            for en_type in self.entropy_types:
                self.columns.append(f"volume_entropy_{en_type}")

        if self.pct_encoding is not None:
            for en_type in self.entropy_types:
                self.columns.append(f"pct_entropy_{en_type}")

    def get_features(
        self,
        verbose: bool = True,
        to_csv: bool = False,
        output_path: Optional[str] = None,
        max_batches: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generate microstructural features from tick data.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress information.

        to_csv : bool, default=False
            Whether to save results to CSV file.

        output_path : Optional[str], default=None
            Path to save CSV file if to_csv=True.

        max_batches : Optional[int], default=None
            Maximum number of batches to process.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame of microstructural features if to_csv=False, else None.

        Raises
        ------
        ValueError
            If to_csv=True but output_path is not provided.

        Examples
        --------
        >>> generator = OptimizedMicrostructuralFeaturesGenerator(tick_data, bar_indices)
        >>> features = generator.get_features(verbose=True)
        >>> features.shape
        (1000, 22)  # Number of bars × number of features
        """
        if to_csv and output_path is None:
            raise ValueError("output_path must be specified when to_csv=True")

        if to_csv:
            open(output_path, "w").close()
            header = True

        all_bars = []
        batch_count = 0

        for batch in self.generator_object:
            if verbose:
                print(f"Processing batch {batch_count + 1}...")

            # Process batch
            batch_bars, stop_flag = self._process_batch_vectorized(batch)

            if to_csv:
                pd.DataFrame(batch_bars, columns=self.columns).to_csv(
                    output_path, header=header, index=False, mode="a"
                )
                header = False
            else:
                all_bars.extend(batch_bars)

            batch_count += 1

            # Early stopping conditions
            if stop_flag:
                break
            if max_batches and batch_count >= max_batches:
                break

        # Return results
        if not to_csv and all_bars:
            return pd.DataFrame(all_bars, columns=self.columns)
        return None if to_csv else pd.DataFrame(columns=self.columns)

    def _process_batch_vectorized(self, batch: pd.DataFrame) -> tuple:
        """
        Process an entire batch using vectorized operations.

        Parameters
        ----------
        batch : pd.DataFrame
            Batch of tick data with columns: date_time, price, volume

        Returns
        -------
        tuple
            - List of bar features
            - Boolean indicating if all bars have been processed
        """
        # Convert to numpy for speed
        dates = batch.iloc[:, 0].values
        prices = batch.iloc[:, 1].values.astype(np.float64)
        volumes = batch.iloc[:, 2].values.astype(np.float64)

        # Get bar indices for this batch
        start_idx = self.current_bar_idx
        end_idx = start_idx

        # Find how many bars we can complete in this batch
        cumulative_ticks = 0
        bar_indices_in_batch = []

        while end_idx < len(self.tick_num_array):
            cumulative_ticks += self.tick_num_array[end_idx]
            if cumulative_ticks <= len(prices):
                bar_indices_in_batch.append(cumulative_ticks - 1)  # Zero-indexed
                end_idx += 1
            else:
                break

        if not bar_indices_in_batch:
            return [], False

        # Process ticks with Numba
        results = _process_ticks_numba(
            prices, volumes, np.array(bar_indices_in_batch, dtype=np.int32)
        )

        trade_sizes, tick_rules, dollar_sizes, log_rets, price_diffs, ticks_per_bar = results

        # Compute basic features
        avg_tick_sizes, tick_rule_sums, vwap_values = _compute_basic_features_numba(
            trade_sizes, tick_rules, dollar_sizes, ticks_per_bar
        )

        # Get date for each bar
        bar_dates = dates[bar_indices_in_batch]

        # Process each bar
        bars_list = []

        for i in range(len(bar_indices_in_batch)):
            n_ticks = ticks_per_bar[i]
            if n_ticks == 0:
                continue

            # Prepare bar data
            bar_data = {
                "date_time": bar_dates[i],
                "avg_tick_size": avg_tick_sizes[i],
                "tick_rule_sum": tick_rule_sums[i],
                "vwap": vwap_values[i],
            }

            # Get valid slice for this bar
            bar_tick_rules = tick_rules[i, :n_ticks]
            bar_price_diffs = price_diffs[i, :n_ticks]
            bar_trade_sizes = trade_sizes[i, :n_ticks]
            bar_log_rets = log_rets[i, :n_ticks]
            bar_dollar_sizes = dollar_sizes[i, :n_ticks]

            # Compute lambdas
            kyle_result = get_trades_based_kyle_lambda(
                bar_price_diffs.tolist(), bar_trade_sizes.tolist(), bar_tick_rules.tolist()
            )
            bar_data.update({"kyle_lambda": kyle_result[0], "kyle_lambda_t_value": kyle_result[1]})

            amihud_result = get_trades_based_amihud_lambda(
                bar_log_rets.tolist(), bar_dollar_sizes.tolist()
            )
            bar_data.update(
                {"amihud_lambda": amihud_result[0], "amihud_lambda_t_value": amihud_result[1]}
            )

            hasbrouck_result = get_trades_based_hasbrouck_lambda(
                bar_log_rets.tolist(), bar_dollar_sizes.tolist(), bar_tick_rules.tolist()
            )
            bar_data.update(
                {
                    "hasbrouck_lambda": hasbrouck_result[0],
                    "hasbrouck_lambda_t_value": hasbrouck_result[1],
                }
            )

            # Compute entropy features
            self._compute_entropy_features(bar_data, bar_tick_rules, bar_trade_sizes, bar_log_rets)

            # Convert to list in column order
            bar_row = [bar_data.get(col, np.nan) for col in self.columns]
            bars_list.append(bar_row)

        # Update state
        self.current_bar_idx += len(bar_indices_in_batch)
        stop_flag = self.current_bar_idx >= len(self.tick_num_array)

        return bars_list, stop_flag

    def _compute_entropy_features(
        self, bar_data: Dict, tick_rules: np.ndarray, trade_sizes: np.ndarray, log_rets: np.ndarray
    ):
        """
        Compute entropy features for a bar.

        Parameters
        ----------
        bar_data : Dict
            Dictionary to update with entropy features

        tick_rules : np.ndarray
            Array of tick rules for the bar

        trade_sizes : np.ndarray
            Array of trade sizes for the bar

        log_rets : np.ndarray
            Array of log returns for the bar
        """
        # 1. Tick rule entropy
        tick_rule_msg = self._fast_encode_tick_rule(tick_rules)

        for en_type in self.entropy_types:
            entropy_func = self._get_entropy_function(en_type)
            bar_data[f"tick_rule_entropy_{en_type}"] = entropy_func(tick_rule_msg)

        # 2. Volume entropy (if encoding provided)
        if self.volume_encoding is not None:
            volume_msg = encode_array(trade_sizes.tolist(), self.volume_encoding)
            for en_type in self.entropy_types:
                entropy_func = self._get_entropy_function(en_type)
                bar_data[f"volume_entropy_{en_type}"] = entropy_func(volume_msg)

        # 3. Percentage entropy (if encoding provided)
        if self.pct_encoding is not None:
            pct_msg = encode_array(log_rets.tolist(), self.pct_encoding)
            for en_type in self.entropy_types:
                entropy_func = self._get_entropy_function(en_type)
                bar_data[f"pct_entropy_{en_type}"] = entropy_func(pct_msg)

    def _fast_encode_tick_rule(self, tick_rule_array: np.ndarray) -> str:
        """
        Fast encoding of tick rule arrays using pre-allocation.

        Parameters
        ----------
        tick_rule_array : np.ndarray
            Array of tick rules (-1, 0, 1)

        Returns
        -------
        str
            Encoded message with 'a' for 1, 'b' for -1, 'c' for 0
        """
        n = len(tick_rule_array)
        chars = [""] * n

        for i in range(n):
            val = tick_rule_array[i]
            if val == 1:
                chars[i] = "a"
            elif val == -1:
                chars[i] = "b"
            elif val == 0:
                chars[i] = "c"
            else:
                chars[i] = "c"  # Default

        return "".join(chars)

    def _get_entropy_function(self, entropy_type: str):
        """
        Get the entropy function for a given type.

        Parameters
        ----------
        entropy_type : str
            Type of entropy ('shannon', 'lempel_ziv', 'plug_in', 'konto')

        Returns
        -------
        callable
            Entropy calculation function

        Raises
        ------
        ValueError
            If entropy_type is not supported
        """
        entropy_functions = {
            "shannon": get_shannon_entropy,
            "lempel_ziv": get_lempel_ziv_entropy,
            "plug_in": get_plug_in_entropy,
            "konto": get_konto_entropy,
        }

        if entropy_type not in entropy_functions:
            raise ValueError(f"Unsupported entropy type: {entropy_type}")

        return entropy_functions[entropy_type]

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Validate CSV format.

        Parameters
        ----------
        test_batch : pd.DataFrame
            First row of the dataset

        Raises
        ------
        AssertionError
            If CSV format is invalid
        """
        assert test_batch.shape[1] == 3, "Must have exactly 3 columns: date_time, price, volume"
        assert isinstance(test_batch.iloc[0, 1], (float, np.floating)), "Price column must be float"
        assert not isinstance(test_batch.iloc[0, 2], str), "Volume column must be numeric"

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print("Column 0 is not a valid datetime format:", test_batch.iloc[0, 0])


# ========== SIMPLIFIED VERSION ==========
class SimpleMicrostructuralFeaturesGenerator(OptimizedMicrostructuralFeaturesGenerator):
    """
    Simplified version that only computes Shannon and Lempel-Ziv entropy.

    This is recommended for most applications where you need basic entropy measures
    without the computational overhead of all four entropy types.

    Usage:
    ------
    >>> generator = SimpleMicrostructuralFeaturesGenerator(
    ...     tick_data, bar_indices,
    ...     volume_encoding={'quantile': 10},
    ...     pct_encoding={'sigma': 0.01}
    ... )
    >>> features = generator.get_features()
    """

    def __init__(
        self,
        trades_input: Union[str, pd.DataFrame],
        tick_num_series: pd.Series,
        batch_size: int = 2_000_000,
        volume_encoding: Optional[Dict] = None,
        pct_encoding: Optional[Dict] = None,
    ):
        # Only compute Shannon and Lempel-Ziv entropy
        super().__init__(
            trades_input=trades_input,
            tick_num_series=tick_num_series,
            batch_size=batch_size,
            volume_encoding=volume_encoding,
            pct_encoding=pct_encoding,
            entropy_types=["shannon", "lempel_ziv"],  # Only these two
        )
