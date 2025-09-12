from typing import Tuple, Union

import pandas as pd
from loguru import logger

from ..filters.filters import cusum_filter
from .strategies import BaseStrategy


def get_entries(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    filter_events: bool = False,
    filter_threshold: Union[float, pd.Series] = None,
    on_crossover: bool = True,
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Converts raw strategy signals into continuous trading positions and entry timestamps.

    This function transforms discrete strategy signals (which may have gaps) into
    continuous position series suitable for backtesting and performance analysis.
    It implements position persistence - once a signal is generated, the position
    is held until the next signal change or exit condition.

    Key Distinction:
    - Raw strategy signals: Momentary decisions (e.g., [0, 1, 0, 0, -1, 0])
    - Trading positions: Continuous holdings (e.g., [0, 1, 1, 1, -1, -1])

    Args:
        strategy (BaseStrategy): Trading strategy instance that implements generate_signals().
                                Must return pd.Series with values {-1, 0, 1} where:
                                - 1 = Long signal
                                - -1 = Short signal
                                - 0 = No signal/neutral

        data (pd.DataFrame): Market data DataFrame. Must contain 'close' column if
                            filter_events=True. Index must be DatetimeIndex.

        filter_events (bool, optional): Whether to apply CUSUM event filtering.
                                       If True, only signals occurring during significant
                                       price movements (determined by filter_threshold)
                                       are considered. Defaults to False.

        filter_threshold (Union[float, pd.Series], optional): Threshold for CUSUM filter.
                                                             - If float: Fixed threshold
                                                             - If Series: Dynamic threshold
                                                             Only used if filter_events=True.
                                                             Defaults to None.

        on_crossover (bool, optional): Signal filtering mode. Defaults to True.
                                      - True: Only generate positions on signal changes
                                              (recommended for most strategies)
                                      - False: Generate positions for ALL non-zero signals
                                              (may create excessive position entries)

    Returns:
        Tuple[pd.Series, pd.DatetimeIndex]: A tuple containing:
            - side (pd.Series): Continuous position series with same index as data.
                               Values are {-1, 0, 1} representing:
                               - 1: Long position
                               - -1: Short position
                               - 0: No position/flat
                               Forward-filled to maintain positions between signals.

            - t_events (pd.DatetimeIndex): Timestamps where new positions were initiated.
                                          These represent actual trade entry points,
                                          not every signal occurrence.

    Usage Examples:

        # Basic usage - get trading positions
        >>> strategy = BollingerMeanReversionStrategy()
        >>> side, t_events = get_entries(strategy, data)
        >>> print(f"Generated {len(t_events)} trades")
        >>> print(f"Days with positions: {(side != 0).sum()}")

        # Performance analysis
        >>> returns = (side.shift(1) * data['close'].pct_change()).dropna()
        >>> total_return = (1 + returns).prod() - 1

        # With CUSUM filtering for significant price moves only
        >>> side, t_events = get_entries(strategy, data,
        ...                             filter_events=True,
        ...                             filter_threshold=0.02)

        # Include all signals (not recommended for most cases)
        >>> side, t_events = get_entries(strategy, data, on_crossover=False)

    Signal Processing Logic:

        1. **Raw Signal Generation**: Calls strategy.generate_signals(data)
        2. **Signal Filtering**: Applied in order of priority:
           a) CUSUM filtering (if filter_events=True)
           b) Crossover filtering (if on_crossover=True)
        3. **Position Creation**: Forward-fills signals to create continuous positions

        Filter Behavior:
        - filter_events=True, on_crossover=True: Most selective (recommended)
        - filter_events=False, on_crossover=True: Standard crossover mode
        - filter_events=False, on_crossover=False: Least selective
        - filter_events=True, on_crossover=False: CUSUM only

    Performance Considerations:

        - Vectorized operations for computational efficiency
        - CUSUM filter adds computational overhead but improves signal quality
        - on_crossover=True significantly reduces number of trade events

    Common Patterns:

        # Standard backtesting workflow
        >>> strategy = MovingAverageCrossoverStrategy()
        >>> side, t_events = get_entries(strategy, data)
        >>>
        >>> # Calculate returns using positions (NOT raw signals)
        >>> returns = (side.shift(1) * data['close'].pct_change()).dropna()
        >>>
        >>> # Analyze individual trades
        >>> trade_ends = timing_of_flattening_and_flips(side)
        >>>
        >>> # Performance metrics
        >>> metrics = calculate_performance_metrics(returns, data.index, positions=side)

        # Signal quality analysis
        >>> raw_signals = strategy.generate_signals(data)
        >>> side, t_events = get_entries(strategy, data)
        >>>
        >>> signal_utilization = len(t_events) / (raw_signals != 0).sum()
        >>> print(f"Signal utilization: {signal_utilization:.1%}")

    Raises:
        Exception: If data doesn't contain 'close' column when filter_events=True
        TypeError: If filter_threshold is not float or pd.Series when filter_events=True

    Notes:
        - The returned 'side' series is what should be used for performance calculations
        - The returned 't_events' represents trade initiation times, useful for analysis
        - For meta-learning applications, use raw strategy signals for label creation
        - For backtesting and performance analysis, always use the returned 'side' series
        - Position changes can be analyzed using timing_of_flattening_and_flips(side)

    See Also:
        timing_of_flattening_and_flips(): Analyzes position exit/reversal events
        calculate_performance_metrics(): Computes strategy performance using positions
        cusum_filter(): Event filtering based on cumulative price movements

    References:
        Based on concepts from "Advances in Financial Machine Learning" by Marcos López de Prado,
        particularly around event-driven backtesting and signal processing.
    """
    primary_signals = strategy.generate_signals(data)
    signal_mask = primary_signals != 0

    # Vectorized CUSUM filter application
    if filter_events:
        try:
            close = data.close
        except Exception as e:
            logger.error(f"Check your data: {e}")

        if not isinstance(filter_threshold, (pd.Series, float)):
            raise TypeError("filter_threshold must be a Series or a float")
        elif isinstance(filter_threshold, pd.Series):
            filter_threshold = filter_threshold.copy().dropna()
            close = close.reindex(filter_threshold.index)

        filtered_events = cusum_filter(close, filter_threshold)
        signal_mask &= primary_signals.index.isin(filtered_events)
    else:
        # Vectorized signal change detection
        if on_crossover:
            signal_mask &= primary_signals != primary_signals.shift()

    t_events = primary_signals.index[signal_mask]

    side = pd.Series(index=data.index, name="side")
    side.loc[t_events] = primary_signals.loc[t_events]
    side = side.ffill().fillna(0).astype("int8")

    if filter_events:
        s = " generated by CUSUM filter"
    elif on_crossover:
        s = " generated by crossover"
    else:
        s = ""

    logger.info(f"Generated {len(t_events):,} trade events{s}.")

    return side, t_events
