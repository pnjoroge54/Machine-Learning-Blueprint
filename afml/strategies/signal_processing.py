from typing import Tuple, Union

import pandas as pd
from loguru import logger

from ..filters.filters import cusum_filter
from .strategies import BaseStrategy


def get_entries(
    strategy: BaseStrategy,
    data: pd.DataFrame,
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

        filter_threshold (Union[float, pd.Series], optional): Threshold for CUSUM filter.
                                                             - If float: Fixed threshold
                                                             - If Series: Dynamic threshold
                                                             Only used if not None.
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

            - t_events (pd.DatetimeIndex): Timestamps where new positions were initiated.
                                          These represent actual trade entry points,
                                          not every signal occurrence.

    See Also:
        timing_of_flattening_and_flips(): Analyzes position exit/reversal events
        get_positions_from_events(): Gets target positions from events with columns "side" and "t1"
        cusum_filter(): Event filtering based on cumulative price movements

    References:
        Based on concepts from "Advances in Financial Machine Learning" by Marcos López de Prado,
        particularly around event-driven backtesting and signal processing.
    """
    primary_signals = strategy.generate_signals(data)
    signal_mask = primary_signals != 0

    # Vectorized CUSUM filter application
    if filter_threshold is not None:
        try:
            close = data["close"].copy()
        except Exception as e:
            logger.error(f"Check your data: {e}")

        if not isinstance(filter_threshold, (pd.Series, float)):
            raise TypeError("filter_threshold must be a Series or a float")
        elif isinstance(filter_threshold, pd.Series):
            filter_threshold = filter_threshold.copy().dropna()
            close = close.reindex(filter_threshold.index)

        filtered_events = cusum_filter(close, filter_threshold)
        signal_mask &= primary_signals.index.isin(filtered_events)
        thres = (
            f"(threshold = {filter_threshold:.4%})"
            if isinstance(filter_threshold, float)
            else "using series"
        )
        msg = f" selected by CUSUM filter {thres}"
    else:
        # Vectorized signal change detection
        if on_crossover:
            signal_mask &= primary_signals != primary_signals.shift()
            msg = " generated from crossovers"
        else:
            msg = ""

    t_events = primary_signals.index[signal_mask]
    n = len(t_events)
    logger.info(
        f"{strategy.get_strategy_name()} | {n:,} ({n / sum(primary_signals != 0):.2%}) trade events{msg}."
    )

    return primary_signals, t_events
