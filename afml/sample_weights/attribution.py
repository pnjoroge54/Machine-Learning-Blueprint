"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
"""

import numpy as np
import pandas as pd

from ..cache import smart_cacheable
from ..sampling.concurrent import (
    get_av_uniqueness_from_triple_barrier,
    num_concurrent_events,
)
from ..util.multiprocess import mp_pandas_obj


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.10, page 69.

    Determination of Sample Weight by Absolute Return Attribution

    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) Close prices
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Sample weights based on number return and concurrency for molecule
    """

    ret = np.log(close_series).diff()  # Log-returns, so that they are additive

    weights = {}
    for t_in, t_out in label_endtime.loc[molecule].items():
        # Weights depend on returns and label concurrency
        weights[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()

    weights = pd.Series(weights)
    return weights.abs()


@smart_cacheable
def get_weights_by_return(
    triple_barrier_events,
    close_series,
    num_threads=5,
    num_conc_events=None,
    verbose=True,
):
    """
    Determination of Sample Weight by Absolute Return Attribution
    Modified to ensure compatibility with precomputed num_conc_events

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices
    :param num_threads: (int) Number of threads
    :param num_conc_events: (pd.Series) Precomputed concurrent events count
    :param verbose: (bool) Report progress
    :return: (pd.Series) Sample weights
    """
    # Validate input
    assert not triple_barrier_events.isnull().values.any(), "NaN values in events"
    assert not triple_barrier_events.index.isnull().any(), "NaN values in index"

    # Create processing pipeline for num_conc_events
    def process_concurrent_events(ce):
        """Process concurrent events to ensure proper format and indexing."""
        ce = ce.loc[~ce.index.duplicated(keep="last")]
        ce = ce.reindex(close_series.index).fillna(0)
        if isinstance(ce, pd.Series):
            ce = ce.to_frame()
        return ce

    # Handle num_conc_events (whether provided or computed)
    if num_conc_events is None:
        num_conc_events = mp_pandas_obj(
            num_concurrent_events,
            ("molecule", triple_barrier_events.index),
            num_threads,
            close_series_index=close_series.index,
            label_endtime=triple_barrier_events["t1"],
            verbose=verbose,
        )
        processed_ce = process_concurrent_events(num_conc_events)
    else:
        # Ensure precomputed value matches expected format
        processed_ce = process_concurrent_events(num_conc_events.copy())

        # Verify index compatibility
        missing_in_close = processed_ce.index.difference(close_series.index)
        assert missing_in_close.empty, (
            f"num_conc_events contains {len(missing_in_close)} " "indices not in close_series"
        )

    # Compute weights using processed concurrent events
    weights = mp_pandas_obj(
        _apply_weight_by_return,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=processed_ce,  # Use processed version
        close_series=close_series,
        verbose=verbose,
    )

    # Normalize weights
    weights *= weights.shape[0] / weights.sum()
    return weights


def get_weights_by_time_decay(
    triple_barrier_events,
    close_series,
    num_threads=4,
    last_weight=1,
    linear=True,
    av_uniqueness=None,
    verbose=True,
):
    """
    Advances in Financial Machine Learning, Snippet 4.11, page 70.
    Implementation of Time Decay Factors
    """
    assert (
        bool(triple_barrier_events.isnull().values.any()) is False
        and bool(triple_barrier_events.index.isnull().any()) is False
    ), "NaN values in triple_barrier_events, delete nans"

    # Get average uniqueness if not provided
    if av_uniqueness is None:
        av_uniqueness = get_av_uniqueness_from_triple_barrier(
            triple_barrier_events, close_series, num_threads, verbose=verbose
        )
    elif isinstance(av_uniqueness, pd.Series):
        av_uniqueness = av_uniqueness.to_frame()

    # Calculate cumulative time weights
    cum_time_weights = av_uniqueness["tW"].sort_index().cumsum()

    if linear:
        # Apply linear decay (your existing linear code is correct)
        if last_weight >= 0:
            slope = (1 - last_weight) / cum_time_weights.iloc[-1]
        else:
            slope = 1 / ((last_weight + 1) * cum_time_weights.iloc[-1])
        const = 1 - slope * cum_time_weights.iloc[-1]
        weights = const + slope * cum_time_weights
        weights[weights < 0] = 0
        return weights
    else:
        # Apply exponential decay
        if last_weight == 1:
            return pd.Series(1.0, index=cum_time_weights.index)

        if cum_time_weights.iloc[-1] == 0:
            return pd.Series(1.0, index=cum_time_weights.index)

        # Calculate normalized position (0 = newest, 1 = oldest)
        if last_weight >= 0:
            # For last_weight >= 0, use standard exponential decay
            normalized_position = (cum_time_weights - cum_time_weights.iloc[0]) / (
                cum_time_weights.iloc[-1] - cum_time_weights.iloc[0]
            )
            weights = last_weight**normalized_position
        else:
            # For last_weight < 0, implement cutoff (similar to linear case)
            # This is more complex for exponential - you might want to reconsider this case
            cutoff_threshold = abs(last_weight)
            normalized_position = (cum_time_weights - cum_time_weights.iloc[0]) / (
                cum_time_weights.iloc[-1] - cum_time_weights.iloc[0]
            )
            weights = (1 - cutoff_threshold) ** normalized_position
            weights[weights < 0] = 0

        return weights
