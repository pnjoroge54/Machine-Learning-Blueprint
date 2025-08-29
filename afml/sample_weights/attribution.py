"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
"""

import numpy as np
import pandas as pd

from ..cache import cacheable
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


@cacheable
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
        ce = ce.loc[~ce.index.duplicated(keep="last")]
        return ce.reindex(close_series.index).fillna(0)

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
    num_threads=5,
    decay=1,
    linear=True,
    av_uniqueness=None,
    verbose=True,
):
    """
    Advances in Financial Machine Learning, Snippet 4.11, page 70.

    Implementation of Time Decay Factors

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param decay: (int) Decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regardless of how old
        - decay = 0 means that weights converge linearly (exponenentially) to zero, as they become older
        - decay < 0 means that the oldest portion c of the observations receive zero weight (i.e they are erased from memory)
    :param linear: (bool) If True, linear decay is applied, else exponential decay
    :param av_uniqueness: (pd.Series) Average uniqueness of events
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Sample weights based on time decay factors
    """
    assert (
        bool(triple_barrier_events.isnull().values.any()) is False
        and bool(triple_barrier_events.index.isnull().any()) is False
    ), "NaN values in triple_barrier_events, delete nans"

    # Apply piecewise-linear or exponential decay to observed uniqueness
    # Newest observation gets weight=1, oldest observation gets weight=decay
    if av_uniqueness is None:
        av_uniqueness = get_av_uniqueness_from_triple_barrier(
            triple_barrier_events, close_series, num_threads, verbose=verbose
        )
    else:
        av_uniqueness = av_uniqueness.copy()
    decay_w = av_uniqueness["tW"].sort_index().cumsum()

    if linear:
        # Apply linear decay
        if decay >= 0:
            slope = (1 - decay) / decay_w.iloc[-1]
        else:
            slope = 1 / ((decay + 1) * decay_w.iloc[-1])
        const = 1 - slope * decay_w.iloc[-1]
        decay_w = const + slope * decay_w
        decay_w[decay_w < 0] = 0  # Weights can't be negative
        return decay_w
    else:
        # Apply exponential decay
        # Handle edge cases
        if decay == 1:
            return pd.Series(1.0, index=decay_w.index)
        if decay_w.iloc[-1] == 0:  # Avoid division by zero if all tW are 0
            return pd.Series(1.0, index=decay_w.index)
        age = decay_w.iloc[-1] - decay_w
        norm_age = age / age.max()  # Scale age to be in [0, 1]
        exp_decay_w = decay**norm_age
        return exp_decay_w
