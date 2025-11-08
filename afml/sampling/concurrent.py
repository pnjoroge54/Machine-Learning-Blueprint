"""
Logic regarding concurrent labels from chapter 4.
"""

import pandas as pd

from ..cache import robust_cacheable
from ..util.multiprocess import mp_pandas_obj


def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched)
    to compute the number of concurrent events per bar.

    :param close_series_index: (pd.Series) Close prices index
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.Series) Number concurrent labels for each datetime index
    """
    # Find events that span the period [molecule[0], molecule[1]]
    label_endtime = label_endtime.fillna(
        close_series_index[-1]
    )  # Unclosed events still must impact other weights
    label_endtime = label_endtime[
        label_endtime >= molecule[0]
    ]  # Events that end at or after molecule[0]
    # Events that start at or before t1[molecule].max()
    label_endtime = label_endtime.loc[: label_endtime[molecule].max()]

    # Count events spanning a bar
    nearest_index = close_series_index.searchsorted(
        pd.DatetimeIndex([label_endtime.index[0], label_endtime.max()])
    )
    count = pd.Series(0, index=close_series_index[nearest_index[0] : nearest_index[1] + 1])
    for t_in, t_out in label_endtime.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0] : label_endtime[molecule].max()]


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched)
    to compute the number of concurrent events per bar.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """
    wght = {}
    for t_in, t_out in label_endtime.loc[molecule].items():
        wght[t_in] = (1.0 / num_conc_events.loc[t_in:t_out]).mean()

    wght = pd.Series(wght)
    return wght


@robust_cacheable
def get_num_conc_events(events, close, num_threads=4, verbose=True):
    num_conc_events = mp_pandas_obj(
        num_concurrent_events,
        ("molecule", events.index),
        num_threads,
        close_series_index=close.index,
        label_endtime=events["t1"],
        verbose=verbose,
    )
    return num_conc_events


@robust_cacheable
def get_av_uniqueness_from_triple_barrier(
    triple_barrier_events, close_series, num_threads, num_conc_events=None, verbose=True
):
    """
    This function is the orchestrator to derive average sample uniqueness from a dataset labeled by the triple barrier
    method.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param num_conc_events: (pd.Series) Number concurrent labels for each datetime index
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Average uniqueness over event's lifespan for each index in triple_barrier_events
    """
    out = pd.DataFrame()

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
        num_conc_events = get_num_conc_events(
            triple_barrier_events, close_series, num_threads, verbose
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

    out["tW"] = mp_pandas_obj(
        _get_average_uniqueness,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=processed_ce,
        verbose=verbose,
    )
    return out
