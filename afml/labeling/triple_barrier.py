"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd
from numba import njit, prange

from ..sample_weights.optimized_attribution import (
    get_weights_by_return_optimized,
    get_weights_by_time_decay_optimized,
)
from ..sampling.optimized_concurrent import (
    get_av_uniqueness_from_triple_barrier_optimized,
    get_num_conc_events_optimized,
)

# pylint: disable=invalid-name, too-many-arguments, too-many-locals, too-many-statements, too-many-branches


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1_optimized(close, events, pt_sl):
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method (Numba Optimized)

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function for more details)
    :param pt_sl: (list) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    # 1. Prepare data for Numba
    # Get integer locations for events and vertical barriers relative to the `close` series
    event_locs = close.index.get_indexer(events.index)
    t1_locs = close.index.get_indexer(events["t1"])

    # Convert pandas objects to NumPy arrays for Numba compatibility
    close_val = close.values
    t1_locs[t1_locs == -1] = len(close_val) - 1  # Handle NaT in t1
    trgt_val = events["trgt"].values
    side_val = events["side"].values
    pt_sl_arr = np.array(pt_sl, dtype=float)

    # 2. Call the Numba-jitted function
    sl_hit_locs, pt_hit_locs = _find_barrier_hits(
        close_val, event_locs, t1_locs, trgt_val, side_val, pt_sl_arr
    )

    # 3. Process results: Convert integer locations back to timestamps
    out = events[["t1"]].copy()
    out["sl"] = pd.NaT
    out["pt"] = pd.NaT

    # Find where hits occurred (locs are not -1)
    sl_hit_mask = sl_hit_locs != -1
    pt_hit_mask = pt_hit_locs != -1

    # Get index labels and timestamps for events where a barrier was hit
    sl_idx_labels = events.index[sl_hit_mask]
    pt_idx_labels = events.index[pt_hit_mask]
    sl_timestamps = close.index[sl_hit_locs[sl_hit_mask]]
    pt_timestamps = close.index[pt_hit_locs[pt_hit_mask]]

    # Assign the timestamps to the correct event rows
    out.loc[sl_idx_labels, "sl"] = sl_timestamps
    out.loc[pt_idx_labels, "pt"] = pt_timestamps

    return out


@njit(parallel=True, cache=True)
def _find_barrier_hits(close_val, event_locs, t1_locs, trgt, side, pt_sl_arr):
    """
    Core Numba-jitted logic to find the first time barriers are touched.
    Operates entirely on NumPy arrays for maximum performance.
    """
    pt_level = pt_sl_arr[0]
    sl_level = pt_sl_arr[1]

    # Use arrays of int64 to store integer index locations of hits
    sl_hit_locs = np.full(event_locs.shape[0], -1, dtype=np.int64)
    pt_hit_locs = np.full(event_locs.shape[0], -1, dtype=np.int64)

    # Numba can parallelize this loop automatically
    for i in prange(event_locs.shape[0]):
        start_loc = event_locs[i]
        end_loc = t1_locs[i]

        # Skip if the event start is not found in the price series
        if start_loc == -1:
            continue

        start_price = close_val[start_loc]
        event_side = side[i]

        # Set profit-taking and stop-loss levels for the current event
        pt = np.log(1 + pt_level * trgt[i]) if pt_level > 0 else np.inf
        sl = np.log(1 - sl_level * trgt[i]) if sl_level > 0 else -np.inf

        # Iterate through the price path for the event
        for j in range(start_loc + 1, end_loc + 1):
            # Calculate path-return
            ret = np.log(close_val[j] / start_price) * event_side

            # Check for stop-loss hit (if not already found)
            if sl_hit_locs[i] == -1 and ret <= sl:
                sl_hit_locs[i] = j

            # Check for profit-taking hit (if not already found)
            if pt_hit_locs[i] == -1 and ret >= pt:
                pt_hit_locs[i] = j

            # If both barriers have been hit, we can stop searching for this event
            if sl_hit_locs[i] != -1 and pt_hit_locs[i] != -1:
                break

    return sl_hit_locs, pt_hit_locs


# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events, close, num_bars=0, **time_delta_kwargs):
    """
    Advances in Financial Machine Learning, Enhanced Implementation.

    Adding a Vertical Barrier

    For each event in t_events, finds the timestamp of the next price bar at or immediately after:
    - A fixed number of bars (for activity-based sampling), OR
    - A time delta (for time-based sampling)

    This function creates a series of vertical barrier timestamps aligned with the original events index.
    Out-of-bound barriers are marked with NaT for downstream handling.

    :param t_events: (pd.Series) Series of event timestamps (e.g., from symmetric CUSUM filter)
    :param close: (pd.Series) Close price series with DateTimeIndex
    :param num_bars: (int) Number of bars for vertical barrier (activity-based mode).
                     Takes precedence over time delta parameters when > 0.
    :param time_delta_kwargs: Time components for time-based barrier (mutually exclusive with num_bars):
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    :return: (pd.Series) Vertical barrier timestamps with same index as t_events.
             Out-of-bound events return pd.NaT.

    Example:
        ### Activity-bar mode (tick/volume/dollar bars)
        vertical_barriers = add_vertical_barrier(t_events, close, num_bars=10)

        ### Time-based mode
        vertical_barriers = add_vertical_barrier(t_events, close, days=1, hours=3)
    """
    # Validate inputs
    if num_bars and time_delta_kwargs:
        raise ValueError("Use either num_bars OR time deltas, not both")

    # BAR-BASED VERTICAL BARRIERS
    if num_bars > 0:
        indices = close.index.get_indexer(t_events, method="nearest")
        t1 = []
        for i in indices:
            if i == -1:  # Event not found
                t1.append(pd.NaT)
            else:
                end_loc = i + num_bars
                t1.append(close.index[end_loc] if end_loc < len(close) else pd.NaT)
        return pd.Series(t1, index=t_events, name="t1")

    # TIME-BASED VERTICAL BARRIERS
    td = pd.Timedelta(**time_delta_kwargs) if time_delta_kwargs else pd.Timedelta(0)
    barrier_times = t_events + td

    # Find next index positions
    t1_indices = np.searchsorted(close.index, barrier_times, side="left")
    t1 = []
    for idx in t1_indices:
        if idx < len(close):
            t1.append(close.index[idx])
        else:
            t1.append(pd.NaT)  # Mark out-of-bound for downstream

    return pd.Series(t1, index=t_events, name="t1")


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(
    close,
    t_events,
    pt_sl,
    target,
    min_ret,
    vertical_barrier_times=False,
    side_prediction=None,
):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        E.g.: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param vertical_barrier_times: (pd.Series) Timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Triple-barrier events with the following columns:
    - index: Event start times
    - t1: Event end times
    - trgt: Target volatility
    - side: Optional. Algo's position side
    """

    # 1. Get target
    target = target.reindex(t_events)
    target = target[target > min_ret]  # min_ret

    # 2. Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

    # 3. Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({"t1": vertical_barrier_times, "trgt": target, "side": side_}, axis=1)
    events = events.dropna(subset=["trgt"])

    # Apply Triple Barrier
    first_touch_dates = apply_pt_sl_on_t1_optimized(close, events, pt_sl_)

    events["t1"] = first_touch_dates.dropna(how="all").min(axis=1)  # pd.min ignores nan

    if side_prediction is None:
        events = events.drop("side", axis=1)

    return events


# Snippet 3.9, page 55, Question 3.3
@njit(parallel=True, cache=True)
def barrier_touched(ret, target, pt_sl):
    """
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.

    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param ret: (np.array) Log-returns
    :param target: (np.array) Volatility target
    :param pt_sl: (np.array) Profit-taking and stop-loss multiples
    :return: (np.array) Labels
    """
    N = ret.shape[0]  # Number of events
    store = np.empty(N, dtype=np.int8)  # Store labels in an array

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Iterate through the DataFrame and check if the vertical barrier was reached
    for i in prange(N):
        pt_level_reached = ret[i] > np.log(1 + profit_taking_multiple * target[i])
        sl_level_reached = ret[i] < np.log(1 - stop_loss_multiple * target[i])

        if ret[i] > 0.0 and pt_level_reached:
            # Top barrier reached
            store[i] = 1
        elif ret[i] < 0.0 and sl_level_reached:
            # Bottom barrier reached
            store[i] = -1
        else:
            # Vertical barrier reached
            store[i] = 0

    return store


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def get_bins(triple_barrier_events, close, vertical_barrier_zero=False, pt_sl=None):
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (pd.DataFrame) Events DataFrame with the following structure:
    - **index**: pd.DatetimeIndex of event start times
    - **t1**: (pd.Series) Event end times
    - **trgt**: (pd.Series) Target volatility
    - **pt**: (pd.Series) Profit-taking multiple
    - **sl**: (pd.Series) Stop-loss multiple
    - **side**: (pd.Series, optional) Algo's position side
      Labeling behavior depends on the presence of 'side':
        - Case 1: If 'side' not in events → `bin ∈ {-1, 1}` (label by price action)
        - Case 2: If 'side' is present    → `bin ∈ {0, 1}`  (label by PnL — meta-labeling)
    :param close: (pd.Series) Close prices
    :param vertical_barrier_zero: (bool) If True, sets bin to 0 only for events where the vertical barrier is touched first; otherwise, labeling is determined by the sign of the return.
    :param pt_sl: (list) Profit-taking and stop-loss multiples
    :return: (pd.DataFrame)
    Events DataFrame with the following columns:
    - index: Event start times
    - t1: Event end times
    - trgt: Target volatility
    - side: Optional. Algo's position side
    - ret: Returns of the event
    - bin: Labels for the event, where 1 is a positive return, -1 is a negative return, and 0 is a vertical barrier hit
    """

    # 1. Align prices with their respective events
    events = triple_barrier_events.dropna(subset=["t1"])
    all_dates = events.index.union(other=events["t1"].array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # 2. Create out DataFrame
    out_df = events[["t1"]].copy()
    out_df["ret"] = np.log(prices.loc[events["t1"].array].array / prices.loc[events.index])
    out_df["trgt"] = events["trgt"]

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events:
        out_df["ret"] *= events["side"]  # meta-labeling

    if vertical_barrier_zero:
        # Label 0 when vertical barrier reached
        pt_sl_ = [1, 1] if pt_sl is None else [pt_sl[0], pt_sl[1]]
        pt_sl = np.array(pt_sl_, dtype=float)
        out_df["bin"] = barrier_touched(out_df["ret"].values, out_df["trgt"].values, pt_sl)
    else:
        # Label is the sign of the return
        out_df["bin"] = np.where(out_df["ret"] > 0, 1, -1).astype("int8")

    # Meta labeling: label incorrect events with a 0
    if "side" in events:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    # Add the side to the output. This is useful for when a meta label model must be fit
    if "side" in triple_barrier_events.columns:
        out_df["side"] = events["side"].astype("int8")

    out_df["ret"] = np.exp(out_df["ret"]) - 1  # Convert log returns to simple returns
    return out_df


# Snippet 3.8 page 54
def drop_labels(triple_barrier_events, min_pct=0.05):
    """
    Advances in Financial Machine Learning, Snippet 3.8 page 54.

    This function recursively eliminates rare observations.

    :param triple_barrier_events: (pd.DataFrame) Triple-barrier events.
    :param min_pct: (float) A fraction used to decide if the observation occurs less than that fraction.
    :return: (pd.DataFrame) Triple-barrier events.
    """
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = triple_barrier_events["bin"].value_counts(normalize=True)

        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        print(f"dropped label: {df0.idxmin()} - {df0.min():.4%}")
        triple_barrier_events = triple_barrier_events[triple_barrier_events["bin"] != df0.idxmin()]

    return triple_barrier_events


def triple_barrier_labels(
    close: pd.Series,
    target: pd.Series,
    t_events: pd.DatetimeIndex,
    vertical_barrier_times: bool = False,
    side_prediction: pd.Series = None,
    pt_sl: list = [1, 1],
    min_ret: float = 0.0,
    min_pct: float = 0.05,
    vertical_barrier_zero: bool = False,
    verbose: bool = True,
):
    """
    Get sides or meta-labels created using triple barrier labeling method.

    :param close: (pd.Series) of trading data.
    :param target: Target volatility used to label events.
    :param t_events: Events used to generate labels, e.g. events from CUSUM filter
    :param vertical_barrier_times: Vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param pt_sl: Take-profit & stop-loss thresholds as a function of target volatility.
    :param min_ret: Minimum return allowed in sample.
    :param min_pct: Minimum weight required for item to be allowed as a class in labels bin.
    :param vertical_barrier_zero: Default is False, which sets out['ret'] value in get_bins() to the sign of
            price return when vertical barrier is touched,
            else, if True, sets it to 0 when vertical barrier is touched.
    :param verbose: Log outputs if True.
    :return: (pd.DataFrame)
    Events DataFrame with the following columns:
    - index: Event start times
    - t1: Event end times
    - trgt: Target volatility
    - side: Optional. Algo's position side
    - ret: Returns of the event
    - bin: Labels for the event, where 1 is a positive return, -1 is a negative return, and 0 is a vertical barrier hit
    """
    if verbose:
        time0 = time.perf_counter()
        if vertical_barrier_zero:
            print("Vertical barrier returns set to zero.")

    events = get_events(
        close,
        t_events,
        pt_sl,
        target,
        min_ret,
        vertical_barrier_times,
        side_prediction,
    )
    if verbose:
        print(f"get_events done after {timedelta(seconds=round(time.perf_counter() - time0))}.")
        time1 = time.perf_counter()

    events = get_bins(events, close, vertical_barrier_zero, pt_sl)
    if verbose:
        print(f"get_bins done after {timedelta(seconds=round(time.perf_counter() - time1))}.")
        time1 = time.perf_counter()

    if side_prediction is not None:
        events = drop_labels(events, min_pct)

    if verbose:
        print(f"drop_labels done after {timedelta(seconds=round(time.perf_counter() - time1))}.")

    if verbose:
        N, n_events = close.shape[0], events.shape[0]
        print(
            f"\ntriple_barrier_labels done after {timedelta(seconds=round(time.perf_counter() - time0))}."
        )
        print(f"\npt_sl = {pt_sl}")
        print(f"Sampled {n_events:,} of {N:,} ({n_events / N:.2%}).")
        print(f"Accuracy: {events.bin.value_counts(normalize=True)[1]:.2%}")

    return events


def get_event_weights(
    triple_barrier_events, close, time_decay=1.0, linear_decay=False, verbose=False
):
    """
    :param triple_barrier_events: (pd.DataFrame) Triple-barrier events DataFrame with the following structure:
    - **index**: pd.DatetimeIndex of event start times
    - **t1**: (pd.Series) Event end times
    - **trgt**: (pd.Series) Target volatility
    - **pt**: (pd.Series) Profit-taking multiple
    - **sl**: (pd.Series) Stop-loss multiple
    - **side**: (pd.Series, optional) Algo's position side
    :param close: (pd.Series) Close prices
    :param verbose: (bool) Log outputs if True.
    :param time_decay: (float) Decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regardless of how old
        - decay = 0 means that weights converge linearly (exponenentially) to zero, as they become older
        - decay < 0 means that the oldest portion c of the observations receive zero weight (i.e they are erased from memory)
    :param linear_decay: (bool) If True, linear decay is applied, else exponential decay
    :param verbose: (bool) Log outputs if True.
    :return: (pd.DataFrame) Events DataFrame with additional columns:
        - **tW**: Average uniqueness of the event (time-weighted)
        - **w**: Sample weights scaled by time-decay & return-weighted attribution
    """
    events = triple_barrier_events.copy()
    # Estimate the uniqueness of a triple_barrier_events
    num_conc_events = get_num_conc_events_optimized(close.index, events["t1"], verbose)
    av_uniqueness = get_av_uniqueness_from_triple_barrier_optimized(
        events,
        close,
        num_conc_events,
        verbose,
    )

    # Sample weights scaled by time-decay & return-weighted attribution
    return_weights = get_weights_by_return_optimized(
        events,
        close,
        num_conc_events,
        verbose,
    )
    time_decay = get_weights_by_time_decay_optimized(
        events,
        close,
        time_decay,
        linear_decay,
        av_uniqueness,
        verbose,
    )

    events["tW"] = av_uniqueness
    events["w"] = return_weights * time_decay
    return events
