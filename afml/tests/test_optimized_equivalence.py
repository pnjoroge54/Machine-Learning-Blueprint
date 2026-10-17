"""
The Numba-optimized concurrency and sample-weight functions must reproduce the
reference AFML implementations (Chapter 4) on the same events.
"""

import numpy as np
import pandas as pd
import pytest

from ..sample_weights.attribution import get_weights_by_return, get_weights_by_time_decay
from ..sample_weights.optimized_attribution import (
    get_weights_by_return_optimized,
    get_weights_by_time_decay_optimized,
)
from ..sampling.concurrent import get_av_uniqueness_from_triple_barrier, get_num_conc_events
from ..sampling.optimized_concurrent import (
    get_av_uniqueness_from_triple_barrier_optimized,
    get_num_conc_events_optimized,
)


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-02", periods=1500, freq="15min")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    starts = np.sort(rng.choice(np.arange(10, 1400), size=200, replace=False))
    hold = rng.integers(1, 60, size=len(starts))
    events = pd.DataFrame({"t1": idx[np.minimum(starts + hold, len(idx) - 1)]}, index=idx[starts])
    return close, events


def _as_series(x, name=None):
    if isinstance(x, pd.DataFrame):
        x = x[name] if name in x.columns else x.iloc[:, 0]
    return pd.Series(x).astype(float)


def test_num_concurrent_events(market):
    close, events = market
    ref = _as_series(get_num_conc_events(events, close, num_threads=1, verbose=False))
    opt = _as_series(get_num_conc_events_optimized(close.index, events["t1"]))
    common = ref.index.intersection(opt.index)
    assert len(common) > 1000
    np.testing.assert_allclose(opt.loc[common], ref.loc[common])


def test_average_uniqueness(market):
    close, events = market
    ref = _as_series(get_av_uniqueness_from_triple_barrier(events, close, num_threads=1, verbose=False), "tW")
    opt = _as_series(get_av_uniqueness_from_triple_barrier_optimized(events, close.index), "tW")
    np.testing.assert_allclose(opt.loc[events.index], ref.loc[events.index], rtol=1e-6)
    assert ((opt > 0) & (opt <= 1)).all()


def test_weights_by_return(market):
    close, events = market
    ref = _as_series(get_weights_by_return(events, close, num_threads=1, verbose=False))
    opt = _as_series(get_weights_by_return_optimized(events, close))
    np.testing.assert_allclose(opt.loc[events.index], ref.loc[events.index], rtol=1e-6)


@pytest.mark.parametrize("last_weight, linear", [(1.0, True), (0.5, True), (-0.25, True), (0.5, False), (0.1, False)])
def test_weights_by_time_decay(market, last_weight, linear):
    close, events = market
    ref = _as_series(get_weights_by_time_decay(
        events, close, num_threads=1, last_weight=last_weight, linear=linear, verbose=False))
    opt = _as_series(get_weights_by_time_decay_optimized(
        events, close.index, last_weight=last_weight, linear=linear))
    np.testing.assert_allclose(opt.loc[events.index], ref.loc[events.index], rtol=1e-6, atol=1e-12)
    # Newest observation always carries weight 1 under decay
    assert opt.iloc[-1] == pytest.approx(1.0)


def test_linear_decay_with_negative_last_weight_follows_afml_4_11(market):
    """The oldest |c| share of cumulative uniqueness gets zero weight; the newest keeps 1."""
    close, events = market
    c = -0.25
    av = get_av_uniqueness_from_triple_barrier_optimized(events, close.index)
    w = get_weights_by_time_decay_optimized(events, close.index, last_weight=c, linear=True, av_uniqueness=av)
    cum = av["tW"].sort_index().cumsum()
    share = cum / cum.iloc[-1]
    assert (w[share <= abs(c) - 1e-9] == 0).all()
    assert (w[share > abs(c) + 1e-9] > 0).all()
    assert w.iloc[-1] == pytest.approx(1.0)


def test_exponential_decay_runs_from_oldest_to_newest(market):
    close, events = market
    w = get_weights_by_time_decay_optimized(events, close.index, last_weight=0.2, linear=False)
    assert w.iloc[0] == pytest.approx(0.2)
    assert w.iloc[-1] == pytest.approx(1.0)
    assert w.is_monotonic_increasing
