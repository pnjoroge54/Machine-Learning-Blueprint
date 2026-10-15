"""
Regression tests for afml.cache's joblib-backed decorators.

1. Hit counting: joblib's ``check_call_in_cache`` returns a bool, so treating "did not
   raise" as a hit counted every call as a hit.
2. Key drift: joblib keys on the pickled arguments. A pandas object's pickle changes
   when lazily cached properties are filled in (``DatetimeArray._cache``,
   ``Index._cache``) or when its blocks are consolidated, with no change in values,
   so the same data could miss the cache depending on what had touched it.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

import afml.cache as afml_cache
from afml.cache.robust_cache_keys import create_robust_cacheable, stabilize_pandas_args

CALLS = []


def _frame(n=500, consolidated=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, n), "x": np.arange(float(n))}, index=idx)
    if not consolidated:
        df["n"] = np.arange(n)  # adds a second block
        df["y"] = np.ones(n)    # and a third, unconsolidated float block
    else:
        df["n"] = np.arange(n)
        df["y"] = np.ones(n)
        df._consolidate_inplace()
    return df


def mean_close(df):
    CALLS.append(1)
    return float(df["close"].mean())


@pytest.fixture
def decorate(tmp_path, monkeypatch):
    """A fresh robust_cacheable bound to an isolated joblib store."""
    monkeypatch.setattr(afml_cache, "memory", joblib.Memory(str(tmp_path), verbose=0))
    CALLS.clear()

    def _decorate(func, **kwargs):
        name = f"{func.__module__}.{func.__qualname__}"
        afml_cache.cache_stats._stats.pop(name, None)
        return create_robust_cacheable(**kwargs)(func), name

    yield _decorate


def test_hit_counter_records_misses(decorate):
    cached, name = decorate(mean_close)
    df = _frame()
    cached(df)
    cached(df)
    assert afml_cache.cache_stats.get_stats()[name] == {"hits": 1, "misses": 1}
    assert len(CALLS) == 1


@pytest.mark.parametrize("time_aware", [False, True])
def test_used_datetime_index_still_hits(decorate, time_aware):
    cached, name = decorate(mean_close, use_time_awareness=time_aware)
    cached(_frame())
    df = _frame()
    df.index.searchsorted(df.index[10])        # fills DatetimeArray._cache
    df["close"].loc[df.index[5:9]]              # fills Index._cache
    cached(df)
    assert len(CALLS) == 1, "same data recomputed after its index was used"
    assert afml_cache.cache_stats.get_stats()[name] == {"hits": 1, "misses": 1}


def test_block_layout_does_not_change_key(decorate):
    cached, _ = decorate(mean_close)
    cached(_frame(consolidated=False))
    cached(_frame(consolidated=True))
    assert len(CALLS) == 1


def test_frames_inside_containers_are_stabilized():
    a, b = _frame(), _frame()
    b.index.searchsorted(b.index[3])
    args_a, kwargs_a = ([a],), {"d": {"frame": a}}
    args_b, kwargs_b = ([b],), {"d": {"frame": b}}
    stabilize_pandas_args(args_a, kwargs_a)
    stabilize_pandas_args(args_b, kwargs_b)
    assert joblib.hash((args_a, kwargs_a)) == joblib.hash((args_b, kwargs_b))


def test_stabilize_does_not_change_values():
    df = _frame(consolidated=False)
    before = df.copy(deep=True)
    df.index.searchsorted(df.index[3])
    stabilize_pandas_args((df,), {})
    pd.testing.assert_frame_equal(df, before)
    assert df.index.freq == before.index.freq


def test_unified_cacheable_counts_and_keys_like_robust(tmp_path, monkeypatch):
    from afml.cache import unified_cacheable as uc

    monkeypatch.setattr(uc, "memory", joblib.Memory(str(tmp_path), verbose=0))
    CALLS.clear()
    name = f"{mean_close.__module__}.{mean_close.__qualname__}"
    afml_cache.cache_stats._stats.pop(name, None)
    cached = uc.unified_cacheable()(mean_close)
    cached(_frame())
    df = _frame()
    df.index.searchsorted(df.index[10])
    cached(df)
    assert len(CALLS) == 1
    assert afml_cache.cache_stats.get_stats()[name] == {"hits": 1, "misses": 1}
