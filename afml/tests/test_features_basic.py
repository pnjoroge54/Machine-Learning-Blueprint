"""
Tests for afml.features: returns, moving averages, trading-session encodings.
"""

import numpy as np
import pandas as pd
import pytest

from ..features.moving_averages import calculate_ma_differences, get_ma_crossovers
from ..features.returns import (
    get_lagged_returns,
    get_period_autocorr,
    get_period_returns,
    get_return_dist_features,
    rolling_autocorr_numba,
)
from ..features.trading_session import get_time_features, trading_session_encoded_features


@pytest.fixture(scope="module")
def hourly_close():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    idx = idx[idx.dayofweek < 5]  # weekend gaps, like FX
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx)))), index=idx, name="close")


# --------------------------------------------------------------------------- returns


def test_period_returns_use_the_last_bar_before_the_lookback(hourly_close):
    ret = get_period_returns(hourly_close, hours=5)
    t = ret.index[50]
    prev = hourly_close.loc[: t - pd.Timedelta(hours=5)]
    prev_close = hourly_close.iloc[hourly_close.index.searchsorted(t - pd.Timedelta(hours=5)) - 1]
    assert ret.loc[t] == pytest.approx(hourly_close.loc[t] / prev_close - 1)
    assert prev.index[-1] <= t - pd.Timedelta(hours=5)


def test_period_returns_bridge_weekend_gaps(hourly_close):
    ret = get_period_returns(hourly_close, days=1)
    monday = ret.index[ret.index.dayofweek == 0][30]
    # A day back from Monday morning falls on Sunday: the reference is Friday's last bar
    ref = hourly_close.iloc[hourly_close.index.searchsorted(monday - pd.Timedelta(days=1)) - 1]
    assert hourly_close.index[hourly_close.index.searchsorted(monday - pd.Timedelta(days=1)) - 1].dayofweek == 4
    assert ret.loc[monday] == pytest.approx(hourly_close.loc[monday] / ref - 1)


def test_rolling_autocorr_matches_pandas():
    rng = np.random.default_rng(0)
    x = np.zeros(300)
    for i in range(1, 300):
        x[i] = 0.6 * x[i - 1] + rng.normal()
    lookback = 50
    out = rolling_autocorr_numba(x, lookback)
    assert np.isnan(out[: lookback - 1]).all()
    for i in (lookback - 1, 120, 299):
        w = x[i - lookback + 1 : i + 1]
        assert out[i] == pytest.approx(np.corrcoef(w[:-1], w[1:])[0, 1])
    assert np.nanmean(out) > 0.3  # persistent series


def test_period_autocorr_is_indexed_like_returns(hourly_close):
    ac = get_period_autocorr(hourly_close, lookback=20, hours=1)
    assert ac.index.equals(get_period_returns(hourly_close, hours=1).index)
    assert ac.dropna().between(-1, 1).all()


def test_lagged_returns_are_geometric_per_period_and_shifted():
    growth = 0.001
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    prices = pd.Series(100 * (1 + growth) ** np.arange(200), index=idx)
    df = get_lagged_returns(prices, lags=[1, 5], nperiods=2)
    assert {"returns", "returns_5", "returns_1_lag_1", "returns_5_lag_2"} <= set(df.columns)
    assert df["returns"].dropna().to_numpy() == pytest.approx(growth)
    assert df["returns_5"].dropna().to_numpy() == pytest.approx(growth)
    pd.testing.assert_series_equal(
        df["returns_5_lag_2"], df["returns_5"].shift(10), check_names=False
    )


def test_return_distribution_features(hourly_close):
    df = get_return_dist_features(hourly_close, window=24)
    assert list(df.columns) == ["returns_norm", "returns_skew", "returns_kurt"]
    assert df.index.equals(hourly_close.index)
    assert abs(df["returns_norm"].dropna().mean()) < 0.2


# --------------------------------------------------------------------------- moving averages


def test_ma_differences_columns_and_values(hourly_close):
    df = calculate_ma_differences(hourly_close, windows=[5, 20, 50])
    assert list(df.columns) == ["sma_diff_5_20", "sma_diff_5_50", "sma_diff_20_50"]
    expected = hourly_close.rolling(5).mean() - hourly_close.rolling(20).mean()
    pd.testing.assert_series_equal(df["sma_diff_5_20"], expected, check_names=False, atol=1e-10)


def test_ma_crossovers_are_binary_and_signal_marks_changes(hourly_close):
    state = get_ma_crossovers(hourly_close, windows=[10, 30])
    assert set(state["sma_cross_10_30"].unique()) <= {0, 1}
    signal = get_ma_crossovers(hourly_close, windows=[10, 30], as_signal=True)
    assert set(signal["sma_cross_10_30"].unique()) <= {-1, 0, 1}
    changes = state["sma_cross_10_30"].diff().fillna(0)
    pd.testing.assert_series_equal(signal["sma_cross_10_30"], changes, check_names=False, check_dtype=False)


# --------------------------------------------------------------------------- trading sessions


def test_session_flags_are_binary_and_overlap_is_consistent(hourly_close):
    flags = trading_session_encoded_features(hourly_close.index)
    session_cols = [c for c in flags.columns if c.endswith("_session")]
    assert session_cols
    for c in session_cols + ["session_overlap"]:
        assert set(flags[c].unique()) <= {0, 1}
    overlap = (flags[session_cols].sum(axis=1) > 1).astype(int)
    assert (flags["session_overlap"].to_numpy() == overlap.to_numpy()).all()


def test_time_features_cyclical_pairs_lie_on_the_unit_circle(hourly_close):
    bars = hourly_close.to_frame()
    feats = get_time_features(bars, timeframe="H1", n_terms=2, bar_type="time", forex=True)
    assert len(feats) == len(bars)
    sins = [c for c in feats.columns if "_sin" in c]
    assert sins
    for s in sins:
        c = s.replace("_sin", "_cos")
        np.testing.assert_allclose(feats[s] ** 2 + feats[c] ** 2, 1.0, atol=1e-9)
