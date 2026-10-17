"""
Tests for labeling.trend_scanning.trend_scanning_labels (current API: a span of
candidate windows, look-forward or look-backward, optional volatility filter).
"""

import numpy as np
import pandas as pd
import pytest

from ..labeling.trend_scanning import trend_scanning_labels

COLUMNS = {"t1", "window", "slope", "t_value", "rsquared", "ret", "bin"}


def _close(path):
    idx = pd.date_range("2024-01-01", periods=len(path), freq="D")
    return pd.Series(path, index=idx, name="close")


@pytest.fixture(scope="module")
def up_then_down():
    rng = np.random.default_rng(0)
    trend = np.r_[np.linspace(100, 130, 120), np.linspace(130, 100, 120)]
    return _close(trend * np.exp(rng.normal(0, 0.002, len(trend))))


def test_output_schema(up_then_down):
    labels = trend_scanning_labels(up_then_down, span=(5, 20))
    assert COLUMNS <= set(labels.columns)
    assert labels.index.isin(up_then_down.index).all()
    assert set(labels["bin"].unique()) <= {-1, 0, 1}
    assert labels["window"].between(5, 20).all()
    assert labels["rsquared"].between(0, 1).all()


def test_trends_are_labelled_by_sign(up_then_down):
    labels = trend_scanning_labels(up_then_down, span=(5, 20))
    rising = labels.loc[: up_then_down.index[90]]
    falling = labels.loc[up_then_down.index[130] : up_then_down.index[200]]
    assert (rising["bin"] == 1).mean() > 0.9
    assert (falling["bin"] == -1).mean() > 0.9
    assert (rising["t_value"] >= 0).mean() > 0.95 and (falling["t_value"] <= 0).mean() > 0.95
    np.testing.assert_array_equal(np.sign(labels["slope"]), np.sign(labels["t_value"]))


def test_lookforward_windows_end_after_the_event(up_then_down):
    fwd = trend_scanning_labels(up_then_down, span=(5, 20), lookforward=True)
    assert (pd.to_datetime(fwd["t1"]) > fwd.index).all()
    # Look-backward output carries the fit statistics only (no t1/ret/bin columns)
    back = trend_scanning_labels(up_then_down, span=(5, 20), lookforward=False)
    assert {"window", "slope", "t_value", "rsquared"} <= set(back.columns)


def test_explicit_window_list_is_respected(up_then_down):
    labels = trend_scanning_labels(up_then_down, span=[10, 15])
    assert set(labels["window"].unique()) <= {10, 15}
