"""
Accounting tests for backtester.cpcv_dynamic_backtest._simulate_path.

These pin the bookkeeping identity only (equity = start + gross P&L - costs), not
the P&L units, which the GUI plan (Section 10.1, defect 1) flags separately.
"""

import numpy as np
import pandas as pd
import pytest

from ..backtester.cpcv_dynamic_backtest import BacktestConfig, _simulate_path


class _FixedSizer:
    """Stands in for PropFirmAwareSizer: returns predetermined positions."""

    def __init__(self, positions: pd.Series):
        self.positions = positions

    def size(self, **kwargs):
        return {"final_size": self.positions}


def _inputs(n=96, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04", periods=n, freq="h")  # spans four days
    probs = pd.Series(0.6, index=idx)
    events = pd.DataFrame({"t1": idx + pd.Timedelta(hours=2)}, index=idx)
    returns = pd.Series(rng.normal(0, 0.002, n), index=idx)
    positions = pd.Series(np.where(np.arange(n) // 6 % 2 == 0, 1.0, -0.5), index=idx)
    sides = pd.Series(1, index=idx)
    return idx, probs, events, returns, positions, sides


def test_equity_equals_start_plus_gross_minus_costs_across_days():
    idx, probs, events, returns, positions, sides = _inputs()
    cfg = BacktestConfig()
    res = _simulate_path(0, probs, events, returns, _FixedSizer(positions), cfg, sides)

    gross = positions * returns * cfg.lot_size * cfg.pip_value
    change = (positions - positions.shift(1).fillna(0.0)).abs()
    costs = change * cfg.commission_per_lot + change * cfg.slippage_pips * cfg.pip_value
    expected = cfg.initial_balance + (gross - costs).cumsum()

    np.testing.assert_allclose(res.equity_curve.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-6)
    assert res.final_equity == pytest.approx(expected.iloc[-1])
    assert res.trading_days == 4  # one reset per calendar day, including the first


def test_costs_are_charged_once():
    """Regression: costs were netted into realized P&L and also passed as fees."""
    idx, probs, events, _, positions, sides = _inputs()
    flat = pd.Series(0.0, index=idx)  # no market moves: equity changes only by costs
    cfg = BacktestConfig()
    res = _simulate_path(0, probs, events, flat, _FixedSizer(positions), cfg, sides)
    change = (positions - positions.shift(1).fillna(0.0)).abs()
    costs = change * cfg.commission_per_lot + change * cfg.slippage_pips * cfg.pip_value
    expected = cfg.initial_balance - costs.cumsum()
    # Every bar, not just the last: the double charge showed on bars that trade
    np.testing.assert_allclose(res.equity_curve.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-6)
