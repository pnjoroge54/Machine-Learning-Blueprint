"""
Tests for backtester.cpcv_dynamic_backtest._simulate_path: position accounting,
no lookahead, costs, intrabar breaches, rule-set day boundaries on UTC bars,
the per-trade risk cap and phase targets.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ..backtester.cpcv_dynamic_backtest import BacktestConfig, _simulate_path
from ..prop_firms import FTMO_2STEP, FUNDEDNEXT_STELLAR_2STEP, Phase


class _FixedSizer:
    """Stands in for PropFirmAwareSizer: returns predetermined bet sizes."""

    def __init__(self, sizes: pd.Series, stop_loss_pct=None):
        self.sizes = sizes
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct

    def size(self, **kwargs):
        return {"final_size": self.sizes}


def _bars(n=96, seed=0, start="2025-03-03 00:00", vol=0.0004, wiggle=0.0):
    """UTC hourly OHLC bars around 1.1000."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h")
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    high = close * (1 + wiggle)
    low = close * (1 - wiggle)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def _run(prices, sizes, cfg=None, sizer_stop=None):
    idx = prices.index
    cfg = cfg or BacktestConfig(lot_size=1.0)
    probs = pd.Series(0.6, index=idx)
    events = pd.DataFrame({"t1": idx + pd.Timedelta(hours=2)}, index=idx)
    sides = pd.Series(1, index=idx)
    sizer = _FixedSizer(sizes, stop_loss_pct=sizer_stop)
    return _simulate_path(0, probs, events, prices, sizer, cfg, sides), cfg


def _expected_equity(prices, lots, cfg):
    """Independent mark-to-market: lots set at t's close earn from t+1."""
    vpu = cfg.pip_value / cfg.pip_size
    held = lots.shift(1).fillna(0.0)
    pnl = held * prices["close"].diff().fillna(0.0) * vpu
    costs = lots.diff().fillna(lots).abs() * (cfg.commission_per_lot + cfg.slippage_pips * cfg.pip_value)
    return cfg.initial_balance + (pnl - costs).cumsum()


def _alternating(idx, a=1.0, b=-0.5, period=6):
    return pd.Series(np.where(np.arange(len(idx)) // period % 2 == 0, a, b), index=idx)


def test_equity_matches_mark_to_market_with_costs_and_no_lookahead():
    prices = _bars()
    sizes = _alternating(prices.index)
    res, cfg = _run(prices, sizes)
    expected = _expected_equity(prices, sizes * cfg.lot_size, cfg)
    np.testing.assert_allclose(res.equity_curve.to_numpy(), expected.to_numpy(), atol=1e-6)
    assert res.breach is None


def test_position_does_not_earn_its_own_signal_bar():
    prices = _bars(n=20, vol=0.0)
    prices.loc[prices.index[10]:, ["open", "high", "low", "close"]] *= 1.01  # jump at bar 10
    sizes = pd.Series(0.0, index=prices.index)
    sizes.iloc[10:] = 1.0  # signal appears on the jump bar itself
    cfg = BacktestConfig(lot_size=1.0, commission_per_lot=0.0, slippage_pips=0.0)
    res, _ = _run(prices, sizes, cfg)
    assert res.final_equity == pytest.approx(cfg.initial_balance)


def test_costs_are_charged_once_per_lot_traded():
    prices = _bars(vol=0.0)
    sizes = _alternating(prices.index)
    res, cfg = _run(prices, sizes)
    lots = sizes * cfg.lot_size
    costs = lots.diff().fillna(lots).abs() * (cfg.commission_per_lot + cfg.slippage_pips * cfg.pip_value)
    np.testing.assert_allclose(res.equity_curve.to_numpy(), (cfg.initial_balance - costs.cumsum()).to_numpy(), atol=1e-6)
    assert res.n_trades > 0


def test_pnl_is_in_pips():
    prices = _bars(n=3, vol=0.0)
    prices.iloc[2] = prices.iloc[1] + 0.0010  # +10 pips on the third bar
    sizes = pd.Series([1.0, 1.0, 1.0], index=prices.index)
    cfg = BacktestConfig(lot_size=1.0, commission_per_lot=0.0, slippage_pips=0.0)
    res, _ = _run(prices, sizes, cfg)
    assert res.final_equity - cfg.initial_balance == pytest.approx(100.0)  # 10 pips x $10/pip


def test_intrabar_low_breaches_and_freezes_the_account():
    prices = _bars(n=24, vol=0.0)
    prices.loc[prices.index[12], "low"] -= 0.0600  # a wick far below, close unchanged
    sizes = pd.Series(1.0, index=prices.index)
    res, cfg = _run(prices, sizes, BacktestConfig(lot_size=1.0, commission_per_lot=0.0, slippage_pips=0.0))
    assert res.breach in ("daily", "overall")
    assert res.breach_time == prices.index[12]
    assert (res.equity_curve.iloc[12:] == res.equity_curve.iloc[12]).all()
    assert not res.hit_target


def test_close_only_prices_miss_the_intrabar_wick():
    prices = _bars(n=24, vol=0.0)
    prices.loc[prices.index[12], "low"] -= 0.0600
    sizes = pd.Series(1.0, index=prices.index)
    cfg = BacktestConfig(lot_size=1.0, commission_per_lot=0.0, slippage_pips=0.0)
    res, _ = _run(prices["close"], sizes, cfg)  # Series: no high/low
    assert res.breach is None


def test_days_roll_on_the_rule_set_boundary_from_utc_bars():
    prices = _bars(n=72, start="2025-01-13 00:00")  # three UTC days in winter
    sizes = _alternating(prices.index, a=0.2, b=0.0, period=1)  # trades every other bar
    fn, _ = _run(prices, sizes, BacktestConfig(lot_size=1.0, rule_set=FUNDEDNEXT_STELLAR_2STEP))
    ftmo, _ = _run(prices, sizes, BacktestConfig(lot_size=1.0, rule_set=FTMO_2STEP))
    # NY-close days start 22:00 UTC, Prague days 23:00 UTC: both see 4 trading days
    assert fn.trading_days == 4 and ftmo.trading_days == 4


def test_funded_risk_cap_limits_lots():
    prices = _bars(n=10, vol=0.0)
    sizes = pd.Series(1.0, index=prices.index)
    cfg = BacktestConfig(lot_size=100.0, phase=Phase.FUNDED, commission_per_lot=0.0,
                         slippage_pips=0.0, stop_loss_pct=0.01)
    prices.iloc[-1] = prices.iloc[-2] * 1.001
    res, _ = _run(prices, sizes, cfg)
    vpu = cfg.pip_value / cfg.pip_size
    cap_lots = 3_000.0 / (0.01 * prices["close"].iloc[-2] * vpu)
    expected = cap_lots * (prices["close"].iloc[-1] - prices["close"].iloc[-2]) * vpu
    assert res.final_equity - cfg.initial_balance == pytest.approx(expected, rel=1e-6)


def test_funded_risk_cap_needs_a_stop():
    prices = _bars(n=5)
    sizes = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="stop_loss_pct"):
        _run(prices, sizes, BacktestConfig(phase=Phase.FUNDED))


@pytest.mark.parametrize("rules, hit", [(FUNDEDNEXT_STELLAR_2STEP, True), (FTMO_2STEP, False)])
def test_hit_target_reads_the_rule_set(rules, hit):
    prices = _bars(n=48, vol=0.0)
    sizes = pd.Series(0.0, index=prices.index)
    sizes.iloc[0:30] = 1.0
    prices.iloc[1:30] = prices.iloc[0] + 0.0090  # +90 pips while long: +9%
    prices.iloc[30:] = prices.iloc[29]
    cfg = BacktestConfig(lot_size=10.0, commission_per_lot=0.0, slippage_pips=0.0,
                         rule_set=replace(rules, daily_loss_limit_pct=0.5, overall_loss_limit_pct=0.5))
    res, _ = _run(prices, sizes, cfg)
    assert res.final_balance == pytest.approx(cfg.initial_balance * 1.09, rel=1e-6)
    assert res.hit_target is hit  # FundedNext phase 1 target 8%, FTMO 10%
