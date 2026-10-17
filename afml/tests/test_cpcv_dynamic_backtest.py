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


# ── Defect 2: calibration must not see the labels it is applied to ───────────


def _noise_paths(n=400, n_paths=3, n_groups=8, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    y = pd.Series(rng.integers(0, 2, n))
    paths = [rng.uniform(0, 1, n) for _ in range(n_paths)]
    return paths, y, groups


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_crossfit_calibration_ignores_each_groups_own_labels(method):
    from ..backtester.cpcv_dynamic_backtest import _calibrate_paths_crossfit

    paths, y, groups = _noise_paths()
    flipped = y.copy()
    held = groups == 3
    flipped[held] = 1 - flipped[held]
    a = np.column_stack(_calibrate_paths_crossfit(paths, y, method, groups))
    b = np.column_stack(_calibrate_paths_crossfit(paths, flipped, method, groups))
    np.testing.assert_allclose(a[held], b[held])
    assert not np.allclose(a[~held], b[~held])  # other groups do learn from group 3


def test_pooled_calibration_leaks_labels():
    """The old Phase 2.5: one map on every label, applied back to the same rows."""
    from ..backtester.cpcv_dynamic_backtest import _apply_calibrator, _fit_calibrator

    paths, y, groups = _noise_paths()
    flipped = y.copy()
    held = groups == 3
    flipped[held] = 1 - flipped[held]
    old = [_apply_calibrator(paths[0], _fit_calibrator(paths, lab, "isotonic"), "isotonic")
           for lab in (y, flipped)]
    assert not np.allclose(old[0][held], old[1][held])


def test_backtest_run_calibrates_paths_cross_fitted():
    from ..backtester.cpcv_dynamic_backtest import CPCVDynamicBacktest
    from ..cross_validation.combinatorial import CombinatorialPurgedCV
    from sklearn.linear_model import LogisticRegression

    prices = _bars(n=240, vol=0.0005)
    idx = prices.index
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    events = pd.DataFrame({"t1": idx + pd.Timedelta(hours=2)}, index=idx)
    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, t1=events["t1"], pct_embargo=0.0)
    bt = CPCVDynamicBacktest(cv, LogisticRegression(), _FixedSizer(pd.Series(0.0, index=idx)),
                             BacktestConfig(lot_size=1.0), prices, pd.Series(1, index=idx),
                             calibration_method="isotonic")
    bt.run(X, y, events)
    assert len(bt.results_) == cv.n_test_paths and bt.calibrator_ is not None


# ── Defect 3: PBO is a statement about competing configurations ──────────────


def _trial_returns(n_trials=10, n=800, edge=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(rng.normal(0, 1e-3, size=(n, n_trials)), index=idx,
                      columns=[f"t{i}" for i in range(n_trials)])
    df["t0"] += edge
    return df


def _ran_backtest():
    from ..backtester.cpcv_dynamic_backtest import CPCVDynamicBacktest

    bt = CPCVDynamicBacktest.__new__(CPCVDynamicBacktest)
    bt.results_ = [object()]  # a completed run
    return bt


def test_pbo_audit_refuses_the_paths_of_one_configuration():
    with pytest.raises(ValueError, match="competing configurations"):
        _ran_backtest().pbo_audit()


def test_pbo_across_trials_separates_noise_from_a_real_edge():
    bt = _ran_backtest()
    noise = bt.pbo_audit(_trial_returns(edge=0.0), n_folds=8)
    real = bt.pbo_audit(_trial_returns(edge=5e-4), n_folds=8)
    assert 0.2 <= noise <= 0.8   # selection among noise: the winner is a coin flip OOS
    assert real < 0.1


# ── Defect 4: annualize from the bar frequency, not sqrt(252) ────────────────


@pytest.mark.parametrize("freq, periods", [("h", 252 * 24), ("15min", 252 * 96), ("D", 252)])
def test_path_sharpe_is_annualized_from_the_bar_index(freq, periods):
    prices = _bars(n=200, vol=0.0005)
    prices.index = pd.date_range("2025-03-03", periods=len(prices), freq=freq)
    sizes = _alternating(prices.index)
    res, _ = _run(prices, sizes, BacktestConfig(lot_size=1.0, commission_per_lot=0.0, slippage_pips=0.0))
    r = res.returns
    assert res.sharpe == pytest.approx(r.mean() / r.std() * np.sqrt(periods), rel=1e-9)
