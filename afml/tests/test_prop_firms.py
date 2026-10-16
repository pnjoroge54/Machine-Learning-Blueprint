"""
Tests for afml.prop_firms: rule sets and presets, the balance/equity account
model under FundedNext, FTMO and The5ers rules, trade-level rules (maximum risk,
Quick Strike), the stop-loss ceiling and the sizing modifiers.

Where a firm publishes worked examples, the tests use them.
"""

from dataclasses import replace
from datetime import date, datetime

import pandas as pd
import pytest

from ..prop_firms import (
    FTMO_1STEP,
    FTMO_2STEP,
    FUNDEDNEXT_STELLAR_2STEP,
    NY_CLOSE,
    PRESETS,
    THE5ERS_HIGH_STAKES,
    Phase,
    PropFirmAccountState,
    PropFirmRuleSet,
    derisking_factor,
    kelly_payoff_multiplier,
    news_window_factor,
    trading_day,
    trading_days,
)
from ..prop_firms.ceiling import assert_timescale_match, prop_firm_sl_ceiling

FN = FUNDEDNEXT_STELLAR_2STEP


def _state(rules=FN, phase=Phase.CHALLENGE_PHASE_1, balance=100_000.0):
    return PropFirmAccountState(initial_balance=balance, phase=phase, rule_set=rules)


# --------------------------------------------------------------------------- rule sets


@pytest.mark.parametrize(
    "field, value",
    [
        ("daily_loss_limit_pct", 0.0),
        ("daily_loss_limit_pct", 1.5),
        ("overall_loss_limit_pct", -0.1),
        ("news_profit_credit", 1.1),
        ("min_trading_days", -1),
        ("max_leverage", 0),
        ("daily_reference", "high"),
        ("max_loss_mode", "trailing"),
        ("breach_on", "margin"),
        ("max_risk_per_trade_pct", 0.0),
        ("quick_strike_seconds", 0),
        ("day_boundary", "Mars/Olympus"),
    ],
)
def test_rule_set_rejects_invalid_values(field, value):
    with pytest.raises(Exception):
        replace(FN, **{field: value})


def test_rule_set_is_frozen():
    with pytest.raises(Exception):
        FN.daily_loss_limit_pct = 0.2  # type: ignore[misc]


def test_presets():
    assert set(PRESETS) == {"fundednext_stellar_2step", "ftmo_2step", "ftmo_1step", "the5ers_high_stakes"}
    assert (FN.phase_targets[Phase.CHALLENGE_PHASE_1], FN.phase_targets[Phase.CHALLENGE_PHASE_2]) == (0.08, 0.05)
    assert FN.max_risk_per_trade_pct == 0.03 and FN.max_risk_phases == {Phase.FUNDED}
    assert (FN.quick_strike_seconds, FN.quick_strike_max_profit_share) == (30.0, 0.30)
    assert FTMO_2STEP.phase_targets[Phase.CHALLENGE_PHASE_1] == 0.10 and FTMO_2STEP.min_trading_days == 4
    assert FTMO_2STEP.day_boundary == "Europe/Prague"
    assert FTMO_1STEP.max_loss_mode == "trailing_eod" and FTMO_1STEP.daily_loss_limit_pct == 0.03
    assert THE5ERS_HIGH_STAKES.daily_reference == "max_balance_equity"
    assert THE5ERS_HIGH_STAKES.trading_day_rule == "profitable"
    for rules in PRESETS.values():
        assert rules.breach_on == "equity"


# --------------------------------------------------------------------------- day boundaries


@pytest.mark.parametrize(
    "utc, boundary, expected",
    [
        # Server midnight (GMT+2 winter / GMT+3 summer) = 17:00 New York
        ("2025-01-15 21:59", NY_CLOSE, date(2025, 1, 15)),
        ("2025-01-15 22:00", NY_CLOSE, date(2025, 1, 16)),
        ("2025-07-15 20:59", NY_CLOSE, date(2025, 7, 15)),
        ("2025-07-15 21:00", NY_CLOSE, date(2025, 7, 16)),
        # Prague midnight: 23:00 UTC in winter, 22:00 UTC in summer
        ("2025-01-15 22:59", "Europe/Prague", date(2025, 1, 15)),
        ("2025-01-15 23:00", "Europe/Prague", date(2025, 1, 16)),
        ("2025-07-15 22:00", "Europe/Prague", date(2025, 7, 16)),
    ],
)
def test_trading_day_from_utc(utc, boundary, expected):
    assert trading_day(utc, boundary) == expected
    assert trading_day(pd.Timestamp(utc, tz="UTC"), boundary) == expected


def test_trading_days_vectorised_matches_scalar():
    idx = pd.date_range("2025-03-28", "2025-04-02", freq="h")  # spans the EU DST change
    days = trading_days(idx, "Europe/Prague")
    assert list(days) == [trading_day(t, "Europe/Prague") for t in idx]


# --------------------------------------------------------------------------- FundedNext (help-centre examples)


def test_new_account():
    s = _state()
    assert s.current_equity == s.balance == 100_000.0
    assert s.overall_floor == pytest.approx(90_000.0)
    assert s.daily_floor == pytest.approx(95_000.0)
    assert s.daily_remaining == pytest.approx(5_000.0)
    assert s.risk_budget_pct == pytest.approx(0.05)
    assert s.phase_progress == 0.0 and not s.meets_min_trading_days and not s.is_breached


def test_fundednext_profit_today_widens_the_room():
    """Example 2: $2,000 profit by noon -> $7,000 may be lost before a breach."""
    s = _state()
    s.update(2_000.0, 0.0, 0.0)
    assert s.daily_remaining == pytest.approx(7_000.0)
    assert s.daily_limit == pytest.approx(5_000.0)  # the allowance itself is fixed


def test_fundednext_closed_loss_leaves_the_rest():
    """Case 01: a $3,000 closed loss leaves $2,000, floating losses included."""
    s = _state()
    s.update(-3_000.0, 0.0, 0.0)
    assert s.daily_remaining == pytest.approx(2_000.0)
    s.update(0.0, -1_999.0, 0.0)
    assert not s.is_breached
    s.update(0.0, -2_001.0, 0.0)
    assert s.breach == "daily"


def test_fundednext_open_loss_carried_over_midnight():
    """Case 04: +$2,000 closed and -$6,000 floating is fine today, a breach tomorrow."""
    s = _state()
    s.update(2_000.0, -6_000.0, 0.0, timestamp="2025-03-03 12:00")
    assert not s.is_breached and s.daily_loss_used == pytest.approx(4_000.0)
    # Next server day: the reference is the new balance (102,000); equity is 96,000
    s.update(0.0, -6_000.0, 0.0, timestamp="2025-03-03 22:30")
    assert s.day == date(2025, 3, 4)
    assert s.day_reference == pytest.approx(102_000.0)
    assert s.breach == "daily"


def test_overall_floor_is_static_and_equity_based():
    s = _state()
    for day in range(1, 4):  # stay inside each day's allowance while drifting down
        s.update(-3_000.0, 0.0, 0.0, timestamp=f"2025-03-0{day} 12:00")
    assert s.balance == pytest.approx(91_000.0) and not s.is_breached
    s.update(0.0, -1_500.0, 0.0, timestamp="2025-03-04 12:00")
    assert s.breach == "overall" and s.overall_floor == pytest.approx(90_000.0)


def test_intrabar_low_breaches_even_if_the_bar_recovers():
    s = _state()
    s.update(0.0, -1_000.0, 0.0, worst_unrealized_pnl=-5_200.0)
    assert s.breach == "daily" and s.breach_level == pytest.approx(94_800.0)


def test_breach_closes_the_account():
    s = _state()
    s.update(-6_000.0, 0.0, 0.0)
    assert s.is_breached and s.risk_budget_pct == 0.0
    balance = s.balance
    s.update(10_000.0, 0.0, 0.0)
    assert s.balance == balance  # further updates are ignored


def test_fees_persist_across_updates():
    """Regression: fees charged on one bar used to reappear as equity on the next."""
    s = _state()
    s.update(0.0, 0.0, 5.0)
    s.update(0.0, 0.0, 0.0)
    assert s.current_equity == pytest.approx(99_995.0)
    s.update(100.0, 20.0, 3.0)
    assert s.current_equity == pytest.approx(100_000.0 + 100.0 + 20.0 - 8.0)


def test_fees_are_banked_once_across_days():
    s = _state()
    s.update(0.0, 0.0, 10.0)
    s.reset_daily()
    assert s.day_start_balance == pytest.approx(99_990.0) and s.intraday_fees == 0.0
    s.update(0.0, 0.0, 2.0)
    assert s.current_equity == pytest.approx(99_988.0)


def test_open_profit_is_not_counted_twice_overnight():
    s = _state()
    s.update(0.0, 100.0, 0.0, timestamp="2025-03-03 12:00")
    s.update(0.0, 100.0, 0.0, timestamp="2025-03-04 12:00")
    assert s.current_equity == pytest.approx(100_100.0)


def test_only_days_with_trades_count_and_progress_uses_closed_profit():
    s = _state()
    s.update(0.0, 0.0, 0.0, timestamp="2025-03-03 10:00")  # no trade
    s.update(4_000.0, 0.0, 0.0, timestamp="2025-03-04 10:00")
    s.update(0.0, 3_000.0, 0.0, timestamp="2025-03-05 10:00")  # open profit only
    s.reset_daily()
    assert s.trading_days_completed == 1
    assert s.phase_progress == pytest.approx(0.5)  # 4% closed of an 8% target


# --------------------------------------------------------------------------- FTMO


def test_ftmo_daily_floor_from_prague_midnight_balance():
    s = _state(FTMO_2STEP)
    s.update(3_000.0, 0.0, 0.0, timestamp="2025-01-15 12:00")
    s.update(0.0, 0.0, 0.0, timestamp="2025-01-15 23:30")  # 00:30 Prague
    assert s.day_reference == pytest.approx(103_000.0)
    assert s.daily_floor == pytest.approx(98_000.0)  # 5% of initial below the new reference
    s.update(0.0, -5_100.0, 0.0, timestamp="2025-01-16 08:00")
    assert s.breach == "daily"


def test_ftmo_1step_floor_trails_end_of_day_balance_and_never_falls():
    s = _state(FTMO_1STEP)
    s.update(5_000.0, 0.0, 0.0, timestamp="2025-01-15 12:00")
    assert s.overall_floor == pytest.approx(90_000.0)  # moves only at end of day
    s.update(0.0, 0.0, 0.0, timestamp="2025-01-16 12:00")
    assert s.overall_floor == pytest.approx(95_000.0)
    s.update(-2_000.0, 0.0, 0.0, timestamp="2025-01-16 13:00")
    s.update(0.0, 0.0, 0.0, timestamp="2025-01-17 12:00")
    assert s.overall_floor == pytest.approx(95_000.0)
    assert s.daily_loss_amount == pytest.approx(3_000.0)


def test_trailing_floor_can_stop_at_initial_balance():
    rules = replace(FTMO_1STEP, max_loss_trail_stops_at_initial=True)
    s = _state(rules)
    s.update(20_000.0, 0.0, 0.0, timestamp="2025-01-15 12:00")
    s.update(0.0, 0.0, 0.0, timestamp="2025-01-16 12:00")
    assert s.overall_floor == pytest.approx(100_000.0)


def test_trailing_intraday_follows_equity_peaks():
    rules = replace(FTMO_2STEP, max_loss_mode="trailing_intraday", max_loss_trail_source="equity")
    s = _state(rules)
    s.update(0.0, 4_000.0, 0.0)
    assert s.overall_floor == pytest.approx(94_000.0)


# --------------------------------------------------------------------------- The5ers


def test_the5ers_reference_is_higher_of_balance_and_equity():
    s = _state(THE5ERS_HIGH_STAKES)
    s.update(0.0, 2_000.0, 0.0, timestamp="2025-03-03 12:00")  # open profit at the reset
    s.update(0.0, 2_000.0, 0.0, timestamp="2025-03-03 23:00")
    assert s.day_reference == pytest.approx(102_000.0)
    assert s.daily_loss_amount == pytest.approx(5_100.0)  # 5% of the reference
    assert s.daily_floor == pytest.approx(96_900.0)


def test_the5ers_profitable_day_definition():
    s = _state(THE5ERS_HIGH_STAKES)
    s.update(400.0, 0.0, 0.0, timestamp="2025-03-03 12:00")  # 0.4%: not enough
    s.update(600.0, 0.0, 0.0, timestamp="2025-03-04 12:00")  # 0.6%: counts
    s.update(0.0, 900.0, 0.0, timestamp="2025-03-05 12:00")  # open profit only: does not
    s.reset_daily()
    assert s.trading_days_completed == 1


# --------------------------------------------------------------------------- trade-level rules


def test_max_risk_applies_to_funded_fundednext_only():
    assert _state().max_risk_amount is None
    funded = _state(phase=Phase.FUNDED)
    assert funded.max_risk_amount == pytest.approx(3_000.0)
    assert funded.check_trade_risk(2_999.0) and not funded.check_trade_risk(3_001.0)
    assert funded.max_lots_for_risk(loss_per_lot_at_stop=1_000.0) == pytest.approx(3.0)
    assert _state(FTMO_2STEP, phase=Phase.FUNDED).max_risk_amount is None


def test_quick_strike_share():
    s = _state()
    t0 = pd.Timestamp("2025-03-03 10:00")
    s.record_trade(t0, t0 + pd.Timedelta(seconds=10), 400.0)  # quick
    s.record_trade(t0, t0 + pd.Timedelta(minutes=5), 600.0)
    assert s.quick_strike_share == pytest.approx(0.4)
    assert not s.quick_strike_ok
    s.record_trade(t0, t0 + pd.Timedelta(hours=1), 1_000.0)
    assert s.quick_strike_share == pytest.approx(0.2)  # 400 of 2,000
    assert s.quick_strike_ok
    assert _state(FTMO_2STEP).quick_strike_ok  # rule not in force


def test_rule_set_is_the_only_source_of_limits():
    custom = PropFirmRuleSet(
        phase_targets={Phase.CHALLENGE_PHASE_1: 0.10, Phase.CHALLENGE_PHASE_2: 0.05, Phase.FUNDED: None},
        daily_loss_limit_pct=0.04,
        overall_loss_limit_pct=0.08,
        min_trading_days=3,
        max_leverage=50,
        breach_on="balance",
    )
    s = _state(custom, balance=50_000.0)
    assert s.daily_floor == pytest.approx(48_000.0)
    assert s.overall_floor == pytest.approx(46_000.0)
    s.update(0.0, -3_000.0, 0.0)  # balance-based: floating losses do not breach
    assert not s.is_breached


# --------------------------------------------------------------------------- stop-loss ceiling and modifiers


def test_sl_ceiling_formula():
    assert prop_firm_sl_ceiling(25_000, 0.05, 0.5, 2, 10.0) == pytest.approx(125.0)
    assert prop_firm_sl_ceiling(25_000, 0.05, 0.5, 2, 10.0, aggregate_exposure=True) == pytest.approx(250.0)
    assert prop_firm_sl_ceiling(25_000, 0.05, 0.5, 2, 10.0, bet_size=-0.5) == pytest.approx(250.0)
    assert prop_firm_sl_ceiling(bet_size=0.0) == float("inf")


@pytest.mark.parametrize("kwargs", [{"bet_size": 1.5}, {"lot_size": 0}, {"n_concurrent": 0}])
def test_sl_ceiling_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        prop_firm_sl_ceiling(**kwargs)


def test_assert_timescale_match():
    assert_timescale_match(10, 10)
    with pytest.raises(ValueError, match="max_hold"):
        assert_timescale_match(10, 12)


def test_kelly_multiplier_bounds_and_no_edge():
    assert kelly_payoff_multiplier(0.45, avg_win_loss_ratio=1.0) == 0.0
    for p in (0.55, 0.65, 0.8, 0.95):
        assert 0.0 <= kelly_payoff_multiplier(p, avg_win_loss_ratio=2.0, max_amplification=1.5) <= 1.5
    assert kelly_payoff_multiplier(0.6, 2.0) >= kelly_payoff_multiplier(0.6, 1.0)


@pytest.mark.parametrize("met, floor", [(True, 0.10), (False, 0.30)])
def test_derisking_factor(met, floor):
    assert derisking_factor(0.5, met) == 1.0
    assert derisking_factor(0.8, met) == pytest.approx(1.0)
    assert derisking_factor(1.0, met) == pytest.approx(floor)
    assert derisking_factor(0.9, met) == pytest.approx((1.0 + floor) / 2)


def test_news_window_factor_applies_only_in_funded_window():
    news = [datetime(2025, 1, 10, 13, 30)]
    inside, outside = datetime(2025, 1, 10, 13, 33), datetime(2025, 1, 10, 13, 40)
    assert news_window_factor(inside, news, 0.4, 5.0, Phase.FUNDED) == 0.4
    assert news_window_factor(outside, news, 0.4, 5.0, Phase.FUNDED) == 1.0
    assert news_window_factor(inside, news, 0.4, 5.0, Phase.CHALLENGE_PHASE_1) == 1.0
    assert news_window_factor(inside, [], 0.4, 5.0, Phase.FUNDED) == 1.0
    assert news_window_factor(inside, news, 1.0, 5.0, Phase.FUNDED) == 1.0
