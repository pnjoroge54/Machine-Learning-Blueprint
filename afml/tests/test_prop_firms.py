"""
Tests for afml.prop_firms: rule sets, account state, stop-loss ceiling and the
sizing modifiers.
"""

from dataclasses import replace
from datetime import datetime

import pytest

from ..prop_firms import (
    FUNDEDNEXT_STELLAR_2STEP,
    Phase,
    PropFirmAccountState,
    PropFirmRuleSet,
    derisking_factor,
    kelly_payoff_multiplier,
    news_window_factor,
)
from ..prop_firms.ceiling import assert_timescale_match, prop_firm_sl_ceiling

RULES = FUNDEDNEXT_STELLAR_2STEP


def _state(phase=Phase.CHALLENGE_PHASE_1, rules=RULES, balance=100_000.0):
    return PropFirmAccountState(initial_balance=balance, phase=phase, rule_set=rules)


# --------------------------------------------------------------------------- rule set


def test_stellar_preset_values():
    assert RULES.phase_targets[Phase.CHALLENGE_PHASE_1] == 0.08
    assert RULES.phase_targets[Phase.CHALLENGE_PHASE_2] == 0.05
    assert RULES.phase_targets[Phase.FUNDED] is None
    assert RULES.daily_loss_is_dynamic is True
    assert (RULES.daily_loss_limit_pct, RULES.overall_loss_limit_pct) == (0.05, 0.10)


@pytest.mark.parametrize(
    "field, value",
    [
        ("daily_loss_limit_pct", 0.0),
        ("daily_loss_limit_pct", 1.5),
        ("overall_loss_limit_pct", -0.1),
        ("news_profit_credit", 1.1),
        ("min_trading_days", -1),
        ("max_leverage", 0),
    ],
)
def test_rule_set_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(RULES, **{field: value})


def test_rule_set_is_frozen():
    with pytest.raises(Exception):
        RULES.daily_loss_limit_pct = 0.2  # type: ignore[misc]


# --------------------------------------------------------------------------- account state


def test_new_account_state():
    s = _state()
    assert s.current_equity == s.banked_equity == 100_000.0
    assert s.overall_floor == pytest.approx(90_000.0)
    assert s.daily_limit == pytest.approx(5_000.0)
    assert s.risk_budget_pct == pytest.approx(0.05)
    assert s.phase_progress == 0.0
    assert not s.meets_min_trading_days


def test_dynamic_daily_limit_expands_with_realized_profit_only():
    s = _state()
    s.update(realized_pnl_delta=1_000.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.daily_limit == pytest.approx(6_000.0)
    s.update(realized_pnl_delta=0.0, unrealized_pnl=3_000.0, fees_and_swaps=0.0)
    assert s.daily_limit == pytest.approx(6_000.0)  # floating profit does not expand it


def test_static_daily_limit_ignores_profit():
    s = _state(rules=replace(RULES, daily_loss_is_dynamic=False))
    s.update(realized_pnl_delta=2_000.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.daily_limit == pytest.approx(5_000.0)


def test_losses_consume_daily_budget_and_overall_floor_binds():
    s = _state()
    s.update(realized_pnl_delta=-2_000.0, unrealized_pnl=0.0, fees_and_swaps=10.0)
    assert s.daily_loss_used == pytest.approx(2_010.0)
    assert s.daily_remaining == pytest.approx(2_990.0)
    # Near the overall floor the overall budget becomes the binding constraint
    s.update(realized_pnl_delta=0.0, unrealized_pnl=-6_500.0, fees_and_swaps=0.0)
    assert s.overall_remaining == pytest.approx(s.current_equity - 90_000.0)
    assert s.risk_budget_pct == pytest.approx(min(s.daily_remaining, s.overall_remaining) / 100_000.0)


def test_reset_daily_banks_profit_and_clears_daily_counters():
    s = _state()
    s.update(realized_pnl_delta=3_000.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    s.reset_daily()
    assert s.banked_equity == pytest.approx(103_000.0)
    assert s.intraday_realized_pnl == 0.0 and s.daily_loss_used == 0.0
    assert s.trading_days_completed == 1
    s.update(realized_pnl_delta=0.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.current_equity == pytest.approx(103_000.0)  # profit persists across days


def test_phase_progress_and_min_trading_days():
    s = _state()
    s.update(realized_pnl_delta=4_000.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.phase_progress == pytest.approx(0.5)  # 4% of an 8% target
    s.update(realized_pnl_delta=8_000.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.phase_progress == 1.0  # capped
    for _ in range(RULES.min_trading_days):
        s.reset_daily()
    assert s.meets_min_trading_days
    assert _state(phase=Phase.FUNDED).phase_progress == 0.0  # no target when funded


def test_fees_persist_across_updates():
    """Regression: fees charged on one bar used to reappear as equity on the next."""
    s = _state()
    s.update(realized_pnl_delta=0.0, unrealized_pnl=0.0, fees_and_swaps=5.0)
    s.update(realized_pnl_delta=0.0, unrealized_pnl=0.0, fees_and_swaps=0.0)
    assert s.current_equity == pytest.approx(99_995.0)
    s.update(realized_pnl_delta=100.0, unrealized_pnl=20.0, fees_and_swaps=3.0)
    assert s.current_equity == pytest.approx(100_000.0 + 100.0 + 20.0 - 8.0)
    assert s.daily_loss_used == pytest.approx(5.0)  # the realized gain offset the second fee


def test_fees_are_banked_once_across_days():
    s = _state()
    s.update(realized_pnl_delta=0.0, unrealized_pnl=0.0, fees_and_swaps=10.0)
    s.reset_daily()
    assert s.banked_equity == pytest.approx(99_990.0)
    assert s.intraday_fees == 0.0
    s.update(realized_pnl_delta=0.0, unrealized_pnl=0.0, fees_and_swaps=2.0)
    assert s.current_equity == pytest.approx(99_988.0)  # yesterday's fee not charged twice


# --------------------------------------------------------------------------- stop-loss ceiling


def test_sl_ceiling_formula():
    # 5% of 25k = 1,250 USD over 2 x 0.5 lots x $10/pip = 125 pips
    assert prop_firm_sl_ceiling(25_000, 0.05, 0.5, 2, 10.0) == pytest.approx(125.0)
    # Aggregated exposure: n_concurrent does not divide
    assert prop_firm_sl_ceiling(25_000, 0.05, 0.5, 2, 10.0, aggregate_exposure=True) == pytest.approx(250.0)
    # Ceiling scales as 1 / |bet_size|
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


# --------------------------------------------------------------------------- sizing modifiers


def test_kelly_multiplier_bounds_and_no_edge():
    assert kelly_payoff_multiplier(0.45, avg_win_loss_ratio=1.0) == 0.0  # no edge
    for p in (0.55, 0.65, 0.8, 0.95):
        m = kelly_payoff_multiplier(p, avg_win_loss_ratio=2.0, max_amplification=1.5)
        assert 0.0 <= m <= 1.5
    # A better payoff ratio never lowers the multiplier at the same probability
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


def test_rule_set_is_the_only_source_of_limits():
    custom = PropFirmRuleSet(
        phase_targets={Phase.CHALLENGE_PHASE_1: 0.10, Phase.CHALLENGE_PHASE_2: 0.05, Phase.FUNDED: None},
        daily_loss_limit_pct=0.04, daily_loss_is_dynamic=False, overall_loss_limit_pct=0.08,
        min_trading_days=3, max_leverage=50,
    )
    s = _state(rules=custom, balance=50_000.0)
    assert s.daily_limit == pytest.approx(2_000.0)
    assert s.overall_floor == pytest.approx(46_000.0)
