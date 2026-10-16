"""
Mutable per-account state for prop-firm evaluation and funded trading.

The account keeps two levels, as the firms do:

* balance: initial balance plus closed P&L, net of commissions and swaps.
* equity: balance plus the floating P&L of open positions.

Every rule in a PropFirmRuleSet is evaluated from these two numbers, so the
same code serves balance- or equity-based, static or trailing drawdown rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .phase import Phase
from .rule_set import PropFirmRuleSet, trading_day


@dataclass
class TradeRecord:
    """A closed trade, for trade-level rules such as Quick Strike."""

    open_time: pd.Timestamp
    close_time: pd.Timestamp
    pnl: float

    @property
    def hold_seconds(self) -> float:
        return (pd.Timestamp(self.close_time) - pd.Timestamp(self.open_time)).total_seconds()


@dataclass
class PropFirmAccountState:
    """
    Tracks balance, equity and the rule floors bar by bar.

    Parameters
    ----------
    initial_balance : float
        Account balance at the start of the phase.
    phase : Phase
        Current phase (indexes rule_set.phase_targets and phase-scoped rules).
    rule_set : PropFirmRuleSet
        The program's rules.

    Usage
    -----
    Call ``update()`` after each bar or fill, passing the bar's timestamp: the
    account starts a new trading day itself when the timestamp crosses the rule
    set's day boundary. Without timestamps, call ``reset_daily()`` at each day
    boundary instead.

    Once a floor is breached the account is closed: ``breach`` records the rule,
    further updates are ignored and ``risk_budget_pct`` is 0.
    """

    initial_balance: float
    phase: Phase
    rule_set: PropFirmRuleSet

    balance: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    day: date | None = field(init=False)
    day_start_balance: float = field(init=False)
    day_start_equity: float = field(init=False)
    day_reference: float = field(init=False)
    high_water: float = field(init=False)
    intraday_realized_pnl: float = field(init=False)
    intraday_fees: float = field(init=False)
    trading_days_completed: int = field(init=False)
    breach: str | None = field(init=False)
    breach_time: pd.Timestamp | None = field(init=False)
    breach_level: float | None = field(init=False)
    trades: list[TradeRecord] = field(init=False)
    _traded_today: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        self.balance = float(self.initial_balance)
        self.unrealized_pnl = 0.0
        self.day = None
        self.high_water = float(self.initial_balance)
        self.trading_days_completed = 0
        self.breach = None
        self.breach_time = None
        self.breach_level = None
        self.trades = []
        self._open_day()

    # -- Levels ------------------------------------------------------------

    @property
    def current_equity(self) -> float:
        """Balance plus open P&L."""
        return self.balance + self.unrealized_pnl

    @property
    def banked_equity(self) -> float:
        """Balance at the start of the current trading day (former name)."""
        return self.day_start_balance

    def _measured(self, balance: float, unrealized: float) -> float:
        return balance + unrealized if self.rule_set.breach_on == "equity" else balance

    @property
    def measured_level(self) -> float:
        """The level the floors are compared with (equity or balance, per the rules)."""
        return self._measured(self.balance, self.unrealized_pnl)

    # -- Floors ------------------------------------------------------------

    @property
    def daily_loss_amount(self) -> float:
        """Today's loss allowance from the day's reference level."""
        rs = self.rule_set
        basis = self.initial_balance if rs.daily_loss_basis == "initial_balance" else self.day_reference
        return rs.daily_loss_limit_pct * basis

    @property
    def daily_limit(self) -> float:
        """Alias of daily_loss_amount."""
        return self.daily_loss_amount

    @property
    def daily_floor(self) -> float:
        """Level the measured balance/equity may not fall below today."""
        return self.day_reference - self.daily_loss_amount

    @property
    def overall_floor(self) -> float:
        """Maximum-loss floor; static, or trailing the high-water mark."""
        rs = self.rule_set
        amount = rs.overall_loss_limit_pct * self.initial_balance
        static = self.initial_balance - amount
        if rs.max_loss_mode == "static":
            return static
        floor = self.high_water - amount
        if rs.max_loss_trail_stops_at_initial:
            floor = min(floor, float(self.initial_balance))
        return max(floor, static)

    @property
    def daily_loss_used(self) -> float:
        """How far the measured level is below today's reference."""
        return max(0.0, self.day_reference - self.measured_level)

    @property
    def daily_remaining(self) -> float:
        """Room above today's floor (FundedNext's "today's permitted loss")."""
        return max(0.0, self.measured_level - self.daily_floor)

    @property
    def overall_remaining(self) -> float:
        return max(0.0, self.measured_level - self.overall_floor)

    @property
    def risk_budget_pct(self) -> float:
        """
        Fraction of initial balance still available as risk capacity: the
        binding one of the daily and overall rooms, and 0 once breached.
        """
        if self.breach is not None:
            return 0.0
        return min(self.daily_remaining, self.overall_remaining) / self.initial_balance

    @property
    def is_breached(self) -> bool:
        return self.breach is not None

    # -- Progress ------------------------------------------------------------

    @property
    def phase_profit_pct(self) -> float:
        """Closed profit as a fraction of the initial balance (0 when in loss)."""
        return max(0.0, (self.balance - self.initial_balance) / self.initial_balance)

    @property
    def phase_progress(self) -> float:
        """
        Fraction of the phase profit target achieved. 0.0 on a new account and
        for phases with no target (e.g. funded).
        """
        target = self.rule_set.phase_targets.get(self.phase)
        if target is None or target == 0.0:
            return 0.0
        return min(self.phase_profit_pct / target, 1.0)

    @property
    def meets_min_trading_days(self) -> bool:
        """Whether enough qualifying days (per trading_day_rule) are complete."""
        return self.trading_days_completed >= self.rule_set.min_trading_days

    # -- Trade-level rules ---------------------------------------------------

    @property
    def max_risk_amount(self) -> float | None:
        """Largest potential loss allowed per trade in this phase, or None."""
        rs = self.rule_set
        if rs.max_risk_per_trade_pct is None or self.phase not in rs.max_risk_phases:
            return None
        return rs.max_risk_per_trade_pct * self.initial_balance

    def check_trade_risk(self, potential_loss: float) -> bool:
        """True if a trade risking ``potential_loss`` at its stop is allowed."""
        cap = self.max_risk_amount
        return cap is None or abs(potential_loss) <= cap + 1e-9

    def max_lots_for_risk(self, loss_per_lot_at_stop: float) -> float | None:
        """Largest position (in lots) whose loss at the stop stays within the cap."""
        cap = self.max_risk_amount
        if cap is None:
            return None
        if loss_per_lot_at_stop <= 0:
            raise ValueError("loss_per_lot_at_stop must be > 0")
        return cap / loss_per_lot_at_stop

    def record_trade(self, open_time, close_time, pnl: float) -> None:
        """Log a closed trade for trade-level rules."""
        self.trades.append(TradeRecord(pd.Timestamp(open_time), pd.Timestamp(close_time), float(pnl)))

    @property
    def quick_strike_share(self) -> float:
        """
        Share of total profit made by trades held under quick_strike_seconds.
        0.0 when the rule is disabled or nothing qualifies; 1.0 when quick trades
        made money but the account has no net profit.
        """
        seconds = self.rule_set.quick_strike_seconds
        if seconds is None or not self.trades:
            return 0.0
        quick = sum(max(t.pnl, 0.0) for t in self.trades if t.hold_seconds < seconds)
        if quick == 0.0:
            return 0.0
        total = sum(t.pnl for t in self.trades)
        return 1.0 if total <= 0 else quick / total

    @property
    def quick_strike_ok(self) -> bool:
        rs = self.rule_set
        if rs.quick_strike_seconds is None or self.phase not in rs.quick_strike_phases:
            return True
        return self.quick_strike_share < rs.quick_strike_max_profit_share

    # -- Mutation ------------------------------------------------------------

    def update(
        self,
        realized_pnl_delta: float,
        unrealized_pnl: float,
        fees_and_swaps: float,
        *,
        worst_unrealized_pnl: float | None = None,
        timestamp=None,
        traded: bool | None = None,
    ) -> None:
        """
        Update the account after a bar or a fill.

        Parameters
        ----------
        realized_pnl_delta : float
            Gross P&L of trade(s) closed since the last call.
        unrealized_pnl : float
            Floating P&L of all open positions now.
        fees_and_swaps : float
            Commissions and swaps accrued since the last call, as a positive
            number; subtracted from the balance.
        worst_unrealized_pnl : float, optional
            Lowest floating P&L reached since the last call (e.g. at the bar's
            low for a long). Checked against the floors before the call's fills
            settle, so an intrabar breach is caught even if the bar closes
            above the floor.
        timestamp : optional
            Time of the update. Starts a new trading day when it crosses the
            rule set's day boundary. Timezone-naive values are treated as UTC.
        traded : bool, optional
            Whether a position was opened or changed. Defaults to True when
            the call carries realized P&L or fees.
        """
        if self.breach is not None:
            return
        if timestamp is not None:
            d = trading_day(timestamp, self.rule_set.day_boundary)
            if self.day is None:
                self.day = d
            elif d != self.day:
                self._roll_to(d)
            ts = pd.Timestamp(timestamp)
        else:
            ts = None

        if worst_unrealized_pnl is not None:
            self._check(self._measured(self.balance, worst_unrealized_pnl), ts)
            if self.breach is not None:
                return

        if traded is None:
            traded = realized_pnl_delta != 0.0 or fees_and_swaps != 0.0
        if traded:
            self._traded_today = True
        self.balance += realized_pnl_delta - fees_and_swaps
        self.intraday_realized_pnl += realized_pnl_delta
        self.intraday_fees += fees_and_swaps
        self.unrealized_pnl = float(unrealized_pnl)

        if self.rule_set.max_loss_mode == "trailing_intraday":
            self.high_water = max(self.high_water, self._trail_level())
        self._check(self.measured_level, ts)

    def reset_daily(self) -> None:
        """
        Close the current trading day and start the next one. Use when calling
        ``update()`` without timestamps; with timestamps days roll automatically.
        """
        self._close_day()
        self._open_day()

    # -- Internals -----------------------------------------------------------

    def _trail_level(self) -> float:
        if self.rule_set.max_loss_trail_source == "equity":
            return self.current_equity
        return self.balance

    def _check(self, level: float, ts) -> None:
        if level < self.overall_floor - 1e-9:
            rule = "overall"
        elif level < self.daily_floor - 1e-9:
            rule = "daily"
        else:
            return
        self.breach, self.breach_time, self.breach_level = rule, ts, level

    def _roll_to(self, new_day: date) -> None:
        self._close_day()
        self.day = new_day
        self._open_day()

    def _close_day(self) -> None:
        rs = self.rule_set
        if rs.trading_day_rule == "any_trade":
            qualifies = self._traded_today
        else:
            day_profit = min(self.balance, self.current_equity) - self.day_start_balance
            qualifies = day_profit >= rs.profitable_day_min_pct * self.initial_balance - 1e-9
        if qualifies:
            self.trading_days_completed += 1
        if rs.max_loss_mode == "trailing_eod":
            self.high_water = max(self.high_water, self._trail_level())

    def _open_day(self) -> None:
        self.day_start_balance = self.balance
        self.day_start_equity = self.current_equity
        ref = self.rule_set.daily_reference
        if ref == "balance":
            self.day_reference = self.day_start_balance
        elif ref == "equity":
            self.day_reference = self.day_start_equity
        else:
            self.day_reference = max(self.day_start_balance, self.day_start_equity)
        self.intraday_realized_pnl = 0.0
        self.intraday_fees = 0.0
        self._traded_today = False
