"""
Mutable per-account state for prop-firm evaluation and funded trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .phase import Phase
from .rule_set import PropFirmRuleSet


@dataclass
class PropFirmAccountState:
    """
    Tracks the mutable account state that changes bar-by-bar.

    Parameters
    ----------
    initial_balance : float
        Account balance at the start of the phase.
    phase : Phase
        Current phase (indexes rule_set.phase_targets).
    rule_set : PropFirmRuleSet
        The prop-firm program's rule set. Determines profit targets,
        loss limits, and the news-window profit-credit haircut.

    Notes
    -----
    The daily loss limit expands with intraday realized profit only
    when rule_set.daily_loss_is_dynamic is True (FundedNext-style).
    This is the most commonly misunderstood rule for programs that
    use it. The overall floor is permanently anchored at
    initial_balance * (1 - rule_set.overall_loss_limit_pct) and does
    not move.

    current_equity is banked_equity (cumulative equity as of the start
    of the current day) plus today's intraday activity. banked_equity
    is captured by reset_daily() from the prior day's ending
    current_equity, so multi-day profit persists across day
    boundaries. daily_loss_used and intraday_realized_pnl remain
    correctly scoped to the current day only — see reset_daily().
    """

    initial_balance: float
    phase: Phase
    rule_set: PropFirmRuleSet

    # Running account metrics — updated by PropFirmAccountState.update()
    current_equity: float = field(init=False)
    banked_equity: float = field(init=False)
    daily_loss_used: float = field(init=False)
    intraday_realized_pnl: float = field(init=False)
    trading_days_completed: int = field(init=False)
    phase_profit_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_equity = self.initial_balance
        self.banked_equity = self.initial_balance
        self.daily_loss_used = 0.0
        self.intraday_realized_pnl = 0.0
        self.trading_days_completed = 0
        self.phase_profit_pct = 0.0

    # -- Properties --------------------------------------------------------

    @property
    def overall_floor(self) -> float:
        """Balance/equity floor — permanently anchored, never moves."""
        return self.initial_balance * (1.0 - self.rule_set.overall_loss_limit_pct)

    @property
    def daily_limit(self) -> float:
        """
        Daily loss limit.

        Expands by the amount of intraday realized profit already made
        today when rule_set.daily_loss_is_dynamic is True. Commissions,
        swaps, and fees count against it.
        """
        base = self.initial_balance * self.rule_set.daily_loss_limit_pct
        if self.rule_set.daily_loss_is_dynamic:
            return base + max(0.0, self.intraday_realized_pnl)
        return base

    @property
    def daily_remaining(self) -> float:
        return max(0.0, self.daily_limit - self.daily_loss_used)

    @property
    def overall_remaining(self) -> float:
        return max(0.0, self.current_equity - self.overall_floor)

    @property
    def risk_budget_pct(self) -> float:
        """
        Fraction of initial balance still available as risk capacity.

        Takes the minimum of daily and overall remaining budget so that
        the binding constraint always governs position sizing.
        """
        return min(self.daily_remaining, self.overall_remaining) / self.initial_balance

    @property
    def phase_progress(self) -> float:
        """
        Fraction of the phase profit target achieved. 0.0 on a new account.
        Returns 0.0 for phases with no target (e.g. funded).
        """
        target = self.rule_set.phase_targets.get(self.phase)
        if target is None or target == 0.0:
            return 0.0
        return min(self.phase_profit_pct / target, 1.0)

    @property
    def meets_min_trading_days(self) -> bool:
        """
        Whether the rule set's minimum trading-days requirement is met.

        Addition beyond the original module: min_trading_days was
        previously a declared-but-unused constant. This is the first
        consumer of it; it does not change any sizing behavior.
        """
        return self.trading_days_completed >= self.rule_set.min_trading_days

    # -- Mutation ------------------------------------------------------------

    def update(
        self,
        realized_pnl_delta: float,
        unrealized_pnl: float,
        fees_and_swaps: float,
    ) -> None:
        """
        Update account state after each bar or closed trade.

        Parameters
        ----------
        realized_pnl_delta : float
            P&L of any trade(s) closed since the last update call.
        unrealized_pnl : float
            Total floating P&L of all currently open positions.
        fees_and_swaps : float
            Commissions and overnight swaps accrued since the last
            update. Must be passed as a positive number; the method
            subtracts it.
        """
        self.intraday_realized_pnl += realized_pnl_delta
        net_change = realized_pnl_delta - fees_and_swaps
        if net_change < 0:
            self.daily_loss_used += abs(net_change)

        self.current_equity = (
            self.banked_equity + self.intraday_realized_pnl + unrealized_pnl - fees_and_swaps
        )
        self.phase_profit_pct = max(
            0.0,
            (self.current_equity - self.initial_balance) / self.initial_balance,
        )

    def reset_daily(self) -> None:
        """
        Call at the start of each new trading day.

        Banks the current equity (today's close, per the last update()
        call) as tomorrow's baseline BEFORE zeroing intraday_realized_pnl,
        so a profitable day's gains persist into the next day's
        current_equity and phase_profit_pct instead of being discarded.
        """
        self.banked_equity = self.current_equity
        self.daily_loss_used = 0.0
        self.intraday_realized_pnl = 0.0
        self.trading_days_completed += 1
