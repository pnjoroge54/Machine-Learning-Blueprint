"""
Configurable prop-firm rule sets.

Generalizes the module-level constants previously hardcoded in
bet_sizing/prop_firm_sizer.py (_PHASE_TARGETS, _DAILY_LOSS_LIMIT_PCT,
_OVERALL_LOSS_LIMIT_PCT, _NEWS_PROFIT_CREDIT, _MIN_TRADING_DAYS,
_MAX_LEVERAGE, _COMMISSION_PER_LOT) into an explicit, swappable
PropFirmRuleSet. PropFirmAccountState and the sizing modifiers consume
a PropFirmRuleSet instance instead of reading module constants, so the
risk-sizing math is invariant to which program's rules are in force.

Note on scope: min_trading_days, max_leverage, and commission_per_lot
are carried through as data but are not consumed by any method in this
subpackage, matching the behavior of the original module (where the
equivalent constants were also declared but unused by
PropFirmAccountState or PropFirmAwareSizer). PropFirmAccountState now
exposes a meets_min_trading_days property as one small addition; see
account_state.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .phase import Phase


@dataclass(frozen=True)
class PropFirmRuleSet:
    """
    Constraint set for a prop-firm evaluation/funded program.

    Parameters
    ----------
    phase_targets : dict[Phase, float | None]
        Profit target per phase, as a fraction of initial balance.
        None for phases with no profit target (e.g. funded).
    daily_loss_limit_pct : float
        Maximum daily loss as a fraction of initial balance.
    daily_loss_is_dynamic : bool
        If True, the daily limit expands with intraday realized profit
        (FundedNext-style: base + max(0, intraday_realized_pnl)).
        If False, the daily limit is fixed at
        initial_balance * daily_loss_limit_pct regardless of intraday P&L.
    overall_loss_limit_pct : float
        Permanent floor, as a fraction of initial_balance. Does not move.
    min_trading_days : int
        Minimum number of trading days required before the phase can be
        passed or a payout requested. Not enforced elsewhere in the
        pipeline; exposed via PropFirmAccountState.meets_min_trading_days.
    max_leverage : int
        Maximum leverage permitted by the program. Not currently
        consumed by any sizing or account-state logic; carried as
        documentation of the constraint pending a leverage check.
    news_profit_credit : float
        Fraction of P&L credited inside a news window, funded stage
        only. 1.0 disables the haircut (no news-window adjustment).
    news_window_minutes : float
        Half-width of the news window in minutes.
    commission_per_lot : float
        Commission per lot assumed by the program's fee schedule. Not
        currently consumed here; execution-level commission simulation
        lives separately in each backtester's own config (e.g.
        BacktestConfig.commission_per_lot in cpcv_dynamic_backtest.py).
        Kept here only as documentation of the program's stated fee;
        do not treat this as the source of truth for P&L simulation.
    """

    phase_targets: dict[Phase, float | None]
    daily_loss_limit_pct: float
    daily_loss_is_dynamic: bool
    overall_loss_limit_pct: float
    min_trading_days: int
    max_leverage: int
    news_profit_credit: float = 1.0
    news_window_minutes: float = 5.0
    commission_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.daily_loss_limit_pct <= 1.0:
            raise ValueError(
                f"daily_loss_limit_pct must be in (0, 1], got {self.daily_loss_limit_pct}"
            )
        if not 0.0 < self.overall_loss_limit_pct <= 1.0:
            raise ValueError(
                f"overall_loss_limit_pct must be in (0, 1], got {self.overall_loss_limit_pct}"
            )
        if not 0.0 <= self.news_profit_credit <= 1.0:
            raise ValueError(f"news_profit_credit must be in [0, 1], got {self.news_profit_credit}")
        if self.min_trading_days < 0:
            raise ValueError(f"min_trading_days must be >= 0, got {self.min_trading_days}")
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {self.max_leverage}")


# ── Presets ─────────────────────────────────────────────────────────────────

FUNDEDNEXT_STELLAR_2STEP = PropFirmRuleSet(
    phase_targets={
        Phase.CHALLENGE_PHASE_1: 0.08,
        Phase.CHALLENGE_PHASE_2: 0.05,
        Phase.FUNDED: None,
    },
    daily_loss_limit_pct=0.05,
    daily_loss_is_dynamic=True,
    overall_loss_limit_pct=0.10,
    min_trading_days=5,
    max_leverage=100,
    news_profit_credit=0.40,
    news_window_minutes=5.0,
    commission_per_lot=5.0,
)
