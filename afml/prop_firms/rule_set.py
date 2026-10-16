"""
Configurable prop-firm rule sets.

A PropFirmRuleSet describes one program's rules as data, so the same account
model, sizer and backtester work for any firm. The loss rules vary along a few
independent axes, each a field here:

* Daily loss: how large the allowance is (a share of the initial balance or of
  the day's reference level), what level the day's loss is measured from (the
  balance, equity, or the higher of the two at the daily reset), and when the
  day resets (a timezone, or the New York close used by most MT4/MT5 servers).
* Overall loss: a static floor below the initial balance, or a floor that trails
  the highest end-of-day or intraday balance or equity.
* Breach measurement: equity (balance plus open P&L, net of fees and swaps) or
  balance only. Equity breaches are also checked inside each bar.
* Trading days, per-trade risk caps, short-hold profit caps (FundedNext's Quick
  Strike) and news-window profit credit.

Presets: FUNDEDNEXT_STELLAR_2STEP, FTMO_2STEP, FTMO_1STEP,
THE5ERS_HIGH_STAKES. Each preset's docstring notes its sources and any detail
the firm does not publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .phase import Phase

NY_CLOSE = "NY_CLOSE"
"""
Day boundary at 17:00 New York time, i.e. midnight on the GMT+2 / GMT+3 server
clock most MT4/MT5 brokers run (GMT+3 while US daylight saving is in force).
"""

_DAILY_BASES = ("initial_balance", "day_reference")
_DAILY_REFERENCES = ("balance", "equity", "max_balance_equity")
_MAX_LOSS_MODES = ("static", "trailing_eod", "trailing_intraday")
_SOURCES = ("balance", "equity")
_TRADING_DAY_RULES = ("any_trade", "profitable")


@dataclass(frozen=True)
class PropFirmRuleSet:
    """
    Constraint set for a prop-firm evaluation or funded program.

    Parameters
    ----------
    phase_targets : dict[Phase, float | None]
        Profit target per phase as a fraction of the initial balance; None for
        phases without a target (e.g. funded). Measured on closed balance.
    daily_loss_limit_pct : float
        Daily loss allowance as a fraction of ``daily_loss_basis``.
    overall_loss_limit_pct : float
        Maximum loss as a fraction of the initial balance.
    min_trading_days : int
        Qualifying days required before a phase can be passed.
    max_leverage : int
        Program leverage. Carried as data; not enforced here.
    daily_loss_basis : {"initial_balance", "day_reference"}
        What ``daily_loss_limit_pct`` is a fraction of.
    daily_reference : {"balance", "equity", "max_balance_equity"}
        Level at the daily reset that the day's loss is measured from. The
        breach floor for the day is ``reference - allowance``. Profit made
        during the day therefore widens the room to the floor automatically.
    day_boundary : str
        IANA timezone whose midnight starts a new trading day, or NY_CLOSE.
        Bar timestamps without a timezone are treated as UTC.
    max_loss_mode : {"static", "trailing_eod", "trailing_intraday"}
        Static floor at ``initial * (1 - pct)``, or a floor ``pct * initial``
        below the highest end-of-day (or intraday) ``max_loss_trail_source``.
    max_loss_trail_source : {"balance", "equity"}
        Level a trailing floor follows.
    max_loss_trail_stops_at_initial : bool
        If True a trailing floor stops rising once it reaches the initial
        balance (common on futures programs).
    breach_on : {"equity", "balance"}
        Level compared with both floors. Equity includes open P&L and is net of
        commissions and swaps.
    trading_day_rule : {"any_trade", "profitable"}
        A day counts toward ``min_trading_days`` if any position was opened, or
        only if it was profitable by at least ``profitable_day_min_pct``.
    profitable_day_min_pct : float
        Minimum day profit, as a fraction of the initial balance, for a
        "profitable" day. The day's profit is ``min(balance, equity)`` at the
        reset minus the previous reset's balance (The5ers' definition).
    max_risk_per_trade_pct : float | None
        Cap on a trade's potential loss at its stop, as a fraction of the
        initial balance; None for no cap.
    max_risk_phases : frozenset[Phase]
        Phases in which ``max_risk_per_trade_pct`` applies.
    quick_strike_seconds : float | None
        Trades held for less than this many seconds are "quick strikes"; None
        disables the rule.
    quick_strike_max_profit_share : float
        Quick strikes must make up less than this share of total profit.
    quick_strike_phases : frozenset[Phase]
        Phases in which the Quick Strike rule applies.
    news_profit_credit : float
        Share of profit credited for trades in a news window, funded stage
        only. 1.0 disables the haircut.
    news_window_minutes : float
        Half-width of the news window in minutes.
    commission_per_lot : float
        The program's stated commission. Documentation only; backtests take
        costs from their own configuration.
    name : str
        Label for reports.
    """

    phase_targets: dict[Phase, float | None]
    daily_loss_limit_pct: float
    overall_loss_limit_pct: float
    min_trading_days: int
    max_leverage: int
    daily_loss_basis: str = "initial_balance"
    daily_reference: str = "balance"
    day_boundary: str = "Europe/Prague"
    max_loss_mode: str = "static"
    max_loss_trail_source: str = "balance"
    max_loss_trail_stops_at_initial: bool = False
    breach_on: str = "equity"
    trading_day_rule: str = "any_trade"
    profitable_day_min_pct: float = 0.0
    max_risk_per_trade_pct: float | None = None
    max_risk_phases: frozenset = field(default_factory=frozenset)
    quick_strike_seconds: float | None = None
    quick_strike_max_profit_share: float = 0.30
    quick_strike_phases: frozenset = field(default_factory=frozenset)
    news_profit_credit: float = 1.0
    news_window_minutes: float = 5.0
    commission_per_lot: float = 0.0
    name: str = "custom"

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
        for name, value, allowed in (
            ("daily_loss_basis", self.daily_loss_basis, _DAILY_BASES),
            ("daily_reference", self.daily_reference, _DAILY_REFERENCES),
            ("max_loss_mode", self.max_loss_mode, _MAX_LOSS_MODES),
            ("max_loss_trail_source", self.max_loss_trail_source, _SOURCES),
            ("breach_on", self.breach_on, _SOURCES),
            ("trading_day_rule", self.trading_day_rule, _TRADING_DAY_RULES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.max_risk_per_trade_pct is not None and not 0.0 < self.max_risk_per_trade_pct <= 1.0:
            raise ValueError(
                f"max_risk_per_trade_pct must be in (0, 1], got {self.max_risk_per_trade_pct}"
            )
        if self.quick_strike_seconds is not None and self.quick_strike_seconds <= 0:
            raise ValueError("quick_strike_seconds must be > 0")
        if not 0.0 < self.quick_strike_max_profit_share <= 1.0:
            raise ValueError("quick_strike_max_profit_share must be in (0, 1]")
        if self.profitable_day_min_pct < 0:
            raise ValueError("profitable_day_min_pct must be >= 0")
        # Fail early on an unknown timezone rather than mid-backtest
        trading_day(pd.Timestamp("2024-01-01"), self.day_boundary)


def trading_day(ts, day_boundary: str) -> date:
    """
    Trading day a timestamp belongs to under ``day_boundary``.

    Timestamps without a timezone are treated as UTC.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if day_boundary == NY_CLOSE:
        # 17:00 New York becomes 00:00 of the next trading day
        return (ts.tz_convert("America/New_York") + pd.Timedelta(hours=7)).date()
    return ts.tz_convert(day_boundary).date()


def trading_days(index: pd.DatetimeIndex, day_boundary: str) -> pd.Index:
    """Vectorised ``trading_day`` for a DatetimeIndex."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    if day_boundary == NY_CLOSE:
        local = idx.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    else:
        local = idx.tz_convert(day_boundary)
    return pd.Index(local.date)


# ── Presets ─────────────────────────────────────────────────────────────────

_ALL_PHASES = frozenset(Phase)

FUNDEDNEXT_STELLAR_2STEP = PropFirmRuleSet(
    name="FundedNext Stellar 2-Step",
    phase_targets={
        Phase.CHALLENGE_PHASE_1: 0.08,
        Phase.CHALLENGE_PHASE_2: 0.05,
        Phase.FUNDED: None,
    },
    # 5% of initial balance, measured from the balance at the server-midnight
    # reset; open P&L, commissions and swaps count; profit made today widens the
    # room (help.fundednext.com, "How can I calculate the daily loss limit?")
    daily_loss_limit_pct=0.05,
    daily_loss_basis="initial_balance",
    daily_reference="balance",
    # Server time is GMT+2, GMT+3 in summer
    day_boundary=NY_CLOSE,
    # Balance or equity must never fall below 90% of the initial balance
    overall_loss_limit_pct=0.10,
    max_loss_mode="static",
    breach_on="equity",
    min_trading_days=5,
    trading_day_rule="any_trade",
    max_leverage=100,
    # Funded accounts: a trade's potential loss is capped at 3% of initial balance
    max_risk_per_trade_pct=0.03,
    max_risk_phases=frozenset({Phase.FUNDED}),
    # Trades held under 30 s must stay below 30% of total profit (all stages)
    quick_strike_seconds=30.0,
    quick_strike_max_profit_share=0.30,
    quick_strike_phases=_ALL_PHASES,
    # Funded: news trades within 5 min of high-impact events credit 40% of profit
    news_profit_credit=0.40,
    news_window_minutes=5.0,
    commission_per_lot=5.0,
)
"""
FundedNext Stellar 2-Step. Sources: help.fundednext.com articles 8021076,
17255124, 8019811 and 8019914.
"""

FTMO_2STEP = PropFirmRuleSet(
    name="FTMO 2-Step",
    phase_targets={
        Phase.CHALLENGE_PHASE_1: 0.10,
        Phase.CHALLENGE_PHASE_2: 0.05,
        Phase.FUNDED: None,
    },
    # Equity may not fall below (balance at 00:00 CE(S)T) - 5% of initial capital
    daily_loss_limit_pct=0.05,
    daily_loss_basis="initial_balance",
    daily_reference="balance",
    day_boundary="Europe/Prague",
    # Static: equity may not fall below 90% of initial capital
    overall_loss_limit_pct=0.10,
    max_loss_mode="static",
    breach_on="equity",
    min_trading_days=4,
    trading_day_rule="any_trade",
    max_leverage=100,
)
"""FTMO 2-Step Challenge and Verification. Source: ftmo.com/en/trading-objectives."""

FTMO_1STEP = PropFirmRuleSet(
    name="FTMO 1-Step",
    phase_targets={
        Phase.CHALLENGE_PHASE_1: 0.10,
        Phase.CHALLENGE_PHASE_2: None,
        Phase.FUNDED: None,
    },
    daily_loss_limit_pct=0.03,
    daily_loss_basis="initial_balance",
    daily_reference="balance",
    day_boundary="Europe/Prague",
    # Floor is 10% of initial capital below the highest end-of-day balance (or
    # the initial capital, if higher); it only rises
    overall_loss_limit_pct=0.10,
    max_loss_mode="trailing_eod",
    max_loss_trail_source="balance",
    breach_on="equity",
    min_trading_days=0,
    max_leverage=100,
)
"""
FTMO 1-Step. Source: ftmo.com/en/trading-objectives. The Best Day rule (best
day at most 50% of total profit) is not modelled.
"""

THE5ERS_HIGH_STAKES = PropFirmRuleSet(
    name="The5ers High Stakes",
    phase_targets={
        Phase.CHALLENGE_PHASE_1: 0.10,
        Phase.CHALLENGE_PHASE_2: 0.05,
        Phase.FUNDED: None,
    },
    # 5% of the higher of balance and equity at the server-midnight reset
    daily_loss_limit_pct=0.05,
    daily_loss_basis="day_reference",
    daily_reference="max_balance_equity",
    day_boundary=NY_CLOSE,
    overall_loss_limit_pct=0.10,
    max_loss_mode="static",
    breach_on="equity",
    # Three profitable days of at least 0.5% of initial balance, measured as
    # min(midnight balance, midnight equity) - previous day balance
    min_trading_days=3,
    trading_day_rule="profitable",
    profitable_day_min_pct=0.005,
    max_leverage=100,
)
"""
The5ers High Stakes. Sources: the5ers.com/high-stakes (targets, limits,
profitable-day definition) and the5ers.com/faqs/2-step-plan-rules-specifications
("previous day's closing balance or equity, whichever is higher"). The5ers does
not publish the High Stakes daily-loss reference itself; this preset assumes it
matches the 2-Step plan. No trades within 2 minutes of high-impact news is a
prohibition, not a profit credit, and is not modelled.
"""

PRESETS = {
    "fundednext_stellar_2step": FUNDEDNEXT_STELLAR_2STEP,
    "ftmo_2step": FTMO_2STEP,
    "ftmo_1step": FTMO_1STEP,
    "the5ers_high_stakes": THE5ERS_HIGH_STAKES,
}
