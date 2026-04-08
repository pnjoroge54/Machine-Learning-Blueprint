"""
Two-stage hybrid position sizer with FundedNext Stellar 2-Step rule integration.

Architecture
────────────
Stage 1  – get_signal / bet_size_probability (concurrency-corrected, confidence-weighted)
Stage 2  – Kelly payoff multiplier (payoff-ratio adjustment)
Modifier – w-param calibration chain (drawdown budget → sigmoid w → position scale)
Modifier – Profit-target proximity de-risking factor
Modifier – News-window adjustment (funded stage only)

The final position size is:

    final_size = stage1_signal × kelly_multiplier × w_modifier
                 × derisking_factor × news_factor

All modifiers are multiplicative and independent. Each can be inspected
individually in the output DataFrame's diagnostic columns.

Dependencies
────────────
    afml.bet_sizing.bet_sizing     : bet_size_probability
    afml.bet_sizing.ch10_snippets  : get_w, bet_size_sigmoid
    afml.cross_validation.cross_validation : PurgedKFold (used externally)
    scipy, numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..bet_sizing.bet_sizing import bet_size_probability
from ..bet_sizing.ch10_snippets import get_w, bet_size_sigmoid


# ── Phase enumeration ─────────────────────────────────────────────────────────

class Phase(Enum):
    CHALLENGE_PHASE_1 = "challenge_phase_1"
    CHALLENGE_PHASE_2 = "challenge_phase_2"
    FUNDED            = "funded"


# ── FundedNext Stellar 2-Step rule constants ──────────────────────────────────

_PHASE_TARGETS = {
    Phase.CHALLENGE_PHASE_1: 0.08,
    Phase.CHALLENGE_PHASE_2: 0.05,
    Phase.FUNDED:            None,   # no profit target in funded stage
}

_DAILY_LOSS_LIMIT_PCT   = 0.05   # 5% of initial balance (expandable)
_OVERALL_LOSS_LIMIT_PCT = 0.10   # 10% of initial balance (floor fixed)
_NEWS_PROFIT_CREDIT     = 0.40   # only 40% of profits in ±5 min window count
_MIN_TRADING_DAYS       = 5
_MAX_LEVERAGE           = 100
_COMMISSION_PER_LOT     = 5.0    # USD


# ── Account state ─────────────────────────────────────────────────────────────

@dataclass
class PropFirmAccountState:
    """
    Tracks the mutable account state that changes bar-by-bar.

    Parameters
    ----------
    initial_balance : float
        Account balance at the start of the phase.
    phase : Phase
        Current phase (determines profit target and rule set).

    Notes
    -----
    The daily loss limit is dynamic: it equals the base limit plus any
    intraday realized profit made so far today. This is the most commonly
    misunderstood FundedNext rule. The overall floor is permanently anchored
    at initial_balance * (1 - _OVERALL_LOSS_LIMIT_PCT) and does not move.
    """

    initial_balance: float
    phase: Phase

    # Running account metrics — updated by PropFirmAccountState.update()
    current_equity:         float = field(init=False)
    daily_loss_used:        float = field(init=False)
    intraday_realized_pnl:  float = field(init=False)
    trading_days_completed: int   = field(init=False)
    phase_profit_pct:       float = field(init=False)

    def __post_init__(self) -> None:
        self.current_equity         = self.initial_balance
        self.daily_loss_used        = 0.0
        self.intraday_realized_pnl  = 0.0
        self.trading_days_completed = 0
        self.phase_profit_pct       = 0.0

    # -- Properties ------------------------------------------------------------

    @property
    def overall_floor(self) -> float:
        """Balance/equity floor — permanently anchored, never moves."""
        return self.initial_balance * (1.0 - _OVERALL_LOSS_LIMIT_PCT)

    @property
    def daily_limit(self) -> float:
        """
        Dynamic daily loss limit.

        Expands by the amount of intraday realized profit already made today.
        Commissions, swaps, and fees count against it.
        """
        base = self.initial_balance * _DAILY_LOSS_LIMIT_PCT
        return base + max(0.0, self.intraday_realized_pnl)

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
        Returns 0.0 for the funded stage (no target).
        """
        target = _PHASE_TARGETS.get(self.phase)
        if target is None or target == 0.0:
            return 0.0
        return min(self.phase_profit_pct / target, 1.0)

    # -- Mutation --------------------------------------------------------------

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
            Commissions and overnight swaps accrued since the last update.
            Must be passed as a positive number; the method subtracts it.
        """
        self.intraday_realized_pnl += realized_pnl_delta
        net_change = realized_pnl_delta - fees_and_swaps
        if net_change < 0:
            self.daily_loss_used += abs(net_change)

        self.current_equity = (
            self.initial_balance
            + self.intraday_realized_pnl
            + unrealized_pnl
            - fees_and_swaps
        )
        self.phase_profit_pct = max(
            0.0,
            (self.current_equity - self.initial_balance) / self.initial_balance,
        )

    def reset_daily(self) -> None:
        """Call at the start of each new trading day."""
        self.daily_loss_used       = 0.0
        self.intraday_realized_pnl = 0.0
        self.trading_days_completed += 1


# ── w-param calibrator ────────────────────────────────────────────────────────

class WParamCalibrator:
    """
    Calibrates the sigmoid w parameter from the remaining drawdown budget.

    The chain is:

        risk_budget_pct → cal_bet_size → w_param → sigmoid modifier

    A smaller risk budget produces a larger w, which flattens the sigmoid
    function and causes the same ML signal to produce a smaller position.
    The strategy de-risks automatically as the budget is consumed.

    Parameters
    ----------
    stop_loss_pct : float
        Fraction of account balance risked per trade at the stop.
    safety_factor : float
        Fraction of calculated budget to use. The remainder (1 - safety_factor)
        absorbs commissions, swaps, and gap risk.
    cal_divergence : float
        Reference divergence for calibration. The w is set so that at this
        divergence level the sigmoid outputs cal_bet_size.
    """

    def __init__(
        self,
        stop_loss_pct:  float = 0.01,
        safety_factor:  float = 0.70,
        cal_divergence: float = 0.90,
    ) -> None:
        self.stop_loss_pct  = stop_loss_pct
        self.safety_factor  = safety_factor
        self.cal_divergence = cal_divergence

    def calibrate(self, state: PropFirmAccountState) -> float:
        """
        Compute the sigmoid w parameter from the current account state.

        Returns
        -------
        float
            w parameter for bet_size_sigmoid. Returns np.inf (flat sigmoid,
            zero position) when the budget is exhausted.
        """
        risk_budget_pct = state.risk_budget_pct

        # Maximum allowable bet size at full signal strength:
        #   position × stop_loss ≤ risk_budget × safety_factor
        cal_bet_size = (risk_budget_pct * self.safety_factor) / self.stop_loss_pct
        cal_bet_size = float(np.clip(cal_bet_size, 0.0, 0.98))

        if cal_bet_size < 0.02:
            return np.inf   # budget exhausted — sigmoid collapses to zero

        return get_w(
            price_div=self.cal_divergence,
            m_bet_size=cal_bet_size,
            func="sigmoid",
        )


# ── Kelly payoff multiplier ───────────────────────────────────────────────────

def kelly_payoff_multiplier(
    prob:              float,
    avg_win_loss_ratio: float = 1.0,
    kelly_fraction:    float = 0.5,
    max_amplification: float = 1.5,
) -> float:
    """
    Kelly multiplier relative to get_signal at the same probability.

    Computes the ratio of the fractional Kelly fraction to the get_signal
    output at the same probability. The result is applied as a second-stage
    multiplier on the Stage 1 signal.

    Parameters
    ----------
    prob : float
        Predicted probability from the classifier (0.5, 1.0].
    avg_win_loss_ratio : float
        b in the Kelly formula f* = (p*b - q) / b. Estimated from the
        live or backtest trade log; update monthly.
    kelly_fraction : float
        Fraction of full Kelly to use. 0.5 (half-Kelly) is the standard
        default under estimation uncertainty (Masters, 1995).
    max_amplification : float
        Hard upper bound on the multiplier. Prevents Kelly from recommending
        more than max_amplification × the Stage 1 signal at any observation.

    Returns
    -------
    float
        Multiplier in [0.0, max_amplification].
        Returns 0.0 when Kelly f* ≤ 0 (no economic edge at this probability
        given the payoff ratio — close any open position).

    Notes
    -----
    When b = 1 (symmetric payoffs), the multiplier is approximately 1.0
    throughout and the two stages are consistent. When b > 1 (favorable
    payoffs), the multiplier exceeds 1.0 at moderate probabilities.
    """
    b = avg_win_loss_ratio
    q = 1.0 - prob
    f_star = (prob * b - q) / b

    if f_star <= 0:
        return 0.0

    z      = (prob - 0.5) / np.sqrt(prob * (1.0 - prob))
    signal = max(2.0 * norm.cdf(z) - 1.0, 1e-6)
    raw    = (kelly_fraction * f_star) / signal
    return float(np.clip(raw, 0.0, max_amplification))


# ── Modifier functions ────────────────────────────────────────────────────────

def derisking_factor(phase_progress: float) -> float:
    """
    Profit-target proximity de-risking factor.

    Reduces position sizes as the phase profit target approaches, protecting
    accumulated progress. Scaling begins at 80% of target and reaches 30%
    at 100% progress. The 30% floor allows continued trading to satisfy
    the minimum 5 trading days rule without material drawdown risk.

    Parameters
    ----------
    phase_progress : float
        Fraction of the phase profit target achieved, in [0.0, 1.0].
        Use PropFirmAccountState.phase_progress.

    Returns
    -------
    float
        Multiplicative factor in [0.30, 1.0].
    """
    if phase_progress < 0.80:
        return 1.0
    # Linear decay from 1.0 at 80% progress to 0.30 at 100%
    slope = (0.30 - 1.0) / (1.0 - 0.80)
    return float(np.clip(1.0 + slope * (phase_progress - 0.80), 0.30, 1.0))


def news_window_factor(
    current_time: datetime,
    news_times:   list[datetime],
    window_minutes: float = 5.0,
    phase: Phase = Phase.CHALLENGE_PHASE_1,
) -> float:
    """
    News-window position size adjustment (funded stage only).

    FundedNext credits only 40% of profits made within ±5 minutes of
    high-impact news events. Sizing positions as if they will contribute
    the full P&L causes the strategy to systematically fall short of
    payout targets. The 0.40 factor sizes positions to deliver their
    expected contribution after the profit-credit haircut.

    Parameters
    ----------
    current_time : datetime
        Current bar time (UTC).
    news_times : list[datetime]
        List of high-impact news event times for today (UTC).
    window_minutes : float
        Half-width of the news window in minutes. Default 5.0.
    phase : Phase
        Only applies the factor during Phase.FUNDED.

    Returns
    -------
    float
        _NEWS_PROFIT_CREDIT (0.40) if in a news window and funded,
        1.0 otherwise.
    """
    if phase != Phase.FUNDED or not news_times:
        return 1.0

    window = pd.Timedelta(minutes=window_minutes)
    ct     = pd.Timestamp(current_time).tz_localize(None)
    for nt in news_times:
        nt_ts = pd.Timestamp(nt).tz_localize(None)
        if abs(ct - nt_ts) <= window:
            return _NEWS_PROFIT_CREDIT

    return 1.0


# ── Full sizer ────────────────────────────────────────────────────────────────

class PropFirmAwareSizer:
    """
    Full two-stage hybrid sizer with prop firm risk integration.

    Combines:
        Stage 1 : bet_size_probability (concurrency-corrected signal)
        Stage 2 : kelly_payoff_multiplier (payoff-ratio adjustment)
        Modifier: WParamCalibrator (drawdown budget → sigmoid scale)
        Modifier: derisking_factor (profit-target proximity)
        Modifier: news_window_factor (funded stage only)

    Parameters
    ----------
    w_calibrator : WParamCalibrator
    avg_win_loss_ratio : float
        b in the Kelly formula. Estimate from the live trade log.
    kelly_fraction : float
        Half-Kelly (0.5) is the recommended default.
    max_amplification : float
        Upper bound on the Kelly multiplier.
    step_size : float
        Discretization step for bet_size_probability.
    news_window_minutes : float
        Half-width of the news window in minutes.
    """

    def __init__(
        self,
        w_calibrator:       WParamCalibrator,
        avg_win_loss_ratio: float = 1.0,
        kelly_fraction:     float = 0.5,
        max_amplification:  float = 1.5,
        step_size:          float = 0.05,
        news_window_minutes: float = 5.0,
    ) -> None:
        self.w_calibrator        = w_calibrator
        self.avg_win_loss_ratio  = avg_win_loss_ratio
        self.kelly_fraction      = kelly_fraction
        self.max_amplification   = max_amplification
        self.step_size           = step_size
        self.news_window_minutes = news_window_minutes

    def size(
        self,
        events:       pd.DataFrame,
        prob:         pd.Series,
        pred:         pd.Series,
        state:        PropFirmAccountState,
        news_times:   list[datetime] | None = None,
        current_time: datetime | None = None,
        average_active: bool = True,
    ) -> pd.DataFrame:
        """
        Compute final discretized position sizes with full diagnostic record.

        Parameters
        ----------
        events : pd.DataFrame
            Events DataFrame with DatetimeIndex and 't1' column
            (label end times), as produced by the triple-barrier labeller.
        prob : pd.Series
            Predicted probabilities from the classifier, aligned with events.
        pred : pd.Series
            Primary side predictions (+1 long, -1 short), aligned with events.
        state : PropFirmAccountState
            Current account state. Must be updated before each call.
        news_times : list[datetime], optional
            High-impact news event times for today (UTC). Pass None or []
            if no news filter is required.
        current_time : datetime, optional
            Current bar time. Used for the news window check.
        average_active : bool
            Whether to apply avg_active_signals concurrency correction.
            Should always be True for triple-barrier labels.

        Returns
        -------
        pd.DataFrame
            One row per observation with columns:
            stage1_signal, kelly_multiplier, w_param, max_bet_size,
            derisking, news, combined_modifier, final_size.
        """
        news_times   = news_times or []
        current_time = current_time or datetime.now(tz=timezone.utc)

        # -- Stage 1: concurrency-corrected signal ----------------------------
        stage1 = bet_size_probability(
            events=events,
            prob=prob,
            num_classes=2,
            pred=pred,
            step_size=self.step_size,
            average_active=average_active,
        )

        # -- Stage 2: Kelly payoff multiplier ---------------------------------
        kelly_mult = prob.map(
            lambda p: kelly_payoff_multiplier(
                prob=p,
                avg_win_loss_ratio=self.avg_win_loss_ratio,
                kelly_fraction=self.kelly_fraction,
                max_amplification=self.max_amplification,
            )
        )

        # -- Modifier: w-param calibration chain ------------------------------
        w_param = self.w_calibrator.calibrate(state)
        if np.isinf(w_param):
            # Budget exhausted — return zero-sized result immediately
            return self._zero_result(events.index, stage1, kelly_mult, w_param)

        max_bet = float(np.clip(
            (state.risk_budget_pct * self.w_calibrator.safety_factor)
            / self.w_calibrator.stop_loss_pct,
            0.0, 0.98,
        ))

        # Apply sigmoid modifier: scale the stage1 signal by the budget curve
        # The sigmoid evaluated at stage1 (used as divergence proxy) shrinks
        # the signal in proportion to available risk capacity.
        sigmoid_scale = pd.Series(
            [bet_size_sigmoid(w_param, float(s)) for s in stage1],
            index=stage1.index,
        )

        # -- Modifier: de-risking and news window -----------------------------
        derisking = derisking_factor(state.phase_progress)
        news      = news_window_factor(
            current_time=current_time,
            news_times=news_times,
            window_minutes=self.news_window_minutes,
            phase=state.phase,
        )

        combined_modifier = sigmoid_scale * derisking * news

        # -- Final size -------------------------------------------------------
        raw_final   = stage1 * kelly_mult * combined_modifier
        final_size  = (raw_final / self.step_size).round() * self.step_size
        final_size  = final_size.clip(-1.0, 1.0)

        return pd.DataFrame({
            "stage1_signal":     stage1,
            "kelly_multiplier":  kelly_mult,
            "w_param":           w_param,
            "max_bet_size":      max_bet,
            "sigmoid_scale":     sigmoid_scale,
            "derisking":         derisking,
            "news":              news,
            "combined_modifier": combined_modifier,
            "final_size":        final_size,
        }, index=events.index)

    def _zero_result(
        self,
        index:      pd.Index,
        stage1:     pd.Series,
        kelly_mult: pd.Series,
        w_param:    float,
    ) -> pd.DataFrame:
        z = pd.Series(0.0, index=index)
        return pd.DataFrame({
            "stage1_signal":     stage1,
            "kelly_multiplier":  kelly_mult,
            "w_param":           w_param,
            "max_bet_size":      0.0,
            "sigmoid_scale":     z,
            "derisking":         0.0,
            "news":              1.0,
            "combined_modifier": z,
            "final_size":        z,
        }, index=index)


# ── Factory function ──────────────────────────────────────────────────────────

def make_stellar_2step_sizer(
    stop_loss_pct:      float = 0.01,
    safety_factor:      float = 0.70,
    cal_divergence:     float = 0.90,
    avg_win_loss_ratio: float = 1.0,
    kelly_fraction:     float = 0.5,
    max_amplification:  float = 1.5,
    step_size:          float = 0.05,
    news_window_minutes: float = 5.0,
) -> PropFirmAwareSizer:
    """
    Convenience factory for the FundedNext Stellar 2-Step sizer.

    Parameters
    ----------
    stop_loss_pct : float
        Fraction of account balance risked per trade at the stop (default 1%).
    safety_factor : float
        Fraction of available budget to use; remainder absorbs frictions.
    cal_divergence : float
        Calibration divergence for the sigmoid w parameter.
    avg_win_loss_ratio : float
        Estimated average win-to-loss ratio from the live trade log.
        Update monthly. b=1 for symmetric payoffs.
    kelly_fraction : float
        Fraction of full Kelly to apply (default 0.5, half-Kelly).
    max_amplification : float
        Upper bound on the Kelly multiplier.
    step_size : float
        Discretization step for position sizes (default 0.05 = 5%).
    news_window_minutes : float
        Half-width of the news window in minutes (default 5.0).

    Returns
    -------
    PropFirmAwareSizer

    Example
    -------
    >>> from prop_firm_sizer import PropFirmAccountState, Phase, make_stellar_2step_sizer
    >>>
    >>> state = PropFirmAccountState(initial_balance=100_000.0,
    ...                              phase=Phase.CHALLENGE_PHASE_1)
    >>> sizer = make_stellar_2step_sizer(stop_loss_pct=0.01,
    ...                                  avg_win_loss_ratio=1.2,
    ...                                  kelly_fraction=0.5)
    >>>
    >>> # On every bar:
    >>> state.update(realized_pnl_delta=closed_pnl,
    ...              unrealized_pnl=floating_pnl,
    ...              fees_and_swaps=today_fees)
    >>> result = sizer.size(events=events_df, prob=probs, pred=sides,
    ...                     state=state, average_active=True)
    >>> position_sizes = result['final_size']
    """
    return PropFirmAwareSizer(
        w_calibrator=WParamCalibrator(
            stop_loss_pct=stop_loss_pct,
            safety_factor=safety_factor,
            cal_divergence=cal_divergence,
        ),
        avg_win_loss_ratio=avg_win_loss_ratio,
        kelly_fraction=kelly_fraction,
        max_amplification=max_amplification,
        step_size=step_size,
        news_window_minutes=news_window_minutes,
    )
