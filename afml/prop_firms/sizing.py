"""
Two-stage hybrid position sizer with configurable prop-firm rule integration.

Architecture
------------
Stage 1  - bet_size_probability (concurrency-corrected, confidence-weighted)
Stage 2  - Kelly payoff multiplier (payoff-ratio adjustment)
Modifier - w-param calibration chain (drawdown budget -> sigmoid w -> position scale)
Modifier - Profit-target proximity de-risking factor
Modifier - News-window adjustment (funded stage only)

The final position size is:

    final_size = stage1_signal x kelly_multiplier x w_modifier
                 x derisking_factor x news_factor

All modifiers are multiplicative and independent. Each can be inspected
individually in the output DataFrame's diagnostic columns.

Refactor note
-------------
derisking_factor's thresholds (de-risk starting at 80% of target, floor
at 30%) are a strategy design choice, not a prop-firm rule, so they are
NOT parameterized by PropFirmRuleSet. news_window_factor's profit-credit
haircut IS a firm rule and now takes it explicitly rather than reading
a module constant. PropFirmAwareSizer.size() sources both
news_profit_credit and news_window_minutes from state.rule_set, so the
rule set is the single source of truth; the sizer no longer has its own
news_window_minutes constructor parameter (public-API change from the
original module).

Dependencies
------------
    afml.bet_sizing.bet_sizing     : bet_size_probability
    afml.bet_sizing.ch10_snippets  : get_w, bet_size_sigmoid
    scipy, numpy, pandas
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..bet_sizing.bet_sizing import bet_size_probability
from ..bet_sizing.ch10_snippets import bet_size_sigmoid, get_w
from .account_state import PropFirmAccountState
from .phase import Phase

# ── w-param calibrator ───────────────────────────────────────────────────────


class WParamCalibrator:
    """
    Calibrates the sigmoid w parameter from the remaining drawdown budget.

    The chain is:

        risk_budget_pct -> cal_bet_size -> w_param -> sigmoid modifier

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
        stop_loss_pct: float = 0.01,
        safety_factor: float = 0.70,
        cal_divergence: float = 0.90,
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.safety_factor = safety_factor
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
        #   position x stop_loss <= risk_budget x safety_factor
        cal_bet_size = (risk_budget_pct * self.safety_factor) / self.stop_loss_pct
        cal_bet_size = float(np.clip(cal_bet_size, 0.0, 0.98))

        if cal_bet_size < 0.02:
            return np.inf  # budget exhausted - sigmoid collapses to zero

        return get_w(
            price_div=self.cal_divergence,
            m_bet_size=cal_bet_size,
            func="sigmoid",
        )


# ── Kelly payoff multiplier ──────────────────────────────────────────────────


def kelly_payoff_multiplier(
    prob: float,
    avg_win_loss_ratio: float = 1.0,
    kelly_fraction: float = 0.5,
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
        more than max_amplification x the Stage 1 signal at any observation.

    Returns
    -------
    float
        Multiplier in [0.0, max_amplification]. Returns 0.0 when Kelly
        f* <= 0 (no economic edge at this probability given the payoff
        ratio - close any open position).

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

    z = (prob - 0.5) / np.sqrt(prob * (1.0 - prob))
    signal = max(2.0 * norm.cdf(z) - 1.0, 1e-6)
    raw = (kelly_fraction * f_star) / signal
    return float(np.clip(raw, 0.0, max_amplification))


# ── Modifier functions ───────────────────────────────────────────────────────


def derisking_factor(
    phase_progress: float,
    meets_min_trading_days: bool,
    floor_days_met: float = 0.10,
    floor_days_unmet: float = 0.30,
) -> float:
    """
    Reduces position sizes as the phase profit target approaches, protecting
    accumulated progress. Scaling begins at 80% of target; which floor it
    reaches at 100% progress depends on whether the minimum trading-days
    requirement is already satisfied.

    If the requirement is not yet met, the account still has to generate
    qualifying trading days even after reaching the profit target, so the
    floor stays at floor_days_unmet (0.30 by default) -- enough continued
    trading activity to accumulate more days without material drawdown
    risk. If the requirement is already met, there is no remaining
    obligation to keep trading once the profit target is reached, so the
    floor can drop further (floor_days_met, 0.10 by default) to protect
    more of the progress already made.

    Not parameterized by PropFirmRuleSet: the 80% threshold and both
    floors are a strategy design choice, independent of which firm's
    rules are active. meets_min_trading_days is different in kind from
    those two -- it is not a tunable strategy choice, it is a fact about
    the account's current progress against a firm rule
    (PropFirmRuleSet.min_trading_days) -- which is why it is a required
    argument rather than a third tunable default: this function has no
    correct answer without it.

    Parameters
    ----------
    phase_progress : float
        Fraction of the phase profit target achieved, in [0.0, 1.0].
        Use PropFirmAccountState.phase_progress.
    meets_min_trading_days : bool
        Whether the rule set's minimum trading-days requirement is
        already satisfied. Use PropFirmAccountState.meets_min_trading_days.
    floor_days_met : float
        Floor applied once the trading-days requirement is satisfied.
    floor_days_unmet : float
        Floor applied while the trading-days requirement is still open.

    Returns
    -------
    float
        Multiplicative factor in [floor, 1.0], where floor is
        floor_days_met or floor_days_unmet depending on
        meets_min_trading_days.
    """
    if phase_progress < 0.80:
        return 1.0
    floor = floor_days_met if meets_min_trading_days else floor_days_unmet
    slope = (floor - 1.0) / (1.0 - 0.80)
    return float(np.clip(1.0 + slope * (phase_progress - 0.80), floor, 1.0))


def news_window_factor(
    current_time: datetime,
    news_times: list[datetime],
    news_profit_credit: float,
    window_minutes: float = 5.0,
    phase: Phase = Phase.CHALLENGE_PHASE_1,
) -> float:
    """
    News-window position size adjustment (funded stage only).

    Some prop-firm programs credit only a fraction of profits made
    within a window around high-impact news events. Sizing positions as
    if they will contribute the full P&L causes the strategy to
    systematically fall short of payout targets. news_profit_credit
    sizes positions to deliver their expected contribution after the
    profit-credit haircut.

    Parameters
    ----------
    current_time : datetime
        Current bar time (UTC).
    news_times : list[datetime]
        List of high-impact news event times for today (UTC).
    news_profit_credit : float
        Fraction of P&L credited inside the news window, from
        PropFirmRuleSet.news_profit_credit. 1.0 disables the haircut.
    window_minutes : float
        Half-width of the news window in minutes. Default 5.0.
    phase : Phase
        Only applies the factor during Phase.FUNDED.

    Returns
    -------
    float
        news_profit_credit if in a news window and funded, 1.0 otherwise.
    """
    if phase != Phase.FUNDED or not news_times or news_profit_credit >= 1.0:
        return 1.0

    window = pd.Timedelta(minutes=window_minutes)
    ct = pd.Timestamp(current_time).tz_localize(None)
    for nt in news_times:
        nt_ts = pd.Timestamp(nt).tz_localize(None)
        if abs(ct - nt_ts) <= window:
            return news_profit_credit

    return 1.0


# ── Full sizer ────────────────────────────────────────────────────────────────


class PropFirmAwareSizer:
    """
    Full two-stage hybrid sizer with prop firm risk integration.

    Combines:
        Stage 1 : bet_size_probability (concurrency-corrected signal)
        Stage 2 : kelly_payoff_multiplier (payoff-ratio adjustment)
        Modifier: WParamCalibrator (drawdown budget -> sigmoid scale)
        Modifier: derisking_factor (profit-target proximity)
        Modifier: news_window_factor (funded stage only, rule-set driven)

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
    """

    def __init__(
        self,
        w_calibrator: WParamCalibrator,
        avg_win_loss_ratio: float = 1.0,
        kelly_fraction: float = 0.5,
        max_amplification: float = 1.5,
        step_size: float = 0.05,
    ) -> None:
        self.w_calibrator = w_calibrator
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_amplification = max_amplification
        self.step_size = step_size

    def size(
        self,
        events: pd.DataFrame,
        prob: pd.Series,
        pred: pd.Series,
        state: PropFirmAccountState,
        news_times: list[datetime] | None = None,
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
            Current account state. Must be updated before each call. Its
            rule_set supplies the news-window profit credit and window
            width; there is no separate sizer-level news parameter.
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
        news_times = news_times or []
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
            return self._zero_result(events.index, stage1, kelly_mult, w_param)

        max_bet = float(
            np.clip(
                (state.risk_budget_pct * self.w_calibrator.safety_factor)
                / self.w_calibrator.stop_loss_pct,
                0.0,
                0.98,
            )
        )

        # A magnitude modifier: the side comes from stage 1 alone. Left signed, it
        # multiplies a short signal's sign by itself and turns every short long.
        sigmoid_scale = pd.Series(
            [abs(bet_size_sigmoid(w_param, float(s))) for s in stage1],
            index=stage1.index,
        )

        # -- Modifier: de-risking and news window -----------------------------
        derisking = derisking_factor(state.phase_progress, state.meets_min_trading_days)
        news = news_window_factor(
            current_time=current_time,
            news_times=news_times,
            news_profit_credit=state.rule_set.news_profit_credit,
            window_minutes=state.rule_set.news_window_minutes,
            phase=state.phase,
        )

        combined_modifier = sigmoid_scale * derisking * news

        # -- Final size ---------------------------------------------------------
        raw_final = stage1 * kelly_mult * combined_modifier
        final_size = (raw_final / self.step_size).round() * self.step_size
        final_size = final_size.clip(-1.0, 1.0)

        return pd.DataFrame(
            {
                "stage1_signal": stage1,
                "kelly_multiplier": kelly_mult,
                "w_param": w_param,
                "max_bet_size": max_bet,
                "sigmoid_scale": sigmoid_scale,
                "derisking": derisking,
                "news": news,
                "combined_modifier": combined_modifier,
                "final_size": final_size,
            },
            index=events.index,
        )

    def _zero_result(
        self,
        index: pd.Index,
        stage1: pd.Series,
        kelly_mult: pd.Series,
        w_param: float,
    ) -> pd.DataFrame:
        z = pd.Series(0.0, index=index)
        return pd.DataFrame(
            {
                "stage1_signal": stage1,
                "kelly_multiplier": kelly_mult,
                "w_param": w_param,
                "max_bet_size": 0.0,
                "sigmoid_scale": z,
                "derisking": 0.0,
                "news": 1.0,
                "combined_modifier": z,
                "final_size": z,
            },
            index=index,
        )


# ── Factory function ─────────────────────────────────────────────────────────


def make_sizer(
    stop_loss_pct: float = 0.01,
    safety_factor: float = 0.70,
    cal_divergence: float = 0.90,
    avg_win_loss_ratio: float = 1.0,
    kelly_fraction: float = 0.5,
    max_amplification: float = 1.5,
    step_size: float = 0.05,
) -> PropFirmAwareSizer:
    """
    Convenience factory for a PropFirmAwareSizer.

    Renamed from make_stellar_2step_sizer: the sizer itself has no
    firm-specific logic left (news parameters now come from the
    PropFirmRuleSet on the PropFirmAccountState passed to .size() at
    call time), so a firm-specific factory name was no longer accurate.

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

    Returns
    -------
    PropFirmAwareSizer

    Example
    -------
    >>> from afml.prop_firm import (
    ...     PropFirmAccountState, Phase, make_sizer, FUNDEDNEXT_STELLAR_2STEP,
    ... )
    >>>
    >>> state = PropFirmAccountState(initial_balance=100_000.0,
    ...                              phase=Phase.CHALLENGE_PHASE_1,
    ...                              rule_set=FUNDEDNEXT_STELLAR_2STEP)
    >>> sizer = make_sizer(stop_loss_pct=0.01, avg_win_loss_ratio=1.2,
    ...                     kelly_fraction=0.5)
    >>>
    >>> # On every bar:
    >>> state.update(realized_pnl_delta=closed_pnl,
    ...              unrealized_pnl=floating_pnl,
    ...              fees_and_swaps=bar_fees)  # accrued since the last update
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
    )


# ── Stop-loss ceiling ────────────────────────────────────────────────────────
#
# Canonical home for a function previously duplicated verbatim (modulo one
# variable rename) in backtester/optimal_trading_rules.py and
# backtester/synthetic_backtest.py. Both call sites should import it from
# here; see MIGRATION.md.


def prop_firm_sl_ceiling(
    account_usd: float = 25_000.0,
    daily_loss_pct: float = 0.05,
    lot_size: float = 0.5,
    n_concurrent: int = 2,
    pip_value_per_lot: float = 10.0,
    bet_size: float = 1.0,
    aggregate_exposure: bool = False,
) -> float:
    """
    Maximum stop-loss per trade in pips under a prop-firm daily loss cap.

    Most funded-account programs cap daily losses at a fixed percentage of
    the account balance (commonly 4-5%). If n_concurrent positions can all
    hit their stop-losses on the same day, each stop-loss must be sized so
    that the total loss stays within the daily limit.

    Parameters
    ----------
    account_usd : float
        Funded account size in USD.
    daily_loss_pct : float
        Maximum daily loss as a fraction (e.g. 0.05 = 5%). Pass
        state.risk_budget_pct for a ceiling valid at the current point in
        the session, not rule_set.daily_loss_limit_pct (see the Production
        Problem section of this article).
    lot_size : float
        Notional at FULL signal strength (|final_size| = 1), not the lot
        actually sent to the broker.
    n_concurrent : int
        Maximum number of simultaneously open positions. Ignored when
        aggregate_exposure=True (see below).
    pip_value_per_lot : float
        USD value of one pip per standard lot (10.0 for most EURUSD
        broker configurations).
    bet_size : float
        PropFirmAwareSizer output for this trade, in [-1, 1]. Executed
        notional is lot_size * |bet_size|; the ceiling scales as
        1 / |bet_size|. Defaults to 1.0 for backward compatibility, which
        reproduces the pre-existing behavior exactly -- but 1.0 is the one
        bet size PropFirmAwareSizer.size() almost never emits.
    aggregate_exposure : bool
        True  -> avg_active_signals() semantics: concurrent bets are
                 averaged into one net position, so exposure never exceeds
                 lot_size * |bet_size| and n_concurrent does NOT divide.
        False -> legacy behavior: each bet held at its own notional, total
                 exposure is n_concurrent * lot_size * |bet_size|. Correct
                 only if positions are actually sized and executed
                 independently rather than netted by average_active=True.

    Returns
    -------
    float
        Maximum stop-loss per trade in pips. inf when bet_size is 0.
    """
    if not 0.0 <= abs(bet_size) <= 1.0:
        raise ValueError(f"bet_size must lie in [-1, 1]; got {bet_size!r}")
    if lot_size <= 0 or pip_value_per_lot <= 0 or n_concurrent < 1:
        raise ValueError("lot_size, pip_value_per_lot must be > 0; n_concurrent >= 1")

    exposure_lots = lot_size * abs(bet_size)
    if not aggregate_exposure:
        exposure_lots *= n_concurrent
    if exposure_lots == 0.0:
        return float("inf")

    daily_limit_usd = account_usd * daily_loss_pct
    return daily_limit_usd / (exposure_lots * pip_value_per_lot)
