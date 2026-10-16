"""
CPCV dynamic backtest orchestrator with fresh account state per path.

Architecture (four-phase)
─────────────────────────
  Phase 1 — Train:      Fit the base estimator on every C(N, k) split.
                          The estimator must be a plain classifier, NOT a
                          CalibratorCV (which carries an internal PurgedKFold
                          with a full-length t1 that breaks on subsets).

  Phase 2 — Assemble:   Recombine per-split OOF predictions into φ[N, k]
                          backtest paths via recombine_test_predictions().

  Phase 2.5 — Calibrate: Optionally fit an isotonic or Platt calibration map
                          on the pooled OOF predictions and apply it to each
                          path.  This replaces CalibratorCV's internal CV loop
                          while using the same OOF predictions that CPCV
                          already produced.

  Phase 3 — Simulate:   Run a bar-by-bar P&L simulation on each assembled
                          (and optionally calibrated) path with a fresh
                          PropFirmAccountState.

Dependencies
────────────
    afml.cross_validation.combinatorial : CombinatorialPurgedCV,
                                          optimal_folds_number
    afml.bet_sizing.prop_firm_sizer     : PropFirmAwareSizer,
                                          PropFirmAccountState, Phase
    joblib, numpy, pandas, sklearn
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..cross_validation.combinatorial import CombinatorialPurgedCV
from ..prop_firms import (
    FUNDEDNEXT_STELLAR_2STEP,
    Phase,
    PropFirmAccountState,
    PropFirmAwareSizer,
    PropFirmRuleSet,
)

# ── Backtest configuration ────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """
    Static configuration shared across all CPCV paths.

    Parameters
    ----------
    initial_balance : float
        Starting account balance for every path simulation.
    phase : Phase
        Prop firm phase. Selects the profit target and phase-scoped rules.
    pip_value : float
        Account-currency value of one pip per standard lot.
    pip_size : float
        Price change that makes one pip (0.0001 for most FX pairs, 0.01 for
        JPY pairs and many CFDs).
    lot_size : float
        Lots held at full signal strength (|bet size| = 1).
    commission_per_lot : float
        Commission per lot traded, charged on every change in position.
    slippage_pips : float
        Assumed slippage in pips per lot traded.
    stop_loss_pct : float or None
        Stop distance as a fraction of price. Needed only when the rule set
        caps per-trade risk in this phase; falls back to the sizer's
        ``stop_loss_pct``.
    n_jobs : int
        Number of parallel workers. -1 = all physical cores.
    random_state : int
        Seed for any stochastic components.
    rule_set : PropFirmRuleSet
        Program rules (FundedNext, FTMO, The5ers presets, or custom).
    """

    initial_balance: float = 100_000.0
    phase: Phase = Phase.CHALLENGE_PHASE_1
    pip_value: float = 10.0
    pip_size: float = 0.0001
    lot_size: float = 0.01
    commission_per_lot: float = 5.0
    slippage_pips: float = 0.5
    stop_loss_pct: float | None = None
    n_jobs: int = -1
    random_state: int = 42
    rule_set: PropFirmRuleSet = field(default_factory=lambda: FUNDEDNEXT_STELLAR_2STEP)

    @property
    def value_per_price_unit(self) -> float:
        """Account-currency P&L of a one-unit price move on one lot."""
        return self.pip_value / self.pip_size


# ── Per-path result container ─────────────────────────────────────────────────


@dataclass
class PathResult:
    """Result of a single CPCV path simulation."""

    path_id: int
    equity_curve: pd.Series
    returns: pd.Series
    sharpe: float
    max_drawdown: float
    daily_breaches: int
    overall_breach: bool
    trading_days: int
    hit_target: bool
    final_equity: float
    breach: str | None = None
    breach_time: pd.Timestamp | None = None
    final_balance: float | None = None
    quick_strike_ok: bool = True
    n_trades: int = 0


# ── Phase 1 helper: fit one split ─────────────────────────────────────────────


def _fit_predict_split(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_weight: pd.Series | None = None,
) -> np.ndarray:
    """
    Fit the estimator on one split's training set and return test predictions.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must be a plain classifier (not CalibratorCV).  Will be cloned.
    X, y : pd.DataFrame, pd.Series
        Full feature matrix and labels.
    train_idx, test_idx : np.ndarray
        Positional indices for this split's training and (concatenated) test set.
    sample_weight : pd.Series, optional
        Per-observation sample weights aligned with X.

    Returns
    -------
    np.ndarray
        Predicted probabilities (class 1) for the test observations.
    """
    model = clone(estimator)

    if sample_weight is not None:
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            sample_weight=sample_weight.iloc[train_idx],
        )
    else:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

    return model.predict_proba(X.iloc[test_idx])[:, 1]


# ── Phase 2.5 helper: calibrate path predictions ─────────────────────────────


def _fit_calibrator(
    path_predictions: list[np.ndarray],
    y: pd.Series,
    method: str,
) -> IsotonicRegression | LogisticRegression:
    """
    Fit a calibration map on pooled OOF predictions from all paths.

    Each observation appears in exactly one path, so pooling all path
    predictions produces a full-length OOF probability array suitable
    for fitting the calibration map without data leakage.

    Parameters
    ----------
    path_predictions : list of np.ndarray
        Raw OOF probabilities per path (from recombine_test_predictions).
    y : pd.Series
        True labels aligned with X.
    method : str
        'isotonic' or 'platt'.

    Returns
    -------
    calibrator : fitted IsotonicRegression or LogisticRegression
    """
    # Pool: average across paths at each observation.
    # recombine_test_predictions returns one array per path, each of length
    # n_observations.  At each index the observation appeared OOS in exactly
    # one path, so averaging recovers the single OOF prediction per bar.
    stacked = np.column_stack(path_predictions)  # (n_obs, n_paths)
    pooled = np.nanmean(stacked, axis=1)

    valid = ~np.isnan(pooled)
    y_arr = y.values[valid].astype(int)
    p_arr = pooled[valid]

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(p_arr, y_arr)
    elif method == "platt":
        cal = LogisticRegression(solver="lbfgs", max_iter=1000)
        cal.fit(p_arr.reshape(-1, 1), y_arr)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}. Use 'isotonic' or 'platt'.")

    return cal


def _apply_calibrator(
    probs: np.ndarray,
    calibrator: IsotonicRegression | LogisticRegression,
    method: str,
) -> np.ndarray:
    """Apply a fitted calibrator to raw probabilities."""
    if method == "isotonic":
        return calibrator.transform(probs)
    else:
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]


# ── Phase 3 helper: simulate one assembled path ──────────────────────────────


def _as_prices(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Close prices, plus high/low when given (needed for intrabar breach checks)."""
    if isinstance(prices, pd.Series):
        df = prices.to_frame("close")
    else:
        df = prices.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df.columns:
            raise KeyError("prices must have a 'close' column")
    for col in ("high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    return df[["close", "high", "low"]].astype(float)


def _simulate_path(
    path_id: int,
    oof_probs: pd.Series,
    events: pd.DataFrame,
    prices: pd.Series | pd.DataFrame,
    sizer: PropFirmAwareSizer,
    cfg: BacktestConfig,
    primary_sides: pd.Series,
) -> PathResult:
    """
    Simulate one fully-assembled backtest path bar-by-bar.

    Accounting
    ----------
    * The target position for bar t comes from the prediction at bar t and is
      traded at bar t's close, so it earns from bar t+1 onward (no lookahead).
    * Positions are held in lots (``bet size * cfg.lot_size``) with an average
      entry price. P&L is ``lots * price change / pip_size * pip_value``;
      commission and slippage are charged per lot traded, once.
    * Closing or reducing a position realizes P&L into the balance; open
      positions are marked at each close as floating P&L (equity).
    * Each bar's worst price for the open position (low for longs, high for
      shorts) is checked against the rule set's floors before the bar's trade,
      so equity breaches inside a bar are caught. Without high/low columns the
      check falls back to closes, which understates breaches.
    * Days roll on the rule set's day boundary; timestamps without a timezone
      are treated as UTC. A breach closes the account: equity is frozen at the
      breach level and the path stops trading.

    Parameters
    ----------
    path_id : int
        Index of this combinatorial path.
    oof_probs : pd.Series
        Out-of-fold predicted probabilities (raw or calibrated).
    events : pd.DataFrame
        Events table aligned with X.
    prices : pd.Series or pd.DataFrame
        Close prices, or OHLC bars with at least close (high/low recommended),
        indexed like X.
    sizer : PropFirmAwareSizer
        Fully configured sizer.
    cfg : BacktestConfig
    primary_sides : pd.Series
        Side predictions (+1/-1) aligned with X.

    Returns
    -------
    PathResult
    """
    # Drop bars with no prediction (purged/embargoed gaps)
    valid = oof_probs.notna()
    oof_probs = oof_probs.loc[valid]
    sides_valid = primary_sides.loc[oof_probs.index]
    events_valid = events.loc[oof_probs.index]
    px = _as_prices(prices).loc[oof_probs.index]

    state = PropFirmAccountState(
        initial_balance=cfg.initial_balance,
        phase=cfg.phase,
        rule_set=cfg.rule_set,
    )

    # Size positions from OOF predictions
    sizing_result = sizer.size(
        events=events_valid,
        prob=oof_probs,
        pred=sides_valid,
        state=state,
        average_active=True,
    )
    target_lots = sizing_result["final_size"].reindex(px.index).fillna(0.0) * cfg.lot_size

    vpu = cfg.value_per_price_unit
    # Per-trade risk cap (e.g. FundedNext's 3% maximum risk on funded accounts)
    if state.max_risk_amount is not None:
        stop_pct = cfg.stop_loss_pct or getattr(sizer, "stop_loss_pct", None)
        if not stop_pct:
            raise ValueError(
                "The rule set caps per-trade risk in this phase; set "
                "BacktestConfig.stop_loss_pct (stop distance as a fraction of price)."
            )
        max_lots = state.max_risk_amount / (stop_pct * px["close"] * vpu)
        target_lots = target_lots.clip(lower=-max_lots, upper=max_lots)

    cost_per_lot = cfg.commission_per_lot + cfg.slippage_pips * cfg.pip_value
    pos, entry = 0.0, 0.0
    open_time, trade_pnl = None, 0.0
    equity_values = []

    for ts, close, high, low, target in zip(
        px.index,
        px["close"].to_numpy(),
        px["high"].to_numpy(),
        px["low"].to_numpy(),
        target_lots.to_numpy(),
    ):
        # 1) Mark the position held through this bar, checking its worst price
        if pos != 0.0:
            worst = low if pos > 0 else high
            state.update(
                0.0,
                pos * (close - entry) * vpu,
                0.0,
                worst_unrealized_pnl=pos * (worst - entry) * vpu,
                timestamp=ts,
                traded=False,
            )
        else:
            state.update(0.0, 0.0, 0.0, timestamp=ts, traded=False)
        if state.is_breached:
            break

        # 2) Trade at this bar's close to the target from this bar's signal
        delta = target - pos
        if abs(delta) > 1e-12:
            new_pos = pos + delta
            realized, closed_lots = 0.0, 0.0
            if pos != 0.0 and np.sign(delta) != np.sign(pos):
                closed_lots = min(abs(delta), abs(pos))
                realized = np.sign(pos) * closed_lots * (close - entry) * vpu
            fees = abs(delta) * cost_per_lot

            if pos == 0.0:
                open_time, trade_pnl = ts, -fees
                entry = close
            elif new_pos == 0.0 or np.sign(new_pos) != np.sign(pos):
                # Closed (and possibly reversed): log the closed trade
                close_fees = closed_lots * cost_per_lot
                state.record_trade(open_time, ts, trade_pnl + realized - close_fees)
                if new_pos != 0.0:
                    open_time, trade_pnl = ts, -(fees - close_fees)
                    entry = close
                else:
                    open_time, trade_pnl, entry = None, 0.0, 0.0
            elif abs(new_pos) > abs(pos):
                entry = (pos * entry + delta * close) / new_pos  # added to the position
                trade_pnl -= fees
            else:
                trade_pnl += realized - fees  # partial close

            state.update(realized, new_pos * (close - entry) * vpu, fees, timestamp=ts, traded=True)
            pos = new_pos

        equity_values.append(state.current_equity)
        if state.is_breached:
            break

    if state.is_breached:
        # The account is closed at the breach; freeze equity there
        if len(equity_values) < len(px):
            equity_values.append(state.breach_level)
        equity_values += [state.breach_level] * (len(px) - len(equity_values))
    else:
        state.reset_daily()  # count the final day toward trading days

    equity_curve = pd.Series(equity_values, index=px.index[: len(equity_values)])
    returns = equity_curve.pct_change().fillna(0.0)

    # Path metrics
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 1e-9 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    target = cfg.rule_set.phase_targets.get(cfg.phase)
    hit_target = target is not None and not state.is_breached and state.phase_profit_pct >= target

    return PathResult(
        path_id=path_id,
        equity_curve=equity_curve,
        returns=returns,
        sharpe=float(sharpe),
        max_drawdown=max_drawdown,
        daily_breaches=int(state.breach == "daily"),
        overall_breach=state.breach == "overall",
        trading_days=state.trading_days_completed,
        hit_target=bool(hit_target),
        final_equity=float(equity_curve.iloc[-1]),
        breach=state.breach,
        breach_time=state.breach_time,
        final_balance=float(state.balance),
        quick_strike_ok=state.quick_strike_ok,
        n_trades=len(state.trades),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────


class CPCVDynamicBacktest:
    """
    CPCV dynamic backtest with fresh account state per path.

    Four-phase architecture:

    1. **Train** across all C(N, k) combinatorial splits in parallel.
    2. **Recombine** per-split OOF predictions into φ backtest paths via
       ``CombinatorialPurgedCV.recombine_test_predictions()``.
    3. **Calibrate** (optional) — fit an isotonic or Platt calibration map
       on the pooled OOF predictions and apply it to each path.
    4. **Simulate** each assembled path bar-by-bar with a fresh account state.

    .. warning::
       The ``estimator`` must be a plain classifier (e.g. ``BaggingClassifier``,
       ``RandomForestClassifier``).  Do **not** pass a ``CalibratorCV`` — it
       carries an internal ``PurgedKFold`` whose ``t1`` is bound to the full
       dataset and will raise ``ValueError: Lengths must match`` when fitted
       on a CPCV training subset.  Use ``calibration_method`` instead.

    Parameters
    ----------
    cv_gen : CombinatorialPurgedCV
        Configured combinatorial CV generator.
    estimator : sklearn-compatible estimator
        Base classifier.  Cloned and retrained for every split.
    sizer : PropFirmAwareSizer
        Fully configured sizer.
    cfg : BacktestConfig
    close_prices : pd.Series or pd.DataFrame
        Bar-level close prices aligned with X, or OHLC bars with open/high/low/
        close columns. High and low are used for intrabar breach checks.
    primary_sides : pd.Series
        Side predictions (+1 long, -1 short) aligned with X.
    calibration_method : str or None, default=None
        If set, calibrate OOF probabilities before simulation.
        ``'isotonic'`` — non-parametric; requires ~200+ OOF observations.
        ``'platt'`` — logistic (Platt scaling); stable on small sets.
        ``None`` — skip calibration; use raw probabilities.

    Example
    -------
    >>> from afml.cross_validation.combinatorial import (
    ...     CombinatorialPurgedCV, optimal_folds_number
    ... )
    >>> from afml.bet_sizing.prop_firm_sizer import Phase, make_stellar_2step_sizer
    >>> from afml.backtester.cpcv_dynamic_backtest import (
    ...     CPCVDynamicBacktest, BacktestConfig
    ... )
    >>>
    >>> N, k = optimal_folds_number(
    ...     n_observations=len(X),
    ...     target_train_size=int(len(X) * 0.60),
    ...     target_n_test_paths=5,
    ... )
    >>> cv_gen = CombinatorialPurgedCV(
    ...     n_folds=N, n_test_folds=k, t1=t1, pct_embargo=0.01
    ... )
    >>> cfg = BacktestConfig(
    ...     initial_balance=100_000.0,
    ...     phase=Phase.CHALLENGE_PHASE_1,
    ...     pip_value=10.0,
    ...     lot_size=0.01,
    ...     n_jobs=-1,
    ... )
    >>> sizer = make_stellar_2step_sizer(
    ...     stop_loss_pct=0.01,
    ...     avg_win_loss_ratio=1.2,
    ...     kelly_fraction=0.5,
    ... )
    >>> backtest = CPCVDynamicBacktest(
    ...     cv_gen=cv_gen,
    ...     estimator=clf,         # plain classifier, NOT CalibratorCV
    ...     sizer=sizer,
    ...     cfg=cfg,
    ...     close_prices=close_prices,
    ...     primary_sides=sides,
    ...     calibration_method='isotonic',
    ... )
    >>> backtest.run(X=X, y=y, events=events, sample_weight=sw)
    >>> backtest.distribution_report()
    >>> backtest.pbo_audit(n_folds=8)
    >>> backtest.plot_equity_distribution()
    """

    def __init__(
        self,
        cv_gen: CombinatorialPurgedCV,
        estimator,
        sizer: PropFirmAwareSizer,
        cfg: BacktestConfig,
        close_prices: pd.Series | pd.DataFrame,
        primary_sides: pd.Series,
        calibration_method: str | None = None,
    ) -> None:
        # Guard against passing CalibratorCV
        est_class = type(estimator).__name__
        if est_class == "CalibratorCV":
            raise TypeError(
                "Do not pass a CalibratorCV as the estimator — its internal "
                "PurgedKFold carries a full-length t1 that breaks on CPCV "
                "training subsets.  Pass the base classifier and set "
                "calibration_method='isotonic' or 'platt' instead."
            )

        if calibration_method is not None and calibration_method not in (
            "isotonic",
            "platt",
        ):
            raise ValueError(
                f"calibration_method must be 'isotonic', 'platt', or None; "
                f"got {calibration_method!r}."
            )

        self.cv_gen = cv_gen
        self.estimator = estimator
        self.sizer = sizer
        self.cfg = cfg
        self.close_prices = close_prices
        self.primary_sides = primary_sides
        self.calibration_method = calibration_method
        self.results_: list[PathResult] = []
        self.calibrator_ = None

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        events: pd.DataFrame,
        price_returns: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        Train, recombine, calibrate (optional), and simulate all φ[N, k] paths.

        Parameters
        ----------
        X, y, events : pd.DataFrame, pd.Series, pd.DataFrame
            Full feature matrix, labels, and events.
        price_returns : pd.Series, optional
            Deprecated and ignored: P&L is computed from close_prices (price
            levels are needed to convert moves into pips).
        sample_weight : pd.Series, optional
            Per-observation sample weights for training.
        """
        if price_returns is not None:
            warnings.warn(
                "price_returns is ignored; P&L is computed from close_prices",
                DeprecationWarning,
                stacklevel=2,
            )

        # ── Phase 1: Train across all C(N, k) splits ────────────────────
        splits = [
            (train, np.concatenate(test_list)) for train, test_list in self.cv_gen.split(X, y)
        ]

        split_predictions = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(_fit_predict_split)(self.estimator, X, y, train, test, sample_weight)
            for train, test in splits
        )

        # ── Phase 2: Recombine into φ backtest paths ────────────────────
        path_predictions = self.cv_gen.recombine_test_predictions(split_predictions)

        # ── Phase 2.5: Calibrate (optional) ─────────────────────────────
        if self.calibration_method is not None:
            self.calibrator_ = _fit_calibrator(path_predictions, y, self.calibration_method)
            path_predictions = [
                _apply_calibrator(preds, self.calibrator_, self.calibration_method)
                for preds in path_predictions
            ]

        # ── Phase 3: Simulate each assembled path ───────────────────────
        self.results_ = [
            _simulate_path(
                path_id=pid,
                oof_probs=pd.Series(preds, index=X.index, name=f"path_{pid}"),
                events=events,
                prices=self.close_prices,
                sizer=self.sizer,
                cfg=self.cfg,
                primary_sides=self.primary_sides,
            )
            for pid, preds in enumerate(path_predictions)
        ]

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def distribution_report(self) -> pd.DataFrame:
        """
        Print and return a summary of the path distribution.

        Returns
        -------
        pd.DataFrame
            One row per path with sharpe, max_drawdown, daily_breaches,
            overall_breach, trading_days, hit_target, final_equity, breach,
            breach_time, quick_strike_ok, n_trades.
        """
        if not self.results_:
            raise RuntimeError("Call run() before distribution_report().")

        rows = [
            {
                "path_id": r.path_id,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "daily_breaches": r.daily_breaches,
                "overall_breach": r.overall_breach,
                "trading_days": r.trading_days,
                "hit_target": r.hit_target,
                "final_equity": r.final_equity,
                "breach": r.breach,
                "breach_time": r.breach_time,
                "quick_strike_ok": r.quick_strike_ok,
                "n_trades": r.n_trades,
            }
            for r in self.results_
        ]
        df = pd.DataFrame(rows).set_index("path_id")

        print("\n── CPCV Path Distribution ────────────────────────────")
        print(f"  Paths simulated    : {len(df)}")
        print(f"  Median Sharpe      : {df['sharpe'].median():.3f}")
        print(f"  Sharpe std         : {df['sharpe'].std():.3f}")
        print(f"  Paths > 0 Sharpe   : {(df['sharpe'] > 0).sum()} / {len(df)}")
        print(f"  Median max DD      : {df['max_drawdown'].median():.2%}")
        print(f"  Daily limit breach : {df['daily_breaches'].sum()} path(s)")
        print(f"  Overall breach     : {df['overall_breach'].sum()} path(s)")
        print(f"  Hit phase target   : {df['hit_target'].sum()} / {len(df)}")
        print(f"  Median final equity: ${df['final_equity'].median():,.0f}")
        print("─────────────────────────────────────────────────────\n")

        return df

    def pbo_audit(self, n_folds: int = 8) -> float:
        """
        Compute the Probability of Backtest Overfitting (CSCV) from path returns.

        Parameters
        ----------
        n_folds : int
            Number of subsets S for CSCV (typically 8-16).

        Returns
        -------
        float
            PBO in [0, 1].
        """
        try:
            from ..cross_validation.pbo import compute_pbo
        except ImportError:
            raise ImportError(
                "compute_pbo not found. Ensure the Unified Validation Pipeline "
                "module is installed in afml.cross_validation.pbo."
            )

        if not self.results_:
            raise RuntimeError("Call run() before pbo_audit().")

        returns_matrix = pd.concat(
            [r.returns.rename(r.path_id) for r in self.results_],
            axis=1,
        ).fillna(0.0)

        t1_neutral = pd.Series(returns_matrix.index, index=returns_matrix.index)

        result = compute_pbo(returns_matrix, t1=t1_neutral, n_folds=n_folds)
        pbo = result["pbo"]
        print(f"\n  PBO (CSCV, S={n_folds}): {pbo:.4f}")
        return pbo

    def plot_equity_distribution(
        self,
        figsize: tuple[float, float] = (7.5, 4.5),
        alpha: float = 0.25,
    ) -> None:
        """
        Plot all φ equity curves on a single normalized axis.

        Parameters
        ----------
        figsize : tuple
        alpha : float
            Opacity of individual path curves.
        """
        import matplotlib.pyplot as plt

        if not self.results_:
            raise RuntimeError("Call run() before plot_equity_distribution().")

        fig, ax = plt.subplots(figsize=figsize)

        for r in self.results_:
            normalized = r.equity_curve / r.equity_curve.iloc[0]
            ax.plot(
                normalized.index,
                normalized.values,
                alpha=alpha,
                linewidth=0.8,
                color="#58a6ff",
            )

        # Median path
        all_curves = pd.concat(
            [r.equity_curve.rename(r.path_id) / r.equity_curve.iloc[0] for r in self.results_],
            axis=1,
        ).ffill()

        median_curve = all_curves.median(axis=1)
        ax.plot(
            median_curve.index,
            median_curve.values,
            color="#e6edf3",
            linewidth=2.0,
            label="Median path",
        )

        ax.axhline(
            1.0,
            color="#8b949e",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.set_title(
            f"CPCV Equity Distribution — {len(self.results_)} paths",
            fontsize=10,
        )
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Normalized equity", fontsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
