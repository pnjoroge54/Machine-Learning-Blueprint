"""
CPCV dynamic backtest orchestrator with fresh account state per path.

Architecture (three-phase)
──────────────────────────
  Phase 1 — Train:  Fit the estimator on every C(N, k) combinatorial split
                     produced by CombinatorialPurgedCV.split().  Each split
                     yields purged+embargoed train indices and a list of test
                     fold arrays.  Training is parallelised across splits.

  Phase 2 — Assemble:  Recombine per-split OOF predictions into the φ[N, k]
                        backtest paths via recombine_test_predictions().

  Phase 3 — Simulate:  Run a bar-by-bar P&L simulation on each assembled
                        path with a fresh PropFirmAccountState.

This mirrors the CPCVAnalyzer pattern: training is per-split, path assembly
uses CombinatorialPurgedCV's own recombination logic, and simulation
consumes the fully-assembled path predictions.

Dependencies
────────────
    afml.cross_validation.combinatorial : CombinatorialPurgedCV,
                                          optimal_folds_number
    afml.bet_sizing.prop_firm_sizer     : PropFirmAwareSizer,
                                          PropFirmAccountState, Phase
    joblib, numpy, pandas, sklearn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from ..cross_validation.combinatorial import CombinatorialPurgedCV
from ..bet_sizing.prop_firm_sizer import PropFirmAccountState, PropFirmAwareSizer, Phase


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
        Prop firm phase.  Determines profit target and rule set.
    pip_value : float
        Dollar value of one pip per standard lot.
    lot_size : float
        Lot size used to translate bet size fractions into notional positions.
    commission_per_lot : float
        Commission per lot (both sides).  Counts against the daily limit.
    slippage_pips : float
        Assumed slippage per trade in pips.  Applied to each entry and exit.
    n_jobs : int
        Number of parallel workers.  -1 = all physical cores.
    random_state : int
        Seed for any stochastic components.
    """

    initial_balance: float = 100_000.0
    phase: Phase = Phase.CHALLENGE_PHASE_1
    pip_value: float = 10.0
    lot_size: float = 0.01
    commission_per_lot: float = 5.0
    slippage_pips: float = 0.5
    n_jobs: int = -1
    random_state: int = 42


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
        Will be cloned before fitting.
    X, y : pd.DataFrame, pd.Series
        Full feature matrix and labels.
    train_idx, test_idx : np.ndarray
        Positional indices for this split's training and (concatenated) test set.
    sample_weight : pd.Series, optional
        Per-observation sample weights aligned with X.

    Returns
    -------
    np.ndarray
        Predicted probabilities (class 1) for the test observations, ordered
        to match test_idx.
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


# ── Phase 3 helper: simulate one assembled path ──────────────────────────────

def _simulate_path(
    path_id: int,
    oof_probs: pd.Series,
    events: pd.DataFrame,
    price_returns: pd.Series,
    sizer: PropFirmAwareSizer,
    cfg: BacktestConfig,
    primary_sides: pd.Series,
) -> PathResult:
    """
    Simulate one fully-assembled backtest path bar-by-bar.

    Parameters
    ----------
    path_id : int
        Index of this combinatorial path.
    oof_probs : pd.Series
        Out-of-fold predicted probabilities for every observation on this path,
        produced by recombine_test_predictions().
    events : pd.DataFrame
        Events table aligned with X.
    price_returns : pd.Series
        Bar-level close-to-close returns aligned with X.
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
    ret_valid = price_returns.loc[oof_probs.index]

    state = PropFirmAccountState(
        initial_balance=cfg.initial_balance,
        phase=cfg.phase,
    )

    # Size positions from OOF predictions
    sizing_result = sizer.size(
        events=events_valid,
        prob=oof_probs,
        pred=sides_valid,
        state=state,
        average_active=True,
    )
    position_sizes = sizing_result["final_size"]

    # Bar-by-bar P&L simulation
    equity_values = []
    daily_breaches = 0
    current_day = None

    for ts in position_sizes.index:
        pos = position_sizes.loc[ts]
        ret = ret_valid.loc[ts]

        # Gross bar P&L
        gross_pnl = pos * ret * cfg.lot_size * cfg.pip_value

        # Transaction costs on position change
        if ts == position_sizes.index[0]:
            prev_pos = 0.0
        else:
            prev_pos = position_sizes.shift(1).loc[ts]

        position_chg = abs(pos - prev_pos)
        commission = position_chg * cfg.commission_per_lot
        slippage = position_chg * cfg.slippage_pips * cfg.pip_value
        total_cost = commission + slippage
        realized_delta = gross_pnl - total_cost

        # Daily reset
        bar_day = pd.Timestamp(ts).date()
        if bar_day != current_day:
            state.reset_daily()
            current_day = bar_day

        state.update(
            realized_pnl_delta=realized_delta,
            unrealized_pnl=0.0,
            fees_and_swaps=total_cost,
        )

        if state.daily_remaining <= 0:
            daily_breaches += 1

        equity_values.append(state.current_equity)

    equity_curve = pd.Series(equity_values, index=position_sizes.index)
    returns = equity_curve.pct_change().fillna(0.0)

    # Path metrics
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 1e-9
        else 0.0
    )

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    overall_breach = bool((equity_curve < state.overall_floor).any())

    hit_target = (
        (state.phase_profit_pct >= 0.08)
        if cfg.phase == Phase.CHALLENGE_PHASE_1
        else (state.phase_profit_pct >= 0.05)
    )

    return PathResult(
        path_id=path_id,
        equity_curve=equity_curve,
        returns=returns,
        sharpe=float(sharpe),
        max_drawdown=max_drawdown,
        daily_breaches=daily_breaches,
        overall_breach=overall_breach,
        trading_days=state.trading_days_completed,
        hit_target=hit_target,
        final_equity=float(equity_curve.iloc[-1]),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CPCVDynamicBacktest:
    """
    CPCV dynamic backtest with fresh account state per path.

    Follows the same two-phase pattern as CPCVAnalyzer:

    1. **Train** across all C(N, k) combinatorial splits in parallel.
    2. **Recombine** per-split OOF predictions into φ backtest paths via
       ``CombinatorialPurgedCV.recombine_test_predictions()``.
    3. **Simulate** each assembled path bar-by-bar with a fresh account state.

    Parameters
    ----------
    cv_gen : CombinatorialPurgedCV
        Configured combinatorial CV generator.
    estimator : sklearn-compatible estimator
        Base classifier.  Cloned and retrained for every split.
    sizer : PropFirmAwareSizer
        Fully configured sizer.
    cfg : BacktestConfig
    close_prices : pd.Series
        Bar-level close prices aligned with X.
    primary_sides : pd.Series
        Side predictions (+1 long, -1 short) aligned with X.

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
    ...     estimator=clf,
    ...     sizer=sizer,
    ...     cfg=cfg,
    ...     close_prices=close_prices,
    ...     primary_sides=sides,
    ... )
    >>> backtest.run(X=X, y=y, events=events)
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
        close_prices: pd.Series,
        primary_sides: pd.Series,
    ) -> None:
        self.cv_gen = cv_gen
        self.estimator = estimator
        self.sizer = sizer
        self.cfg = cfg
        self.close_prices = close_prices
        self.primary_sides = primary_sides
        self.results_: list[PathResult] = []

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        events: pd.DataFrame,
        price_returns: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        Train, recombine, and simulate all φ[N, k] paths.

        Parameters
        ----------
        X, y, events : pd.DataFrame, pd.Series, pd.DataFrame
            Full feature matrix, labels, and events.
        price_returns : pd.Series, optional
            Bar-level returns.  Computed from close_prices if not provided.
        sample_weight : pd.Series, optional
            Per-observation sample weights for training.
        """
        if price_returns is None:
            price_returns = self.close_prices.pct_change().fillna(0.0)

        # ── Phase 1: Train across all C(N, k) splits ────────────────────
        # Exhaust the generator eagerly so cv_gen stores index_train_test_
        # and _fold_index_num, which recombine_test_predictions requires.
        splits = [
            (train, np.concatenate(test_list))
            for train, test_list in self.cv_gen.split(X, y)
        ]

        split_predictions = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(_fit_predict_split)(
                self.estimator, X, y, train, test, sample_weight
            )
            for train, test in splits
        )

        # ── Phase 2: Recombine into φ backtest paths ────────────────────
        path_predictions = self.cv_gen.recombine_test_predictions(
            split_predictions
        )

        # ── Phase 3: Simulate each assembled path ───────────────────────
        self.results_ = [
            _simulate_path(
                path_id=pid,
                oof_probs=pd.Series(preds, index=X.index, name=f"path_{pid}"),
                events=events,
                price_returns=price_returns,
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
            overall_breach, trading_days, hit_target, final_equity.
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
        print(f"  Daily limit breach : {df['daily_breaches'].sum()} bar(s) total")
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

        t1_neutral = pd.Series(
            returns_matrix.index, index=returns_matrix.index
        )

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
                normalized.index, normalized.values,
                alpha=alpha, linewidth=0.8, color="#58a6ff",
            )

        # Median path
        all_curves = pd.concat(
            [
                r.equity_curve.rename(r.path_id) / r.equity_curve.iloc[0]
                for r in self.results_
            ],
            axis=1,
        ).ffill()

        median_curve = all_curves.median(axis=1)
        ax.plot(
            median_curve.index, median_curve.values,
            color="#e6edf3", linewidth=2.0, label="Median path",
        )

        ax.axhline(
            1.0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6,
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
