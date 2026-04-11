"""
CPCV dynamic backtest orchestrator with fresh account state per path.

Each of the φ[N, k] combinatorial paths receives:
  - A fresh PropFirmAccountState (same initial balance and limits)
  - A distinct sequence of OOF predictions (from a different training fold combination)
  - A bar-by-bar simulation loop that updates account state at each step

Paths are simulated in parallel via joblib.Parallel.

The output is a distribution of equity curves and per-path metrics from
which the PBO audit (CSCV) can be computed via the Unified Validation
Pipeline's compute_pbo function.

Dependencies
────────────
    afml.cross_validation.cross_validation : PurgedKFold
    afml.cross_validation.combinatorial    : CombinatorialPurgedCV,
                                             optimal_folds_number
    prop_firm_sizer                        : PropFirmAwareSizer,
                                             PropFirmAccountState, Phase
    joblib, numpy, pandas, sklearn
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        Prop firm phase. Determines profit target and rule set.
    pip_value : float
        Dollar value of one pip per standard lot.
    lot_size : float
        Lot size used to translate bet size fractions into notional positions.
    commission_per_lot : float
        Commission per lot (both sides). Counts against the daily limit.
    slippage_pips : float
        Assumed slippage per trade in pips. Applied to each entry and exit.
    n_jobs : int
        Number of parallel workers. -1 = all physical cores.
    random_state : int
        Seed for any stochastic components.
    """

    initial_balance:    float = 100_000.0
    phase:              Phase = Phase.CHALLENGE_PHASE_1
    pip_value:          float = 10.0
    lot_size:           float = 0.01
    commission_per_lot: float = 5.0
    slippage_pips:      float = 0.5
    n_jobs:             int   = -1
    random_state:       int   = 42


# ── Per-path result container ─────────────────────────────────────────────────

@dataclass
class PathResult:
    """Result of a single CPCV path simulation."""

    path_id:        int
    equity_curve:   pd.Series          # bar-by-bar equity indexed by timestamp
    returns:        pd.Series          # bar-by-bar returns
    sharpe:         float
    max_drawdown:   float
    daily_breaches: int                # number of bars where daily limit was breached
    overall_breach: bool               # whether overall floor was ever breached
    trading_days:   int
    hit_target:     bool               # whether phase profit target was reached
    final_equity:   float


# ── Path simulation (runs in a single worker) ─────────────────────────────────

def simulate_path(
    path_id:       int,
    train_indices: list[np.ndarray],
    test_indices:  list[np.ndarray],
    X:             pd.DataFrame,
    y:             pd.Series,
    events:        pd.DataFrame,
    price_returns: pd.Series,
    estimator,
    sizer:         PropFirmAwareSizer,
    cfg:           BacktestConfig,
    primary_sides: pd.Series,
) -> PathResult:
    """
    Simulate one CPCV path bar-by-bar with a fresh account state.

    Parameters
    ----------
    path_id : int
        Index of this combinatorial path.
    train_indices, test_indices : list of np.ndarray
        Index arrays for each fold's training and test sets on this path.
        Produced by CombinatorialPurgedCV.
    X, y, events : pd.DataFrame, pd.Series, pd.DataFrame
        Full feature matrix, labels, and events aligned by DatetimeIndex.
    price_returns : pd.Series
        Bar-level returns (close-to-close), aligned with X.
    estimator : sklearn-compatible estimator
        Base classifier. Cloned and retrained on each fold's training set.
    sizer : PropFirmAwareSizer
        Fully configured sizer. The same sizer instance is used across all
        paths; account state is provided fresh per path.
    cfg : BacktestConfig
    primary_sides : pd.Series
        Side predictions (+1/-1), aligned with X.

    Returns
    -------
    PathResult
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
        
    state = PropFirmAccountState(
        initial_balance=cfg.initial_balance,
        phase=cfg.phase,
    )

    # -- Collect OOF predictions across folds for this path ------------------
    oof_probs = pd.Series(np.nan, index=X.index)

    for tr_idx, te_idx in zip(train_indices, test_indices):
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        probs = model.predict_proba(X.iloc[te_idx])[:, 1]
        oof_probs.iloc[te_idx] = probs

    # Drop bars with no OOF prediction (purged/embargoed bars between folds)
    valid_mask  = oof_probs.notna().index
    oof_probs = oof_probs.loc[valid_mask]
    sides_valid = primary_sides.loc[valid_mask]
    events_valid = events.loc[valid_mask]
    ret_valid = price_returns.loc[valid_mask]

    # -- Size positions from OOF predictions ----------------------------------
    sizing_result = sizer.size(
        events=events_valid,
        prob=oof_probs,
        pred=sides_valid,
        state=state,
        average_active=True,
    )

    position_sizes = sizing_result["final_size"]

    # -- Bar-by-bar P&L simulation --------------------------------------------
    equity_values = []
    daily_breaches = 0
    current_day    = None

    for ts in position_sizes.index:
        pos  = position_sizes.loc[ts]
        ret  = ret_valid.loc[ts]

        # Gross bar P&L from position × price return
        gross_pnl = pos * ret * cfg.lot_size * cfg.pip_value

        # Transaction costs: commission + slippage (applied on position change)
        prev_pos     = position_sizes.shift(1).loc[ts] if ts != position_sizes.index[0] else 0.0
        position_chg = abs(pos - prev_pos)
        commission   = position_chg * cfg.commission_per_lot
        slippage     = position_chg * cfg.slippage_pips * cfg.pip_value
        total_cost   = commission + slippage

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
    returns      = equity_curve.pct_change().fillna(0.0)

    # -- Path metrics ---------------------------------------------------------
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 1e-9 else 0.0
    )

    rolling_max  = equity_curve.cummax()
    drawdown     = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    overall_breach = bool((equity_curve < state.overall_floor).any())

    target_pct = (
        (state.phase_profit_pct >= state.initial_balance * 0.08 / state.initial_balance)
        if cfg.phase == Phase.CHALLENGE_PHASE_1
        else (state.phase_profit_pct >= state.initial_balance * 0.05 / state.initial_balance)
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
        hit_target=target_pct,
        final_equity=float(equity_curve.iloc[-1]),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CPCVDynamicBacktest:
    """
    CPCV dynamic backtest with fresh account state per path.

    Generates all φ[N, k] combinatorial paths from CombinatorialPurgedCV,
    simulates each in parallel, and aggregates the path distribution.

    Parameters
    ----------
    cv_gen : CombinatorialPurgedCV
        Configured combinatorial CV generator. N and k should be chosen
        with optimal_folds_number to produce the desired number of paths.
    estimator : sklearn-compatible estimator
        Base classifier. Cloned and retrained for every fold on every path.
    sizer : PropFirmAwareSizer
        Fully configured sizer.
    cfg : BacktestConfig
    close_prices : pd.Series
        Bar-level close prices, aligned with X. Used to compute price_returns
        if not provided to run().
    primary_sides : pd.Series
        Side predictions (+1 long, -1 short), aligned with X.

    Example
    -------
    >>> from afml.cross_validation.combinatorial import (
    ...     CombinatorialPurgedCV, optimal_folds_number
    ... )
    >>> from prop_firm_sizer import Phase, make_stellar_2step_sizer
    >>> from cpcv_dynamic_backtest import CPCVDynamicBacktest, BacktestConfig
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
        cv_gen:        CombinatorialPurgedCV,
        estimator,     
        sizer:         PropFirmAwareSizer,
        cfg:           BacktestConfig,
        close_prices:  pd.Series,
        primary_sides: pd.Series,
    ) -> None:
        self.cv_gen        = cv_gen
        self.estimator     = estimator
        self.sizer         = sizer
        self.cfg           = cfg
        self.close_prices  = close_prices
        self.primary_sides = primary_sides
        self.results_: list[PathResult] = []

    def run(
        self,
        X:             pd.DataFrame,
        y:             pd.Series,
        events:        pd.DataFrame,
        price_returns: pd.Series | None = None,
    ) -> None:
        """
        Simulate all φ[N, k] paths and store results.

        Parameters
        ----------
        X, y, events : pd.DataFrame, pd.Series, pd.DataFrame
            Full feature matrix, labels, and events.
        price_returns : pd.Series, optional
            Bar-level returns. Computed from close_prices if not provided.
        """
        if price_returns is None:
            price_returns = self.close_prices.pct_change().fillna(0.0)

        # Collect all (train_indices, test_indices) pairs across paths.
        # CombinatorialPurgedCV.split() yields one split per fold;
        # the φ paths are reconstructed by recombining fold OOF predictions.
        # Each element of paths is a list of (train_idx, test_idx) tuples —
        # one per fold within that path.
        paths = list(self.cv_gen.split(X, y))

        self.results_ = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(simulate_path)(
                path_id=pid,
                train_indices=[s[0] for s in path_splits],
                test_indices=[s[1] for s in path_splits],
                X=X,
                y=y,
                events=events,
                price_returns=price_returns,
                estimator=self.estimator,
                sizer=self.sizer,
                cfg=self.cfg,
                primary_sides=self.primary_sides,
            )
            for pid, path_splits in enumerate(paths)
        )

    def distribution_report(self) -> pd.DataFrame:
        """
        Print and return a summary of the path distribution.

        Returns
        -------
        pd.DataFrame
            One row per path with: path_id, sharpe, max_drawdown,
            daily_breaches, overall_breach, trading_days, hit_target,
            final_equity.
        """
        if not self.results_:
            raise RuntimeError("Call run() before distribution_report().")

        rows = [
            {
                "path_id":        r.path_id,
                "sharpe":         r.sharpe,
                "max_drawdown":   r.max_drawdown,
                "daily_breaches": r.daily_breaches,
                "overall_breach": r.overall_breach,
                "trading_days":   r.trading_days,
                "hit_target":     r.hit_target,
                "final_equity":   r.final_equity,
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
  
      Delegates to compute_pbo from the Unified Validation Pipeline module.
  
      Parameters
      ----------
      n_folds : int
          Number of subsets S for CSCV (typically 8–16).
  
      Returns
      -------
      float
          PBO in [0, 1]. Values near 0 indicate low overfitting risk;
          values near 0.5 indicate the result is consistent with chance.
      """
      try:
          from ..cross_validation.pbo import compute_pbo
      except ImportError:
          raise ImportError(
              "compute_pbo not found. Ensure the Unified Validation Pipeline "
              "module is installed in afml.cross_validation.combinatorial."
          )
  
      if not self.results_:
          raise RuntimeError("Call run() before pbo_audit().")
  
      # Build returns matrix: columns = paths, rows = bars
      returns_matrix = pd.concat(
          [r.returns.rename(r.path_id) for r in self.results_],
          axis=1,
      ).fillna(0.0)
  
      # Create a neutral t1 series aligned with the returns_matrix index.
      # This satisfies the interface requirement; pct_embargo is set to 0.0 inside compute_pbo.
      t1_neutral = pd.Series(returns_matrix.index, index=returns_matrix.index)
  
      result = compute_pbo(returns_matrix, t1=t1_neutral, n_folds=n_folds)
      pbo = result["pbo"]
      print(f"\n  PBO (CSCV, S={n_folds}): {pbo:.4f}")
      return pbo
      
    def plot_equity_distribution(
        self,
        figsize: tuple[float, float] = (7.5, 4.5),
        alpha:   float = 0.25,
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
            ax.plot(normalized.index, normalized.values, alpha=alpha,
                    linewidth=0.8, color="#58a6ff")

        # Median path
        all_curves = pd.concat(
            [r.equity_curve.rename(r.path_id) / r.equity_curve.iloc[0]
             for r in self.results_],
            axis=1,
        ).fillna(method="ffill")

        median_curve = all_curves.median(axis=1)
        ax.plot(median_curve.index, median_curve.values,
                color="#e6edf3", linewidth=2.0, label="Median path")

        ax.axhline(1.0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"CPCV Equity Distribution — {len(self.results_)} paths",
                     fontsize=10)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Normalized equity", fontsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
