"""
Optimal Triple Barrier Method Implementation
Based on Chapter 13 of "Advances in Financial Machine Learning" by Marcos López de Prado

This module provides:
1. Parameter estimation for Ornstein-Uhlenbeck process from strategy signals
2. Monte Carlo simulation for optimal barrier determination
3. Integration with BaseStrategy signal generators
"""

import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm import tqdm

from ..cache import cacheable
from ..production.model_development import load_and_prepare_training_data
from ..strategies.trading_strategies import BaseStrategy


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters"""

    phi: float  # Mean reversion speed
    forecast: float  # Expected value E0[Pi,Ti]
    sigma: float  # Volatility
    half_life: float  # Half-life of mean reversion


@dataclass
class OptimalBarriers:
    """Optimal barrier configuration"""

    profit_taking: float  # Optimal profit-taking barrier (π̄)
    stop_loss: float  # Optimal stop-loss barrier (π)
    pt_sigma_multiple: float  # Optimal profit-taking barrier reltaive to sigma (π̄)
    sl_sigma_multiple: float  # Optimal stop-loss barrier reltaive to sigma(π)
    sharpe_ratio: float  # Expected Sharpe ratio
    mean_return: float  # Expected mean return
    std_return: float  # Expected return volatility
    win_rate: float  # Percentage of profitable trades
    avg_holding_period: float  # Average bars held
    max_holding_period: int  # Maximum bars held


@dataclass
class BarrierOptimizationResult:
    """Complete optimization results"""

    optimal: OptimalBarriers
    grid_results: pd.DataFrame  # Full grid of (π, π̄) -> metrics
    ou_params: OUParameters
    strategy_name: str


# ============================================================================
# ORNSTEIN-UHLENBECK PARAMETER ESTIMATION
# ============================================================================


def estimate_ou_parameters(
    price_series: pd.Series,
    signals: pd.Series,
    forecast_method: str = "signal_based",
    forecast_window: int = 20,
    custom_forecast: Optional[float] = None,
) -> OUParameters:
    """
    Estimate O-U parameters from strategy signals

    This implements Step 1 from AFML Chapter 13.5.1:
    Pi,t = E0[Pi,Ti] + φ(Pi,t−1 − E0[Pi,Ti]) + ξt

    Args:
        price_series: Price time series
        signals: Trading signals (1, -1, 0) from BaseStrategy
        forecast_method: How to calculate E0[Pi,Ti]:
            - 'signal_based': Use signal direction * rolling vol (default)
            - 'zero': Assume zero expected return (random walk)
            - 'mean_return': Use historical mean return
            - 'moving_average': Use MA of returns over forecast_window
            - 'custom': Use custom_forecast value
        forecast_window: Window for rolling calculations (if applicable)
        custom_forecast: Custom forecast value (if forecast_method='custom')

    Returns:
        OUParameters with estimated {φ, σ, forecast}
    """
    # Calculate returns
    returns = price_series.pct_change().dropna()

    # Calculate forecast E0[Pi,Ti] - the expected terminal value
    if forecast_method == "signal_based":
        # For each signal, forecast is direction * expected move
        # This makes sense: if signal=1 (long), we expect positive return
        # The magnitude is based on recent volatility
        rolling_vol = returns.rolling(window=forecast_window).std()

        # At each signal point, forecast is signal_direction * typical_move
        forecast_series = signals * rolling_vol * np.sqrt(forecast_window)
        forecast = forecast_series[signals != 0].mean()

        if np.isnan(forecast) or forecast == 0:
            forecast = returns.mean() * forecast_window  # Fallback

    elif forecast_method == "zero":
        # Random walk assumption: no drift
        forecast = 0.0

    elif forecast_method == "mean_return":
        # Use historical mean return as forecast
        forecast = returns.mean()

    elif forecast_method == "moving_average":
        # Use moving average of returns (original implementation)
        forecast = returns.rolling(window=forecast_window).mean().mean()

    elif forecast_method == "custom":
        # User-provided forecast
        if custom_forecast is None:
            raise ValueError("custom_forecast must be provided when forecast_method='custom'")
        forecast = custom_forecast

    else:
        raise ValueError(f"Unknown forecast_method: {forecast_method}")

    # Build X and Y vectors as per equation (13.6)
    # X = Pi,t−1 − E0[Pi,Ti]
    # Y = Pi,t
    signal_indices = signals[signals != 0].index

    if len(signal_indices) < 10:
        warnings.warn("Insufficient signal points for robust parameter estimation")
        return OUParameters(phi=0.95, forecast=0.0, sigma=returns.std(), half_life=14)

    indices = returns.index.get_indexer_for(signal_indices)
    indices = indices[indices > 0]  # Ensure we have t-1 available

    returns_vals = returns.values
    X = returns_vals[indices - 1] - forecast
    Y = returns_vals[indices]

    if len(X) < 5:
        warnings.warn("Insufficient data points for parameter estimation")
        return OUParameters(phi=0.95, forecast=forecast, sigma=returns.std(), half_life=14)

    # OLS estimation: φ̂ = cov[Y,X] / cov[X,X]
    phi_hat = np.cov(Y, X)[0, 1] / np.var(X) if np.var(X) > 0 else 0.95

    # Constrain φ ∈ (0, 1) for mean reversion
    phi_hat = max(0.01, min(0.999, phi_hat))

    # Estimate residuals and sigma
    residuals = Y - forecast - phi_hat * X
    sigma_hat = np.std(residuals)

    # Calculate half-life: τ = -log(2) / log(φ)
    half_life = -np.log(2) / np.log(phi_hat) if phi_hat > 0 else np.inf

    return OUParameters(phi=phi_hat, forecast=forecast, sigma=sigma_hat, half_life=half_life)


# ============================================================================
# MONTE CARLO SIMULATION (NUMBA-ACCELERATED)
# ============================================================================


@njit(cache=True)
def simulate_ou_path(
    phi: float, forecast: float, sigma: float, seed: float, max_hp: int
) -> Tuple[np.ndarray, int]:
    """
    Simulate single O-U path

    Args:
        phi: Mean reversion parameter
        forecast: Expected value
        sigma: Volatility
        seed: Starting value (P0)
        max_hp: Maximum holding period

    Returns:
        (path, actual_length)
    """
    path = np.zeros(max_hp + 1)
    path[0] = seed

    for t in range(max_hp):
        # P(t) = (1-φ)*forecast + φ*P(t-1) + σ*ε
        path[t + 1] = (1 - phi) * forecast + phi * path[t] + sigma * np.random.normal()

    return path, max_hp


@njit(cache=True)
def check_barrier_hit(
    path: np.ndarray, profit_taking: float, stop_loss: float, max_hp: int, seed: float
) -> Tuple[float, int]:
    """
    Check which barrier is hit first

    Args:
        path: Simulated price path
        profit_taking: Upper barrier (π̄)
        stop_loss: Lower barrier (π)
        max_hp: Maximum holding period
        seed: Starting value

    Returns:
        (exit_return, holding_period)
    """
    for t in range(1, len(path)):
        cumulative_return = path[t] - seed

        # Check barriers
        if cumulative_return >= profit_taking:
            return cumulative_return, t
        elif cumulative_return <= -stop_loss:
            return cumulative_return, t
        elif t >= max_hp:
            return cumulative_return, t

    return path[-1] - seed, len(path) - 1


@njit(cache=True, parallel=True)
def simulate_barrier_outcomes(
    phi: float,
    forecast: float,
    sigma: float,
    seed: float,
    profit_taking: float,
    stop_loss: float,
    n_iter: int,
    max_hp: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulations for a specific barrier configuration

    This implements Steps 3-4 from AFML Chapter 13.5.1

    Returns:
        (returns_array, holding_periods_array)
    """
    returns = np.zeros(n_iter)
    holding_periods = np.zeros(n_iter)

    for i in prange(n_iter):
        # Generate path
        path, _ = simulate_ou_path(phi, forecast, sigma, seed, max_hp)

        # Check barrier hits
        exit_return, hp = check_barrier_hit(path, profit_taking, stop_loss, max_hp, seed)

        returns[i] = exit_return
        holding_periods[i] = hp

    return returns, holding_periods


# ============================================================================
# BARRIER OPTIMIZATION
# ============================================================================


def optimize_barriers(
    ou_params: OUParameters,
    profit_taking_range: np.ndarray = None,
    stop_loss_range: np.ndarray = None,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    seed: float = 0.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, OptimalBarriers]:
    """
    Find optimal profit-taking and stop-loss barriers

    This implements Steps 2-5a from AFML Chapter 13.5.1

    Args:
        ou_params: Estimated O-U parameters
        profit_taking_range: Array of profit-taking multiples of sigma
        stop_loss_range: Array of stop-loss multiples of sigma
        n_iter: Number of Monte Carlo iterations
        max_holding_period: Maximum bars to hold (vertical barrier)
        seed: Initial price level (typically 0)
        verbose: Print progress

    Returns:
        (grid_results_df, optimal_barriers)
    """
    # Default ranges: Step 2 from AFML
    if profit_taking_range is None:
        profit_taking_range = np.linspace(0.5, 10, 20)

    if stop_loss_range is None:
        stop_loss_range = np.linspace(0.5, 10, 20)

    # Scale by sigma
    pt_scaled = profit_taking_range * ou_params.sigma
    sl_scaled = stop_loss_range * ou_params.sigma

    results = []
    total_combinations = len(pt_scaled) * len(sl_scaled)

    if verbose:
        print(f"Testing {total_combinations} barrier combinations...")
        print(
            f"O-U Parameters: φ={ou_params.phi:.4f}, σ={ou_params.sigma:.4f}, "
            f"forecast={ou_params.forecast:.6f}, half-life={ou_params.half_life:.2f}"
        )

    for pt, sl in tqdm(product(pt_scaled, sl_scaled), total=total_combinations, desc="Progress"):
        # Run Monte Carlo simulations
        returns, holding_periods = simulate_barrier_outcomes(
            ou_params.phi,
            ou_params.forecast,
            ou_params.sigma,
            seed,
            pt,
            sl,
            n_iter,
            max_holding_period,
        )

        # Calculate metrics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
        win_rate = np.sum(returns > 0) / len(returns)
        avg_hp = np.mean(holding_periods)

        results.append(
            {
                "profit_taking": pt,
                "stop_loss": sl,
                "pt_sigma_multiple": pt / ou_params.sigma,
                "sl_sigma_multiple": sl / ou_params.sigma,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "avg_holding_period": avg_hp,
                "total_return": mean_ret * n_iter,
            }
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find optimal (Step 5a)
    optimal_idx = results_df["sharpe_ratio"].idxmax()
    optimal_row = results_df.loc[optimal_idx]

    optimal = OptimalBarriers(
        profit_taking=optimal_row["profit_taking"],
        stop_loss=optimal_row["stop_loss"],
        pt_sigma_multiple=optimal_row["pt_sigma_multiple"],
        sl_sigma_multiple=optimal_row["sl_sigma_multiple"],
        sharpe_ratio=optimal_row["sharpe_ratio"],
        mean_return=optimal_row["mean_return"],
        std_return=optimal_row["std_return"],
        win_rate=optimal_row["win_rate"],
        avg_holding_period=optimal_row["avg_holding_period"],
        max_holding_period=max_holding_period,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMAL BARRIERS FOUND")
        print("=" * 70)
        print(f"Profit Taking: {optimal.profit_taking:.4f} ({optimal.pt_sigma_multiple:.2f}σ)")
        print(f"Stop Loss:     {optimal.stop_loss:.4f} ({optimal.sl_sigma_multiple:.2f}σ)")
        print(f"Max Bars:      {max_holding_period:,}")

        print(f"Sharpe Ratio:  {optimal.sharpe_ratio:.4f}")
        print(f"Win Rate:      {optimal.win_rate:.2%}")
        print(f"Avg Hold:      {optimal.avg_holding_period:.1f} bars")
        print("=" * 70)

    return results_df, optimal


def find_optimal_stop_loss_given_target(
    ou_params: OUParameters,
    profit_target: float,
    stop_loss_range: np.ndarray = None,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    seed: float = 0.0,
) -> Tuple[float, float]:
    """
    Find optimal stop-loss given a fixed profit target

    This implements Step 5b from AFML Chapter 13.5.1

    Args:
        ou_params: Estimated O-U parameters
        profit_target: Fixed profit-taking level
        stop_loss_range: Range of stop-loss values to test
        n_iter: Number of Monte Carlo iterations
        max_holding_period: Maximum bars to hold
        seed: Initial price level

    Returns:
        (optimal_stop_loss, expected_sharpe)
    """
    if stop_loss_range is None:
        stop_loss_range = np.linspace(0.5, 10, 20) * ou_params.sigma

    best_sharpe = -np.inf
    best_sl = None

    for sl in stop_loss_range:
        returns, _ = simulate_barrier_outcomes(
            ou_params.phi,
            ou_params.forecast,
            ou_params.sigma,
            seed,
            profit_target,
            sl,
            n_iter,
            max_holding_period,
        )

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret if std_ret > 0 else -np.inf

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_sl = sl

    return best_sl, best_sharpe


def find_optimal_profit_taking_given_max_loss(
    ou_params: OUParameters,
    max_stop_loss: float,
    profit_taking_range: np.ndarray = None,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    seed: float = 0.0,
) -> Tuple[float, float]:
    """
    Find optimal profit-taking given maximum allowable stop-loss

    This implements Step 5c from AFML Chapter 13.5.1

    Args:
        ou_params: Estimated O-U parameters
        max_stop_loss: Maximum allowable stop-loss
        profit_taking_range: Range of profit-taking values to test
        n_iter: Number of Monte Carlo iterations
        max_holding_period: Maximum bars to hold
        seed: Initial price level

    Returns:
        (optimal_profit_taking, expected_sharpe)
    """
    if profit_taking_range is None:
        profit_taking_range = np.linspace(0.5, 10, 20) * ou_params.sigma

    best_sharpe = -np.inf
    best_pt = None

    for pt in profit_taking_range:
        # Only test with stop-loss <= max_stop_loss
        returns, _ = simulate_barrier_outcomes(
            ou_params.phi,
            ou_params.forecast,
            ou_params.sigma,
            seed,
            pt,
            max_stop_loss,
            n_iter,
            max_holding_period,
        )

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret if std_ret > 0 else -np.inf

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_pt = pt

    return best_pt, best_sharpe


# ============================================================================
# INTEGRATION WITH BaseStrategy
# ============================================================================


def optimize_strategy_barriers(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    forecast_method: str = "signal_based",
    forecast_window: int = 20,
    custom_forecast: Optional[float] = None,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    pt_range: np.ndarray = None,
    sl_range: np.ndarray = None,
    verbose: bool = True,
) -> BarrierOptimizationResult:
    """
    Complete workflow: Generate signals and optimize barriers

    Args:
        strategy: Instance of BaseStrategy (e.g., BollingerStrategy)
        data: OHLCV DataFrame
        forecast_method: Method to calculate E0[Pi,Ti] (see estimate_ou_parameters)
            - 'signal_based': Direction * expected move (DEFAULT, recommended)
            - 'zero': Random walk assumption
            - 'mean_return': Historical mean
            - 'moving_average': Rolling MA
            - 'custom': User-provided value
        forecast_window: Window for rolling calculations (default: 20)
            - For mean reversion: Use your strategy's lookback (e.g., Bollinger window)
            - For trend following: Use your trend detection window
            - For pairs trading: Use half-life of cointegration
        custom_forecast: Custom forecast value (required if forecast_method='custom')
        n_iter: Monte Carlo iterations (default: 100000)
        max_holding_period: Vertical barrier (max bars held, default: 100)
        pt_range: Profit-taking range (multiples of sigma)
        sl_range: Stop-loss range (multiples of sigma)
        verbose: Print progress

    Returns:
        BarrierOptimizationResult with optimal barriers and full grid

    Example:
        # For Bollinger Bands (mean reversion)
        strategy = BollingerStrategy(window=20, std=2.0)
        result = optimize_strategy_barriers(
            strategy=strategy,
            data=data,
            forecast_method='signal_based',  # Recommended
            forecast_window=20  # Match Bollinger window
        )

        # For MA Crossover (trend following)
        strategy = MACrossoverStrategy(fast=10, slow=30)
        result = optimize_strategy_barriers(
            strategy=strategy,
            data=data,
            forecast_method='signal_based',
            forecast_window=30  # Use slower MA window
        )
    """
    if verbose:
        print(f"\nOptimizing barriers for: {strategy.get_strategy_name()}")
        print(f"Strategy objective: {strategy.get_objective()}")

    # Generate signals
    signals = strategy.generate_signals(data)

    if verbose:
        n_signals = (signals != 0).sum()
        print(f"Generated {n_signals:,} trading signals")

    # Estimate O-U parameters
    ou_params = estimate_ou_parameters(
        price_series=data["close"],
        signals=signals,
        forecast_method=forecast_method,
        forecast_window=forecast_window,
        custom_forecast=custom_forecast,
    )

    # Optimize barriers
    grid_results, optimal = optimize_barriers(
        ou_params=ou_params,
        profit_taking_range=pt_range,
        stop_loss_range=sl_range,
        n_iter=n_iter,
        max_holding_period=max_holding_period,
        verbose=verbose,
    )

    return BarrierOptimizationResult(
        optimal=optimal,
        grid_results=grid_results,
        ou_params=ou_params,
        strategy_name=strategy.get_strategy_name(),
    )


@cacheable()
def generate_optimal_label_config(
    data_config: dict,
    strategy: BaseStrategy,
    forecast_window: int = 20,
    forecast_method: str = "signal_based",
    custom_forecast: Optional[float] = None,
    pt_range: np.ndarray = np.linspace(0.5, 10, 20),
    sl_range: np.ndarray = np.linspace(0.5, 10, 20),
    max_holding_period: int = 100,
    min_ret: float = 0.0,
    n_iter: int = 100000,
    vertical_barrier_zero: bool = False,
    filter_as_series: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, float | Dict[str, Any]], BarrierOptimizationResult]:
    """
    Generate an optimal label configuration from synthetic data.

    Args:
        data_config: Configuration to obtain OHLCV DataFrame
        strategy: Instance of BaseStrategy (e.g., BollingerStrategy)
        forecast_window: Window for rolling calculations (default: 20)
            - For mean reversion: Use your strategy's lookback (e.g., Bollinger window)
            - For trend following: Use your trend detection window
            - For pairs trading: Use half-life of cointegration
        forecast_method: Method to calculate E0[Pi,Ti] (see estimate_ou_parameters)
            - 'signal_based': Direction * expected move (DEFAULT, recommended)
            - 'zero': Random walk assumption
            - 'mean_return': Historical mean
            - 'moving_average': Rolling MA
            - 'custom': User-provided value
        custom_forecast: Custom forecast value (required if forecast_method='custom')
        pt_range: Profit-taking range (multiples of sigma)
        sl_range: Stop-loss range (multiples of sigma)
        max_holding_period: Vertical barrier (max bars held, default: 100)
        min_ret: Minimum return to generate triple-barrier events
        n_iter: Monte Carlo iterations (default: 100000)
        vertical_barrier_zero: vertical_barrier_zero setting for label config
        filter_as_series: filter_as_series setting for label config
        verbose: Print progress

    Returns:
        BarrierOptimizationResult with optimal barriers and full grid

    Example:
        # For Bollinger Bands (mean reversion)
        strategy = BollingerStrategy(window=20, std=2.0)
        result = optimize_strategy_barriers(
            strategy=strategy,
            data=data,
            forecast_method='signal_based',  # Recommended
            forecast_window=20  # Match Bollinger window
        )

        # For MA Crossover (trend following)
        strategy = MACrossoverStrategy(fast=10, slow=30)
        result = optimize_strategy_barriers(
            strategy=strategy,
            data=data,
            forecast_method='signal_based',
            forecast_window=30  # Use slower MA window
        )
    """

    data = load_and_prepare_training_data(**data_config)

    result = optimize_strategy_barriers(
        strategy,
        data,
        forecast_method,
        forecast_window,
        custom_forecast,
        n_iter,
        max_holding_period,
        pt_range,
        sl_range,
        verbose,
    )

    optimal_barriers = result.optimal

    label_config = dict(
        profit_target=optimal_barriers.pt_sigma_multiple,
        stop_loss=optimal_barriers.sl_sigma_multiple,
        max_holding_period=dict(num_bars=max_holding_period),
        min_ret=min_ret,
        vertical_barrier_zero=vertical_barrier_zero,
        filter_as_series=filter_as_series,
    )

    return label_config, result


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_barrier_heatmap(
    result,
    metric: str = "sharpe_ratio",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "magma",
    save_path: Optional[str] = None,
    display: bool = True,
):
    """
    Create heatmap of barrier optimization results

    Args:
        result: BarrierOptimizationResult from optimize_strategy_barriers
        metric: Metric to visualize ('sharpe_ratio', 'win_rate', 'mean_return')
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return

    df = result.grid_results

    # Create pivot table for heatmap
    pivot = df.pivot_table(
        values=metric, index="sl_sigma_multiple", columns="pt_sigma_multiple", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=0 if metric == "sharpe_ratio" else None,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
    )

    # Explicitly set tick positions and labels with 4 decimals
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns])

    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{val:.2f}" for val in pivot.index])

    # Mark optimal point
    opt = result.optimal
    opt_pt_mult = opt.profit_taking / result.ou_params.sigma
    opt_sl_mult = opt.stop_loss / result.ou_params.sigma

    # Find closest point in grid
    pt_vals = pivot.columns.values
    sl_vals = pivot.index.values
    pt_idx = np.argmin(np.abs(pt_vals - opt_pt_mult))
    sl_idx = np.argmin(np.abs(sl_vals - opt_sl_mult))

    ax.plot(
        pt_idx + 0.5,
        sl_idx + 0.5,
        "r*",
        markersize=20,
        label=f"Optimal (Sharpe={opt.sharpe_ratio:.3f})",
    )

    ax.set_xlabel("Profit Taking (× σ)", fontsize=12)
    ax.set_ylabel("Stop Loss (× σ)", fontsize=12)
    ax.set_title(
        f"{result.strategy_name}\n{metric.replace('_', ' ').title()} Heatmap",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")

    plt.tight_layout()

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
        plt.close()


def plot_barrier_comparison(
    results: List[BarrierOptimizationResult], figsize: Tuple[int, int] = (14, 10), display: bool = True,
):
    """
    Compare multiple strategies side-by-side

    Args:
        results: List of BarrierOptimizationResult objects
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    n_strategies = len(results)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    metrics = [
        "sharpe_ratio",
        "win_rate",
        "mean_return",
        "std_return",
        "avg_holding_period",
    ]
    metric_labels = [
        "Sharpe Ratio",
        "Win Rate",
        "Mean Return",
        "Std Return",
        "Avg Holding Period",
    ]

    strategy_names = [r.strategy_name for r in results]

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        values = [r.optimal.__dict__[metric] for r in results]

        bars = ax.bar(
            range(n_strategies),
            values,
            alpha=0.7,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, n_strategies)),
        )

        ax.set_xticks(range(n_strategies))
        ax.set_xticklabels(strategy_names, rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.5f}" if abs(val) < 10 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Barrier comparison plot
    ax = axes[5]
    x = np.arange(n_strategies)
    width = 0.35

    pt_vals = [r.optimal.profit_taking / r.ou_params.sigma for r in results]
    sl_vals = [r.optimal.stop_loss / r.ou_params.sigma for r in results]

    ax.bar(x - width / 2, pt_vals, width, label="Profit Taking (×σ)", alpha=0.8, color="green")
    ax.bar(x + width / 2, sl_vals, width, label="Stop Loss (×σ)", alpha=0.8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax.set_ylabel("Barrier Size (× σ)")
    ax.set_title("Optimal Barriers", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Strategy Comparison - Optimal Barrier Performance", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    if display:
        plt.show()


def plot_ou_parameter_distribution(
    result: BarrierOptimizationResult,
    n_paths: int = 100,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    display: bool = True,
):
    """
    Visualize O-U process behavior with estimated parameters

    Args:
        result: BarrierOptimizationResult
        n_paths: Number of sample paths to plot
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    ou = result.ou_params
    opt = result.optimal

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Sample paths with barriers
    ax1 = axes[0]
    paths = []

    for _ in range(n_paths):
        path, _ = simulate_ou_path(ou.phi, ou.forecast, ou.sigma, 0.0, opt.max_holding_period)
        paths.append(path)
        ax1.plot(path, alpha=0.1, color="blue")

    # Plot mean path
    mean_path = np.mean(paths, axis=0)
    ax1.plot(mean_path, color="darkblue", linewidth=2, label="Mean Path")

    # Plot barriers
    ax1.axhline(
        opt.profit_taking,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Profit Taking ({opt.profit_taking:.3f})",
    )
    ax1.axhline(
        -opt.stop_loss,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Stop Loss ({-opt.stop_loss:.3f})",
    )
    ax1.axhline(
        ou.forecast,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Forecast ({ou.forecast:.3f})",
    )

    ax1.set_xlabel("Time Steps", fontsize=11)
    ax1.set_ylabel("Price Return", fontsize=11)
    ax1.set_title(
        f"O-U Process Sample Paths\nφ={ou.phi:.3f}, σ={ou.sigma:.6f}, Half-life={ou.half_life:.1f}",
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # Right plot: Return distribution at barriers
    ax2 = axes[1]

    # Simulate many outcomes
    returns, _ = simulate_barrier_outcomes(
        ou.phi, ou.forecast, ou.sigma, 0.0, opt.profit_taking, opt.stop_loss, 10000, opt.max_holding_period
    )

    # Plot distribution
    ax2.hist(returns, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax2.axvline(
        opt.profit_taking, color="green", linestyle="--", linewidth=2, label="Profit Taking"
    )
    ax2.axvline(-opt.stop_loss, color="red", linestyle="--", linewidth=2, label="Stop Loss")
    ax2.axvline(
        np.mean(returns),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean Return ({np.mean(returns):.4f})",
    )

    ax2.set_xlabel("Exit Return", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title(
        f"Exit Return Distribution (10k simulations)\n"
        f"Win Rate: {opt.win_rate:.1%}, Sharpe: {opt.sharpe_ratio:.3f}",
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{result.strategy_name} - O-U Process Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved O-U analysis plot to {save_path}")
        plt.close()


def create_full_report(result: BarrierOptimizationResult, save_dir: str = None, display: bool = True):
    """
    Generate comprehensive visual report

    Args:
        result: BarrierOptimizationResult
        save_dir: Directory to save plots (optional)
    """
    print("\n" + "=" * 70)
    print(f"BARRIER OPTIMIZATION REPORT: {result.strategy_name}")
    print("=" * 70)

    # Print summary
    print("\nO-U Parameters:")
    print(f"  φ (phi):        {result.ou_params.phi:.4f}")
    print(f"  σ (sigma):      {result.ou_params.sigma:.4f}")
    print(f"  Forecast:       {result.ou_params.forecast:.4f}")
    print(f"  Half-life:      {result.ou_params.half_life:.2f} periods")

    print("\nOptimal Barriers:")
    print(
        f"  Profit Taking:  {result.optimal.profit_taking:.4f} "
        f"({result.optimal.profit_taking / result.ou_params.sigma:.2f}σ)"
    )
    print(
        f"  Stop Loss:      {result.optimal.stop_loss:.4f} "
        f"({result.optimal.stop_loss / result.ou_params.sigma:.2f}σ)"
    )
    print(
        f"  Max Bars:       {result.optimal.max_holding_period:,}"
    )

    print("\nExpected Performance:")
    print(f"  Sharpe Ratio:   {result.optimal.sharpe_ratio:.4f}")
    print(f"  Mean Return:    {result.optimal.mean_return:.4f}")
    print(f"  Std Return:     {result.optimal.std_return:.4f}")
    print(f"  Win Rate:       {result.optimal.win_rate:.2%}")
    print(f"  Avg Hold:       {result.optimal.avg_holding_period:.1f} periods")
    print("=" * 70)

    # Generate plots
    if save_dir:
        save_dir = Path(save_dir, "optimal_barrier_analysis")
        save_dir.mkdir(exist_ok=True, parents=True)
        heatmap_path = save_dir / "heatmap.png"
        ou_path = save_dir / "ou_analysis.png"
    else:
        heatmap_path = None
        ou_path = None

    # Plot 1: Heatmap
    plot_barrier_heatmap(result, save_path=heatmap_path, display=display)

    # Plot 2: O-U Analysis
    plot_ou_parameter_distribution(result, save_path=ou_path, display=display)

    # Save grid results
    if save_dir:
        csv_path = save_dir / "grid_results.csv"
        result.grid_results.to_csv(csv_path, index=False)
        print(f"\nSaved grid results to {csv_path}")
