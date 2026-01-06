"""
Correct Implementation of Chapter 13: Backtesting on Synthetic Data
From "Advances in Financial Machine Learning" by Marcos López de Prado

INTEGRATION WITH EXISTING TRIPLE BARRIER INFRASTRUCTURE:
1. Use existing get_events() to get historical outcomes
2. Extract realized price paths from these events
3. Estimate O-U parameters from actual price dynamics
4. Optimize barriers via Monte Carlo on synthetic paths

NO GUESSING - uses actual realized price paths to fit the process.
"""

import warnings
from dataclasses import dataclass
from numba import jit, njit, prange
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from afml.labeling.triple_barrier import add_vertical_barrier
from afml.strategies.signal_processing import get_entries

from ..cache.unified_cache_system import cacheable
from ..strategies.trading_strategies import BaseStrategy


@dataclass
class OUParameters:
    """
    Ornstein-Uhlenbeck process parameters from equation (13.2)

    P_t = (1-φ)E_0[P_T] + φP_{t-1} + σε_t
    """
    phi: float              # Mean reversion speed φ ∈ (0,1)
    forecast: float         # E_0[P_T] - estimated mean reversion target
    sigma: float            # Volatility σ
    half_life: float        # Half-life = -log(2)/log(φ)


@dataclass
class OptimalBarriers:
    """Optimal barrier configuration from Chapter 13 Step 5"""
    profit_taking: float        # π̄ (absolute PnL)
    stop_loss: float            # π (absolute PnL)
    pt_sigma_multiple: float    # π̄/σ (for use with target volatility)
    sl_sigma_multiple: float    # π/σ (for use with target volatility)
    sharpe_ratio: float
    mean_return: float
    std_return: float
    win_rate: float
    avg_holding_period: float


@dataclass
class BarrierOptimizationResult:
    """Complete optimization results"""
    optimal: OptimalBarriers
    grid_results: pd.DataFrame
    ou_params: OUParameters
    strategy_name: str
    n_events: int  # Number of triple-barrier events used


# ============================================================================
# STEP 1: ESTIMATE O-U PARAMETERS FROM TRIPLE-BARRIER EVENTS
# ============================================================================
def estimate_ou_parameters_from_triple_barrier(
    close: pd.Series,
    triple_barrier_events: pd.DataFrame,
    equilibrium_method: str = 'ar1_regression',
    rolling_window: int = None
) -> OUParameters:
    """
    Estimate O-U parameters directly from triple-barrier events using equation (13.5):

    P_{i,t} = E_0[P_{i,T_i}] + φ(P_{i,t-1} - E_0[P_{i,T_i}]) + ξ_t

    Supports two equilibrium estimation methods:
    1. 'ar1_regression': Global equilibrium via AR(1) regression (no free parameters)
    2. 'rolling_window': Time-varying equilibrium via rolling mean (requires window choice)

    Args:
        close: Price series
        triple_barrier_events: Output from get_bins() with columns:
            - index: Event start times
            - t1: Event end times
            - ret: Realized return
        equilibrium_method: 'ar1_regression' or 'rolling_window'
        rolling_window: Window length (required if method='rolling_window')

    Returns:
        OUParameters with estimated {φ, E_0[P_T], σ}
    """
    if len(triple_barrier_events) < 10:
        warnings.warn("Insufficient events (<10) for robust parameter estimation")
        return OUParameters(phi=0.95, forecast=0.0, sigma=1.0, half_life=14)

    entry_times = close.index.get_indexer(triple_barrier_events.index)
    exit_times = close.index.get_indexer_for(triple_barrier_events.index)

    if equilibrium_method == 'rolling_window':
        if rolling_window is None:
            raise ValueError("rolling_window must be specified when method='rolling_window'")

        # Calculate rolling mean equilibrium series
        equilibrium_series = close.rolling(window=rolling_window, min_periods=1).mean()
        equilibrium_series = equilibrium_series.ffill()

        # Build X, Y, Z vectors with time-varying equilibrium
        X_arrays, Y_arrays, Z_arrays = [], [], []

        for entry_time, exit_time in zip(entry_times, exit_times):
            price_path = close.iloc[entry_time:exit_time].values

            if len(price_path) < 2:
                continue

            # Use rolling mean at entry time (no look-ahead)
            forecast_level = equilibrium_series.iloc[entry_time].values

            X_arrays.append(price_path[:-1] - forecast_level)
            Y_arrays.append(price_path[1:])
            Z_arrays.append(np.full(len(price_path) - 1, forecast_level))

        if len(X_arrays) < 10:
            warnings.warn("Insufficient events after filtering (<10)")
            return OUParameters(phi=0.95, forecast=0.0, sigma=1.0, half_life=14)

        X = np.concatenate(X_arrays)
        Y = np.concatenate(Y_arrays)
        Z = np.concatenate(Z_arrays)

        # Estimate φ: cov(Y,X) / var(X)
        cov_YX = np.cov(Y, X)[0, 1]
        var_X = np.var(X)

        if var_X < 1e-10:
            phi_hat = 0.95
        else:
            phi_hat = cov_YX / var_X

        phi_hat = max(0.01, min(0.999, phi_hat))

        # Residuals and σ
        residuals = Y - Z - phi_hat * X
        sigma_hat = np.std(residuals) if np.std(residuals) > 0 else 1.0

        # Average forecast across all events
        forecast = np.mean(Z)

    elif equilibrium_method == 'ar1_regression':
        # Global equilibrium via AR(1) regression (original method)
        all_prices_t = []
        all_prices_t_minus_1 = []

        for entry_time, exit_time in zip(entry_times, exit_times):
            price_path = close.iloc[entry_time:exit_time].values

            if len(price_path) < 2:
                continue

            all_prices_t.extend(price_path[1:])
            all_prices_t_minus_1.extend(price_path[:-1])

        if len(all_prices_t) < 10:
            warnings.warn("Insufficient price observations (<10)")
            return OUParameters(phi=0.95, forecast=0.0, sigma=1.0, half_life=14)

        P_t = np.array(all_prices_t)
        P_t_minus_1 = np.array(all_prices_t_minus_1)

        # AR(1): P_t = α + φP_{t-1} + ξ
        X_with_const = np.column_stack([np.ones(len(P_t_minus_1)), P_t_minus_1])

        try:
            beta = np.linalg.lstsq(X_with_const, P_t, rcond=None)[0]
            alpha = beta[0]
            phi_hat = beta[1]
        except np.linalg.LinAlgError:
            warnings.warn("OLS regression failed, using defaults")
            return OUParameters(phi=0.95, forecast=0.0, sigma=1.0, half_life=14)

        phi_hat = max(0.01, min(0.999, phi_hat))

        # Back out equilibrium: μ = α/(1-φ)
        if abs(1 - phi_hat) > 1e-6:
            forecast = alpha / (1 - phi_hat)
        else:
            forecast = np.mean(P_t)

        # Residuals
        predicted = alpha + phi_hat * P_t_minus_1
        residuals = P_t - predicted
        sigma_hat = np.std(residuals) if np.std(residuals) > 0 else 1.0

    else:
        raise ValueError(f"Unknown equilibrium_method: {equilibrium_method}")

    # Half-life
    half_life = -np.log(2) / np.log(phi_hat) if 0 < phi_hat < 1 else np.inf

    return OUParameters(
        phi=phi_hat,
        forecast=forecast,
        sigma=sigma_hat,
        half_life=half_life
    )


# ============================================================================
# STEPS 3-4: MONTE CARLO SIMULATION
# ============================================================================

@njit(cache=True)
def simulate_ou_price_path(
    phi: float,
    forecast: float,
    sigma: float,
    entry_price: float,
    max_hp: int
) -> np.ndarray:
    """
    Simulate O-U process on PRICE LEVELS.

    Equation (13.2): P_t = (1-φ)E_0[P_T] + φP_{t-1} + σε_t

    Args:
        phi: Mean reversion speed
        forecast: E_0[P_{i,T_i}] target price level
        sigma: Volatility (in price units)
        entry_price: P_{i,0} starting price
        max_hp: Maximum holding period

    Returns:
        Array of simulated prices [P_0, P_1, ..., P_max_hp]
    """
    path = np.empty(max_hp + 1)
    path[0] = entry_price

    for t in range(max_hp):
        # O-U process on price level
        path[t + 1] = (1 - phi) * forecast + phi * path[t] + sigma * np.random.normal()

    return path


@njit(cache=True)
def check_barrier_hit_pnl(
    price_path: np.ndarray,
    entry_price: float,
    position_size: np.ndarray,
    profit_taking: float,
    stop_loss: float,
    max_hp: int
) -> Tuple[float, int]:
    """
    Check which barrier is hit first, operating on PnL.

    PnL = m_i * (P_t - P_{i,0})

    Args:
        price_path: Simulated price path
        entry_price: P_{i,0}
        position_size: m_i (1 for long, -1 for short)
        profit_taking: π̄ > 0 (absolute PnL level)
        stop_loss: π > 0 (absolute PnL level, will be negated)
        max_hp: Maximum holding period

    Returns:
        (exit_pnl, holding_period)
    """
    for t in range(1, min(len(price_path), max_hp + 1)):
        price_change = price_path[t] - entry_price
        pnl = position_size[t] * price_change

        # Check barriers
        if pnl >= profit_taking:
            return pnl, t
        elif pnl <= -stop_loss:
            return pnl, t

    # Hit vertical barrier (time limit)
    final_idx = min(max_hp, len(price_path) - 1)
    final_pnl = position_size * (price_path[final_idx] - entry_price)
    return final_pnl, max_hp


@jit(cache=True, parallel=True, forceobj=True)
def simulate_barrier_outcomes(
    strategy: BaseStrategy,
    phi: float,
    forecast: float,
    sigma: float,
    entry_price: float,
    position_size: int,
    profit_taking: float,
    stop_loss: float,
    n_iter: int,
    max_hp: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulations for a specific barrier configuration.

    This implements Steps 3-4 from Chapter 13.

    Returns:
        (pnl_array, holding_periods_array)
    """
    returns = np.zeros(n_iter)
    holding_periods = np.zeros(n_iter)

    for i in prange(n_iter):
        # Generate price path
        price_path = simulate_ou_price_path(phi, forecast, sigma, entry_price, max_hp)
        data = pd.Series(price_path, name="close").to_frame()
        position_size = strategy.generate_signals(data)

        # Check barrier hits on PnL
        exit_pnl, hp = check_barrier_hit_pnl(
            price_path, entry_price, position_size,
            profit_taking, stop_loss, max_hp
        )

        returns[i] = exit_pnl
        holding_periods[i] = hp

    return returns, holding_periods


# ============================================================================
# STEPS 2-5: BARRIER OPTIMIZATION
# ============================================================================

def optimize_barriers(
    close: pd.Series,
    triple_barrier_events: pd.DataFrame,
    ou_params: OUParameters,
    profit_taking_range: np.ndarray = None,
    stop_loss_range: np.ndarray = None,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    verbose: bool = True
) -> Tuple[pd.DataFrame, OptimalBarriers]:
    """
    Find optimal profit-taking and stop-loss barriers via Monte Carlo.

    This implements Steps 2-5a from Chapter 13.

    Args:
        close: Price series
        triple_barrier_events: Events from get_bins()
        ou_params: Estimated O-U parameters
        profit_taking_range: Array of PT multiples of sigma (Step 2)
        stop_loss_range: Array of SL multiples of sigma (Step 2)
        n_iter: Number of Monte Carlo iterations (Step 3)
        max_holding_period: Vertical barrier
        verbose: Print progress

    Returns:
        (grid_results_df, optimal_barriers)
    """
    # Step 2: Default ranges from Chapter 13
    if profit_taking_range is None:
        profit_taking_range = np.linspace(0.5, 10, 20)

    if stop_loss_range is None:
        stop_loss_range = np.linspace(0.5, 10, 20)

    # Scale by sigma to get absolute PnL levels
    pt_scaled = profit_taking_range * ou_params.sigma
    sl_scaled = stop_loss_range * ou_params.sigma

    results = []
    total_combinations = len(pt_scaled) * len(sl_scaled)

    if verbose:
        print(f"\n{'='*70}")
        print("BARRIER OPTIMIZATION")
        print(f"{'='*70}")
        print(f"O-U Parameters Estimated from {len(triple_barrier_events):,} Events:")
        print(f"  φ (phi):          {ou_params.phi:.4f}")
        print(f"  σ (sigma):        {ou_params.sigma:.4f}")
        print(f"  E_0[P_T]:         {ou_params.forecast:.6f}")
        print(f"  Half-life:        {ou_params.half_life:.2f} bars")
        print(f"\nTesting {total_combinations:,} barrier combinations...")
        print(f"Monte Carlo iterations per combination: {n_iter:,}")

    # Use representative event for simulation (middle of dataset)
    mid_idx = len(triple_barrier_events) // 2
    entry_time = triple_barrier_events.index[mid_idx]
    entry_price = close.loc[entry_time]

    # Get position size (vectorized check)
    if 'side' in triple_barrier_events.columns:
        position_size = int(triple_barrier_events.iloc[mid_idx]['side'])
    else:
        position_size = 1  # Default to long

    progress_bar = tqdm(
        product(pt_scaled, sl_scaled),
        total=total_combinations,
        desc="Optimizing barriers"
    ) if verbose else product(pt_scaled, sl_scaled)

    for pt, sl in progress_bar:
        # Step 3-4: Run Monte Carlo simulations

        returns, holding_periods = simulate_barrier_outcomes(
            ou_params.phi,
            ou_params.forecast,
            ou_params.sigma,
            entry_price,
            position_size,
            pt, sl, n_iter, max_holding_period
        )

        # Calculate metrics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret if std_ret > 0 else -np.inf
        win_rate = np.sum(returns > 0) / len(returns)
        avg_hp = np.mean(holding_periods)

        results.append({
            'profit_taking': pt,
            'stop_loss': sl,
            'pt_sigma_multiple': pt / ou_params.sigma,
            'sl_sigma_multiple': sl / ou_params.sigma,
            'mean_return': mean_ret,
            'std_return': std_ret,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_holding_period': avg_hp,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Step 5a: Find optimal (maximize Sharpe ratio)
    optimal_idx = results_df['sharpe_ratio'].idxmax()
    optimal_row = results_df.loc[optimal_idx]

    optimal = OptimalBarriers(
        profit_taking=optimal_row['profit_taking'],
        stop_loss=optimal_row['stop_loss'],
        pt_sigma_multiple=optimal_row['pt_sigma_multiple'],
        sl_sigma_multiple=optimal_row['sl_sigma_multiple'],
        sharpe_ratio=optimal_row['sharpe_ratio'],
        mean_return=optimal_row['mean_return'],
        std_return=optimal_row['std_return'],
        win_rate=optimal_row['win_rate'],
        avg_holding_period=optimal_row['avg_holding_period']
    )

    if verbose:
        print(f"\n{'='*70}")
        print("OPTIMAL BARRIERS FOUND")
        print(f"{'='*70}")
        print(f"Profit Taking: {optimal.profit_taking:.4%} "
              f"({optimal.pt_sigma_multiple:.2f}σ)")
        print(f"Stop Loss:     {optimal.stop_loss:.4%} "
              f"({optimal.sl_sigma_multiple:.2f}σ)")
        print("\nExpected Performance (Monte Carlo):")
        print(f"  Sharpe Ratio:  {optimal.sharpe_ratio:.4f}")
        print(f"  Win Rate:      {optimal.win_rate:.2%}")
        print(f"  Mean Return:   {optimal.mean_return:.4%}")
        print(f"  Std Return:    {optimal.std_return:.4%}")
        print(f"  Avg Hold:      {optimal.avg_holding_period:.1f} bars")
        print(f"{'='*70}\n")

    return results_df, optimal


# ============================================================================
# MAIN WORKFLOW: INTEGRATION WITH BaseStrategy + Triple Barrier
# ============================================================================

@cacheable()
def optimize_strategy_barriers_from_triple_barrier(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    target: pd.Series,
    vertical_barrier_zero: bool = True,
    initial_pt_sl: list = [2, 2],
    min_ret: float = 0.0,
    n_iter: int = 100000,
    max_holding_period: int = 100,
    pt_range: np.ndarray = None,
    sl_range: np.ndarray = None,
    equilibrium_method: str = 'ar1_regression',
    rolling_window: int = None,
    verbose: bool = True
) -> BarrierOptimizationResult:
    """
    Complete Chapter 13 workflow using existing triple-barrier infrastructure:

    1. Generate signals from strategy
    2. Run triple-barrier with initial barriers to get REALIZED outcomes
    3. Estimate O-U parameters directly from realized price paths
    4. Optimize barriers via Monte Carlo

    Args:
        strategy: Instance of BaseStrategy
        data: OHLCV DataFrame with 'close' column
        target: Target volatility series (e.g., from daily_vol)
        vertical_barrier_zero: If True, return is zero when vertical barrier is hit, else sign of return
        initial_pt_sl: Initial barriers for triple-barrier [pt, sl]
            These are just for getting historical outcomes, will be optimized
        min_ret: Minimum return filter
        n_iter: Monte Carlo iterations for optimization
        max_holding_period: Vertical barrier for optimization
        pt_range: Profit-taking range (multiples of σ)
        sl_range: Stop-loss range (multiples of σ)
        equilibrium_method: 'ar1_regression' (default) or 'rolling_window'
            - 'ar1_regression': Estimates single global equilibrium (no free parameters)
            - 'rolling_window': Uses time-varying equilibrium (requires rolling_window)
        rolling_window: Window length for rolling mean (required if method='rolling_window')
            Recommendation: Choose based on stationarity testing, not Sharpe optimization
        verbose: Print progress

    Returns:
        BarrierOptimizationResult with optimal barriers

    Example:
        from triple_barrier import get_events, get_bins, add_vertical_barrier
        from sampling import cusum_filter
        from volatility import get_daily_vol

        # Setup
        strategy = BollingerStrategy(window=20, std=2.0)
        signals = strategy.generate_signals(data)

        # Get events for triple-barrier
        daily_vol = get_daily_vol(data['close'])
        t_events = cusum_filter(data['close'], daily_vol.mean())
        vertical_barriers = add_vertical_barrier(t_events, data['close'], num_bars=100)

        # Optimize
        result = optimize_strategy_barriers_from_triple_barrier(
            strategy=strategy,
            data=data,
            target=daily_vol,
            t_events=t_events,
            vertical_barrier_times=vertical_barriers,
            initial_pt_sl=[2, 2],  # Initial guess
            n_iter=100000
        )

        # Use optimal barriers going forward
        optimal_pt = result.optimal.pt_sigma_multiple
        optimal_sl = result.optimal.sl_sigma_multiple
    """
    from ..labeling.triple_barrier import triple_barrier_labels

    if verbose:
        print(f"\n{'='*70}")
        print("OPTIMAL TRADING RULE")
        print(f"{'='*70}")
        print(f"Strategy: {strategy.get_strategy_name()}")
        print(f"Objective: {strategy.get_objective()}")
        print(f"\nStep 1: Running triple-barrier with initial barriers {initial_pt_sl}")
        print("        to capture realized price paths...")

    # Generate signals
    signals = strategy.generate_signals(data)
    t_events = signals[signals != 0].index
    # filter_threshold = target if filter_as_series else target.mean()
    # signals, t_events = get_entries(strategy, data, filter_threshold)

    # Step 1: Run triple-barrier with INITIAL barriers to get realized outcomes
    # (We need some barriers to get events, doesn't matter what they are)
    vertical_barrier_times = add_vertical_barrier(t_events, data['close'], num_bars=max_holding_period)
    triple_barrier_events = triple_barrier_labels(
        close=data['close'],
        t_events=t_events,
        pt_sl=initial_pt_sl,
        target=target,
        min_ret=min_ret,
        vertical_barrier_times=vertical_barrier_times,
        vertical_barrier_zero=vertical_barrier_zero,
        side_prediction=signals,  # Use strategy signals for position side
        verbose=False,
    )

    if verbose:
        print(f"        Generated {len(triple_barrier_events):,} triple-barrier events")
        avg_hp = (triple_barrier_events['t1'] - triple_barrier_events.index).mean().round("1s")
        avg_ret = triple_barrier_events['ret'].mean()
        print(f"        Avg holding period: {str(avg_hp).replace('0 days ', '')}")
        print(f"        Avg realized return: {avg_ret:.4%}")

    close = np.log(data['close'])

    # Step 2: Estimate O-U parameters from events
    if verbose:
        method_desc = 'AR(1) regression' if equilibrium_method == 'ar1_regression' else f'rolling window ({rolling_window})'
        print(f"\nStep 2: Estimating O-U parameters using {method_desc}...")

    ou_params = estimate_ou_parameters_from_triple_barrier(
        close=close,
        triple_barrier_events=triple_barrier_events,
        equilibrium_method=equilibrium_method,
        rolling_window=rolling_window
    )

    # Step 3-4: Optimize barriers via Monte Carlo
    if verbose:
        print("\nStep 3-4: Optimizing barriers via Monte Carlo simulation...")

    grid_results, optimal = optimize_barriers(
        close=close,
        triple_barrier_events=triple_barrier_events,
        ou_params=ou_params,
        profit_taking_range=pt_range,
        stop_loss_range=sl_range,
        n_iter=int(n_iter),
        max_holding_period=max_holding_period,
        verbose=verbose
    )

    return BarrierOptimizationResult(
        optimal=optimal,
        grid_results=grid_results,
        ou_params=ou_params,
        strategy_name=strategy.get_strategy_name(),
        n_events=len(triple_barrier_events)
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    Example usage with existing triple-barrier infrastructure:

    from triple_barrier import get_events, get_bins, add_vertical_barrier
    from sampling import cusum_filter
    from volatility import get_daily_vol
    from strategies import BollingerStrategy
    import pandas as pd

    # Load data
    data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

    # Setup strategy
    strategy = BollingerStrategy(window=20, std=2.0)

    # Get volatility and event sampling
    daily_vol = get_daily_vol(data['close'])
    t_events = cusum_filter(data['close'], daily_vol.mean())
    vertical_barriers = add_vertical_barrier(t_events, data['close'], num_bars=100)

    # Method 1: AR(1) Regression (default - no free parameters)
    result_ar1 = optimize_strategy_barriers_from_triple_barrier(
        strategy=strategy,
        data=data,
        target=daily_vol,
        t_events=t_events,
        vertical_barrier_times=vertical_barriers,
        initial_pt_sl=[2, 2],
        equilibrium_method='ar1_regression'  # Global equilibrium
    )

    # Method 2: Rolling Window (time-varying equilibrium)
    # CRITICAL: Choose window based on stationarity/half-life, NOT Sharpe ratio!
    result_rolling = optimize_strategy_barriers_from_triple_barrier(
        strategy=strategy,
        data=data,
        target=daily_vol,
        t_events=t_events,
        vertical_barrier_times=vertical_barriers,
        initial_pt_sl=[2, 2],
        equilibrium_method='rolling_window',
        rolling_window=200  # Choose via ADF test, not optimization
    )

    # Compare results
    print(f"\\nAR(1) Method:")
    print(f"  φ: {result_ar1.ou_params.phi:.4f}")
    print(f"  Half-life: {result_ar1.ou_params.half_life:.2f} bars")
    print(f"  Optimal PT: {result_ar1.optimal.pt_sigma_multiple:.2f}σ")
    print(f"  Expected Sharpe: {result_ar1.optimal.sharpe_ratio:.4f}")

    print(f"\\nRolling Window Method:")
    print(f"  φ: {result_rolling.ou_params.phi:.4f}")
    print(f"  Half-life: {result_rolling.ou_params.half_life:.2f} bars")
    print(f"  Optimal PT: {result_rolling.optimal.pt_sigma_multiple:.2f}σ")
    print(f"  Expected Sharpe: {result_rolling.optimal.sharpe_ratio:.4f}")

    # Use optimal barriers in production
    optimal_pt_sl = [
        result_ar1.optimal.pt_sigma_multiple,
        result_ar1.optimal.sl_sigma_multiple
    ]
    """)
