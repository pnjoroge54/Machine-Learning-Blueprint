from itertools import product
from random import gauss

import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_ou_parameters(price_series, moving_avg_series, dt=1.0):
    """
    Estimates the parameters of an Ornstein-Uhlenbeck process using the correct
    discrete-time formulation: P_t = (1-φ)μ + φ*P_{t-1} + σ*ε_t
    
    Args:
        price_series: Historical price series
        moving_avg_series: Moving average (long-term mean μ)
        dt: Time step (default=1 for daily data)
    
    Returns:
        tuple: (phi, sigma, mu_mean) where phi is persistence, sigma is volatility
    """
    # Align series and remove NaNs
    data = pd.DataFrame({"price": price_series, "mu": moving_avg_series}).dropna()
    
    # Set up regression: P_t = (1-φ)μ + φ*P_{t-1} + ε_t
    # Rearrange to: P_t - μ = φ*(P_{t-1} - μ) + ε_t
    # Or: y_t = φ * y_{t-1} + ε_t, where y_t = P_t - μ
    
    mu_mean = data["mu"].mean()  # Use average of moving average as long-term mean
    
    # Create mean-adjusted series
    y = (data["price"] - mu_mean).dropna()
    y_lag = y.shift(1).dropna()
    
    # Align the series
    y = y.loc[y_lag.index]
    
    # Regression without intercept: y_t = φ * y_{t-1} + ε_t
    model = sm.OLS(y, y_lag)
    results = model.fit()
    
    phi = results.params.iloc[0]
    sigma = results.resid.std()
    
    # Ensure phi is between 0 and 1 for mean reversion
    phi = max(min(phi, 0.999), 0.001)
    
    # Calculate mean reversion speed
    if phi > 0 and phi < 1:
        theta = -np.log(phi) / dt
        half_life = np.log(2) / theta
    else:
        theta = np.nan
        half_life = np.inf
    
    print(f"\nEstimated Phi (persistence): {phi:.4f}")
    print(f"Estimated Theta (mean reversion speed): {theta:.4f}")
    print(f"Estimated Sigma (volatility): {sigma:.4f}")
    print(f"Estimated Mu (long-term mean): {mu_mean:.4f}")
    print(f"Implied half-life: {half_life:.2f} periods")
    
    return phi, sigma, mu_mean


def estimate_standardized_ou_parameters(price_series, moving_avg_series, vol_series):
    """
    Estimates O-U parameters on volatility-standardized data.
    This approach is more robust when volatility changes over time.
    """
    data = pd.DataFrame({
        "price": price_series, 
        "mu": moving_avg_series, 
        "vol": vol_series
    }).dropna()
    
    # Standardize the price deviations
    data["std_dev"] = (data["price"] - data["mu"]) / data["vol"]
    
    # Estimate on standardized series
    y = data["std_dev"].dropna()
    y_lag = y.shift(1).dropna()
    y = y.loc[y_lag.index]
    
    # Regression: std_dev_t = φ * std_dev_{t-1} + ε_t
    model = sm.OLS(y, y_lag)
    results = model.fit()
    
    phi = max(min(results.params.iloc[0], 0.999), 0.001)
    sigma_std = results.resid.std()
    
    print(f"\n--- Standardized Parameters ---")
    print(f"Phi (persistence): {phi:.4f}")
    print(f"Sigma (standardized): {sigma_std:.4f}")
    
    if phi < 1:
        half_life = -np.log(2) / np.log(phi)
        print(f"Half-life: {half_life:.2f} periods")
    
    return phi, sigma_std


def run_triple_barrier_simulation(
    phi, sigma, mu, forecast_horizon_return=0, 
    nIter=10000, maxHP=100, 
    pt_levels=np.linspace(0.5, 5.0, 10), 
    sl_levels=np.linspace(0.5, 5.0, 10)
):
    """
    Monte Carlo simulation for triple barrier method with correct O-U dynamics.
    
    Args:
        phi: Persistence parameter (0 < phi < 1 for mean reversion)
        sigma: Volatility parameter
        mu: Long-term mean level
        forecast_horizon_return: Expected return over the forecast horizon
        nIter: Number of simulation paths
        maxHP: Maximum holding period (time barrier)
        pt_levels: Profit-taking levels to test
        sl_levels: Stop-loss levels to test
    
    Returns:
        DataFrame with results for each (PT, SL) combination
    """
    results = []
    
    for pt in pt_levels:
        for sl in sl_levels:
            pnl_list = []
            
            for _ in range(int(nIter)):
                # Initialize price at current level (relative to mean)
                p = 0  # Start at mean (deviation = 0)
                hp = 0
                
                # Add the forecast component as a drift
                drift_per_period = forecast_horizon_return / maxHP if maxHP > 0 else 0
                
                while hp < maxHP:
                    # O-U evolution: P_t = φ*P_{t-1} + σ*ε_t + drift
                    p = phi * p + sigma * gauss(0, 1) + drift_per_period
                    hp += 1
                    
                    # Check barriers
                    if p >= pt:  # Profit target hit
                        pnl_list.append(p)
                        break
                    elif p <= -sl:  # Stop loss hit
                        pnl_list.append(p)
                        break
                else:
                    # Time barrier hit
                    pnl_list.append(p)
            
            # Calculate performance metrics
            mean_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list)
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
            
            # Additional metrics
            hit_rate = np.mean([pnl > 0 for pnl in pnl_list])
            profit_factor = (np.mean([pnl for pnl in pnl_list if pnl > 0]) * 
                           sum(1 for pnl in pnl_list if pnl > 0)) / \
                          (abs(np.mean([pnl for pnl in pnl_list if pnl < 0])) * 
                           sum(1 for pnl in pnl_list if pnl < 0)) if any(pnl < 0 for pnl in pnl_list) else float('inf')
            
            results.append({
                'pt_level': pt,
                'sl_level': sl,
                'mean_pnl': mean_pnl,
                'std_pnl': std_pnl,
                'sharpe_ratio': sharpe,
                'hit_rate': hit_rate,
                'profit_factor': profit_factor,
                'num_trades': len(pnl_list)
            })
    
    return pd.DataFrame(results)


def find_optimal_barriers(results_df, metric='sharpe_ratio'):
    """
    Find the optimal barrier combination based on specified metric.
    """
    optimal_idx = results_df[metric].idxmax()
    optimal_params = results_df.loc[optimal_idx]
    
    print(f"\nOptimal barriers based on {metric}:")
    print(f"Profit Target: {optimal_params['pt_level']:.2f}")
    print(f"Stop Loss: {optimal_params['sl_level']:.2f}")
    print(f"Sharpe Ratio: {optimal_params['sharpe_ratio']:.4f}")
    print(f"Hit Rate: {optimal_params['hit_rate']:.4f}")
    print(f"Mean PnL: {optimal_params['mean_pnl']:.4f}")
    
    return optimal_params


# Example usage and testing
def example_usage():
    """
    Example of how to use the corrected implementation.
    """
    # Generate synthetic mean-reverting price data for testing
    np.random.seed(42)
    n_periods = 1000
    true_phi = 0.95
    true_sigma = 0.02
    true_mu = 100.0
    
    # Generate synthetic O-U process
    prices = [true_mu]
    for _ in range(n_periods):
        next_price = (1 - true_phi) * true_mu + true_phi * prices[-1] + true_sigma * gauss(0, 1)
        prices.append(next_price)
    
    price_series = pd.Series(prices[1:])  # Remove initial value
    # Simple moving average as proxy for equilibrium level
    moving_avg = price_series.rolling(window=20).mean()
    
    # Estimate parameters
    phi_est, sigma_est, mu_est = estimate_ou_parameters(price_series, moving_avg)
    
    # Run barrier optimization
    print("\nRunning barrier optimization...")
    results = run_triple_barrier_simulation(
        phi_est, sigma_est, mu_est,
        forecast_horizon_return=0.01,  # 1% expected return
        nIter=5000,
        pt_levels=np.linspace(0.5, 3.0, 6),
        sl_levels=np.linspace(0.5, 3.0, 6)
    )
    
    # Find optimal barriers
    optimal = find_optimal_barriers(results)
    
    return results, optimal


if __name__ == "__main__":
    results, optimal = example_usage()