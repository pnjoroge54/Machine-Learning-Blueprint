import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import statsmodels.api as sm

@dataclass
class RGPParams:
    mu: float            # unconditional mean of log-returns
    sigma: float         # innovation std dev
    phi: Optional[float] # AR(1) coefficient
    intercept: Optional[float]  # AR(1) intercept
    jump_lambda: Optional[float]
    jump_mu: Optional[float] 
    jump_sigma: Optional[float]

def estimate_rgp_from_prices(prices: pd.Series,
                           fit_ar1: bool = True,
                           fit_jumps: bool = False,
                           min_samples: int = 100) -> RGPParams:
    """Improved estimation with better jump detection"""
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    
    lp = np.log(prices.dropna())
    r = lp.diff().dropna()
    n = len(r)
    
    if n < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {n}")
    
    # Basic parameters
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    phi = None
    intercept = None
    
    # AR(1) estimation
    if fit_ar1 and n >= 2:
        r_lag = r.shift(1).dropna()
        r_trunc = r.iloc[1:]  # align with lagged values
        X = sm.add_constant(r_lag.values)
        model = sm.OLS(r_trunc.values, X).fit()
        intercept = float(model.params[0])
        phi = float(model.params[1])
        # Recompute sigma from residuals
        sigma = float(model.resid.std(ddof=1))
    
    # Improved jump detection
    jump_lambda = jump_mu = jump_sigma = None
    if fit_jumps:
        # Use residual returns for jump detection if AR(1) was fitted
        if phi is not None:
            # Calculate residuals from AR(1)
            predicted = intercept + phi * r.shift(1)
            residuals = r - predicted
            returns_for_jumps = residuals.dropna()
        else:
            returns_for_jumps = r
        
        # More robust jump detection using median absolute deviation
        median_ret = returns_for_jumps.median()
        mad = (returns_for_jumps - median_ret).abs().median()
        threshold = 3 * 1.4826 * mad  # Convert MAD to approx std dev
        
        jumps = returns_for_jumps[np.abs(returns_for_jumps - median_ret) > threshold]
        k = len(jumps)
        jump_lambda = float(k / len(returns_for_jumps))
        
        if k > 1:
            jump_mu = float(jumps.mean())
            jump_sigma = float(jumps.std(ddof=1))
        else:
            jump_mu = 0.0
            jump_sigma = 0.0
    
    return RGPParams(mu=mu, sigma=sigma, phi=phi, intercept=intercept,
                   jump_lambda=jump_lambda, jump_mu=jump_mu, jump_sigma=jump_sigma)

def simulate_rgp_path(start_price: float,
                    steps: int,
                    params: RGPParams,
                    random_state: Optional[int] = None) -> np.ndarray:
    """Improved simulation with corrected AR(1) dynamics"""
    rng = np.random.default_rng(random_state)
    logp = np.empty(steps + 1, dtype=float)
    logp[0] = np.log(start_price)
    
    # Initialize previous return
    prev_ret = params.mu  # Start with unconditional mean
    
    for t in range(1, steps + 1):
        # Base innovation
        eps = rng.normal(0.0, params.sigma)
        
        # AR(1) dynamics
        if params.phi is not None and params.intercept is not None:
            ret = params.intercept + params.phi * prev_ret + eps
        else:
            # Simple random walk with drift
            ret = params.mu + eps
        
        # Jump component
        if (params.jump_lambda is not None and 
            params.jump_lambda > 0 and 
            rng.random() < params.jump_lambda):
            jump = rng.normal(params.jump_mu, params.jump_sigma)
            ret += jump
        
        logp[t] = logp[t-1] + ret
        prev_ret = ret
    
    return np.exp(logp)

def simulate_multiple_paths(start_price: float,
                          steps: int,
                          params: RGPParams,
                          n_paths: int = 1000,
                          random_state: Optional[int] = None) -> np.ndarray:
    """Simulate multiple paths for Monte Carlo analysis"""
    rng = np.random.default_rng(random_state)
    paths = np.empty((n_paths, steps + 1))
    
    for i in range(n_paths):
        paths[i] = simulate_rgp_path(start_price, steps, params, 
                                   random_state=rng.integers(0, 2**32))
    
    return paths