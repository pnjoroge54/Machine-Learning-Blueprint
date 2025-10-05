import numpy as np
import pandas as pd
from numba import njit, prange
import numba as nb
from dataclasses import dataclass
from typing import Optional,Tuple
import statsmodels.api as sm


from ..util.volatility import get_period_vol


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



# Numba-optimized single path simulation
@njit(cache=True)
def simulate_single_path_numba(start_log_price: float, steps: int,
                             mu: float, sigma: float, phi: float, 
                             intercept: float, jump_lambda: float,
                             jump_mu: float, jump_sigma: float,
                             use_ar1: bool, use_jumps: bool,
                             seed: int) -> np.ndarray:
    """Numba-optimized single path simulation"""
    np.random.seed(seed)
    logp = np.empty(steps + 1, dtype=np.float64)
    logp[0] = start_log_price
    prev_ret = mu
    
    for t in range(1, steps + 1):
        # Base innovation
        eps = np.random.normal(0.0, sigma)
        
        # AR(1) dynamics
        if use_ar1:
            ret = intercept + phi * prev_ret + eps
        else:
            ret = mu + eps
        
        # Jump component
        if use_jumps and np.random.random() < jump_lambda:
            jump = np.random.normal(jump_mu, jump_sigma)
            ret += jump
        
        logp[t] = logp[t-1] + ret
        prev_ret = ret
    
    return np.exp(logp)

def simulate_multiple_paths_numba(start_price: float, steps: int,
                                params: RGPParams, n_paths: int = 1000,
                                random_state: Optional[int] = None) -> np.ndarray:
    """Multiple paths using Numba optimization"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Extract parameters and flags
    use_ar1 = params.phi is not None and params.intercept is not None
    use_jumps = (params.jump_lambda is not None and 
                params.jump_lambda > 0 and 
                params.jump_mu is not None)
    
    phi = params.phi if use_ar1 else 0.0
    intercept = params.intercept if use_ar1 else 0.0
    jump_lambda = params.jump_lambda if use_jumps else 0.0
    jump_mu = params.jump_mu if use_jumps else 0.0
    jump_sigma = params.jump_sigma if use_jumps else 0.0
    
    start_log_price = np.log(start_price)
    paths = np.empty((n_paths, steps + 1), dtype=np.float64)
    
    for i in range(n_paths):
        seed = np.random.randint(0, 2**31 - 1)  # Different seed for each path
        paths[i] = simulate_single_path_numba(
            start_log_price, steps, params.mu, params.sigma,
            phi, intercept, jump_lambda, jump_mu, jump_sigma,
            use_ar1, use_jumps, seed
        )
    
    return paths


@njit(parallel=True, cache=True)
def simulate_all_paths_numba_parallel(start_log_price: float, steps: int,
                                    mu: float, sigma: float, phi: float,
                                    intercept: float, jump_lambda: float,
                                    jump_mu: float, jump_sigma: float,
                                    use_ar1: bool, use_jumps: bool,
                                    n_paths: int) -> np.ndarray:
    """Ultra-fast version using Numba's parallel capabilities"""
    paths = np.empty((n_paths, steps + 1), dtype=np.float64)
    
    # Numba's parallel loop - each iteration can run on different cores
    for i in prange(n_paths):
        # Each thread gets its own random state
        seed = i  # Simple seeding strategy
        np.random.seed(seed)
        
        logp = np.empty(steps + 1, dtype=np.float64)
        logp[0] = start_log_price
        prev_ret = mu
        
        for t in range(1, steps + 1):
            eps = np.random.normal(0.0, sigma)
            
            if use_ar1:
                ret = intercept + phi * prev_ret + eps
            else:
                ret = mu + eps
            
            if use_jumps and np.random.random() < jump_lambda:
                jump = np.random.normal(jump_mu, jump_sigma)
                ret += jump
            
            logp[t] = logp[t-1] + ret
            prev_ret = ret
        
        paths[i] = np.exp(logp)
    
    return paths


def simulate_multiple_paths_ultra(start_price: float, steps: int,
                                params: RGPParams, n_paths: int = 1000) -> np.ndarray:
    """Ultra-fast combined approach"""
    use_ar1 = params.phi is not None and params.intercept is not None
    use_jumps = (params.jump_lambda is not None and 
                params.jump_lambda > 0 and 
                params.jump_mu is not None)
    
    phi = params.phi if use_ar1 else 0.0
    intercept = params.intercept if use_ar1 else 0.0
    jump_lambda = params.jump_lambda if use_jumps else 0.0
    jump_mu = params.jump_mu if use_jumps else 0.0
    jump_sigma = params.jump_sigma if use_jumps else 0.0
    
    start_log_price = np.log(start_price)
    
    return simulate_all_paths_numba_parallel(
        start_log_price, steps, params.mu, params.sigma,
        phi, intercept, jump_lambda, jump_mu, jump_sigma,
        use_ar1, use_jumps, n_paths
    )


from typing import Sequence, Tuple, Callable, Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger
import mlflow
from mlfinlab.cross_validation import PurgedWalkForward
from mlfinlab.util.multiprocess import mp_pandas_obj

def calculate_volatility(price_series: pd.Series, 
                        lookback_period: int = 20,
                        method: str = 'returns') -> pd.Series:
    """
    Calculate volatility of price series using different methods.
    
    Args:
        price_series: Series of prices
        lookback_period: Period for volatility calculation
        method: 'returns' for standard deviation of returns, 
                'atr' for average true range,
                'ewma' for exponentially weighted moving average
    
    Returns:
        Series of volatility values
    """
    if method == 'daily':
        volatility = get_period_vol(price_series, days=lookback_period)
    elif method == 'hourly':
        volatility = get_period_vol(price_series, hours=lookback_period)
    else:
        raise ValueError("Method must be 'daily' or 'hourly'")
    
    return volatility


def run_trade_path_on_series_volatility(
    price_array: np.ndarray,
    volatility_array: np.ndarray,
    entry_idx: int,
    pt_multiple: float, 
    sl_multiple: float, 
    max_holding: int
) -> Tuple[float, int, str]:
    """
    Run one trade with volatility-adjusted barriers.
    
    Args:
        price_array: Array of prices
        volatility_array: Array of volatility values at each point
        entry_idx: Entry index
        pt_multiple: Profit-taking multiple of volatility (e.g., 1.5 means 1.5 * volatility)
        sl_multiple: Stop-loss multiple of volatility (e.g., 0.5 means 0.5 * volatility)
        max_holding: Maximum holding period
    
    Returns:
        Tuple of (return, holding_period, exit_reason)
    """
    entry_price = price_array[entry_idx]
    entry_volatility = volatility_array[entry_idx]
    
    # Calculate absolute barrier levels based on volatility
    pt_level = entry_price * (1 + pt_multiple * entry_volatility)
    sl_level = entry_price * (1 - sl_multiple * entry_volatility)
    
    n = len(price_array)
    
    for h in range(1, max_holding + 1):
        t = entry_idx + h
        if t >= n:
            exit_price = price_array[-1]
            ret = (exit_price / entry_price) - 1.0
            return ret, h, 'last'
        
        current_price = price_array[t]
        
        # Check profit-taking barrier
        if current_price >= pt_level:
            ret = (pt_level / entry_price) - 1.0
            return ret, h, 'tp'
        
        # Check stop-loss barrier  
        if current_price <= sl_level:
            ret = (sl_level / entry_price) - 1.0
            return ret, h, 'sl'
    
    # Max holding period reached
    exit_price = price_array[min(entry_idx + max_holding, n-1)]
    ret = (exit_price / entry_price) - 1.0
    return ret, max_holding, 'max_hold'


def validate_rule_oos_volatility(
    prices: pd.Series,
    entry_times: Sequence[pd.Timestamp],
    rule_search_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]],
    volatility_lookback: int = 20,
    volatility_method: str = 'daily',
    pt_sl_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_holding: int = 50,
    n_splits: int = 5,
    purge_window: int = 10,
    sides: pd.Series = None,
    experiment_name: str = "volatility_barrier_validation"
) -> Dict:
    """
    Chronological cross-validation with volatility-adjusted barriers.
    
    Args:
        prices: pd.Series of prices indexed by time
        entry_times: Sequence of entry timestamps
        rule_search_fn: Function that takes (price_array, volatility_array, train_entries) 
                       and returns (best_pt_multiple, best_sl_multiple)
        volatility_lookback: Lookback period for volatility calculation
        volatility_method: Method for volatility calculation ('daily', 'hourly')
        pt_sl_grid: Optional grid of (pt_multiples, sl_multiples)
        max_holding: Maximum holding period
        n_splits: Number of CV folds
        purge_window: Purge window for avoiding data leakage,
        sides: Predicted direction of trades
        experiment_name: MLflow experiment name
    
    Returns:
        Dictionary with validation results
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "volatility_lookback": volatility_lookback,
            "volatility_method": volatility_method,
            "max_holding": max_holding,
            "n_splits": n_splits,
            "purge_window": purge_window
        })
        
        if isinstance(prices, pd.Series):
            price_arr = prices.values
            index = prices.index
        else:
            price_arr = np.asarray(prices)
            index = pd.Index(range(len(price_arr)))
        
        # Calculate volatility series
        price_series = pd.Series(price_arr, index=index)
        volatility_series = calculate_volatility(
            price_series, 
            lookback_period=volatility_lookback,
            method=volatility_method
        ).reindex(index).fillna(method='bfill')
        
        volatility_arr = volatility_series.values
        
        logger.info(f"Volatility stats - Mean: {volatility_arr.mean():.6f}, "
                   f"Std: {volatility_arr.std():.6f}, "
                   f"Min: {volatility_arr.min():.6f}, "
                   f"Max: {volatility_arr.max():.6f}")
        
        # Map entry_times to integer indices
        entry_idx_all = np.array([index.get_loc(t) for t in entry_times], dtype=int)
        entry_idx_all = np.sort(entry_idx_all)
        
        # Remove entries where volatility is not available
        valid_entries = entry_idx_all[entry_idx_all >= volatility_lookback]
        if len(valid_entries) < len(entry_idx_all):
            logger.warning(f"Removed {len(entry_idx_all) - len(valid_entries)} entries due to insufficient volatility data")
        
        entry_idx_all = valid_entries
        
        # Use MLFinLab's PurgedWalkForward
        cv = PurgedWalkForward(
            n_splits=n_splits,
            min_train_size=len(entry_idx_all) // (n_splits + 1),
            test_size=len(entry_idx_all) // n_splits,
            purge_window=purge_window
        )
        
        results = {'folds': [], 'aggregate_test_metrics': {}}
        all_test_trades = []
        
        splits = list(cv.split(entry_idx_all))
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            with mlflow.start_run(nested=True, run_name=f"fold_{fold_idx}"):
                logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
                
                train_entries = entry_idx_all[train_indices]
                test_entries = entry_idx_all[test_indices]
                
                if len(train_entries) == 0:
                    logger.warning(f"Skipping fold {fold_idx} - no training data")
                    continue
                
                # In-sample search for optimal volatility multiples
                if pt_sl_grid is None:
                    best_pt_multiple, best_sl_multiple = rule_search_fn(
                        price_arr, volatility_arr, train_entries
                    )
                else:
                    best_pt_multiple, best_sl_multiple = rule_search_fn(
                        price_arr, volatility_arr, train_entries, pt_sl_grid
                    )
                
                logger.info(f"Fold {fold_idx} - Best PT multiple: {best_pt_multiple:.2f}, "
                           f"Best SL multiple: {best_sl_multiple:.2f}")
                
                # Evaluate on test set with volatility-adjusted barriers
                trades = []
                run_trade_path_fn = run_trade_path_on_series_volatility if sides is None else run_trade_path_with_direction
                for entry_idx in test_entries:
                    ret, holding, reason = run_trade_path_fn(
                        price_arr, volatility_arr, int(entry_idx),
                        best_pt_multiple, best_sl_multiple, max_holding
                    )
                    entry_vol = volatility_arr[entry_idx]
                    trades.append({
                        'entry_idx': int(entry_idx),
                        'entry_time': index[entry_idx],
                        'entry_price': price_arr[entry_idx],
                        'entry_volatility': entry_vol,
                        'pt_level': price_arr[entry_idx] * (1 + best_pt_multiple * entry_vol),
                        'sl_level': price_arr[entry_idx] * (1 - best_sl_multiple * entry_vol),
                        'return': float(ret),
                        'holding': int(holding),
                        'exit_reason': reason,
                        'fold': fold_idx,
                        'pt_multiple': best_pt_multiple,
                        'sl_multiple': best_sl_multiple
                    })
                
                trades_df = pd.DataFrame(trades)
                all_test_trades.extend(trades)
                
                # Calculate metrics
                test_metrics = calculate_trading_metrics(trades_df)
                
                # Log fold results
                mlflow.log_params({
                    "fold_pt_multiple": best_pt_multiple,
                    "fold_sl_multiple": best_sl_multiple
                })
                mlflow.log_metrics(test_metrics)
                
                if len(trades_df) > 0:
                    trades_df.to_csv(f"fold_{fold_idx}_trades.csv", index=False)
                    mlflow.log_artifact(f"fold_{fold_idx}_trades.csv")
                
                results['folds'].append({
                    'fold_index': fold_idx,
                    'train_entries_range': (train_entries[0], train_entries[-1]),
                    'test_entries_range': (test_entries[0], test_entries[-1]),
                    'chosen_rule': {
                        'pt_multiple': float(best_pt_multiple), 
                        'sl_multiple': float(best_sl_multiple)
                    },
                    'test_metrics': test_metrics,
                    'trades_df': trades_df
                })
                
                logger.success(f"Fold {fold_idx} completed: {len(trades_df)} trades, "
                              f"Sharpe: {test_metrics['sharpe_ratio']:.4f}")
        
        # Aggregate results
        all_trades_df = pd.DataFrame(all_test_trades)
        aggregate_metrics = calculate_trading_metrics(all_trades_df)
        
        # Log aggregate metrics with volatility context
        mlflow.log_metrics({f"aggregate_{k}": v for k, v in aggregate_metrics.items()})
        
        # Log volatility statistics of trades
        if len(all_trades_df) > 0:
            vol_stats = {
                "mean_entry_volatility": all_trades_df['entry_volatility'].mean(),
                "std_entry_volatility": all_trades_df['entry_volatility'].std(),
                "min_entry_volatility": all_trades_df['entry_volatility'].min(),
                "max_entry_volatility": all_trades_df['entry_volatility'].max()
            }
            mlflow.log_metrics(vol_stats)
            
            all_trades_df.to_csv("all_trades.csv", index=False)
            mlflow.log_artifact("all_trades.csv")
        
        results['aggregate_test_metrics'] = aggregate_metrics
        results['all_trades_df'] = all_trades_df
        results['volatility_stats'] = vol_stats if len(all_trades_df) > 0 else {}
        
        logger.success(f"Validation completed: {len(all_trades_df)} total trades, "
                      f"Aggregate Sharpe: {aggregate_metrics['sharpe_ratio']:.4f}")
        
        return results


def calculate_trading_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive trading metrics for volatility-adjusted strategy"""
    if len(trades_df) == 0:
        return {
            'n_trades': 0, 'mean_return': float('nan'), 'std_return': float('nan'), 
            'sharpe_ratio': float('nan'), 'win_rate': float('nan'),
            'avg_holding_period': float('nan'), 'max_drawdown': float('nan'),
            'profit_factor': float('nan'), 'avg_pt_multiple': float('nan'),
            'avg_sl_multiple': float('nan')
        }
    
    returns = trades_df['return']
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    mean_return = returns.mean()
    std_return = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_holding_period = trades_df['holding'].mean()
    
    # Calculate cumulative returns for drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Volatility multiple stats
    avg_pt_multiple = trades_df['pt_multiple'].mean() if 'pt_multiple' in trades_df.columns else float('nan')
    avg_sl_multiple = trades_df['sl_multiple'].mean() if 'sl_multiple' in trades_df.columns else float('nan')
    
    return {
        'n_trades': len(trades_df),
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'sharpe_ratio': float(sharpe_ratio),
        'win_rate': float(win_rate),
        'avg_holding_period': float(avg_holding_period),
        'max_drawdown': float(max_drawdown),
        'profit_factor': float(profit_factor),
        'avg_pt_multiple': float(avg_pt_multiple),
        'avg_sl_multiple': float(avg_sl_multiple)
    }


# Example volatility-aware rule search function
def volatility_rule_search_fn(
    price_array: np.ndarray, 
    volatility_array: np.ndarray,
    train_entries: np.ndarray,
    pt_sl_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[float, float]:
    """
    Example rule search function that finds optimal volatility multiples.
    """
    logger.info(f"Searching for optimal volatility multiples on {len(train_entries)} training entries")
    
    # Define parameter grid for volatility multiples
    if pt_sl_grid is None:
        pt_multiples = sl_multiples = np.linspace(0.5, 10.0, 20)  # PT: 0.5x to 10.0x volatility
    else:
        pt_multiples, sl_multiples = pt_sl_grid
    
    best_sharpe = -np.inf
    best_pt, best_sl = 0.0, 0.0
    
    for pt in pt_multiples:
        for sl in sl_multiples:
            fold_returns = []
            for entry_idx in train_entries:
                ret, _, _ = run_trade_path_on_series_volatility(
                    price_array, volatility_array, entry_idx, pt, sl, 50
                )
                fold_returns.append(ret)
            
            if len(fold_returns) > 1:
                sharpe = np.mean(fold_returns) / np.std(fold_returns) if np.std(fold_returns) > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_pt, best_sl = pt, sl
    
    logger.info(f"Optimal multiples found: PT={best_pt:.2f}x vol, SL={best_sl:.2f}x vol, Sharpe={best_sharpe:.4f}")
    return best_pt, best_sl


def run_trade_path_with_direction(
    price_array: np.ndarray,
    volatility_array: np.ndarray,
    entry_idx: int,
    direction: int,  # +1 for long, -1 for short
    pt_multiple: float,
    sl_multiple: float,
    max_holding: int
) -> Tuple[float, int, str]:
    """
    Run one trade with direction-aware volatility barriers.
    """
    entry_price = price_array[entry_idx]
    entry_volatility = volatility_array[entry_idx]
    
    # Direction-aware barrier levels
    if direction == 1:  # Long trade
        pt_level = entry_price * (1 + pt_multiple * entry_volatility)
        sl_level = entry_price * (1 - sl_multiple * entry_volatility)
    else:  # Short trade
        pt_level = entry_price * (1 - pt_multiple * entry_volatility)  # Profit if price goes down
        sl_level = entry_price * (1 + sl_multiple * entry_volatility)  # Stop loss if price goes up
    
    n = len(price_array)
    
    for h in range(1, max_holding + 1):
        t = entry_idx + h
        if t >= n:
            exit_price = price_array[-1]
            ret = (exit_price / entry_price - 1.0) * direction
            return ret, h, 'last'
        
        current_price = price_array[t]
        
        if direction == 1:  # Long
            if current_price >= pt_level:
                ret = (pt_level / entry_price) - 1.0
                return ret, h, 'tp'
            if current_price <= sl_level:
                ret = (sl_level / entry_price) - 1.0
                return ret, h, 'sl'
        else:  # Short
            if current_price <= pt_level:
                ret = (entry_price / pt_level) - 1.0  # Inverse for short
                return ret, h, 'tp'
            if current_price >= sl_level:
                ret = (entry_price / sl_level) - 1.0  # Inverse for short
                return ret, h, 'sl'
    
    # Max holding period reached
    exit_price = price_array[min(entry_idx + max_holding, n-1)]
    ret = (exit_price / entry_price - 1.0) * direction
    return ret, max_holding, 'max_hold'


def validate_trend_strategy_oos(
    prices: pd.Series,
    entry_signal_fn: Callable[[pd.Series], pd.Series],  # Function that generates direction signals
    rule_search_fn: Callable,
    volatility_lookback: int = 20,
    max_holding: int = 50,
    n_splits: int = 5,
    purge_window: int = 10
) -> Dict:
    """
    Validate trend-following strategy with directional trading.
    """
    # Generate direction signals
    signals = entry_signal_fn(prices)
    entry_times = get_entry_times_from_signals(signals)
    
    if isinstance(prices, pd.Series):
        price_arr = prices.values
        signal_arr = signals.values
        index = prices.index
    else:
        price_arr = np.asarray(prices)
        signal_arr = np.asarray(signals)
        index = pd.Index(range(len(price_arr)))
    
    # Calculate volatility
    price_series = pd.Series(price_arr, index=index)
    volatility_series = calculate_volatility(price_series, volatility_lookback)
    volatility_arr = volatility_series.fillna(method='bfill').values
    
    # Map entry times to indices
    entry_idx_all = np.array([index.get_loc(t) for t in entry_times], dtype=int)
    entry_idx_all = np.sort(entry_idx_all)
    
    # Use MLFinLab cross-validation
    cv = PurgedWalkForward(
        n_splits=n_splits,
        min_train_size=len(entry_idx_all) // (n_splits + 1),
        test_size=len(entry_idx_all) // n_splits,
        purge_window=purge_window
    )
    
    results = {'folds': [], 'aggregate_test_metrics': {}}
    all_test_trades = []
    
    splits = list(cv.split(entry_idx_all))
    
    for fold_idx, (train_indices, test_indices) in enumerate(splits):
        with mlflow.start_run(nested=True, run_name=f"trend_fold_{fold_idx}"):
            train_entries = entry_idx_all[train_indices]
            test_entries = entry_idx_all[test_indices]
            
            if len(train_entries) == 0:
                continue
            
            # Find optimal barriers for trend strategy
            best_pt, best_sl = rule_search_fn(
                price_arr, volatility_arr, train_entries, signal_arr
            )
            
            # Test on out-of-sample data
            trades = []
            for entry_idx in test_entries:
                direction = signal_arr[entry_idx]
                ret, holding, reason = run_trade_path_with_direction(
                    price_arr, volatility_arr, entry_idx, 
                    direction, best_pt, best_sl, max_holding
                )
                
                trades.append({
                    'entry_idx': int(entry_idx),
                    'entry_time': index[entry_idx],
                    'direction': int(direction),
                    'return': float(ret),
                    'holding': int(holding),
                    'exit_reason': reason,
                    'fold': fold_idx,
                    'pt_multiple': best_pt,
                    'sl_multiple': best_sl
                })
            
            trades_df = pd.DataFrame(trades)
            all_test_trades.extend(trades)
            
            # Calculate directional metrics
            test_metrics = calculate_directional_metrics(trades_df)
            
            results['folds'].append({
                'fold_index': fold_idx,
                'chosen_rule': {'pt_multiple': best_pt, 'sl_multiple': best_sl},
                'test_metrics': test_metrics,
                'trades_df': trades_df
            })
    
    # Aggregate results
    all_trades_df = pd.DataFrame(all_test_trades)
    aggregate_metrics = calculate_directional_metrics(all_trades_df)
    results['aggregate_test_metrics'] = aggregate_metrics
    results['all_trades_df'] = all_trades_df
    
    return results


def calculate_directional_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate metrics for directional trading strategy"""
    if len(trades_df) == 0:
        return {'n_trades': 0, 'long_win_rate': 0, 'short_win_rate': 0, 'overall_sharpe': 0}
    
    long_trades = trades_df[trades_df['direction'] == 1]
    short_trades = trades_df[trades_df['direction'] == -1]
    
    long_win_rate = (long_trades['return'] > 0).mean() if len(long_trades) > 0 else 0
    short_win_rate = (short_trades['return'] > 0).mean() if len(short_trades) > 0 else 0
    
    overall_returns = trades_df['return']
    overall_sharpe = overall_returns.mean() / overall_returns.std() if overall_returns.std() > 0 else 0
    
    return {
        'n_trades': len(trades_df),
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'overall_sharpe': overall_sharpe,
        'mean_return_long': long_trades['return'].mean() if len(long_trades) > 0 else 0,
        'mean_return_short': short_trades['return'].mean() if len(short_trades) > 0 else 0
    }
