import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class SignalConditionalBarrierOptimizer:
    """
    Optimizes barriers only on periods where trading signals are present,
    with dynamic volatility estimation methods.
    """
    
    def __init__(self, price_series, signal_series, returns_series=None):
        """
        Args:
            price_series (pd.Series): Historical prices
            signal_series (pd.Series): Binary/categorical signals (1=long, -1=short, 0=no signal)
            returns_series (pd.Series): Optional pre-calculated returns
        """
        self.prices = price_series
        self.signals = signal_series
        self.returns = returns_series if returns_series is not None else price_series.pct_change()
        
        # Align all series
        self.data = pd.DataFrame({
            'price': self.prices,
            'signal': self.signals,
            'returns': self.returns
        }).dropna()
        
        # Extract signal periods only
        self.signal_periods = self.data[self.data['signal'] != 0].copy()
        
        print(f"Total observations: {len(self.data)}")
        print(f"Signal periods: {len(self.signal_periods)} ({len(self.signal_periods)/len(self.data)*100:.1f}%)")
    
    def estimate_volatility_methods(self, lookback_windows=[20, 50, 100]):
        """
        Compare different volatility estimation methods and select the best one.
        
        Methods compared:
        1. Simple rolling standard deviation
        2. EWMA (Exponentially Weighted Moving Average)
        3. GARCH-like volatility
        4. Realized volatility (if high-frequency data)
        5. Yang-Zhang estimator (if OHLC data available)
        """
        vol_methods = {}
        
        for window in lookback_windows:
            # Method 1: Simple rolling volatility
            vol_methods[f'rolling_std_{window}'] = (
                self.data['returns'].rolling(window=window).std() * np.sqrt(252)
            )
            
            # Method 2: EWMA volatility
            vol_methods[f'ewma_{window}'] = (
                self.data['returns'].ewm(span=window).std() * np.sqrt(252)
            )
            
            # Method 3: Simple realized volatility (sum of squared returns)
            vol_methods[f'realized_{window}'] = np.sqrt(
                self.data['returns'].rolling(window=window).apply(
                    lambda x: np.sum(x**2) * 252
                )
            )
        
        # Method 4: GARCH(1,1) approximation using rolling estimates
        if len(self.data) > 100:
            vol_methods['garch_approx'] = self._estimate_garch_volatility()
        
        return vol_methods
    
    def _estimate_garch_volatility(self, window=100):
        """
        Simple GARCH(1,1) approximation using rolling window estimation.
        """
        returns = self.data['returns'].dropna()
        garch_vol = pd.Series(index=returns.index, dtype=float)
        
        # Initial volatility estimate
        initial_vol = returns.std() * np.sqrt(252)
        
        # Rolling GARCH parameters estimation
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            # Simple GARCH approximation: σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
            # Use rough estimates: ω=0.01, α=0.1, β=0.85
            if i == window:
                vol_estimate = initial_vol
            else:
                prev_vol = garch_vol.iloc[i-1] if not pd.isna(garch_vol.iloc[i-1]) else initial_vol
                prev_return = returns.iloc[i-1]
                
                # GARCH update
                omega = 0.01 / 252  # Small constant
                alpha = 0.1  # React to recent shocks
                beta = 0.85   # Persistence
                
                vol_squared = omega + alpha * prev_return**2 + beta * (prev_vol/np.sqrt(252))**2
                vol_estimate = np.sqrt(vol_squared * 252)
            
            garch_vol.iloc[i] = vol_estimate
        
        return garch_vol
    
    def optimize_volatility_method(self, vol_methods, forecast_horizon=5):
        """
        Select the best volatility method based on out-of-sample forecasting performance.
        """
        print("Optimizing volatility estimation method...")
        
        # Only use signal periods for evaluation
        signal_data = self.signal_periods.copy()
        
        if len(signal_data) < 50:
            print("Warning: Too few signal periods for robust volatility optimization")
            return 'rolling_std_20'  # Default fallback
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=max(10, len(signal_data)//5))
        
        method_scores = {}
        
        for method_name, vol_series in vol_methods.items():
            if vol_series is None or vol_series.isna().all():
                continue
                
            scores = []
            
            # Align volatility with signal periods
            signal_vol = vol_series.reindex(signal_data.index).dropna()
            
            if len(signal_vol) < 20:
                continue
            
            for train_idx, test_idx in tscv.split(signal_vol):
                try:
                    train_vol = signal_vol.iloc[train_idx]
                    test_vol = signal_vol.iloc[test_idx]
                    
                    if len(train_vol) == 0 or len(test_vol) == 0:
                        continue
                    
                    # Use simple persistence model for volatility forecasting
                    # Forecast = last observed volatility
                    vol_forecast = train_vol.iloc[-1]
                    
                    # Calculate forecast errors
                    forecast_errors = (test_vol - vol_forecast) ** 2
                    mse = np.mean(forecast_errors)
                    scores.append(mse)
                
                except Exception as e:
                    print(f"Error evaluating {method_name}: {e}")
                    continue
            
            if scores:
                method_scores[method_name] = np.mean(scores)
        
        if not method_scores:
            print("Could not evaluate volatility methods. Using default.")
            return 'rolling_std_20'
        
        # Select method with lowest MSE
        best_method = min(method_scores.items(), key=lambda x: x[1])
        
        print(f"\nVolatility Method Performance:")
        for method, score in sorted(method_scores.items(), key=lambda x: x[1]):
            print(f"  {method}: MSE = {score:.6f}")
        
        print(f"\nBest volatility method: {best_method[0]}")
        return best_method[0]
    
    def estimate_signal_conditional_ou_parameters(self, volatility_method='rolling_std_20', 
                                                 separate_by_signal_type=True):
        """
        Estimate O-U parameters separately for different signal types and using optimal volatility.
        """
        vol_methods = self.estimate_volatility_methods()
        
        if volatility_method not in vol_methods:
            print(f"Volatility method {volatility_method} not found. Using rolling_std_20")
            volatility_method = 'rolling_std_20'
        
        optimal_vol = vol_methods[volatility_method]
        
        # Add volatility to signal periods data
        self.signal_periods['volatility'] = optimal_vol.reindex(self.signal_periods.index)
        self.signal_periods = self.signal_periods.dropna()
        
        ou_params = {}
        
        if separate_by_signal_type:
            # Separate estimation for long and short signals
            signal_types = self.signal_periods['signal'].unique()
            
            for signal_type in signal_types:
                if signal_type == 0:  # Skip no-signal periods
                    continue
                    
                signal_subset = self.signal_periods[
                    self.signal_periods['signal'] == signal_type
                ].copy()
                
                if len(signal_subset) < 20:
                    print(f"Warning: Only {len(signal_subset)} observations for signal type {signal_type}")
                    continue
                
                # Calculate moving average for this signal type
                signal_subset['moving_avg'] = signal_subset['price'].rolling(window=20).mean()
                signal_subset = signal_subset.dropna()
                
                if len(signal_subset) < 10:
                    continue
                
                params = self._estimate_ou_params_with_vol(
                    signal_subset['price'], 
                    signal_subset['moving_avg'],
                    signal_subset['volatility']
                )
                
                ou_params[f'signal_{signal_type}'] = params
                print(f"\nSignal Type {signal_type} Parameters:")
                print(f"  Phi: {params['phi']:.4f}")
                print(f"  Sigma: {params['sigma']:.4f}")
                print(f"  Vol-adjusted Sigma: {params['sigma_vol_adj']:.4f}")
        
        else:
            # Combined estimation for all signal periods
            self.signal_periods['moving_avg'] = self.signal_periods['price'].rolling(window=20).mean()
            clean_data = self.signal_periods.dropna()
            
            params = self._estimate_ou_params_with_vol(
                clean_data['price'],
                clean_data['moving_avg'],
                clean_data['volatility']
            )
            
            ou_params['combined'] = params
        
        return ou_params, optimal_vol
    
    def _estimate_ou_params_with_vol(self, price_series, ma_series, vol_series):
        """
        Estimate O-U parameters accounting for time-varying volatility.
        """
        data = pd.DataFrame({
            'price': price_series,
            'ma': ma_series,
            'vol': vol_series
        }).dropna()
        
        if len(data) < 10:
            raise ValueError("Insufficient data for parameter estimation")
        
        # Method 1: Standard estimation
        mu_mean = data['ma'].mean()
        y = (data['price'] - mu_mean).dropna()
        y_lag = y.shift(1).dropna()
        y = y.loc[y_lag.index]
        
        model = sm.OLS(y, y_lag)
        results = model.fit()
        phi_standard = max(min(results.params.iloc[0], 0.999), 0.001)
        sigma_standard = results.resid.std()
        
        # Method 2: Volatility-adjusted estimation
        # Normalize by volatility to get more stable parameters
        data['normalized_price'] = (data['price'] - data['ma']) / data['vol']
        y_norm = data['normalized_price'].dropna()
        y_norm_lag = y_norm.shift(1).dropna()
        y_norm = y_norm.loc[y_norm_lag.index]
        
        if len(y_norm) > 5:
            model_vol = sm.OLS(y_norm, y_norm_lag)
            results_vol = model_vol.fit()
            phi_vol_adj = max(min(results_vol.params.iloc[0], 0.999), 0.001)
            sigma_vol_adj = results_vol.resid.std()
        else:
            phi_vol_adj = phi_standard
            sigma_vol_adj = sigma_standard / data['vol'].mean()
        
        return {
            'phi': phi_standard,
            'sigma': sigma_standard,
            'phi_vol_adj': phi_vol_adj,
            'sigma_vol_adj': sigma_vol_adj,
            'mu': mu_mean,
            'n_obs': len(data)
        }
    
    def optimize_barriers_by_signal_type(self, ou_params, vol_series, 
                                       nIter=10000, maxHP=100,
                                       pt_range=(0.5, 5.0, 10), 
                                       sl_range=(0.5, 5.0, 10)):
        """
        Optimize barriers separately for each signal type using their specific O-U parameters.
        """
        pt_levels = np.linspace(*pt_range)
        sl_levels = np.linspace(*sl_range)
        
        all_results = {}
        
        for signal_type, params in ou_params.items():
            print(f"\nOptimizing barriers for {signal_type}...")
            
            # Use volatility-adjusted parameters for more robust results
            phi = params['phi_vol_adj']
            sigma = params['sigma_vol_adj']
            
            # Get signal-specific periods for realistic drift estimation
            if 'signal_' in signal_type:
                signal_num = float(signal_type.split('_')[1])
                signal_subset = self.signal_periods[
                    self.signal_periods['signal'] == signal_num
                ]
                
                # Estimate forward-looking return expectation
                forward_returns = signal_subset['returns'].shift(-1)  # Next period return
                expected_return = forward_returns.mean()
                
            else:
                expected_return = self.signal_periods['returns'].shift(-1).mean()
            
            # Run simulation with signal-conditional parameters
            results = self._run_signal_simulation(
                phi, sigma, expected_return, pt_levels, sl_levels, 
                nIter, maxHP, signal_type
            )
            
            all_results[signal_type] = results
        
        return all_results
    
    def _run_signal_simulation(self, phi, sigma, expected_return, pt_levels, sl_levels,
                              nIter, maxHP, signal_type):
        """
        Run Monte Carlo simulation for a specific signal type.
        """
        drift_per_period = expected_return  # Already per-period
        results = []
        
        if NUMBA_AVAILABLE:
            # Use numba for speed
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                
                for pt in pt_levels:
                    for sl in sl_levels:
                        future = executor.submit(
                            self._simulate_single_combo_numba, 
                            phi, sigma, drift_per_period, maxHP, pt, sl, nIter
                        )
                        futures.append((future, pt, sl))
                
                for future, pt, sl in futures:
                    try:
                        result = future.result()
                        result.update({'pt_level': pt, 'sl_level': sl, 'signal_type': signal_type})
                        results.append(result)
                    except Exception as e:
                        print(f"Error simulating PT={pt}, SL={sl}: {e}")
        
        else:
            # Fallback to standard Python
            for pt in pt_levels:
                for sl in sl_levels:
                    result = self._simulate_single_combo_python(
                        phi, sigma, drift_per_period, maxHP, pt, sl, nIter
                    )
                    result.update({'pt_level': pt, 'sl_level': sl, 'signal_type': signal_type})
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _simulate_single_combo_numba(self, phi, sigma, drift, maxHP, pt, sl, nIter):
        """Numba-accelerated simulation helper"""
        if not NUMBA_AVAILABLE:
            return self._simulate_single_combo_python(phi, sigma, drift, maxHP, pt, sl, nIter)
        
        @jit(nopython=True, parallel=True)
        def simulate_paths(phi, sigma, drift, maxHP, pt, sl, nIter):
            pnl_results = np.zeros(nIter)
            
            for i in prange(nIter):
                p = 0.0
                for hp in range(maxHP):
                    p = phi * p + sigma * np.random.normal() + drift
                    if p >= pt or p <= -sl:
                        break
                pnl_results[i] = p
            
            return pnl_results
        
        pnl_results = simulate_paths(phi, sigma, drift, maxHP, pt, sl, int(nIter))
        
        return self._calculate_metrics(pnl_results)
    
    def _simulate_single_combo_python(self, phi, sigma, drift, maxHP, pt, sl, nIter):
        """Python fallback simulation"""
        pnl_results = []
        
        for _ in range(int(nIter)):
            p = 0.0
            for hp in range(maxHP):
                p = phi * p + sigma * np.random.normal() + drift
                if p >= pt or p <= -sl:
                    break
            pnl_results.append(p)
        
        return self._calculate_metrics(np.array(pnl_results))
    
    def _calculate_metrics(self, pnl_results):
        """Calculate performance metrics from PnL results"""
        mean_pnl = np.mean(pnl_results)
        std_pnl = np.std(pnl_results)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        hit_rate = np.mean(pnl_results > 0)
        
        profit_trades = pnl_results[pnl_results > 0]
        loss_trades = pnl_results[pnl_results < 0]
        
        if len(loss_trades) > 0 and len(profit_trades) > 0:
            profit_factor = np.sum(profit_trades) / abs(np.sum(loss_trades))
        else:
            profit_factor = float('inf') if len(profit_trades) > 0 else 0
        
        # Additional risk metrics
        max_loss = np.min(pnl_results)
        max_gain = np.max(pnl_results)
        var_95 = np.percentile(pnl_results, 5)  # 95% VaR
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'sharpe_ratio': sharpe,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'max_loss': max_loss,
            'max_gain': max_gain,
            'var_95': var_95,
            'num_trades': len(pnl_results)
        }
    
    def get_optimal_barriers_summary(self, all_results, metric='sharpe_ratio'):
        """
        Summarize optimal barriers for each signal type.
        """
        summary = {}
        
        for signal_type, results_df in all_results.items():
            if len(results_df) == 0:
                continue
                
            optimal_idx = results_df[metric].idxmax()
            optimal_result = results_df.loc[optimal_idx]
            
            summary[signal_type] = {
                'optimal_pt': optimal_result['pt_level'],
                'optimal_sl': optimal_result['sl_level'],
                'sharpe_ratio': optimal_result['sharpe_ratio'],
                'hit_rate': optimal_result['hit_rate'],
                'mean_pnl': optimal_result['mean_pnl'],
                'var_95': optimal_result['var_95']
            }
        
        return summary


# Example usage and demonstration
def demonstrate_signal_conditional_optimization():
    """
    Demonstrate the signal-conditional barrier optimization process.
    """
    # Generate synthetic data with realistic signal patterns
    np.random.seed(42)
    n_days = 1000
    
    # Base price series (geometric Brownian motion with mean reversion)
    prices = [100.0]
    for i in range(n_days):
        drift = 0.0001 - 0.01 * (prices[-1] - 100) / 100  # Mean reversion to 100
        shock = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + drift + shock)
        prices.append(new_price)
    
    price_series = pd.Series(prices[1:], index=pd.date_range('2022-01-01', periods=n_days))
    
    # Generate realistic trading signals
    # Signals based on price relative to moving average + some noise
    ma_20 = price_series.rolling(20).mean()
    price_deviation = (price_series - ma_20) / ma_20
    
    signals = np.zeros(len(price_series))
    
    # Long signals when price is significantly below MA (mean reversion opportunity)
    long_threshold = -0.02
    short_threshold = 0.02
    
    signals[price_deviation < long_threshold] = 1   # Long signal
    signals[price_deviation > short_threshold] = -1  # Short signal
    
    # Add some randomness and ensure signals don't change too frequently
    for i in range(1, len(signals)):
        if np.random.random() < 0.8:  # 80% chance to keep previous signal
            signals[i] = signals[i-1] if signals[i-1] != 0 else signals[i]
    
    signal_series = pd.Series(signals, index=price_series.index)
    
    print("Synthetic data generated:")
    print(f"Price range: {price_series.min():.2f} - {price_series.max():.2f}")
    print(f"Long signals: {np.sum(signals == 1)}")
    print(f"Short signals: {np.sum(signals == -1)}")
    print(f"No signal: {np.sum(signals == 0)}")
    
    # Initialize optimizer
    optimizer = SignalConditionalBarrierOptimizer(price_series, signal_series)
    
    # Step 1: Optimize volatility method
    vol_methods = optimizer.estimate_volatility_methods()
    best_vol_method = optimizer.optimize_volatility_method(vol_methods)
    
    # Step 2: Estimate signal-conditional O-U parameters
    ou_params, optimal_vol = optimizer.estimate_signal_conditional_ou_parameters(
        volatility_method=best_vol_method,
        separate_by_signal_type=True
    )
    
    # Step 3: Optimize barriers for each signal type
    barrier_results = optimizer.optimize_barriers_by_signal_type(
        ou_params, optimal_vol,
        nIter=5000,  # Reduced for demo speed
        pt_range=(0.5, 3.0, 6),
        sl_range=(0.5, 3.0, 6)
    )
    
    # Step 4: Get summary of optimal barriers
    optimal_summary = optimizer.get_optimal_barriers_summary(barrier_results)
    
    print("\n" + "="*50)
    print("OPTIMAL BARRIERS SUMMARY")
    print("="*50)
    
    for signal_type, summary in optimal_summary.items():
        print(f"\n{signal_type.upper()}:")
        print(f"  Optimal PT: {summary['optimal_pt']:.2f}")
        print(f"  Optimal SL: {summary['optimal_sl']:.2f}")
        print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
        print(f"  Hit Rate: {summary['hit_rate']:.3f}")
        print(f"  Mean PnL: {summary['mean_pnl']:.4f}")
        print(f"  95% VaR: {summary['var_95']:.4f}")
    
    return optimizer, barrier_results, optimal_summary


if __name__ == "__main__":
    # Run demonstration
    optimizer, results, summary = demonstrate_signal_conditional_optimization()