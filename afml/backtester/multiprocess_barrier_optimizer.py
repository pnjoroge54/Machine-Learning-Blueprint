import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import erf
import statsmodels.api as sm
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class StochasticProcess(ABC):
    """Base class for all stochastic processes"""
    
    @abstractmethod
    def estimate_parameters(self, price_series, **kwargs):
        """Estimate process parameters from historical data"""
        pass
    
    @abstractmethod
    def simulate_path(self, params, n_steps, dt=1.0):
        """Simulate a single price path"""
        pass
    
    @abstractmethod
    def get_process_name(self):
        """Return process name"""
        pass
    
    def goodness_of_fit(self, price_series, params):
        """Calculate goodness of fit metrics"""
        pass


class OrnsteinUhlenbeck(StochasticProcess):
    """
    Ornstein-Uhlenbeck (Mean Reversion) Process
    dX_t = θ(μ - X_t)dt + σdW_t
    
    USE WHEN:
    - Pairs trading / statistical arbitrage
    - Mean-reverting spreads
    - Interest rate models
    - Commodities with storage costs
    """
    
    def estimate_parameters(self, price_series, ma_series=None):
        if ma_series is None:
            ma_series = price_series.rolling(window=20).mean()
        
        data = pd.DataFrame({'price': price_series, 'ma': ma_series}).dropna()
        mu = data['ma'].mean()
        
        y = (data['price'] - mu).dropna()
        y_lag = y.shift(1).dropna()
        y = y.loc[y_lag.index]
        
        model = sm.OLS(y, y_lag)
        results = model.fit()
        
        phi = max(min(results.params.iloc[0], 0.999), 0.001)
        theta = -np.log(phi)
        sigma = results.resid.std()
        
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        return {
            'mu': mu,
            'theta': theta,
            'sigma': sigma,
            'phi': phi,
            'half_life': half_life,
            'process': 'Ornstein-Uhlenbeck'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        mu, theta, sigma, phi = params['mu'], params['theta'], params['sigma'], params['phi']
        
        path = np.zeros(n_steps + 1)
        path[0] = mu
        
        for t in range(n_steps):
            path[t + 1] = phi * path[t] + (1 - phi) * mu + sigma * np.random.normal()
        
        return path
    
    def get_process_name(self):
        return "Ornstein-Uhlenbeck (Mean Reversion)"


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion (Trending)
    dS_t = μS_t dt + σS_t dW_t
    
    USE WHEN:
    - Trend-following strategies
    - Stock prices (Black-Scholes assumption)
    - Index trading
    - Long-term equity strategies
    """
    
    def estimate_parameters(self, price_series):
        returns = price_series.pct_change().dropna()
        
        # Drift (μ) and volatility (σ)
        mu = returns.mean()
        sigma = returns.std()
        
        # Annualized metrics (assuming daily data)
        mu_annual = mu * 252
        sigma_annual = sigma * np.sqrt(252)
        
        # Sharpe ratio estimation
        sharpe = mu / sigma if sigma > 0 else 0
        
        return {
            'mu': mu,
            'sigma': sigma,
            'mu_annual': mu_annual,
            'sigma_annual': sigma_annual,
            'sharpe': sharpe,
            'process': 'Geometric Brownian Motion'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        mu, sigma = params['mu'], params['sigma']
        
        S0 = 100  # Starting price
        path = np.zeros(n_steps + 1)
        path[0] = S0
        
        for t in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t + 1] = path[t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
        return path
    
    def get_process_name(self):
        return "Geometric Brownian Motion (Trend)"


class JumpDiffusion(StochasticProcess):
    """
    Merton's Jump Diffusion Process
    dS_t = μS_t dt + σS_t dW_t + S_t dJ_t
    
    USE WHEN:
    - Event-driven strategies
    - Earnings announcements
    - News-based trading
    - Crisis/tail risk hedging
    - High volatility assets
    """
    
    def estimate_parameters(self, price_series, jump_threshold=2.5):
        returns = price_series.pct_change().dropna()
        
        # Identify jumps (returns beyond threshold standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        
        jump_indicator = np.abs(returns - mean_return) > jump_threshold * std_return
        jump_returns = returns[jump_indicator]
        normal_returns = returns[~jump_indicator]
        
        # Jump intensity (lambda)
        lambda_jump = len(jump_returns) / len(returns)
        
        # Jump magnitude parameters
        if len(jump_returns) > 0:
            jump_mean = jump_returns.mean()
            jump_std = jump_returns.std()
        else:
            jump_mean = 0
            jump_std = std_return
        
        # Continuous component parameters
        mu_continuous = normal_returns.mean() if len(normal_returns) > 0 else mean_return
        sigma_continuous = normal_returns.std() if len(normal_returns) > 0 else std_return
        
        return {
            'mu': mu_continuous,
            'sigma': sigma_continuous,
            'lambda_jump': lambda_jump,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'n_jumps_observed': len(jump_returns),
            'process': 'Jump Diffusion'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        mu = params['mu']
        sigma = params['sigma']
        lambda_jump = params['lambda_jump']
        jump_mean = params['jump_mean']
        jump_std = params['jump_std']
        
        S0 = 100
        path = np.zeros(n_steps + 1)
        path[0] = S0
        
        for t in range(n_steps):
            # Continuous component
            dW = np.random.normal(0, np.sqrt(dt))
            continuous_change = (mu - 0.5 * sigma**2) * dt + sigma * dW
            
            # Jump component
            jump_occurs = np.random.random() < lambda_jump * dt
            jump_size = np.random.normal(jump_mean, jump_std) if jump_occurs else 0
            
            path[t + 1] = path[t] * np.exp(continuous_change + jump_size)
        
        return path
    
    def get_process_name(self):
        return "Jump Diffusion (Events)"


class GARCH(StochasticProcess):
    """
    GARCH(1,1) Process with time-varying volatility
    r_t = σ_t * ε_t
    σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
    
    USE WHEN:
    - Volatility clustering present
    - Options trading
    - VIX-based strategies
    - Risk management
    - High-frequency trading
    """
    
    def estimate_parameters(self, price_series, initial_params=[0.01, 0.1, 0.85]):
        returns = price_series.pct_change().dropna().values
        
        def garch_likelihood(params):
            omega, alpha, beta = params
            
            # Constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)
            
            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            # Log-likelihood
            likelihood = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
            return -likelihood
        
        # Optimize
        result = optimize.minimize(
            garch_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-6, 1), (0, 1), (0, 1)]
        )
        
        omega, alpha, beta = result.x
        
        # Calculate unconditional variance
        uncond_var = omega / (1 - alpha - beta)
        
        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'unconditional_vol': np.sqrt(uncond_var),
            'persistence': alpha + beta,
            'process': 'GARCH(1,1)'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        omega = params['omega']
        alpha = params['alpha']
        beta = params['beta']
        
        S0 = 100
        path = np.zeros(n_steps + 1)
        path[0] = S0
        
        # Initialize variance
        sigma2 = omega / (1 - alpha - beta)
        
        for t in range(n_steps):
            # Generate return with time-varying volatility
            epsilon = np.random.normal()
            return_t = np.sqrt(sigma2) * epsilon
            
            path[t + 1] = path[t] * (1 + return_t)
            
            # Update variance
            sigma2 = omega + alpha * (return_t**2) + beta * sigma2
        
        return path
    
    def get_process_name(self):
        return "GARCH(1,1) (Vol Clustering)"


class HestonModel(StochasticProcess):
    """
    Heston Stochastic Volatility Model
    dS_t = μS_t dt + √v_t S_t dW^S_t
    dv_t = κ(θ - v_t)dt + σ_v√v_t dW^v_t
    
    USE WHEN:
    - Options pricing
    - Volatility surface modeling
    - Leverage effect important
    - Long-dated derivatives
    """
    
    def estimate_parameters(self, price_series, vol_series=None):
        """
        Simplified estimation. For production, use MLE or method of moments.
        """
        returns = price_series.pct_change().dropna()
        
        if vol_series is None:
            # Estimate realized volatility
            vol_series = returns.rolling(window=20).std() * np.sqrt(252)
        
        vol_data = vol_series.dropna()
        
        # Estimate volatility mean reversion
        v = vol_data ** 2  # Variance
        v_lag = v.shift(1).dropna()
        v = v.loc[v_lag.index]
        
        model = sm.OLS(v, sm.add_constant(v_lag))
        results = model.fit()
        
        intercept, phi = results.params
        theta = intercept / (1 - phi) if phi < 1 else v.mean()
        kappa = -np.log(phi) if phi > 0 and phi < 1 else 0.5
        
        # Volatility of volatility
        sigma_v = results.resid.std()
        
        # Drift
        mu = returns.mean()
        
        # Correlation (simplified)
        return_vol_corr = returns.corr(vol_series.pct_change())
        
        return {
            'mu': mu,
            'kappa': kappa,
            'theta': theta,
            'sigma_v': sigma_v,
            'rho': return_vol_corr,
            'v0': v.iloc[-1],
            'process': 'Heston'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        mu = params['mu']
        kappa = params['kappa']
        theta = params['theta']
        sigma_v = params['sigma_v']
        rho = params.get('rho', -0.5)
        v0 = params['v0']
        
        S0 = 100
        S = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        S[0] = S0
        v[0] = v0
        
        for t in range(n_steps):
            # Correlated Brownian motions
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
            
            # Variance process (with Feller condition check)
            v[t + 1] = max(v[t] + kappa * (theta - v[t]) * dt + sigma_v * np.sqrt(max(v[t], 0)) * dW2, 0)
            
            # Price process
            S[t + 1] = S[t] * np.exp((mu - 0.5 * v[t]) * dt + np.sqrt(max(v[t], 0)) * dW1)
        
        return S
    
    def get_process_name(self):
        return "Heston (Stochastic Vol)"


class CEVProcess(StochasticProcess):
    """
    Constant Elasticity of Variance (CEV) Model
    dS_t = μS_t dt + σS_t^γ dW_t
    
    USE WHEN:
    - Leverage effect modeling
    - Small-cap stocks
    - Commodities
    - Assets with volatility smile
    """
    
    def estimate_parameters(self, price_series, gamma_guess=1.5):
        returns = price_series.pct_change().dropna()
        prices = price_series.loc[returns.index].values
        returns_values = returns.values
        
        # Estimate gamma through regression
        # log(|returns|) ≈ constant + γ*log(price)
        log_abs_returns = np.log(np.abs(returns_values) + 1e-10)
        log_prices = np.log(prices)
        
        # Filter out infinities
        valid = np.isfinite(log_abs_returns) & np.isfinite(log_prices)
        
        if np.sum(valid) > 10:
            model = sm.OLS(log_abs_returns[valid], sm.add_constant(log_prices[valid]))
            results = model.fit()
            gamma = results.params[1]
        else:
            gamma = gamma_guess
        
        # Estimate drift and volatility coefficient
        mu = returns.mean()
        
        # σ from standardized returns
        std_returns = returns_values / (prices ** (gamma - 1))
        sigma = np.std(std_returns[np.isfinite(std_returns)])
        
        return {
            'mu': mu,
            'sigma': sigma,
            'gamma': gamma,
            'process': 'CEV'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        mu = params['mu']
        sigma = params['sigma']
        gamma = params['gamma']
        
        S0 = 100
        path = np.zeros(n_steps + 1)
        path[0] = S0
        
        for t in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S_t = path[t]
            
            # CEV dynamics
            drift = mu * S_t * dt
            diffusion = sigma * (S_t ** gamma) * dW
            
            path[t + 1] = max(S_t + drift + diffusion, 0.01)  # Prevent negative prices
        
        return path
    
    def get_process_name(self):
        return "CEV (Leverage Effect)"


class RegimeSwitching(StochasticProcess):
    """
    Markov Regime-Switching Model
    Two-state model with different dynamics in each regime
    
    USE WHEN:
    - Market cycles (bull/bear)
    - Crisis periods
    - Multiple strategy regimes
    - Adaptive strategies
    """
    
    def estimate_parameters(self, price_series, n_regimes=2):
        returns = price_series.pct_change().dropna()
        
        # Simple 2-regime estimation using threshold
        median_vol = returns.rolling(20).std().median()
        current_vol = returns.rolling(20).std()
        
        # Regime 1: Low volatility, Regime 2: High volatility
        regime = (current_vol > median_vol).astype(int)
        
        # Estimate parameters for each regime
        regime1_returns = returns[regime == 0]
        regime2_returns = returns[regime == 1]
        
        params_regime1 = {
            'mu': regime1_returns.mean() if len(regime1_returns) > 0 else 0,
            'sigma': regime1_returns.std() if len(regime1_returns) > 0 else 0.01
        }
        
        params_regime2 = {
            'mu': regime2_returns.mean() if len(regime2_returns) > 0 else 0,
            'sigma': regime2_returns.std() if len(regime2_returns) > 0 else 0.02
        }
        
        # Transition probabilities
        transitions = regime.diff().fillna(0)
        p11 = np.sum((regime[:-1] == 0) & (regime[1:] == 0)) / max(np.sum(regime[:-1] == 0), 1)
        p22 = np.sum((regime[:-1] == 1) & (regime[1:] == 1)) / max(np.sum(regime[:-1] == 1), 1)
        
        return {
            'regime1_mu': params_regime1['mu'],
            'regime1_sigma': params_regime1['sigma'],
            'regime2_mu': params_regime2['mu'],
            'regime2_sigma': params_regime2['sigma'],
            'p11': p11,  # Prob of staying in regime 1
            'p22': p22,  # Prob of staying in regime 2
            'initial_regime': int(regime.iloc[-1]),
            'process': 'Regime Switching'
        }
    
    def simulate_path(self, params, n_steps, dt=1.0):
        S0 = 100
        path = np.zeros(n_steps + 1)
        path[0] = S0
        
        regime = params['initial_regime']
        
        for t in range(n_steps):
            # Determine current regime
            if regime == 0:
                mu = params['regime1_mu']
                sigma = params['regime1_sigma']
                stay_prob = params['p11']
            else:
                mu = params['regime2_mu']
                sigma = params['regime2_sigma']
                stay_prob = params['p22']
            
            # Regime transition
            if np.random.random() > stay_prob:
                regime = 1 - regime
            
            # Generate return
            return_t = mu * dt + sigma * np.random.normal(0, np.sqrt(dt))
            path[t + 1] = path[t] * (1 + return_t)
        
        return path
    
    def get_process_name(self):
        return "Regime Switching (Cycles)"


class ProcessSelector:
    """
    Automatically select the best stochastic process for given data
    """
    
    def __init__(self):
        self.processes = [
            OrnsteinUhlenbeck(),
            GeometricBrownianMotion(),
            JumpDiffusion(),
            GARCH(),
            HestonModel(),
            CEVProcess(),
            RegimeSwitching()
        ]
    
    def test_all_processes(self, price_series):
        """
        Test all processes and rank by goodness of fit
        """
        results = []
        
        for process in self.processes:
            try:
                print(f"Testing {process.get_process_name()}...")
                params = process.estimate_parameters(price_series)
                
                # Calculate goodness of fit metrics
                metrics = self._calculate_fit_metrics(price_series, process, params)
                
                results.append({
                    'process': process.get_process_name(),
                    'params': params,
                    'metrics': metrics,
                    'process_obj': process
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Sort by AIC (lower is better)
        results.sort(key=lambda x: x['metrics']['aic'])
        
        return results
    
    def _calculate_fit_metrics(self, price_series, process, params, n_simulations=100):
        """
        Calculate AIC, BIC, and other fit metrics
        """
        returns = price_series.pct_change().dropna()
        
        # Simulate and compare distributions
        simulated_returns = []
        
        for _ in range(n_simulations):
            sim_path = process.simulate_path(params, len(returns))
            sim_returns = pd.Series(sim_path).pct_change().dropna()
            simulated_returns.extend(sim_returns.values)
        
        # KS test
        ks_stat, ks_pval = stats.ks_2samp(returns.values, simulated_returns)
        
        # Calculate log-likelihood (approximation)
        n_params = len([k for k, v in params.items() if isinstance(v, (int, float))])
        n_obs = len(returns)
        
        # Simple Gaussian likelihood for comparison
        log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + 2 * np.log(returns.std()) + 1)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood
        
        return {
            'aic': aic,
            'bic': bic,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'n_parameters': n_params
        }
    
    def recommend_process(self, price_series, strategy_type=None):
        """
        Recommend best process based on data and strategy type
        """
        results = self.test_all_processes(price_series)
        
        print("\n" + "="*60)
        print("PROCESS SELECTION RESULTS")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['process']}")
            print(f"   AIC: {result['metrics']['aic']:.2f}")
            print(f"   BIC: {result['metrics']['bic']:.2f}")
            print(f"   KS p-value: {result['metrics']['ks_pvalue']:.4f}")
        
        best_process = results[0]
        print(f"\n{'='*60}")
        print(f"RECOMMENDED: {best_process['process']}")
        print(f"{'='*60}")
        
        return best_process


# Example usage
def demonstrate_process_selection():
    """
    Demonstrate automatic process selection on synthetic data
    """
    np.random.seed(42)
    
    # Generate data with regime switching characteristics
    n_days = 500
    prices = [100]
    
    for i in range(n_days):
        # Simulate regime switching
        if i < 200:  # Low vol regime
            ret = np.random.normal(0.0005, 0.01)
        else:  # High vol regime with drift
            ret = np.random.normal(0.001, 0.025)
        
        # Add occasional jumps
        if np.random.random() < 0.02:
            ret += np.random.normal(0, 0.05)
        
        prices.append(prices[-1] * (1 + ret))
    
    price_series = pd.Series(prices, index=pd.date_range('2022-01-01', periods=len(prices)))
    
    # Select best process
    selector = ProcessSelector()
    best_result = selector.recommend_process(price_series)
    
    return best_result


if __name__ == "__main__":
    result = demonstrate_process_selection()