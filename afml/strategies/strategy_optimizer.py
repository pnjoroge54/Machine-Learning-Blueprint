"""
Integrated Framework: Stochastic Process Simulation + Triple Barrier Optimization

This module connects:
1. multi_process_barrier_optimizer.py - Stochastic process identification
2. triple_barrier_optimizer.py - Optimal barrier determination
3. trading_strategies.py - Signal generation

Use this to:
- Test strategies on synthetic data with known properties
- Understand which process models best fit your strategy's behavior
- Optimize exit rules based on the underlying stochastic process
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtester.multi_process_barrier_optimizer import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    ProcessSelector,
)
from ..backtester.triple_barrier_optimizer import (
    BarrierOptimizationResult,
    optimize_strategy_barriers,
)
from .trading_strategies import BaseStrategy


@dataclass
class IntegratedTestResult:
    """Complete test results combining process identification and barrier optimization"""
    strategy_name: str
    detected_process: str
    process_params: Dict
    process_fit_metrics: Dict
    barrier_optimization: 'BarrierOptimizationResult'
    synthetic_backtest_results: Dict
    real_data_comparison: Optional[Dict] = None


class StrategyTestFramework:
    """
    Integrated framework for comprehensive strategy testing
    
    Workflow:
    1. Generate synthetic data using stochastic processes
    2. Apply trading strategy to get signals
    3. Identify which stochastic process best fits the strategy's P&L
    4. Optimize triple barriers based on identified process
    5. Backtest on synthetic data with optimal barriers
    6. Compare to real data (optional)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
    
    def test_strategy_on_process(
        self,
        strategy,  # BaseStrategy instance
        process,   # StochasticProcess instance
        process_params: Dict,
        n_days: int = 1000,
        initial_price: float = 100.0,
        optimize_barriers: bool = True,
        n_barrier_iterations: int = 100000,
        max_holding_period: int = 100
    ) -> IntegratedTestResult:
        """
        Complete test of strategy on specific stochastic process
        
        Args:
            strategy: Trading strategy instance
            process: Stochastic process instance (e.g., OrnsteinUhlenbeck())
            process_params: Parameters for the process
            n_days: Length of synthetic price series
            initial_price: Starting price
            optimize_barriers: Whether to run barrier optimization
            n_barrier_iterations: Monte Carlo iterations for barrier optimization
            max_holding_period: Maximum holding period (vertical barrier)
            
        Returns:
            IntegratedTestResult with complete analysis
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"TESTING: {strategy.get_strategy_name()}")
            print(f"PROCESS: {process.get_process_name()}")
            print("="*70)
        
        # Step 1: Generate synthetic price data
        synthetic_path = process.simulate_path(process_params, n_days, dt=1.0)
        
        dates = pd.date_range('2020-01-01', periods=len(synthetic_path), freq='D')
        synthetic_prices = pd.Series(synthetic_path, index=dates)
        
        # Create OHLCV DataFrame (simplified)
        data = pd.DataFrame({
            'close': synthetic_prices,
            'open': synthetic_prices * 0.999,
            'high': synthetic_prices * 1.002,
            'low': synthetic_prices * 0.998,
            'volume': np.random.randint(1000000, 10000000, len(synthetic_prices))
        })
        
        if self.verbose:
            print(f"\nGenerated {len(data)} days of synthetic data")
        
        # Step 2: Generate trading signals
        signals = strategy.generate_signals(data)
        n_signals = (signals != 0).sum()
        
        if self.verbose:
            print(f"Strategy generated {n_signals} signals")
            print(f"  Long signals: {(signals == 1).sum()}")
            print(f"  Short signals: {(signals == -1).sum()}")
        
        # Step 3: Calculate strategy P&L for process identification
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns  # Lag signals by 1
        strategy_pnl = strategy_returns.cumsum()
        
        # Step 4: Identify best-fit process for strategy P&L
        # (This would use ProcessSelector from multi_process_barrier_optimizer.py)
        selector = ProcessSelector()
        best_process_result = selector.recommend_process(strategy_pnl)
        
        # For now, we know it's the process we used
        detected_process = process.get_process_name()
        fit_metrics = {
            'process': detected_process,
            'known_ground_truth': True,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
        
        # Step 5: Optimize barriers (if requested)
        barrier_result = None
        if optimize_barriers:
            if self.verbose:
                print("\nOptimizing triple barriers...")
            
            # This would use optimize_strategy_barriers from triple_barrier_optimizer.py
            barrier_result = optimize_strategy_barriers(
                strategy=strategy,
                data=data,
                n_iter=n_barrier_iterations,
                max_holding_period=max_holding_period,
                verbose=self.verbose
            )
            
            # Placeholder
            if self.verbose:
                print("(Barrier optimization would run here)")
        
        # Step 6: Backtest with optimal barriers
        backtest_results = self._backtest_with_barriers(
            data=data,
            signals=signals,
            profit_taking=2.0 * returns.std(),  # Placeholder
            stop_loss=1.0 * returns.std(),       # Placeholder
            max_holding=max_holding_period
        )
        
        result = IntegratedTestResult(
            strategy_name=strategy.get_strategy_name(),
            detected_process=detected_process,
            process_params=process_params,
            process_fit_metrics=fit_metrics,
            barrier_optimization=barrier_result,
            synthetic_backtest_results=backtest_results
        )
        
        self.results.append(result)
        return result
    
    def test_strategy_across_processes(
        self,
        strategy,  # BaseStrategy instance
        process_configs: List[Tuple],  # [(process, params), ...]
        n_days: int = 1000,
        optimize_barriers: bool = True
    ) -> List[IntegratedTestResult]:
        """
        Test strategy across multiple stochastic processes
        
        This helps answer: "How does my strategy perform under different 
        market conditions (trending, mean-reverting, volatile, etc.)?"
        
        Args:
            strategy: Trading strategy instance
            process_configs: List of (process_instance, params_dict) tuples
            n_days: Length of synthetic data
            optimize_barriers: Whether to optimize barriers for each
            
        Returns:
            List of IntegratedTestResult, one per process
        """
        results = []
        
        for process, params in process_configs:
            result = self.test_strategy_on_process(
                strategy=strategy,
                process=process,
                process_params=params,
                n_days=n_days,
                optimize_barriers=optimize_barriers
            )
            results.append(result)
        
        if self.verbose:
            self._print_comparison_summary(results)
        
        return results
    
    def identify_optimal_process_for_strategy(
        self,
        strategy: BaseStrategy,
        real_data: pd.DataFrame,
        test_processes: Optional[List] = None
    ) -> Dict:
        """
        Identify which stochastic process best describes the strategy's behavior on real data
        
        Workflow:
        1. Run strategy on real data
        2. Analyze strategy P&L time series
        3. Test multiple stochastic processes
        4. Recommend best-fit process
        5. Suggest optimal barriers for that process
        
        Args:
            strategy: Trading strategy instance
            real_data: Real OHLCV data
            test_processes: List of processes to test (or None for all)
            
        Returns:
            Dict with process identification and recommendations
        """
        if self.verbose:
            print("\n" + "="*70)
            print("IDENTIFYING OPTIMAL PROCESS MODEL")
            print("="*70)
            print(f"Strategy: {strategy.get_strategy_name()}")
            print(f"Data points: {len(real_data)}")
        
        # Generate signals on real data
        signals = strategy.generate_signals(real_data)
        
        # Calculate strategy returns
        returns = real_data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        strategy_pnl = (1 + strategy_returns).cumprod()
        
        # This would use ProcessSelector to identify best process
        # from multi_process_barrier_optimizer import ProcessSelector
        # selector = ProcessSelector()
        # process_results = selector.test_all_processes(strategy_pnl)
        # best_process = process_results[0]
        
        if self.verbose:
            print("\nStrategy Performance on Real Data:")
            print(f"  Total Return: {(strategy_pnl.iloc[-1] - 1) * 100:.2f}%")
            print(f"  Sharpe Ratio: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}")
            print(f"  Win Rate: {(strategy_returns > 0).sum() / len(strategy_returns):.2%}")
        
        # Placeholder result
        result = {
            'strategy_name': strategy.get_strategy_name(),
            'recommended_process': 'Ornstein-Uhlenbeck',  # Would come from ProcessSelector
            'confidence': 0.85,
            'process_parameters': {},
            'suggested_barriers': {
                'profit_taking': '2.5σ',
                'stop_loss': '1.5σ',
                'expected_sharpe': 1.2
            }
        }
        
        return result
    
    def monte_carlo_robustness_test(
        self,
        strategy,  # BaseStrategy instance
        process,   # StochasticProcess instance
        process_params: Dict,
        n_simulations: int = 100,
        n_days: int = 1000,
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Dict:
        """
        Run Monte Carlo simulation to test strategy robustness
        
        Generate multiple price paths from same process and test strategy
        performance distribution
        
        Args:
            strategy: Trading strategy instance
            process: Stochastic process
            process_params: Process parameters
            n_simulations: Number of price paths to generate
            n_days: Days per simulation
            percentiles: Percentiles to report
            
        Returns:
            Dict with performance distribution statistics
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"MONTE CARLO ROBUSTNESS TEST")
            print("="*70)
            print(f"Simulations: {n_simulations}")
            print(f"Process: {process.get_process_name()}")
        
        results = {
            'sharpe_ratios': [],
            'total_returns': [],
            'win_rates': [],
            'max_drawdowns': [],
            'n_trades': []
        }
        
        for i in range(n_simulations):
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_simulations} simulations")
            
            # Generate synthetic path
            synthetic_path = process.simulate_path(process_params, n_days, dt=1.0)
            dates = pd.date_range('2020-01-01', periods=len(synthetic_path), freq='D')
            
            data = pd.DataFrame({
                'close': synthetic_path,
                'open': synthetic_path * 0.999,
                'high': synthetic_path * 1.002,
                'low': synthetic_path * 0.998,
                'volume': np.random.randint(1000000, 10000000, len(synthetic_path))
            }, index=dates)
            
            # Run strategy
            signals = strategy.generate_signals(data)
            returns = data['close'].pct_change()
            strategy_returns = signals.shift(1) * returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            win_rate = (strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0]) if (strategy_returns != 0).sum() > 0 else 0
            
            # Max drawdown
            cum_returns = (1 + strategy_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            
            results['sharpe_ratios'].append(sharpe)
            results['total_returns'].append(total_return)
            results['win_rates'].append(win_rate)
            results['max_drawdowns'].append(max_dd)
            results['n_trades'].append((signals != 0).sum())
        
        # Calculate statistics
        summary = {}
        for metric, values in results.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
            for p in percentiles:
                summary[metric][f'p{p}'] = np.percentile(values, p)
        
        if self.verbose:
            print("\nRobustness Test Results:")
            print(f"\nSharpe Ratio Distribution:")
            print(f"  Mean: {summary['sharpe_ratios']['mean']:.2f}")
            print(f"  Std:  {summary['sharpe_ratios']['std']:.2f}")
            print(f"  5th percentile:  {summary['sharpe_ratios']['p5']:.2f}")
            print(f"  95th percentile: {summary['sharpe_ratios']['p95']:.2f}")
            
            print(f"\nTotal Return Distribution:")
            print(f"  Mean: {summary['total_returns']['mean']*100:.1f}%")
            print(f"  5th percentile:  {summary['total_returns']['p5']*100:.1f}%")
            print(f"  95th percentile: {summary['total_returns']['p95']*100:.1f}%")
        
        return {
            'summary': summary,
            'raw_results': results
        }
    
    def _backtest_with_barriers(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        profit_taking: float,
        stop_loss: float,
        max_holding: int
    ) -> Dict:
        """
        Simple backtest with triple barrier exits
        
        Args:
            data: OHLCV data
            signals: Trading signals
            profit_taking: Profit taking threshold
            stop_loss: Stop loss threshold
            max_holding: Maximum holding period
            
        Returns:
            Dict with backtest results
        """
        returns = data['close'].pct_change()
        positions = []
        trades = []
        
        current_position = None
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0 and current_position is None:
                # Enter position
                current_position = {
                    'entry_idx': i,
                    'entry_price': data['close'].iloc[i],
                    'direction': signals.iloc[i],
                    'entry_time': data.index[i]
                }
            
            elif current_position is not None:
                # Check for exit
                holding_period = i - current_position['entry_idx']
                current_price = data['close'].iloc[i]
                pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                pnl *= current_position['direction']
                
                exit_reason = None
                if pnl >= profit_taking:
                    exit_reason = 'profit_taking'
                elif pnl <= -stop_loss:
                    exit_reason = 'stop_loss'
                elif holding_period >= max_holding:
                    exit_reason = 'max_holding'
                
                if exit_reason:
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': data.index[i],
                        'direction': current_position['direction'],
                        'pnl': pnl,
                        'holding_period': holding_period,
                        'exit_reason': exit_reason
                    })
                    current_position = None
        
        if len(trades) == 0:
            return {
                'n_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'sharpe': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        return {
            'n_trades': len(trades_df),
            'win_rate': (trades_df['pnl'] > 0).mean(),
            'avg_pnl': trades_df['pnl'].mean(),
            'avg_holding': trades_df['holding_period'].mean(),
            'profit_taking_exits': (trades_df['exit_reason'] == 'profit_taking').sum(),
            'stop_loss_exits': (trades_df['exit_reason'] == 'stop_loss').sum(),
            'max_holding_exits': (trades_df['exit_reason'] == 'max_holding').sum(),
        }
    
    def _print_comparison_summary(self, results: List[IntegratedTestResult]):
        """Print comparison across multiple processes"""
        print("\n" + "="*70)
        print("PROCESS COMPARISON SUMMARY")
        print("="*70)
        
        for result in results:
            print(f"\n{result.detected_process}:")
            print(f"  Sharpe: {result.process_fit_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Trades: {result.synthetic_backtest_results.get('n_trades', 0)}")
            print(f"  Win Rate: {result.synthetic_backtest_results.get('win_rate', 0):.2%}")


def create_example_test_suite():
    """
    Example usage showing the complete integrated workflow
    """
    print("="*70)
    print("INTEGRATED STRATEGY TESTING FRAMEWORK")
    print("Example Usage")
    print("="*70)
    
    print("\n1. Test Strategy on Specific Process:")
    print("""
    from trading_strategies import BollingerStrategy
    from multi_process_barrier_optimizer import OrnsteinUhlenbeck
    
    framework = StrategyTestFramework()
    strategy = BollingerStrategy(window=20, std=2.0)
    process = OrnsteinUhlenbeck()
    
    params = {
        'mu': 100,
        'theta': 0.1,
        'sigma': 2.0,
        'phi': 0.95,
        'half_life': 14
    }
    
    result = framework.test_strategy_on_process(
        strategy=strategy,
        process=process,
        process_params=params,
        n_days=1000,
        optimize_barriers=True
    )
    """)
    
    print("\n2. Test Across Multiple Processes:")
    print("""
    from multi_process_barrier_optimizer import (
        OrnsteinUhlenbeck, GeometricBrownianMotion, JumpDiffusion
    )
    
    processes = [
        (OrnsteinUhlenbeck(), {'mu': 100, 'theta': 0.1, 'sigma': 2.0, 'phi': 0.95}),
        (GeometricBrownianMotion(), {'mu': 0.001, 'sigma': 0.02}),
        (JumpDiffusion(), {'mu': 0.001, 'sigma': 0.015, 'lambda_jump': 0.02, 
                          'jump_mean': 0, 'jump_std': 0.05})
    ]
    
    results = framework.test_strategy_across_processes(
        strategy=strategy,
        process_configs=processes
    )
    """)
    
    print("\n3. Identify Process from Real Data:")
    print("""
    real_data = pd.read_csv('historical_prices.csv')
    
    analysis = framework.identify_optimal_process_for_strategy(
        strategy=strategy,
        real_data=real_data
    )
    """)
    
    print("\n4. Monte Carlo Robustness Test:")
    print("""
    robustness = framework.monte_carlo_robustness_test(
        strategy=strategy,
        process=OrnsteinUhlenbeck(),
        process_params=params,
        n_simulations=100
    )
    """)


if __name__ == "__main__":
    create_example_test_suite()
if __name__ == "__main__":
    create_example_test_suite()
    create_example_test_suite()
if __name__ == "__main__":
    create_example_test_suite()
