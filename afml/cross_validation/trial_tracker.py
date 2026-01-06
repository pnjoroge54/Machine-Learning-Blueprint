"""
Strategy Trial Tracker for Deflated Sharpe Ratio (DSR)
Tracks all strategy variations tested during research process
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StrategyTrialTracker:
    """
    Tracks all strategy trials for DSR calculation.

    Usage:
        tracker = StrategyTrialTracker(project_name="momentum_strategy")

        # Log a trial
        tracker.log_trial(
            description="MA crossover 50/200",
            parameters={"fast_ma": 50, "slow_ma": 200},
            features=["returns", "volume"],
            sharpe_ratio=1.2
        )

        # Get trial count for DSR
        n_trials = tracker.get_trial_count()
    """

    def __init__(self, project_name: str, storage_dir: str = "./trial_logs"):
        self.project_name = project_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.log_file = self.storage_dir / f"{project_name}_trials.json"
        self.trials = self._load_trials()

    def _load_trials(self) -> List[Dict]:
        """Load existing trials from disk."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def _save_trials(self):
        """Save trials to disk."""
        with open(self.log_file, 'w') as f:
            json.dump(self.trials, f, indent=2)

    def _generate_trial_hash(self, parameters: Dict, features: List[str],
                            model_type: Optional[str]) -> str:
        """Generate unique hash for trial configuration."""
        config_str = json.dumps({
            'parameters': parameters,
            'features': sorted(features),
            'model_type': model_type
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def log_trial(self,
                  description: str,
                  parameters: Dict[str, Any],
                  features: List[str],
                  sharpe_ratio: Optional[float] = None,
                  model_type: Optional[str] = None,
                  notes: Optional[str] = None,
                  check_duplicate: bool = True) -> int:
        """
        Log a strategy trial.

        Args:
            description: Brief description of the trial
            parameters: Dictionary of strategy parameters
            features: List of features used
            sharpe_ratio: Sharpe ratio if available
            model_type: Type of model (e.g., 'RandomForest', 'LogisticRegression')
            notes: Additional notes
            check_duplicate: If True, warns about duplicate trials but still logs

        Returns:
            Current trial count N
        """
        trial_hash = self._generate_trial_hash(parameters, features, model_type)

        # Check for duplicates
        if check_duplicate:
            duplicates = [t for t in self.trials if t.get('trial_hash') == trial_hash]
            if duplicates:
                print(f"⚠️  Warning: Similar trial already logged (Trial #{duplicates[0]['trial_number']})")
                print(f"   Consider if this is truly a new test or a re-run")

        trial = {
            'trial_number': len(self.trials) + 1,
            'trial_hash': trial_hash,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'model_type': model_type,
            'parameters': parameters,
            'features': features,
            'sharpe_ratio': sharpe_ratio,
            'notes': notes
        }

        self.trials.append(trial)
        self._save_trials()

        print(f"✓ Trial #{trial['trial_number']} logged: {description}")
        return len(self.trials)

    def get_trial_count(self, unique_only: bool = False) -> int:
        """
        Get total number of trials (N for DSR calculation).

        Args:
            unique_only: If True, counts only unique configurations

        Returns:
            Trial count N
        """
        if unique_only:
            unique_hashes = set(t['trial_hash'] for t in self.trials)
            return len(unique_hashes)
        return len(self.trials)

    def get_summary(self) -> Dict:
        """Get summary statistics of trials."""
        if not self.trials:
            return {
                'total_trials': 0,
                'unique_configs': 0,
                'best_sharpe': None,
                'date_range': None
            }

        sharpe_ratios = [t['sharpe_ratio'] for t in self.trials
                        if t['sharpe_ratio'] is not None]

        return {
            'total_trials': len(self.trials),
            'unique_configs': len(set(t['trial_hash'] for t in self.trials)),
            'trials_with_sharpe': len(sharpe_ratios),
            'best_sharpe': max(sharpe_ratios) if sharpe_ratios else None,
            'worst_sharpe': min(sharpe_ratios) if sharpe_ratios else None,
            'mean_sharpe': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else None,
            'date_range': (self.trials[0]['timestamp'], self.trials[-1]['timestamp']),
            'model_types': list(set(t.get('model_type') for t in self.trials if t.get('model_type')))
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"Trial Summary: {self.project_name}")
        print(f"{'='*60}")
        print(f"Total Trials (N):        {summary['total_trials']}")
        print(f"Unique Configurations:   {summary['unique_configs']}")

        if summary['trials_with_sharpe'] > 0:
            print(f"\nSharpe Ratio Statistics:")
            print(f"  Trials with SR:        {summary['trials_with_sharpe']}")
            print(f"  Best SR:               {summary['best_sharpe']:.3f}")
            print(f"  Worst SR:              {summary['worst_sharpe']:.3f}")
            print(f"  Mean SR:               {summary['mean_sharpe']:.3f}")

        if summary['model_types']:
            print(f"\nModel Types Tested:      {', '.join(summary['model_types'])}")

        print(f"\nFirst Trial:             {summary['date_range'][0][:10]}")
        print(f"Last Trial:              {summary['date_range'][1][:10]}")
        print(f"{'='*60}\n")

    def list_trials(self, last_n: Optional[int] = None):
        """List recent trials."""
        trials_to_show = self.trials[-last_n:] if last_n else self.trials

        print(f"\n{'Trial':<6} {'Date':<12} {'Sharpe':<8} {'Description':<40}")
        print("-" * 70)
        for t in trials_to_show:
            sr = f"{t['sharpe_ratio']:.3f}" if t['sharpe_ratio'] is not None else "N/A"
            date = t['timestamp'][:10]
            desc = t['description'][:37] + "..." if len(t['description']) > 40 else t['description']
            print(f"#{t['trial_number']:<5} {date:<12} {sr:<8} {desc:<40}")

    def export_for_dsr(self, output_file: Optional[str] = None) -> Dict:
        """
        Export trial data formatted for DSR calculation.

        Returns:
            Dictionary with N and trial details
        """
        export_data = {
            'project_name': self.project_name,
            'N': len(self.trials),
            'N_unique': len(set(t['trial_hash'] for t in self.trials)),
            'export_timestamp': datetime.now().isoformat(),
            'sharpe_ratios': [t['sharpe_ratio'] for t in self.trials
                            if t['sharpe_ratio'] is not None],
            'trials': self.trials
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✓ Exported to {output_file}")

        return export_data


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = StrategyTrialTracker(project_name="momentum_strategy_v1")

    # Example: Log some trials
    tracker.log_trial(
        description="Simple MA crossover baseline",
        parameters={"fast_ma": 20, "slow_ma": 50, "threshold": 0},
        features=["close_price"],
        model_type="rule_based",
        sharpe_ratio=0.8
    )

    tracker.log_trial(
        description="MA crossover with volume filter",
        parameters={"fast_ma": 20, "slow_ma": 50, "volume_threshold": 1.5},
        features=["close_price", "volume"],
        model_type="rule_based",
        sharpe_ratio=1.1
    )

    tracker.log_trial(
        description="Random Forest with MA features",
        parameters={"n_estimators": 100, "max_depth": 5, "fast_ma": 20, "slow_ma": 50},
        features=["ma_diff", "volume", "volatility"],
        model_type="RandomForest",
        sharpe_ratio=1.4
    )

    # Get trial count for DSR
    N = tracker.get_trial_count()
    print(f"\nTotal trials for DSR calculation: N = {N}")

    # Show summary
    tracker.print_summary()

    # List recent trials
    tracker.list_trials(last_n=5)

    # Export for DSR calculation
    tracker.export_for_dsr("dsr_trial_data.json")
