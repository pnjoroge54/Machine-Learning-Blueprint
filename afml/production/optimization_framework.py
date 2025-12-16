"""
MetaTrader 5 Machine Learning Blueprint (Part 7): Parameter Optimization Framework

This module implements a comprehensive parameter optimization system that explores:
1. Bollinger Band parameters (window, std_dev)
2. Volatility target multipliers for triple-barrier labeling
3. Profit/stop-loss barrier ratios
4. Model hyperparameters (max_depth, min_weight_fraction_leaf)

The framework leverages the caching system from Part 6 to dramatically speed up
the optimization process.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from afml.cache import cv_cacheable, robust_cacheable, time_aware_cacheable
from afml.cross_validation import PurgedKFold, analyze_cross_val_scores


@dataclass
class StrategyConfig:
    """Configuration for Bollinger Band strategy and meta-labeling."""

    # Bollinger Band parameters
    bb_window: int = 20
    bb_std: float = 2.0

    # Triple-barrier parameters
    vol_lookback: int = 100
    vol_multiplier: float = 1.0
    pt_barrier: float = 1.0
    sl_barrier: float = 1.0
    time_horizon: dict = None
    min_ret: float = 0.0

    # Model parameters
    max_depth: int = 4
    min_weight_fraction_leaf: float = 0.05
    n_estimators: int = 100

    def __post_init__(self):
        if self.time_horizon is None:
            self.time_horizon = {"days": 1}

    def to_dict(self) -> dict:
        """Convert to dictionary for hashability."""
        return {
            "bb_window": self.bb_window,
            "bb_std": self.bb_std,
            "vol_lookback": self.vol_lookback,
            "vol_multiplier": self.vol_multiplier,
            "pt_barrier": self.pt_barrier,
            "sl_barrier": self.sl_barrier,
            "max_depth": self.max_depth,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "n_estimators": self.n_estimators,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"BB({self.bb_window},{self.bb_std:.1f})_"
            f"Vol({self.vol_multiplier:.1f})_"
            f"Barriers({self.pt_barrier:.1f},{self.sl_barrier:.1f})_"
            f"Model(d{self.max_depth},w{self.min_weight_fraction_leaf:.2f})"
        )


class ParameterOptimizer:
    """
    Comprehensive parameter optimization system.

    This class explores the parameter space systematically, leveraging caching
    to avoid redundant computations. Key features:

    1. Hierarchical caching:
       - Data loading cached at symbol/timeframe level
       - Feature engineering cached per BB configuration
       - Meta-labels cached per labeling configuration
       - CV scores cached per model configuration

    2. Smart exploration:
       - Coarse grid first, then fine-tune around best regions
       - Early stopping for clearly poor configurations
       - Parallel evaluation where possible
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        strategy,
        feature_engine,
        prepare_training_data_func,
        train_rf_func,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        test_size: float = 0.2,
        n_jobs: int = -1,
    ):
        self.prices_df = prices_df
        self.strategy = strategy
        self.feature_engine = feature_engine
        self.prepare_training_data = prepare_training_data_func
        self.train_rf = train_rf_func
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.test_size = test_size
        self.n_jobs = n_jobs

        # Store results
        self.results = []
        self.best_config = None
        self.best_score = -np.inf

    def define_parameter_grid(
        self,
        bb_windows: List[int] = None,
        bb_stds: List[float] = None,
        vol_multipliers: List[float] = None,
        barrier_ratios: List[Tuple[float, float]] = None,
        model_depths: List[int] = None,
        model_min_weights: List[float] = None,
    ) -> List[StrategyConfig]:
        """
        Define the parameter grid to explore.

        Default parameters chosen to explore key regions:
        - BB windows: Short (10), medium (20), long (50)
        - BB stds: Tight (1.5), standard (2.0), wide (2.5)
        - Vol multipliers: Conservative (0.5), standard (1.0), aggressive (1.5)
        - Barrier ratios: Symmetric (1,1), profit-focused (2,1), risk-focused (1,2)
        - Model complexity: Shallow (3), medium (4,5), deep (6)
        """
        if bb_windows is None:
            bb_windows = [10, 20, 50]
        if bb_stds is None:
            bb_stds = [1.5, 2.0, 2.5]
        if vol_multipliers is None:
            vol_multipliers = [0.5, 1.0, 1.5]
        if barrier_ratios is None:
            barrier_ratios = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
        if model_depths is None:
            model_depths = [3, 4, 5, 6]
        if model_min_weights is None:
            model_min_weights = [0.03, 0.05, 0.07]

        configs = []
        for bb_w, bb_s, vol_m, (pt, sl), depth, min_w in itertools.product(
            bb_windows,
            bb_stds,
            vol_multipliers,
            barrier_ratios,
            model_depths,
            model_min_weights,
        ):
            config = StrategyConfig(
                bb_window=bb_w,
                bb_std=bb_s,
                vol_multiplier=vol_m,
                pt_barrier=pt,
                sl_barrier=sl,
                max_depth=depth,
                min_weight_fraction_leaf=min_w,
            )
            configs.append(config)

        return configs

    def evaluate_configuration(
        self, config: StrategyConfig, weighting_scheme: str = "uniqueness"
    ) -> Dict:
        """
        Evaluate a single configuration.

        This function coordinates all the cached operations to evaluate
        a parameter configuration. The caching hierarchy ensures that:

        1. If we've already tested this exact config → instant return
        2. If we've tested same BB params but different barriers → reuse features
        3. If we've tested same barriers but different model → reuse labels

        Returns:
            Dictionary with CV scores and metadata
        """
        start_time = time.time()

        # Step 1: Prepare training data (cached by BB + barrier params)
        features, events = self.prepare_training_data(
            df=self.prices_df,
            strategy=self.strategy.__class__(
                window=config.bb_window, num_std=config.bb_std
            ),
            feature_engine=self.feature_engine,
            feature_params=dict(bb_period=config.bb_window, bb_std=config.bb_std),
            vol_lookback=config.vol_lookback,
            vol_multiplier=config.vol_multiplier,
            time_horizon=config.time_horizon,
            pt_barrier=config.pt_barrier,
            sl_barrier=config.sl_barrier,
            vertical_barrier_zero=True,
            min_ret=config.min_ret,
        )

        # Step 2: Prepare train/test split
        from afml.cross_validation import PurgedSplit
        from afml.labeling.triple_barrier import get_event_weights

        train_idx = events.index.intersection(features.index)
        cont = events.reindex(train_idx)
        X = features.reindex(train_idx)
        y = cont["bin"]
        t1 = cont["t1"]

        train, test = PurgedSplit(t1, self.test_size).split(X)
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        cont_train = get_event_weights(cont.iloc[train], self.prices_df.close)
        avg_u = cont_train.tW.mean()

        # Get sample weights based on scheme
        if weighting_scheme == "uniqueness":
            sample_weight = cont_train["tW"]
        elif weighting_scheme == "return":
            sample_weight = cont_train["w"]
        else:  # unweighted
            sample_weight = pd.Series(1.0, index=cont_train.index)

        # Step 3: Setup CV
        cv_gen = PurgedKFold(self.n_splits, cont_train["t1"], self.pct_embargo)

        # Step 4: Train and evaluate (cached by full config)
        clf = RandomForestClassifier(
            criterion="entropy",
            n_estimators=config.n_estimators,
            class_weight="balanced_subsample",
            max_samples=avg_u,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_depth=config.max_depth,
            random_state=42,
            n_jobs=self.n_jobs,
        )

        # Get CV scores
        cv_scores, cv_scores_df, cms = analyze_cross_val_scores(
            clf,
            X_train,
            y_train,
            cv_gen,
            sample_weight_train=sample_weight,
            sample_weight_score=sample_weight,
        )

        # Step 5: Train final model and get OOS performance
        clf_final = self.train_rf(
            clf.set_params(oob_score=True), X_train, y_train, sample_weight
        )

        # OOS predictions
        y_pred_proba = clf_final.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        oos_scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        elapsed = time.time() - start_time

        return {
            "config": config,
            "cv_f1_mean": cv_scores["f1"].mean(),
            "cv_f1_std": cv_scores["f1"].std(),
            "cv_accuracy_mean": cv_scores["accuracy"].mean(),
            "cv_precision_mean": cv_scores["precision"].mean(),
            "cv_recall_mean": cv_scores["recall"].mean(),
            "oos_f1": oos_scores["f1"],
            "oos_accuracy": oos_scores["accuracy"],
            "oos_precision": oos_scores["precision"],
            "oos_recall": oos_scores["recall"],
            "n_signals": len(train_idx),
            "avg_uniqueness": avg_u,
            "elapsed_seconds": elapsed,
        }

    def run_optimization(
        self,
        configs: List[StrategyConfig] = None,
        weighting_scheme: str = "uniqueness",
        early_stopping: bool = True,
        min_f1_threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Run the full optimization process.

        Args:
            configs: List of configurations to evaluate. If None, uses default grid.
            weighting_scheme: Sample weighting method
            early_stopping: Skip remaining configs if clearly underperforming
            min_f1_threshold: Minimum F1 to continue (if early_stopping=True)

        Returns:
            DataFrame with all results sorted by CV F1 score
        """
        if configs is None:
            configs = self.define_parameter_grid()

        print(f"Starting optimization of {len(configs)} configurations...")
        print(f"Estimated time without caching: {len(configs) * 5 / 60:.1f} minutes")
        print(f"With caching: Much faster! ⚡\n")

        self.results = []

        for i, config in enumerate(tqdm(configs, desc="Evaluating configs")):
            try:
                result = self.evaluate_configuration(config, weighting_scheme)
                self.results.append(result)

                # Track best
                if result["cv_f1_mean"] > self.best_score:
                    self.best_score = result["cv_f1_mean"]
                    self.best_config = config
                    print(f"\n✨ New best: {config}")
                    print(
                        f"   CV F1: {result['cv_f1_mean']:.4f} (±{result['cv_f1_std']:.4f})"
                    )
                    print(f"   OOS F1: {result['oos_f1']:.4f}\n")

                # Early stopping check
                if early_stopping and i > 20:  # Need some baseline
                    recent_scores = [r["cv_f1_mean"] for r in self.results[-10:]]
                    if all(s < min_f1_threshold for s in recent_scores):
                        print(
                            f"\nEarly stopping: Last 10 configs below threshold {min_f1_threshold:.2f}"
                        )
                        break

            except Exception as e:
                print(f"\n⚠️  Error evaluating {config}: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(
            [
                {
                    **r["config"].to_dict(),
                    "cv_f1_mean": r["cv_f1_mean"],
                    "cv_f1_std": r["cv_f1_std"],
                    "cv_accuracy": r["cv_accuracy_mean"],
                    "cv_precision": r["cv_precision_mean"],
                    "cv_recall": r["cv_recall_mean"],
                    "oos_f1": r["oos_f1"],
                    "oos_accuracy": r["oos_accuracy"],
                    "oos_precision": r["oos_precision"],
                    "oos_recall": r["oos_recall"],
                    "n_signals": r["n_signals"],
                    "avg_uniqueness": r["avg_uniqueness"],
                    "elapsed_sec": r["elapsed_seconds"],
                }
                for r in self.results
            ]
        ).sort_values("cv_f1_mean", ascending=False)

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nBest configuration:")
        print(f"  {self.best_config}")
        print(f"\nBest scores:")
        print(
            f"  CV F1: {results_df.iloc[0]['cv_f1_mean']:.4f} (±{results_df.iloc[0]['cv_f1_std']:.4f})"
        )
        print(f"  OOS F1: {results_df.iloc[0]['oos_f1']:.4f}")
        print(f"\nTop 5 configurations:")
        print(
            results_df.head()[
                [
                    "bb_window",
                    "bb_std",
                    "vol_multiplier",
                    "pt_barrier",
                    "sl_barrier",
                    "max_depth",
                    "cv_f1_mean",
                    "oos_f1",
                ]
            ]
        )

        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze optimization results to identify patterns.

        Returns insights about:
        - Best parameter ranges
        - Sensitive vs robust parameters
        - Trade-offs (e.g., complexity vs performance)
        """
        insights = {}

        # 1. Parameter sensitivities
        for param in ["bb_window", "bb_std", "vol_multiplier", "max_depth"]:
            param_impact = results_df.groupby(param)["cv_f1_mean"].agg(
                ["mean", "std", "count"]
            )
            insights[f"{param}_impact"] = param_impact

        # 2. Best ranges
        top_10_pct = results_df.head(max(1, len(results_df) // 10))
        insights["best_ranges"] = {
            "bb_window": (top_10_pct["bb_window"].min(), top_10_pct["bb_window"].max()),
            "bb_std": (top_10_pct["bb_std"].min(), top_10_pct["bb_std"].max()),
            "vol_multiplier": (
                top_10_pct["vol_multiplier"].min(),
                top_10_pct["vol_multiplier"].max(),
            ),
        }

        # 3. Complexity vs performance
        insights["complexity_vs_performance"] = results_df.groupby("max_depth").agg(
            {"cv_f1_mean": "mean", "oos_f1": "mean", "elapsed_sec": "mean"}
        )

        # 4. Overfitting analysis (CV vs OOS gap)
        results_df["overfit_gap"] = results_df["cv_f1_mean"] - results_df["oos_f1"]
        insights["overfit_stats"] = results_df["overfit_gap"].describe()

        return insights


def print_optimization_insights(insights: Dict):
    """Pretty print optimization insights."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION INSIGHTS")
    print("=" * 80)

    print("\n1. PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 80)
    for key, df in insights.items():
        if "_impact" in key:
            param = key.replace("_impact", "")
            print(f"\n{param.upper()}:")
            print(df.to_string())

    print("\n\n2. BEST PARAMETER RANGES (Top 10%)")
    print("-" * 80)
    for param, (min_val, max_val) in insights["best_ranges"].items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")

    print("\n\n3. COMPLEXITY VS PERFORMANCE")
    print("-" * 80)
    print(insights["complexity_vs_performance"].to_string())

    print("\n\n4. OVERFITTING ANALYSIS (CV F1 - OOS F1)")
    print("-" * 80)
    print(insights["overfit_stats"].to_string())
    print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    """
    Example workflow demonstrating the complete optimization pipeline.
    """

    # This would be imported from your notebook
    # from your_module import (
    #     load_data, prepare_training_data, train_rf,
    #     BollingerStrategy, create_bollinger_features
    # )

    print("=" * 80)
    print("MetaTrader 5 ML Blueprint Part 7: Parameter Optimization")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("1. Systematic exploration of parameter space")
    print("2. Leveraging Part 6's caching for speed")
    print("3. Identifying robust parameter configurations")
    print("4. Analyzing performance trade-offs")
