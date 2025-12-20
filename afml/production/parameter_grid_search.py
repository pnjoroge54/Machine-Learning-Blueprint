from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..strategies.trading_strategies import BaseStrategy
from .model_development import ModelDevelopmentPipeline


class ParameterGridSearch:
    """
    Runs model development pipeline across multiple parameter combinations.
    Supports grid search over specified parameters.
    """

    def __init__(self):
        self.results = []
        self.combinations = []
        self.summary_df = None

    def generate_parameter_grid(self, param_dict):
        """
        Generate all combinations of parameters from a nested dictionary.

        Parameters
        ----------
        param_dict : dict
            dictionary with parameter names as keys and lists of values as values.

        Returns
        -------
        list
            List of dictionaries with all parameter combinations.
        """

        # Flatten the parameter dictionary
        keys = []
        values = []

        for key, value in param_dict.items():
            keys.append(key)
            # Ensure value is a list
            if not isinstance(value, list):
                value = [value]
            values.append(value)

        # Generate all combinations
        combinations = []
        for combo in product(*values):
            param_combo = dict(zip(keys, combo))
            combinations.append(param_combo)

        return combinations

    def run_grid_search(
        self,
        base_config: dict,
        param_grid: dict,
        cache_reports: bool = False,
        save: bool = True,
        parallel: bool = False,
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run model development across multiple parameter combinations.

        Parameters
        ----------
        base_config : dict
            Base configuration with fixed parameters.
            Must contain: symbol, train_start, train_end, strategy,
            data_config (without variable params), feature_config,
            label_config, model_params.
        param_grid : dict
            dictionary specifying which parameters to vary.
            Structure: {
                'data_config': {
                    'bar_type': ['time', 'tick'],
                    'bar_size': ['M1', 'M5', 'M15'],
                    'price': ['mid_price']
                },
                'label_config': {
                    'profit_target': [1.0, 1.5, 2.0],
                    'stop_loss': [1.0, 1.5]
                }
            }
        cache_reports : bool, optional
            Display cache reports (default: False).
        save : bool, optional
            Save individual models (default: True).
        parallel : bool, optional
            Run combinations in parallel (default: False).
        n_jobs : int, optional
            Number of parallel jobs (default: -1, all cores).
        verbose : bool, optional
            Print progress information (default: True).

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with results for all combinations.
        """
        # Extract base configurations
        symbol = base_config["symbol"]
        train_start = base_config["train_start"]
        train_end = base_config["train_end"]
        strategy = base_config["strategy"]
        base_data_config = base_config.get("data_config", {})
        base_feature_config = base_config["feature_config"]
        base_label_config = base_config.get("label_config", {})
        base_model_params = base_config["model_params"]

        # Generate parameter combinations
        self.combinations = self._generate_all_combinations(
            param_grid, base_data_config, base_label_config
        )

        if verbose:
            print(f"\nGenerated {len(self.combinations)} parameter combinations")
            print("=" * 70)

        # Run each combination
        if parallel:
            results = self._run_parallel(
                symbol,
                train_start,
                train_end,
                strategy,
                base_feature_config,
                base_model_params,
                self.combinations,
                cache_reports,
                save,
                n_jobs,
                verbose,
            )
        else:
            results = self._run_sequential(
                symbol,
                train_start,
                train_end,
                strategy,
                base_feature_config,
                base_model_params,
                self.combinations,
                cache_reports,
                save,
                verbose,
            )

        self.results = results
        self.summary_df = self._create_summary_dataframe()

        return self.summary_df

    def _generate_all_combinations(
        self, param_grid, base_data_config, base_label_config
    ):
        """Generate all parameter combinations from grid."""

        combinations = []

        # Prepare data_config combinations
        data_config_combos = [base_data_config.copy()]
        if "data_config" in param_grid:
            data_grid = param_grid["data_config"]
            data_keys = []
            data_values = []

            for key, values in data_grid.items():
                data_keys.append(key)
                if not isinstance(values, list):
                    values = [values]
                data_values.append(values)

            # Generate combinations for data_config
            data_combinations = []
            for combo in product(*data_values):
                data_dict = dict(zip(data_keys, combo))
                data_combinations.append(data_dict)

            # Merge with base data_config
            merged_data_combos = []
            for base in data_config_combos:
                for combo in data_combinations:
                    merged = base.copy()
                    merged.update(combo)
                    merged_data_combos.append(merged)

            data_config_combos = merged_data_combos

        # Prepare label_config combinations
        label_config_combos = [base_label_config.copy()]
        if "label_config" in param_grid:
            label_grid = param_grid["label_config"]
            label_keys = []
            label_values = []

            for key, values in label_grid.items():
                label_keys.append(key)
                if not isinstance(values, list):
                    values = [values]
                label_values.append(values)

            # Generate combinations for label_config
            label_combinations = []
            for combo in product(*label_values):
                label_dict = dict(zip(label_keys, combo))
                label_combinations.append(label_dict)

            # Merge with base label_config
            merged_label_combos = []
            for base in label_config_combos:
                for combo in label_combinations:
                    merged = base.copy()
                    merged.update(combo)
                    merged_label_combos.append(merged)

            label_config_combos = merged_label_combos

        # Combine data and label configs
        for data_config in data_config_combos:
            for label_config in label_config_combos:
                combinations.append(
                    {"data_config": data_config, "label_config": label_config}
                )

        return combinations

    def _run_sequential(
        self,
        symbol,
        train_start,
        train_end,
        strategy,
        base_feature_config,
        base_model_params,
        combinations,
        cache_reports,
        save,
        verbose,
    ):
        """Run combinations sequentially."""
        results = []

        for i, combo in enumerate(combinations, 1):
            if verbose:
                print(f"\n[{i}/{len(combinations)}] Running combination:")
                self._print_combo_summary(combo)

            try:
                # Create pipeline for this combination
                pipeline = ModelDevelopmentPipeline(
                    symbol=symbol,
                    train_start=train_start,
                    train_end=train_end,
                    strategy=strategy,
                    data_config=combo["data_config"],
                    feature_config=base_feature_config,
                    label_config=combo["label_config"],
                    model_params=base_model_params,
                )

                # Run pipeline
                model, features, metrics, config = pipeline.run(
                    cache_reports=False, save=save, verbose=False
                )

                # Store results
                result = {
                    "combination_id": i,
                    "data_config": combo["data_config"],
                    "label_config": combo["label_config"],
                    "pipeline": pipeline,
                    "model": model,
                    "metrics": metrics,
                    "config": config,
                    "features": features,
                    "success": True,
                    "error": None,
                }

                results.append(result)

                if verbose:
                    print(
                        f"  ✓ Success - CV Score: {metrics['cv_results']['best_score']:.4f}"
                    )

            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {str(e)[:100]}...")

                results.append(
                    {
                        "combination_id": i,
                        "data_config": combo["data_config"],
                        "label_config": combo["label_config"],
                        "pipeline": None,
                        "model": None,
                        "metrics": None,
                        "config": None,
                        "features": None,
                        "success": False,
                        "error": str(e),
                    }
                )

        return results

    def _run_parallel(
        self,
        symbol,
        train_start,
        train_end,
        strategy,
        base_feature_config,
        base_model_params,
        combinations,
        cache_reports,
        save,
        n_jobs,
        verbose,
    ):
        """Run combinations in parallel using joblib."""
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("joblib not available, falling back to sequential execution")
            return self._run_sequential(
                symbol,
                train_start,
                train_end,
                strategy,
                base_feature_config,
                base_model_params,
                combinations,
                cache_reports,
                save,
                verbose,
            )

        if verbose:
            print(
                f"Running {len(combinations)} combinations in parallel (n_jobs={n_jobs})..."
            )

        def run_single_combo(i, combo):
            """Run a single parameter combination."""
            try:
                pipeline = ModelDevelopmentPipeline(
                    symbol=symbol,
                    train_start=train_start,
                    train_end=train_end,
                    strategy=strategy,
                    data_config=combo["data_config"],
                    feature_config=base_feature_config,
                    label_config=combo["label_config"],
                    model_params=base_model_params,
                )

                model, features, metrics, config = pipeline.run(
                    cache_reports=False, save=save, verbose=False
                )

                return {
                    "combination_id": i,
                    "data_config": combo["data_config"],
                    "label_config": combo["label_config"],
                    "pipeline": pipeline,
                    "model": model,
                    "metrics": metrics,
                    "config": config,
                    "features": features,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "combination_id": i,
                    "data_config": combo["data_config"],
                    "label_config": combo["label_config"],
                    "pipeline": None,
                    "model": None,
                    "metrics": None,
                    "config": None,
                    "features": None,
                    "success": False,
                    "error": str(e),
                }

        # Run in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run_single_combo)(i, combo)
            for i, combo in enumerate(combinations, 1)
        )

        return results

    def _print_combo_summary(self, combo):
        """Print summary of a parameter combination."""
        data_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in combo["data_config"].items()
                if k not in ["account_name"]
            ]
        )
        label_str = ", ".join([f"{k}: {v}" for k, v in combo["label_config"].items()])

        print(f"  Data: {data_str}")
        if label_str:
            print(f"  Labels: {label_str}")

    def _create_summary_dataframe(self):
        """Create summary DataFrame from all results."""
        summary_data = []

        for result in self.results:
            if result["success"] and result["metrics"]:
                row = {
                    "combination_id": result["combination_id"],
                    "success": result["success"],
                    "cv_score": result["metrics"]["cv_results"]["best_score"],
                    "training_samples": result["metrics"]["training_samples"],
                    "feature_count": result["metrics"]["feature_count"],
                    "best_weighting_scheme": result["metrics"]["best_weighting_scheme"],
                }

                # Add data_config parameters
                for key, value in result["data_config"].items():
                    row[f"data_{key}"] = value

                # Add label_config parameters
                for key, value in result["label_config"].items():
                    row[f"label_{key}"] = value

                # Add error if failed
                if not result["success"]:
                    row["error"] = result["error"]

                summary_data.append(row)

        return pd.DataFrame(summary_data)

    def get_best_combination(self, metric: str = "cv_score", ascending: bool = False):
        """
        Get the best parameter combination based on specified metric.

        Parameters
        ----------
        metric : str, optional
            Metric to use for comparison (default: 'cv_score').
        ascending : bool, optional
            Sort ascending (True) or descending (False) (default: False).

        Returns
        -------
        dict
            Best combination result.
        """
        if self.summary_df is None or self.summary_df.empty:
            raise ValueError("No results available. Run grid search first.")

        # Filter successful runs
        successful_df = self.summary_df[self.summary_df["success"] == True]

        if successful_df.empty:
            raise ValueError("No successful runs found.")

        # Sort by metric
        sorted_df = successful_df.sort_values(metric, ascending=ascending)
        best_id = sorted_df.iloc[0]["combination_id"]

        # Find the corresponding result
        for result in self.results:
            if result["combination_id"] == best_id:
                return result

        return None

    def plot_results(
        self,
        x_param: str,
        y_metric: str = "cv_score",
        group_by: str = None,
        figsize: tuple = (12, 8),
    ):
        """
        Plot results of parameter grid search.

        Parameters
        ----------
        x_param : str
            Parameter to plot on x-axis (e.g., 'data_bar_size').
        y_metric : str, optional
            Metric to plot on y-axis (default: 'cv_score').
        group_by : str, optional
            Parameter to group by (for line plots).
        figsize : tuple, optional
            Figure size (default: (12, 8)).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.summary_df is None:
            raise ValueError("No results to plot. Run grid search first.")

        # Filter successful runs
        plot_df = self.summary_df[self.summary_df["success"] == True].copy()

        if plot_df.empty:
            print("No successful runs to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar plot
        if group_by:
            # Grouped bar plot
            pivot_df = plot_df.pivot_table(
                values=y_metric, index=x_param, columns=group_by, aggfunc="mean"
            )
            pivot_df.plot(kind="bar", ax=axes[0])
            axes[0].set_title(f"{y_metric} by {x_param} (grouped by {group_by})")
        else:
            # Simple bar plot
            avg_scores = plot_df.groupby(x_param)[y_metric].mean().sort_values()
            avg_scores.plot(kind="bar", ax=axes[0])
            axes[0].set_title(f"Average {y_metric} by {x_param}")

        axes[0].set_ylabel(y_metric)
        axes[0].tick_params(axis="x", rotation=45)

        # Scatter plot matrix for top parameters
        # Find top 3 parameters with most variation
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
        param_cols = [
            col for col in plot_df.columns if col.startswith(("data_", "label_"))
        ]

        if len(param_cols) > 1:
            # Select top varied parameters
            top_params = []
            for col in param_cols:
                if col in plot_df.columns:
                    n_unique = plot_df[col].nunique()
                    if n_unique > 1:
                        top_params.append((col, n_unique))

            top_params = sorted(top_params, key=lambda x: x[1], reverse=True)[:3]
            top_param_names = [p[0] for p in top_params]

            if len(top_param_names) >= 2:
                # Create scatter plots
                if len(top_param_names) == 2:
                    axes[1].scatter(
                        plot_df[top_param_names[0]],
                        plot_df[top_param_names[1]],
                        c=plot_df[y_metric],
                        cmap="viridis",
                        s=100,
                        alpha=0.6,
                    )
                    axes[1].set_xlabel(top_param_names[0])
                    axes[1].set_ylabel(top_param_names[1])
                else:
                    # 3D scatter
                    from mpl_toolkits.mplot3d import Axes3D

                    axes[1].remove()
                    ax3d = fig.add_subplot(122, projection="3d")
                    scatter = ax3d.scatter(
                        plot_df[top_param_names[0]],
                        plot_df[top_param_names[1]],
                        plot_df[top_param_names[2]],
                        c=plot_df[y_metric],
                        cmap="viridis",
                        s=100,
                        alpha=0.6,
                    )
                    ax3d.set_xlabel(top_param_names[0])
                    ax3d.set_ylabel(top_param_names[1])
                    ax3d.set_zlabel(top_param_names[2])

                axes[1].set_title(f"Parameter Space vs {y_metric}")
                plt.colorbar(scatter, ax=axes[1], label=y_metric)

        plt.tight_layout()
        plt.show()

    def export_all_results(self, export_dir: Union[str, Path]):
        """
        Export all grid search results.

        Parameters
        ----------
        export_dir : str or Path
            Directory to export results to.
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export summary
        if self.summary_df is not None:
            self.summary_df.to_csv(export_dir / "grid_search_summary.csv", index=False)

        # Export detailed results for each successful combination
        for result in self.results:
            if result["success"] and result["pipeline"] is not None:
                combo_dir = export_dir / f"combination_{result['combination_id']:03d}"
                result["pipeline"].export_results(combo_dir)

                # Save metadata
                import json

                metadata = {
                    "combination_id": result["combination_id"],
                    "data_config": result["data_config"],
                    "label_config": result["label_config"],
                    "metrics": result["metrics"],
                }

                with open(combo_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

        print(f"Exported all results to {export_dir}")


class MultiConfigPipeline:
    """
    Manages multiple ModelDevelopmentPipeline instances for comparison.
    """

    def __init__(self):
        self.pipelines = {}  # name -> ModelDevelopmentPipeline
        self.results = {}  # name -> results
        self.comparison_df = None

    def add_pipeline(
        self,
        name: str,
        symbol: str,
        train_start: str,
        train_end: str,
        strategy: BaseStrategy,
        data_config: dict,
        feature_config: dict,
        label_config: dict,
        model_params: dict,
        run_now: bool = True,
        cache_reports: bool = False,
        save: bool = True,
        verbose: bool = False,
    ):
        """
        Add a pipeline configuration to compare.

        Parameters
        ----------
        name : str
            Unique name for this configuration.
        symbol : str
            Trading instrument symbol.
        train_start : str
            Training start date.
        train_end : str
            Training end date.
        strategy : BaseStrategy
            Signal generating strategy.
        data_config : dict
            Bar construction configuration.
        feature_config : dict
            Feature engineering configuration.
        label_config : dict
            Triple-barrier labeling configuration.
        model_params : dict
            Model training configuration.
        run_now : bool, optional
            Run pipeline immediately (default: True).
        cache_reports : bool, optional
            Display cache reports (default: False).
        save : bool, optional
            Save model and metadata (default: True).
        verbose : bool, optional
            Print progress information (default: False).
        """
        # Create pipeline
        pipeline = ModelDevelopmentPipeline(
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            strategy=strategy,
            data_config=data_config,
            feature_config=feature_config,
            label_config=label_config,
            model_params=model_params,
        )

        self.pipelines[name] = pipeline

        # Run if requested
        if run_now:
            print(f"\nRunning pipeline: {name}")
            print("-" * 50)

            model, features, metrics, config = pipeline.run(
                cache_reports=cache_reports, save=save, verbose=verbose
            )

            self.results[name] = {
                "pipeline": pipeline,
                "model": model,
                "features": features,
                "metrics": metrics,
                "config": config,
            }

    def run_all(
        self, cache_reports: bool = False, save: bool = True, verbose: bool = True
    ):
        """Run all pipelines that haven't been run yet."""
        for name, pipeline in self.pipelines.items():
            if name not in self.results:
                print(f"\nRunning pipeline: {name}")
                print("-" * 50)

                model, features, metrics, config = pipeline.run(
                    cache_reports=cache_reports, save=save, verbose=verbose
                )

                self.results[name] = {
                    "pipeline": pipeline,
                    "model": model,
                    "features": features,
                    "metrics": metrics,
                    "config": config,
                }

    def compare_results(self):
        """Create comparison DataFrame of all pipeline results."""
        comparison_data = []

        for name, result in self.results.items():
            if result["metrics"]:
                row = {
                    "pipeline_name": name,
                    "cv_score": result["metrics"]["cv_results"]["best_score"],
                    "training_samples": result["metrics"]["training_samples"],
                    "feature_count": result["metrics"]["feature_count"],
                    "best_weighting_scheme": result["metrics"]["best_weighting_scheme"],
                    "model_type": type(result["model"].named_steps["clf"]).__name__,
                }

                # Add config parameters
                config = result["config"]
                for key, value in config.items():
                    if key not in ["strategy", "symbol"]:
                        row[key] = value

                comparison_data.append(row)

        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df

    def plot_comparison(self, metric: str = "cv_score", figsize: tuple = (12, 6)):
        """Plot comparison of pipeline results."""
        import matplotlib.pyplot as plt

        if self.comparison_df is None:
            self.compare_results()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar plot of main metric
        self.comparison_df.sort_values(metric, ascending=False).plot(
            x="pipeline_name", y=metric, kind="bar", ax=axes[0], legend=False
        )
        axes[0].set_title(f"{metric} by Pipeline")
        axes[0].set_ylabel(metric)
        axes[0].tick_params(axis="x", rotation=45)

        # Feature importance comparison (top 5 pipelines)
        top_n = min(5, len(self.comparison_df))
        top_pipelines = self.comparison_df.nlargest(top_n, metric)[
            "pipeline_name"
        ].tolist()

        # Collect top features from each pipeline
        top_features = {}
        for name in top_pipelines:
            if name in self.results:
                pipeline = self.results[name]["pipeline"]
                if hasattr(pipeline, "feature_importance"):
                    top_features[name] = pipeline.feature_importance.head(10)

        # Create subplot for feature importance comparison
        if top_features:
            ax = axes[1]
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_features)))

            for (name, fi_df), color in zip(top_features.items(), colors):
                ax.plot(
                    range(len(fi_df)),
                    fi_df["importance"],
                    marker="o",
                    label=name,
                    color=color,
                )

            ax.set_xlabel("Feature Rank")
            ax.set_ylabel("Importance")
            ax.set_title("Top 10 Feature Importance (Best Pipelines)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_best_pipeline(self, metric: str = "cv_score"):
        """Get the best performing pipeline."""
        if self.comparison_df is None:
            self.compare_results()

        best_row = self.comparison_df.loc[self.comparison_df[metric].idxmax()]
        best_name = best_row["pipeline_name"]

        return best_name, self.results[best_name]

    def export_comparison(self, export_dir: Union[str, Path]):
        """Export comparison results."""
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export comparison DataFrame
        if self.comparison_df is not None:
            self.comparison_df.to_csv(
                export_dir / "pipeline_comparison.csv", index=False
            )

        # Export each pipeline's results
        for name, result in self.results.items():
            pipeline_dir = export_dir / name
            result["pipeline"].export_results(pipeline_dir)

        print(f"Exported comparison results to {export_dir}")


# ===========================================================================
#                               EXAMPLE USAGE
# ===========================================================================


'''
def run_parameter_analysis():
    """
    Example of running model development with multiple parameter configurations.
    """
    
    # Base configuration (fixed parameters)
    base_config = {
        'symbol': 'EURUSD',
        'train_start': '2023-01-01',
        'train_end': '2023-12-31',
        'strategy': my_strategy,  # Your strategy instance
        'data_config': {
            'account_name': 'default_account',  # Fixed account
            # Variable parameters will be overridden by param_grid
        },
        'feature_config': {
            'func': calculate_features,  # Your feature calculation function
            'params': {}  # Feature parameters
        },
        'label_config': {
            'target_lookback': 20,  # Fixed
            'max_holding_period': {'days': 1},  # Fixed
            'min_ret': 0.0,  # Fixed
            'vertical_barrier_zero': True,  # Fixed
            'filter_as_series': True  # Fixed
        },
        'model_params': {
            'pipe_clf': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'param_grid': {
                'clf__max_depth': [3, 5, 7],
                'clf__min_samples_split': [2, 5, 10]
            },
            'cv_splits': 5,
            'rnd_search_iter': 10,
            'n_jobs': -1
        }
    }
    
    # Parameter grid to search
    param_grid = {
        'data_config': {
            'bar_type': ['time', 'tick'],
            'bar_size': ['M1', 'M5', 'M15'],
            'price': ['mid_price', 'bid', 'ask']
        },
        'label_config': {
            'profit_target': [1.0, 1.5, 2.0],
            'stop_loss': [1.0, 1.5, 2.0]
        }
    }
    
    # Create grid search instance
    grid_search = ParameterGridSearch()
    
    # Run grid search
    results_df = grid_search.run_grid_search(
        base_config=base_config,
        param_grid=param_grid,
        cache_reports=False,
        save=True,
        parallel=True,  # Run in parallel
        n_jobs=-1,  # Use all cores
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.sort_values('cv_score', ascending=False).head(10))
    
    # Plot results
    grid_search.plot_results(
        x_param='data_bar_size',
        y_metric='cv_score',
        group_by='data_bar_type'
    )
    
    # Get best combination
    best_result = grid_search.get_best_combination()
    print(f"\nBest combination ID: {best_result['combination_id']}")
    print(f"Best CV score: {best_result['metrics']['cv_results']['best_score']:.4f}")
    print(f"Parameters: {best_result['data_config']}")
    
    # Export all results
    grid_search.export_all_results("./grid_search_results")
    
    return grid_search, results_df


# Alternative: Using MultiConfigPipeline for explicit comparisons
def run_explicit_comparisons():
    """
    Compare specific configurations explicitly.
    """
    
    multi_pipeline = MultiConfigPipeline()
    
    # Configuration 1: Time bars with conservative parameters
    multi_pipeline.add_pipeline(
        name="time_bars_conservative",
        symbol="EURUSD",
        train_start="2023-01-01",
        train_end="2023-12-31",
        strategy=my_strategy,
        data_config={
            'account_name': 'default',
            'bar_type': 'time',
            'bar_size': 'M15',
            'price': 'mid_price'
        },
        feature_config=feature_config,
        label_config={
            'target_lookback': 20,
            'profit_target': 1.0,
            'stop_loss': 1.0,
            'max_holding_period': {'days': 1},
            'min_ret': 0.0
        },
        model_params=model_params,
        run_now=True,
        verbose=False
    )
    
    # Configuration 2: Tick bars with aggressive parameters
    multi_pipeline.add_pipeline(
        name="tick_bars_aggressive",
        symbol="EURUSD",
        train_start="2023-01-01",
        train_end="2023-12-31",
        strategy=my_strategy,
        data_config={
            'account_name': 'default',
            'bar_type': 'tick',
            'bar_size': 1000,  # 1000 tick bars
            'price': 'mid_price'
        },
        feature_config=feature_config,
        label_config={
            'target_lookback': 10,
            'profit_target': 2.0,
            'stop_loss': 2.0,
            'max_holding_period': {'hours': 6},
            'min_ret': 0.001
        },
        model_params=model_params,
        run_now=True,
        verbose=False
    )
    
    # Configuration 3: Volume bars
    multi_pipeline.add_pipeline(
        name="volume_bars",
        symbol="EURUSD",
        train_start="2023-01-01",
        train_end="2023-12-31",
        strategy=my_strategy,
        data_config={
            'account_name': 'default',
            'bar_type': 'volume',
            'bar_size': 10000000,  # 10 million volume
            'price': 'mid_price'
        },
        feature_config=feature_config,
        label_config=label_config,
        model_params=model_params,
        run_now=True,
        verbose=False
    )
    
    # Compare all pipelines
    comparison_df = multi_pipeline.compare_results()
    print("\nPipeline Comparison:")
    print(comparison_df.to_string())
    
    # Plot comparison
    multi_pipeline.plot_comparison()
    
    # Get best pipeline
    best_name, best_result = multi_pipeline.get_best_pipeline()
    print(f"\nBest pipeline: {best_name}")
    print(f"CV score: {best_result['metrics']['cv_results']['best_score']:.4f}")
    
    # Export comparison
    multi_pipeline.export_comparison("./pipeline_comparison")
    
    return multi_pipeline, comparison_df


# Simple grid search
grid_search = ParameterGridSearch()
results = grid_search.run_grid_search(
    base_config=base_config,
    param_grid={
        'data_config': {
            'bar_type': ['time', 'tick'],
            'bar_size': ['M1', 'M5', 'M15'],
            'price': ['mid_price']
        }
    },
    parallel=True
)

# Access best pipeline for further analysis
best_result = grid_search.get_best_combination()
best_pipeline = best_result['pipeline']  # ModelDevelopmentPipeline instance

# Analyze intermediate data
print(best_pipeline.get_data_summary())
best_pipeline.plot_feature_importance()
'''
