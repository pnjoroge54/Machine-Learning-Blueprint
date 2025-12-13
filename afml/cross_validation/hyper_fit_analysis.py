import base64
import json
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display


def _get_param_columns(cv_results: pd.DataFrame) -> List[str]:
    """Extract parameter columns from cv_results."""
    return [col for col in cv_results.columns if col.startswith("param_")]


def _get_param_value(cv_results: pd.DataFrame, param_name: str, row_idx: int = 0) -> Any:
    """Safely get parameter value from cv_results."""
    if param_name in cv_results.columns:
        return cv_results.iloc[row_idx][param_name]
    return None


def _safe_groupby_param(cv_results: pd.DataFrame, param_col: str) -> Optional[pd.DataFrame]:
    """Safely group by parameter column, handling missing columns."""
    if param_col not in cv_results.columns:
        return None

    try:
        param_stats = (
            cv_results.groupby(param_col)[["mean_test_score", "std_test_score", "mean_fit_time"]]
            .agg(
                {
                    "mean_test_score": ["mean", "std", "count"],
                    "std_test_score": "mean",
                    "mean_fit_time": "mean",
                }
            )
            .round(4)
        )

        param_stats.columns = ["score_mean", "score_std", "count", "fold_std_mean", "time_mean"]
        param_stats = param_stats.sort_values("score_mean", ascending=False)
        return param_stats
    except Exception as e:
        warnings.warn(f"Failed to group by {param_col}: {e}")
        return None


def analyze_hyperparameter_results(
    cv_results: pd.DataFrame,
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
) -> Dict:
    """
    Comprehensive analysis of hyperparameter search results.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame from GridSearchCV or similar
    target_metric : str, default="mean_test_score"
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time in seconds
    stability_threshold : float, default=0.03
        Maximum standard deviation for stable models

    Returns
    -------
    Dict
        Dictionary with analysis results including generated plots
    """
    analysis = {}
    analysis["plots"] = {}

    print("=" * 80)
    print("HYPERPARAMETER ANALYSIS REPORT")
    print("=" * 80)

    # 1. BASIC METRICS
    top_models = cv_results.sort_values(target_metric, ascending=False).head(10)
    analysis["top_models"] = top_models

    print(f"\n1. TOP PERFORMING MODELS (sorted by {target_metric}):")
    print("-" * 50)

    # Select available columns
    available_columns = []
    for col in ["params", "mean_test_score", "std_test_score", "mean_fit_time"]:
        if col in cv_results.columns:
            available_columns.append(col)

    if available_columns:
        print(top_models[available_columns].to_string())
    else:
        print("No standard columns found in results")

    # 2. PERFORMANCE ANALYSIS
    print(f"\n2. PERFORMANCE SUMMARY:")
    print("-" * 50)

    if target_metric in cv_results.columns:
        mean_score = cv_results[target_metric].mean()
        std_score = cv_results[target_metric].std()
        max_score = cv_results[target_metric].max()
        min_score = cv_results[target_metric].min()

        print(f"Average {target_metric}: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Best {target_metric}: {max_score:.4f}")
        print(f"Worst {target_metric}: {min_score:.4f}")
        print(f"Performance Range: {max_score - min_score:.4f}")
    else:
        print(f"Target metric '{target_metric}' not found in results")
        mean_score = max_score = min_score = std_score = 0

    # 3. STABILITY ANALYSIS
    print(f"\n3. STABILITY ANALYSIS:")
    print("-" * 50)

    if "std_test_score" in cv_results.columns:
        stable_models = cv_results[cv_results["std_test_score"] <= stability_threshold]
        analysis["stable_models"] = stable_models

        if not stable_models.empty:
            print(
                f"Models with stable performance (std ≤ {stability_threshold}): {len(stable_models)}"
            )
            best_stable = stable_models.nlargest(1, target_metric)
            print(
                f"Best stable model: {best_stable[target_metric].iloc[0]:.4f} ± {best_stable['std_test_score'].iloc[0]:.4f}"
            )
        else:
            print(f"No models meet stability threshold of {stability_threshold}")
    else:
        print("No stability metrics available")

    # 4. TIME-EFFICIENCY ANALYSIS
    print(f"\n4. TIME-EFFICIENCY ANALYSIS:")
    print("-" * 50)

    if "mean_fit_time" in cv_results.columns:
        cv_results["efficiency_score"] = cv_results[target_metric] / cv_results["mean_fit_time"]

        if time_constraint:
            time_efficient = cv_results[cv_results["mean_fit_time"] <= time_constraint]
            if not time_efficient.empty:
                best_time_efficient = time_efficient.nlargest(1, target_metric)
                print(f"Best model under {time_constraint}s constraint:")
                print(f"  Score: {best_time_efficient[target_metric].iloc[0]:.4f}")
                print(f"  Time: {best_time_efficient['mean_fit_time'].iloc[0]:.2f}s")
            else:
                print(f"No models meet time constraint of {time_constraint}s")
    else:
        print("No training time data available")

    # 5. HYPERPARAMETER TREND ANALYSIS
    print(f"\n5. HYPERPARAMETER TRENDS:")
    print("-" * 50)

    param_columns = _get_param_columns(cv_results)

    if param_columns:
        for param in param_columns[:5]:  # Limit to first 5 parameters
            param_name = param.replace("param_", "")
            param_stats = _safe_groupby_param(cv_results, param)

            if param_stats is not None and not param_stats.empty:
                print(f"\nParameter: {param_name}")
                print(
                    f"Optimal value: {param_stats.index[0]} (score: {param_stats['score_mean'].iloc[0]:.4f})"
                )
                print(f"Performance by value:")
                print(param_stats.to_string())
    else:
        print("No hyperparameter columns found")

    # 6. CROSS-VALIDATION FOLD CONSISTENCY
    print(f"\n6. CROSS-VALIDATION CONSISTENCY:")
    print("-" * 50)

    fold_columns = [col for col in cv_results.columns if "split" in col and "test" in col]

    if fold_columns:
        fold_scores = cv_results[fold_columns]
        fold_means = fold_scores.mean(axis=0)
        fold_stds = fold_scores.std(axis=0)

        print(f"Fold performance consistency:")
        for i, (fold_mean, fold_std) in enumerate(zip(fold_means, fold_stds)):
            print(f"  Fold {i}: {fold_mean:.4f} ± {fold_std:.4f}")

        problem_folds = [(i, std) for i, std in enumerate(fold_stds) if std > stability_threshold]
        if problem_folds:
            print(f"\n⚠️  High variance folds detected (std > {stability_threshold}):")
            for fold, std in problem_folds:
                print(f"  Fold {fold}: std = {std:.4f}")
    else:
        print("No fold-specific data available")

    # 7. MODEL SELECTION RECOMMENDATION
    print(f"\n7. MODEL SELECTION RECOMMENDATION:")
    print("-" * 50)

    if target_metric in cv_results.columns:
        best_by_score = cv_results.nlargest(1, target_metric)

        if not best_by_score.empty:
            best_score = best_by_score[target_metric].iloc[0]
            best_std = best_by_score.get("std_test_score", pd.Series([0])).iloc[0]

            # Consider stability vs performance trade-off
            if "std_test_score" in cv_results.columns and "stable_models" in analysis:
                stable_models = analysis["stable_models"]
                if len(stable_models) > 0:
                    best_stable = stable_models.nlargest(1, target_metric)
                    stable_score = best_stable[target_metric].iloc[0]
                    stable_std = best_stable["std_test_score"].iloc[0]

                    score_diff = best_score - stable_score
                    if score_diff < 0.005:
                        print(f"✅ RECOMMENDATION: Choose stable model")
                        print(f"   Score: {stable_score:.4f} (vs best: {best_score:.4f})")
                        print(f"   Stability: {stable_std:.4f} (vs best: {best_std:.4f})")
                        print(f"   Performance difference: {score_diff:.4f} (insignificant)")
                        recommended_model = best_stable
                    else:
                        print(f"✅ RECOMMENDATION: Choose best performing model")
                        print(f"   Score: {best_score:.4f} (vs stable: {stable_score:.4f})")
                        print(f"   Stability: {best_std:.4f} (slightly higher variance)")
                        print(f"   Performance gain: {score_diff:.4f} (worth the risk)")
                        recommended_model = best_by_score
                else:
                    print(f"⚠️  RECOMMENDATION: No stable models found")
                    print(f"   Best model: {best_score:.4f} ± {best_std:.4f}")
                    recommended_model = best_by_score
            else:
                print(f"✅ RECOMMENDATION: Choose best performing model")
                print(f"   Score: {best_score:.4f}")
                recommended_model = best_by_score

            # Add practical considerations
            if "params" in recommended_model.columns:
                recommended_params = recommended_model["params"].iloc[0]
                print(f"\n🎯 RECOMMENDED HYPERPARAMETERS:")
                for key, value in recommended_params.items():
                    print(f"   {key}: {value}")
            else:
                print("No parameter information available")
        else:
            print("No models found in results")
            recommended_model = pd.DataFrame()
    else:
        print(f"Target metric '{target_metric}' not found")
        recommended_model = pd.DataFrame()

    # 8. GENERATE VISUALIZATIONS
    print(f"\n8. GENERATING VISUALIZATIONS...")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 8.1 Performance distribution
    ax = axes[0, 0]
    if target_metric in cv_results.columns:
        ax.hist(cv_results[target_metric], bins=20, alpha=0.7)
        if not recommended_model.empty:
            best_score_val = recommended_model[target_metric].iloc[0]
            ax.axvline(
                best_score_val, color="red", linestyle="--", label=f"Best: {best_score_val:.4f}"
            )
        ax.axvline(mean_score, color="green", linestyle="--", label=f"Mean: {mean_score:.4f}")
        ax.set_title("Performance Distribution")
        ax.set_xlabel("Test Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, f"No {target_metric} data", ha="center", va="center", transform=ax.transAxes
        )

    # 8.2 Score vs Stability
    ax = axes[0, 1]
    if target_metric in cv_results.columns and "std_test_score" in cv_results.columns:
        ax.scatter(cv_results[target_metric], cv_results["std_test_score"], alpha=0.5)
        if not recommended_model.empty:
            ax.scatter(
                recommended_model[target_metric].iloc[0],
                (
                    recommended_model["std_test_score"].iloc[0]
                    if "std_test_score" in recommended_model.columns
                    else 0
                ),
                color="red",
                s=100,
                label="Recommended",
            )
        ax.axhline(
            y=stability_threshold, color="orange", linestyle="--", label="Stability threshold"
        )
        ax.set_title("Score vs Stability")
        ax.set_xlabel("Mean Test Score")
        ax.set_ylabel("Std Test Score")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Score/Stability data missing",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 8.3 Score vs Training Time
    ax = axes[0, 2]
    if target_metric in cv_results.columns and "mean_fit_time" in cv_results.columns:
        ax.scatter(cv_results[target_metric], cv_results["mean_fit_time"], alpha=0.5)
        if not recommended_model.empty:
            ax.scatter(
                recommended_model[target_metric].iloc[0],
                (
                    recommended_model["mean_fit_time"].iloc[0]
                    if "mean_fit_time" in recommended_model.columns
                    else 0
                ),
                color="red",
                s=100,
                label="Recommended",
            )
        ax.set_title("Score vs Training Time")
        ax.set_xlabel("Mean Test Score")
        ax.set_ylabel("Training Time (s)")
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "Score/Time data missing", ha="center", va="center", transform=ax.transAxes
        )

    # 8.4-8.5 Parameter importance plots
    param_columns = _get_param_columns(cv_results)

    # Plot first available parameter
    ax = axes[1, 0]
    if param_columns and len(param_columns) >= 1:
        param_col = param_columns[0]
        param_name = param_col.replace("param_", "")
        param_groups = cv_results.groupby(param_col)[target_metric].mean()

        # Try to sort if values are numeric
        try:
            param_groups = param_groups.sort_index()
        except:
            pass

        if len(param_groups) <= 10:  # Only plot if reasonable number of categories
            if param_groups.index.dtype.kind in "iufc":  # Numeric index
                ax.plot(param_groups.index, param_groups.values, marker="o")
            else:
                x_positions = np.arange(len(param_groups))
                ax.bar(x_positions, param_groups.values)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(param_groups.index, rotation=45, ha="right")

            ax.set_title(f"{param_name} vs Performance")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Mean Test Score")
        else:
            ax.text(
                0.5,
                0.5,
                f"Too many {param_name} values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    else:
        ax.text(
            0.5, 0.5, "No parameter data found", ha="center", va="center", transform=ax.transAxes
        )

    # Plot second available parameter
    ax = axes[1, 1]
    if param_columns and len(param_columns) >= 2:
        param_col = param_columns[1]
        param_name = param_col.replace("param_", "")
        param_groups = cv_results.groupby(param_col)[target_metric].mean()

        # Try to sort if values are numeric
        try:
            param_groups = param_groups.sort_index()
        except:
            pass

        if len(param_groups) <= 10:
            if param_groups.index.dtype.kind in "iufc":  # Numeric index
                ax.plot(param_groups.index, param_groups.values, marker="s", color="green")
            else:
                x_positions = np.arange(len(param_groups))
                ax.bar(x_positions, param_groups.values, color="green")
                ax.set_xticks(x_positions)
                ax.set_xticklabels(param_groups.index, rotation=45, ha="right")

            ax.set_title(f"{param_name} vs Performance")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Mean Test Score")
        else:
            ax.text(
                0.5,
                0.5,
                f"Too many {param_name} values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    else:
        ax.text(
            0.5, 0.5, "No second parameter found", ha="center", va="center", transform=ax.transAxes
        )

    # 8.6 Fold consistency
    ax = axes[1, 2]
    if "fold_means" in locals() and len(fold_means) > 0:
        fold_labels = [f"Fold {i}" for i in range(len(fold_means))]
        ax.bar(
            fold_labels,
            fold_means,
            yerr=fold_stds,
            capsize=5,
            alpha=0.7,
            error_kw={"ecolor": "red"},
        )
        ax.set_title("Cross-Validation Fold Performance")
        ax.set_ylabel("Mean Score")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(
            0.5, 0.5, "No fold data available", ha="center", va="center", transform=ax.transAxes
        )

    plt.tight_layout()

    # Save plot to analysis dict
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    analysis["plots"]["main_comparison"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # 9. PRACTICAL INTERPRETATION FOR TRADING
    print(f"\n9. PRACTICAL INTERPRETATION FOR TRADING:")
    print("-" * 50)

    if target_metric in cv_results.columns and not cv_results.empty:
        best_score_val = cv_results[target_metric].max()

        print(f"Expected Strategy Performance:")
        print(f"  • Best {target_metric}: {best_score_val:.4f}")

        if "std_test_score" in cv_results.columns:
            best_std_val = cv_results.loc[cv_results[target_metric].idxmax(), "std_test_score"]
            print(
                f"  • Cross-validation Consistency: {'Good' if best_std_val < 0.03 else 'Moderate'}"
            )

            # Risk assessment
            if best_std_val > 0.04:
                print(f"\n⚠️  RISK WARNING: High variance in CV folds")
                print(f"   Strategy may perform inconsistently in live trading")
            elif best_std_val < 0.02:
                print(f"\n✅ LOW RISK: Excellent consistency across CV folds")
                print(f"   Strategy likely to perform similarly in live trading")

    # 10. FINAL SUMMARY
    analysis["summary"] = {
        "best_model": recommended_model,
        "best_score": best_score if "best_score" in locals() else 0,
        "best_std": best_std if "best_std" in locals() else 0,
        "recommended_params": recommended_params if "recommended_params" in locals() else {},
        "risk_level": (
            "HIGH"
            if best_std > 0.04
            else "MEDIUM" if best_std > 0.02 else "LOW" if "best_std" in locals() else "UNKNOWN"
        ),
    }

    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return analysis


def analyze_your_results(cv_results: pd.DataFrame) -> Dict:
    """
    Custom analysis for your specific hyperparameter search results.
    Handles missing parameters gracefully.
    """
    analysis = {}

    print("SPECIFIC INSIGHTS FROM YOUR RESULTS:")
    print("=" * 80)

    # Your specific observations
    print("\n1. KEY OBSERVATIONS:")
    print("-" * 50)

    # Find best model safely
    if cv_results.empty:
        print("No results to analyze")
        return analysis

    # Determine target metric
    possible_metrics = ["mean_test_score", "mean_train_score", "mean_score"]
    target_metric = next((m for m in possible_metrics if m in cv_results.columns), None)

    if target_metric:
        best_model = cv_results.sort_values(by=target_metric, ascending=False).iloc[0]
    else:
        print("No performance metrics found")
        return analysis

    analysis["best_model"] = best_model

    # Extract parameter values safely - look for common parameter names
    param_patterns = {
        "max_depth": ["max_depth", "depth"],
        "n_estimators": ["n_estimators", "estimators", "trees"],
        "learning_rate": ["learning_rate", "eta"],
        "min_samples_split": ["min_samples_split", "min_split"],
        "min_samples_leaf": ["min_samples_leaf", "min_leaf"],
        "C": ["C", "regularization"],
        "gamma": ["gamma"],
    }

    extracted_params = {}
    param_columns = _get_param_columns(cv_results)

    for param_col in param_columns:
        param_name = param_col.replace("param_", "").lower()
        for key, patterns in param_patterns.items():
            if any(pattern in param_name for pattern in patterns):
                extracted_params[key] = best_model[param_col]
                break

    # Add extracted parameters to analysis
    for key, value in extracted_params.items():
        analysis[key] = value
        print(f"{key}: {value}")

    print(f"Best Model {target_metric}: {best_model[target_metric]:.4f}")

    if "std_test_score" in best_model.index:
        print(f"Standard Deviation: {best_model['std_test_score']:.4f}")

    if "mean_fit_time" in best_model.index:
        print(f"Training Time: {best_model['mean_fit_time']:.2f}s")

    # Compare with simpler models if max_depth parameter exists
    max_depth_col = None
    for param_col in param_columns:
        if "max_depth" in param_col.lower():
            max_depth_col = param_col
            break

    if max_depth_col and max_depth_col in cv_results.columns:
        simple_models = cv_results[cv_results[max_depth_col] <= 4]
        if not simple_models.empty:
            best_simple = simple_models.nlargest(1, target_metric)
            print(f"\nBest Simple Model ({max_depth_col} ≤ 4):")
            print(f"  {max_depth_col}={best_simple[max_depth_col].iloc[0]}")
            print(
                f"  {target_metric}: {best_simple[target_metric].iloc[0]:.4f} (vs best: {best_model[target_metric]:.4f})"
            )

            analysis["best_simple_model"] = best_simple.iloc[0].to_dict()
            analysis["performance_diff"] = (
                best_model[target_metric] - best_simple[target_metric].iloc[0]
            )

    # Performance saturation analysis for depth if available
    if max_depth_col and max_depth_col in cv_results.columns:
        print(f"\n2. PERFORMANCE SATURATION:")
        print("-" * 50)

        depth_groups = cv_results.groupby(max_depth_col)[target_metric].max()
        print("Maximum performance by max_depth:")
        for depth, score in depth_groups.items():
            print(f"  depth={depth}: {score:.4f}")

        analysis["depth_groups"] = depth_groups.to_dict()

    # 3. ACTIONABLE RECOMMENDATIONS
    print(f"\n3. ACTIONABLE RECOMMENDATIONS:")
    print("-" * 50)

    best_score_val = best_model[target_metric]

    if best_score_val > 0.68:
        print("✅ Excellent performance achieved!")
        print("   Consider testing with additional features or ensemble methods")
        analysis["performance_level"] = "EXCELLENT"
    elif best_score_val < 0.65:
        print("⚠️  Performance could be improved")
        print("   Consider: feature engineering, different model architecture, or more data")
        analysis["performance_level"] = "MODERATE"
    else:
        print("✅ Good baseline performance achieved")
        print("   Ready for forward testing with proper risk management")
        analysis["performance_level"] = "GOOD"

    # 4. PRODUCTION CONSIDERATIONS
    print(f"\n4. PRODUCTION CONSIDERATIONS:")
    print("-" * 50)

    if "mean_score_time" in cv_results.columns:
        avg_score_time = cv_results["mean_score_time"].mean()
        print(f"Expected Inference Speed: ~{avg_score_time*1000:.1f}ms per prediction")
        analysis["avg_inference_time"] = avg_score_time * 1000

    if "mean_fit_time" in cv_results.columns:
        print(
            f"Training Time Range: {cv_results['mean_fit_time'].min():.2f}s to {cv_results['mean_fit_time'].max():.2f}s"
        )
        analysis["training_time_range"] = (
            cv_results["mean_fit_time"].min(),
            cv_results["mean_fit_time"].max(),
        )

    # Memory considerations for tree-based models
    n_estimators_col = None
    for param_col in param_columns:
        if "n_estimators" in param_col.lower():
            n_estimators_col = param_col
            break

    if n_estimators_col and n_estimators_col in cv_results.columns:
        avg_estimators = cv_results[n_estimators_col].mean()
        print(f"Average Model Size: ~{avg_estimators:.0f} trees")
        analysis["avg_estimators"] = avg_estimators

    # Add stability rating
    if "std_test_score" in best_model.index:
        stability_score = best_model["std_test_score"]
        analysis["stability"] = (
            "HIGH" if stability_score < 0.02 else "MEDIUM" if stability_score < 0.04 else "LOW"
        )

    return analysis


def generate_hyperparameter_markdown_report(
    cv_results: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    filename: Optional[Union[Path, str]] = None,
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
    return_markdown: bool = False,
) -> Union[Path, Tuple[Path, str]]:
    """
    Generate comprehensive markdown report from hyperparameter analysis.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame
    strategy_config : dict, optional
        Strategy configuration dictionary
    filename : Path or str, optional
        Path to save the markdown report
    target_metric : str, default="mean_test_score"
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time
    stability_threshold : float, default=0.03
        Stability threshold
    return_markdown : bool, default=False
        If True, return markdown string along with path

    Returns
    -------
    Path or Tuple[Path, str]
        Path to the generated markdown file, or tuple of (path, markdown_string)
    """

    # Run analysis functions
    print("🔍 Running hyperparameter analysis...")
    analysis_results = analyze_hyperparameter_results(
        cv_results,
        target_metric=target_metric,
        time_constraint=time_constraint,
        stability_threshold=stability_threshold,
    )

    specific_analysis = analyze_your_results(cv_results)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start building markdown content
    md_content = []

    # Strategy Configuration Table
    if strategy_config:
        md_content.append("## ⚙️ Strategy Configuration")
        md_content.append("")
        md_content.append("| Parameter | Value | Description |")
        md_content.append("|-----------|-------|-------------|")

        config_descriptions = {
            "strategy": "Trading strategy name",
            "symbol": "Trading instrument",
            "account_name": "Trading account identifier",
            "bar_type": "Bar type (tick/volume/time)",
            "bar_size": "Bar timeframe",
            "price": "Price type (bid/ask/mid)",
            "target_lookback": "Target calculation lookback periods",
            "profit_target": "Profit target in risk multiples",
            "stop_loss": "Stop loss in risk multiples",
            "max_holding_period": "Maximum holding period",
            "min_ret": "Minimum return threshold",
            "vertical_barrier_zero": "Vertical barrier at zero crossing",
            "filter_as_series": "Filter as time series",
        }

        for key, value in strategy_config.items():
            description = config_descriptions.get(key, "No description")
            if isinstance(value, dict):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            md_content.append(f"| `{key}` | `{value_str}` | {description} |")

        md_content.append("")
        md_content.append("---")
        md_content.append("")

    # Header
    md_content.append("# 📊 Hyperparameter Tuning Analysis Report")
    md_content.append(f"*Generated on: {timestamp}*  ")
    md_content.append("")

    # Executive Summary
    md_content.append("## 🎯 Executive Summary")
    md_content.append("")

    summary = analysis_results.get("summary", {})
    best_model = summary.get("best_model", pd.Series())

    if not best_model.empty:
        best_score = summary.get("best_score", 0)
        best_std = summary.get("best_std", 0)
        risk_level = summary.get("risk_level", "UNKNOWN")

        md_content.append("### Key Findings")
        md_content.append("")
        md_content.append(f"- **Best Model Performance**: `{best_score:.4f} ± {best_std:.4f}`")
        md_content.append(f"- **Risk Level**: `{risk_level}`")
        md_content.append(
            f"- **Expected Accuracy**: `~{summary.get('expected_accuracy', 0)*100:.1f}%`"
        )
        md_content.append("")

        if risk_level == "HIGH":
            md_content.append(
                "> ⚠️ **Warning**: High variance detected. Model may perform inconsistently in live trading."
            )
        elif risk_level == "LOW":
            md_content.append(
                "> ✅ **Excellent**: Model shows high consistency across validation folds."
            )
        md_content.append("")

    # Main Visualization
    md_content.append("## 📈 Main Visualization")
    md_content.append("")

    if "plots" in analysis_results and "main_comparison" in analysis_results["plots"]:
        md_content.append("### Hyperparameter Analysis Overview")
        md_content.append("")
        md_content.append(
            f'<img src="data:image/png;base64,{analysis_results["plots"]["main_comparison"]}" width="800">'
        )
        md_content.append("")

    # Performance Overview
    md_content.append("## 📊 Performance Overview")
    md_content.append("")

    if target_metric in cv_results.columns:
        md_content.append(f"**Total Models Evaluated**: `{len(cv_results)}`  ")
        md_content.append(
            f"**Performance Range**: `{cv_results[target_metric].max():.4f}` - `{cv_results[target_metric].min():.4f}`  "
        )
        md_content.append(
            f"**Average Performance**: `{cv_results[target_metric].mean():.4f} ± {cv_results[target_metric].std():.4f}`  "
        )
        md_content.append("")

    # Performance Distribution Plot
    if "plots" in analysis_results and "performance_dist" in analysis_results["plots"]:
        md_content.append("### Performance Distribution")
        md_content.append("")
        md_content.append(
            f'<img src="data:image/png;base64,{analysis_results["plots"]["performance_dist"]}" width="600">'
        )
        md_content.append("")

    # Top Models Comparison
    md_content.append("## 🏆 Top Models Comparison")
    md_content.append("")

    if "top_models" in analysis_results:
        top_models = analysis_results["top_models"].head(5)

        md_content.append("| Rank | Mean Score | Std Score | Fit Time (s) | Efficiency |")
        md_content.append("|------|------------|-----------|--------------|------------|")

        for i, (_, row) in enumerate(top_models.iterrows()):
            efficiency = row.get(
                "efficiency_score",
                (
                    row[target_metric] / row["mean_fit_time"]
                    if "mean_fit_time" in row and row["mean_fit_time"] > 0
                    else 0
                ),
            )
            std_score = row.get("std_test_score", 0)
            fit_time = row.get("mean_fit_time", 0)

            md_content.append(
                f"| {i+1} | `{row[target_metric]:.4f}` | `{std_score:.4f}` | `{fit_time:.2f}` | `{efficiency:.2f}` |"
            )

        md_content.append("")

    # Stability Analysis
    md_content.append("## 🛡️ Stability Analysis")
    md_content.append("")

    if "stable_models" in analysis_results:
        stable_models = analysis_results["stable_models"]

        if not stable_models.empty:
            md_content.append(f"**Models meeting stability threshold**: `{len(stable_models)}`  ")
            md_content.append("")

            best_stable = stable_models.nlargest(1, target_metric)
            if not best_stable.empty:
                md_content.append("### Best Stable Model")
                md_content.append(f"- **Score**: `{best_stable[target_metric].iloc[0]:.4f}`  ")
                md_content.append(
                    f"- **Standard Deviation**: `{best_stable['std_test_score'].iloc[0]:.4f}`  "
                )
                md_content.append("")
        else:
            md_content.append("❌ No models meet the stability threshold.  ")
            md_content.append("")

    # Time-Efficiency Analysis
    md_content.append("## ⏱️ Time-Efficiency Analysis")
    md_content.append("")

    if "mean_fit_time" in cv_results.columns:
        time_stats = cv_results["mean_fit_time"].describe()
        md_content.append("### Training Time Statistics")
        md_content.append("")
        md_content.append(f"- **Fastest Model**: `{time_stats['min']:.2f}s`  ")
        md_content.append(f"- **Slowest Model**: `{time_stats['max']:.2f}s`  ")
        md_content.append(f"- **Average Time**: `{time_stats['mean']:.2f}s`  ")
        md_content.append(f"- **Median Time**: `{time_stats['50%']:.2f}s`  ")
        md_content.append("")

    # Hyperparameter Trends
    md_content.append("## 📊 Hyperparameter Trends")
    md_content.append("")

    param_columns = _get_param_columns(cv_results)

    if param_columns:
        md_content.append("### Parameter Impact Analysis")
        md_content.append("")

        for param in param_columns[:3]:  # Show top 3 parameters
            param_name = param.replace("param_", "")

            param_stats = _safe_groupby_param(cv_results, param)

            if param_stats is not None and not param_stats.empty:
                md_content.append(f"#### {param_name}")
                md_content.append("| Value | Mean Score | Score Std | Count | Avg Time (s) |")
                md_content.append("|-------|------------|-----------|-------|--------------|")

                for value, row in param_stats.iterrows():
                    md_content.append(
                        f"| `{value}` | `{row['score_mean']:.4f}` | `{row['score_std']:.4f}` | `{int(row['count'])}` | `{row['time_mean']:.2f}` |"
                    )

                md_content.append("")

    # Model Selection Recommendations
    md_content.append("## 🎯 Model Selection Recommendations")
    md_content.append("")

    if "summary" in analysis_results:
        summary = analysis_results["summary"]
        recommended_params = summary.get("recommended_params", {})

        md_content.append("### Final Recommendation")
        md_content.append("")

        if "best_model" in summary and not summary["best_model"].empty:
            best_model = summary["best_model"]

            if isinstance(best_model, pd.Series):
                md_content.append(
                    f"**Selected Model Performance**: `{best_model.get(target_metric, 0):.4f} ± {best_model.get('std_test_score', 0):.4f}`  "
                )
                md_content.append(
                    f"**Training Time**: `{best_model.get('mean_fit_time', 0):.2f}s`  "
                )
                md_content.append("")

            md_content.append("### Recommended Hyperparameters")
            md_content.append("")
            if recommended_params:
                md_content.append("```python")
                for key, value in recommended_params.items():
                    md_content.append(f"{key} = {value}")
                md_content.append("```")
            else:
                md_content.append("No parameter information available")
            md_content.append("")

    # Specific Insights
    md_content.append("## 🔍 Specific Insights")
    md_content.append("")

    if specific_analysis:
        md_content.append("### Model Architecture Analysis")
        md_content.append("")

        if "performance_level" in specific_analysis:
            perf_level = specific_analysis["performance_level"]
            stability = specific_analysis.get("stability", "UNKNOWN")

            md_content.append(f"**Overall Performance**: `{perf_level}`  ")
            md_content.append(f"**Stability Rating**: `{stability}`  ")
            md_content.append("")

        if "best_model" in specific_analysis:
            best_model = specific_analysis["best_model"]

            md_content.append("### Best Model Details")
            md_content.append("")

            # Extract and display any found parameters
            for key in ["max_depth", "n_estimators", "learning_rate", "C", "gamma"]:
                if key in specific_analysis and specific_analysis[key] is not None:
                    md_content.append(f"- **{key}**: `{specific_analysis[key]}`  ")

            if target_metric in best_model:
                md_content.append(f"- **{target_metric}**: `{best_model[target_metric]:.4f}`  ")

            if "std_test_score" in best_model:
                md_content.append(
                    f"- **Standard Deviation**: `{best_model['std_test_score']:.4f}`  "
                )

            if "mean_fit_time" in best_model:
                md_content.append(f"- **Training Time**: `{best_model['mean_fit_time']:.2f}s`  ")

            md_content.append("")

    # Practical Trading Implications
    md_content.append("## 💼 Practical Trading Implications")
    md_content.append("")

    if "summary" in analysis_results:
        summary = analysis_results["summary"]
        best_score = summary.get("best_score", 0)
        best_std = summary.get("best_std", 0)

        md_content.append("### Performance Expectations")
        md_content.append("")

        expected_win_rate = best_score  # Approximation

        md_content.append(f"- **Expected Win Rate**: `~{expected_win_rate*100:.1f}%`  ")
        md_content.append(
            f"- **Performance Consistency**: `{'High' if best_std < 0.02 else 'Moderate' if best_std < 0.04 else 'Low'}`  "
        )
        md_content.append(f"- **Risk Assessment**: `{summary.get('risk_level', 'UNKNOWN')}`  ")
        md_content.append("")

        # Trading-specific recommendations
        md_content.append("### Trading Strategy Considerations")
        md_content.append("")

        if best_std > 0.04:
            md_content.append("> ⚠️ **High Risk Strategy Detected**  ")
            md_content.append("> - Consider reducing position sizes  ")
            md_content.append("> - Implement strict stop-loss mechanisms  ")
            md_content.append("> - Monitor performance closely during initial deployment  ")
        elif best_std < 0.02:
            md_content.append("> ✅ **Stable Strategy Detected**  ")
            md_content.append("> - Can consider standard position sizing  ")
            md_content.append("> - Strategy likely to perform consistently  ")
            md_content.append("> - Lower monitoring frequency acceptable  ")
        md_content.append("")

    # Detailed Results
    md_content.append("## 📋 Detailed Results")
    md_content.append("")

    # Top 10 models in detail
    if "top_models" in analysis_results:
        md_content.append("### Complete Results (Top 10 Models)")
        md_content.append("")

        top_models_detailed = analysis_results["top_models"].head(10).copy()

        md_content.append("| Rank | Mean Score | Std Score | Fit Time | Parameters |")
        md_content.append("|------|------------|-----------|----------|------------|")

        for i, (_, row) in enumerate(top_models_detailed.iterrows()):
            params = str(row.get("params", ""))
            if len(params) > 80:
                params = params[:77] + "..."

            md_content.append(
                f"| {i+1} | `{row[target_metric]:.4f}` | `{row.get('std_test_score', 0):.4f}` | `{row.get('mean_fit_time', 0):.2f}s` | `{params}` |"
            )

        md_content.append("")

    # Appendix
    md_content.append("## 📚 Appendix")
    md_content.append("")

    md_content.append("### A. Glossary")
    md_content.append("")
    md_content.append("- **Mean Test Score**: Average performance across CV folds  ")
    md_content.append("- **Std Test Score**: Standard deviation across CV folds  ")
    md_content.append("- **Mean Fit Time**: Average training time per model  ")
    md_content.append("- **Stability Threshold**: Maximum acceptable std (default: 0.03)  ")
    md_content.append("- **Efficiency Score**: Performance per unit of training time  ")
    md_content.append("")

    md_content.append("### B. Analysis Methodology")
    md_content.append("")
    md_content.append("1. **Cross-Validation**: Typically 5-fold stratified CV  ")
    md_content.append(f"2. **Scoring Metric**: {target_metric}  ")
    md_content.append("3. **Hyperparameter Search**: GridSearch/RandomizedSearch  ")
    md_content.append(
        f"4. **Stability Analysis**: Models with std ≤ {stability_threshold} considered stable  "
    )
    md_content.append("5. **Time Efficiency**: Pareto frontier analysis  ")
    md_content.append("")

    # Footer
    md_content.append("---")
    md_content.append("*Report generated by Hyperparameter Analysis Module*  ")

    # Join markdown content
    markdown_string = "\n".join(md_content)

    # Save to file if filename provided
    if filename:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown_string)

        print(f"✅ Markdown report generated: {filename}")

        if return_markdown:
            return filename, markdown_string
        else:
            return filename
    elif return_markdown:
        return None, markdown_string
    else:
        print("⚠️ No filename provided, report not saved to file")
        return None


def display_hyperparameter_report(
    cv_results: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
):
    """
    Display hyperparameter analysis report directly in Jupyter notebook.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame
    strategy_config : dict, optional
        Strategy configuration dictionary
    target_metric : str, default="mean_test_score"
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time
    stability_threshold : float, default=0.03
        Stability threshold
    """
    # Generate markdown without saving to file
    _, markdown_content = generate_hyperparameter_markdown_report(
        cv_results=cv_results,
        strategy_config=strategy_config,
        filename=None,
        target_metric=target_metric,
        time_constraint=time_constraint,
        stability_threshold=stability_threshold,
        return_markdown=True,
    )

    # Display in notebook
    display(Markdown(markdown_content))

    # Also run the analysis to show console output
    print("Running analysis...")
    analyze_hyperparameter_results(
        cv_results,
        target_metric=target_metric,
        time_constraint=time_constraint,
        stability_threshold=stability_threshold,
    )


def generate_complete_hyperparameter_report(
    cv_results: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    output_dir: Union[Path, str] = "hyperparameter_report",
    filename: str = "hyperparameter_analysis_report.md",
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
    display_in_notebook: bool = False,
) -> Path:
    """
    Complete hyperparameter analysis workflow with markdown report generation.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame
    strategy_config : dict, optional
        Strategy configuration dictionary
    output_dir : Path or str, default="hyperparameter_report"
        Directory to save all outputs
    filename : str, default="hyperparameter_analysis_report.md"
        Name of the markdown report file
    target_metric : str, default="mean_test_score"
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time
    stability_threshold : float, default=0.03
        Stability threshold
    display_in_notebook : bool, default=False
        If True, also display the report in the notebook

    Returns
    -------
    Path
        Path to the generated markdown report
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define report path
    report_path = output_dir / filename

    print("🔍 Running complete hyperparameter analysis workflow...")

    # Display in notebook if requested
    if display_in_notebook:
        print("\n📊 Displaying report in notebook...")
        display_hyperparameter_report(
            cv_results=cv_results,
            strategy_config=strategy_config,
            target_metric=target_metric,
            time_constraint=time_constraint,
            stability_threshold=stability_threshold,
        )

    # Generate markdown report
    report_file = generate_hyperparameter_markdown_report(
        cv_results=cv_results,
        strategy_config=strategy_config,
        filename=report_path,
        target_metric=target_metric,
        time_constraint=time_constraint,
        stability_threshold=stability_threshold,
    )

    # Also save the raw data as CSV for reference
    csv_path = output_dir / "cv_results.csv"
    cv_results.to_csv(csv_path, index=False)

    print(f"\n✅ Analysis complete! Files saved in: {output_dir}")
    print(f"📊 Report: {report_file}")
    print(f"📁 Raw data: {csv_path}")

    return report_file


# Example usage
if __name__ == "__main__":
    # Example of how to use the functions
    import itertools

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create a sample grid search (simulated)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
    }

    # Create all combinations of parameters
    all_params = list(
        itertools.product(
            param_grid["n_estimators"], param_grid["max_depth"], param_grid["min_samples_split"]
        )
    )

    # Create sample cv_results with different parameter naming conventions
    cv_results = pd.DataFrame(
        {
            "mean_test_score": np.random.uniform(0.6, 0.8, len(all_params)),
            "std_test_score": np.random.uniform(0.01, 0.05, len(all_params)),
            "mean_fit_time": np.random.uniform(0.5, 5.0, len(all_params)),
            "mean_score_time": np.random.uniform(0.01, 0.1, len(all_params)),
            # Test different parameter naming conventions
            "param_n_estimators": [p[0] for p in all_params],
            "param_max_depth": [p[1] for p in all_params],
            "param_min_samples_split": [p[2] for p in all_params],
            "params": [
                {"n_estimators": p[0], "max_depth": p[1], "min_samples_split": p[2]}
                for p in all_params
            ],
        }
    )

    # Add split columns for fold analysis
    for i in range(5):
        cv_results[f"split{i}_test_score"] = np.random.uniform(0.6, 0.8, len(all_params))

    # Define your strategy configuration
    strategy_config = {
        "strategy": "Bollinger_w10_std1.5",
        "symbol": "EURUSD",
        "account_name": "FUNDEDNEXT_STLR2_6K",
        "bar_type": "tick",
        "bar_size": "M1",
        "price": "mid_price",
        "target_lookback": 20,
        "profit_target": 1,
        "stop_loss": 2,
        "max_holding_period": {"days": 1},
        "min_ret": 0,
        "vertical_barrier_zero": True,
        "filter_as_series": False,
    }

    # Test the analysis functions
    print("Testing analyze_hyperparameter_results...")
    analysis = analyze_hyperparameter_results(cv_results)

    print("\n\nTesting analyze_your_results...")
    specific_analysis = analyze_your_results(cv_results)

    # Generate the report
    print("\n\nTesting report generation...")
    report_path = generate_complete_hyperparameter_report(
        cv_results=cv_results,
        strategy_config=strategy_config,
        output_dir=Path("example_hyperparameter_report"),
        filename="example_analysis.md",
    )
    print(f"\n📋 Report generated at: {report_path}")
