import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("dark_background")


def analyze_hyperparameter_results(
    cv_results: pd.DataFrame,
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
) -> Dict:
    """
    Comprehensive analysis of hyperparameter search results.

    Returns dictionary with analysis results including generated plots.
    """
    analysis = {}

    # Store plots in analysis dict for later embedding
    analysis["plots"] = {}

    # 1. BASIC METRICS
    print("=" * 80)
    print("HYPERPARAMETER ANALYSIS REPORT")
    print("=" * 80)

    # Top models
    top_models = cv_results.sort_values(target_metric, ascending=False).head(10)
    analysis["top_models"] = top_models

    print(f"\n1. TOP PERFORMING MODELS (sorted by {target_metric}):")
    print("-" * 50)
    print(top_models[["params", "mean_test_score", "std_test_score", "mean_fit_time"]].to_string())

    # 2. PERFORMANCE ANALYSIS
    print(f"\n2. PERFORMANCE SUMMARY:")
    print("-" * 50)

    # Performance statistics
    mean_score = cv_results["mean_test_score"].mean()
    std_score = cv_results["mean_test_score"].std()
    max_score = cv_results["mean_test_score"].max()
    min_score = cv_results["mean_test_score"].min()

    print(f"Average {target_metric}: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Best {target_metric}: {max_score:.4f}")
    print(f"Worst {target_metric}: {min_score:.4f}")
    print(f"Performance Range: {max_score - min_score:.4f}")

    # 3. STABILITY ANALYSIS
    print(f"\n3. STABILITY ANALYSIS:")
    print("-" * 50)

    # Models with low variance (stable across folds)
    stable_models = cv_results[cv_results["std_test_score"] <= stability_threshold]
    analysis["stable_models"] = stable_models

    if not stable_models.empty:
        print(f"Models with stable performance (std ≤ {stability_threshold}): {len(stable_models)}")
        best_stable = stable_models.nlargest(1, target_metric)
        print(
            f"Best stable model: {best_stable[target_metric].iloc[0]:.4f} ± {best_stable['std_test_score'].iloc[0]:.4f}"
        )
    else:
        print(f"No models meet stability threshold of {stability_threshold}")

    # 4. TIME-EFFICIENCY ANALYSIS
    print(f"\n4. TIME-EFFICIENCY ANALYSIS:")
    print("-" * 50)

    # Pareto frontier analysis (balance between score and time)
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

    # 5. HYPERPARAMETER TREND ANALYSIS
    print(f"\n5. HYPERPARAMETER TRENDS:")
    print("-" * 50)

    # Extract hyperparameter values
    param_columns = [col for col in cv_results.columns if col.startswith("param_")]

    for param in param_columns:
        param_name = param.replace("param_", "")

        # Group by parameter value
        param_stats = (
            cv_results.groupby(param)[["mean_test_score", "std_test_score", "mean_fit_time"]]
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

        print(f"\nParameter: {param_name}")
        print(
            f"Optimal range: {param_stats.index[0]} (score: {param_stats['score_mean'].iloc[0]:.4f})"
        )
        print(f"Performance by value:")
        print(param_stats.to_string())

    # 6. CROSS-VALIDATION FOLD CONSISTENCY
    print(f"\n6. CROSS-VALIDATION CONSISTENCY:")
    print("-" * 50)

    # Check for problematic folds
    fold_columns = [col for col in cv_results.columns if "split" in col and "test" in col]
    fold_scores = cv_results[fold_columns]

    fold_means = fold_scores.mean(axis=0)
    fold_stds = fold_scores.std(axis=0)

    print(f"Fold performance consistency:")
    for i, (fold_mean, fold_std) in enumerate(zip(fold_means, fold_stds)):
        print(f"  Fold {i}: {fold_mean:.4f} ± {fold_std:.4f}")

    # Identify folds with high variance
    problem_folds = [(i, std) for i, std in enumerate(fold_stds) if std > stability_threshold]
    if problem_folds:
        print(f"\n⚠️  High variance folds detected (std > {stability_threshold}):")
        for fold, std in problem_folds:
            print(f"  Fold {fold}: std = {std:.4f}")

    # 7. MODEL SELECTION RECOMMENDATION
    print(f"\n7. MODEL SELECTION RECOMMENDATION:")
    print("-" * 50)

    # Score-based recommendation
    best_by_score = cv_results.nlargest(1, target_metric)
    best_score = best_by_score[target_metric].iloc[0]
    best_std = best_by_score["std_test_score"].iloc[0]

    # Consider stability vs performance trade-off
    if len(stable_models) > 0:
        best_stable = stable_models.nlargest(1, target_metric)
        stable_score = best_stable[target_metric].iloc[0]
        stable_std = best_stable["std_test_score"].iloc[0]

        # Check if the performance difference is statistically significant
        score_diff = best_score - stable_score
        if score_diff < 0.005:  # Less than 0.5% difference
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

    # Add practical considerations
    recommended_params = recommended_model["params"].iloc[0]
    print(f"\n🎯 RECOMMENDED HYPERPARAMETERS:")
    for key, value in recommended_params.items():
        print(f"   {key}: {value}")

    # 8. GENERATE VISUALIZATIONS
    print(f"\n8. GENERATING VISUALIZATIONS...")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 8.1 Performance distribution
    ax = axes[0, 0]
    ax.hist(cv_results["mean_test_score"], bins=20, alpha=0.7)
    ax.axvline(best_score, color="red", linestyle="--", label=f"Best: {best_score:.4f}")
    ax.axvline(mean_score, color="green", linestyle="--", label=f"Mean: {mean_score:.4f}")
    ax.set_title("Performance Distribution")
    ax.set_xlabel("Test Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    # 8.2 Score vs Stability
    ax = axes[0, 1]
    ax.scatter(cv_results["mean_test_score"], cv_results["std_test_score"], alpha=0.5)
    ax.scatter(
        recommended_model[target_metric].iloc[0],
        recommended_model["std_test_score"].iloc[0],
        color="red",
        s=100,
        label="Recommended",
    )
    ax.axhline(y=stability_threshold, color="orange", linestyle="--", label="Stability threshold")
    ax.set_title("Score vs Stability")
    ax.set_xlabel("Mean Test Score")
    ax.set_ylabel("Std Test Score")
    ax.legend()

    # 8.3 Score vs Training Time
    ax = axes[0, 2]
    ax.scatter(cv_results["mean_test_score"], cv_results["mean_fit_time"], alpha=0.5)
    ax.scatter(
        recommended_model[target_metric].iloc[0],
        recommended_model["mean_fit_time"].iloc[0],
        color="red",
        s=100,
        label="Recommended",
    )
    ax.set_title("Score vs Training Time")
    ax.set_xlabel("Mean Test Score")
    ax.set_ylabel("Training Time (s)")
    ax.legend()

    # 8.4 Parameter importance - n_estimators
    ax = axes[1, 0]
    if "param_clf__n_estimators" in cv_results.columns:
        param_groups = cv_results.groupby("param_clf__n_estimators")["mean_test_score"].mean()
        ax.plot(param_groups.index, param_groups.values, marker="o")
        ax.set_title("n_estimators vs Performance")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Mean Test Score")
    else:
        ax.text(
            0.5,
            0.5,
            "No n_estimators parameter found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 8.5 Parameter importance - max_depth
    ax = axes[1, 1]
    if "param_clf__max_depth" in cv_results.columns:
        param_groups = cv_results.groupby("param_clf__max_depth")["mean_test_score"].mean()
        ax.plot(param_groups.index, param_groups.values, marker="s", color="green")
        ax.set_title("max_depth vs Performance")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("Mean Test Score")
    else:
        ax.text(
            0.5,
            0.5,
            "No max_depth parameter found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 8.6 Fold consistency
    ax = axes[1, 2]
    if len(fold_means) > 0:
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

    # Generate individual plots for detailed view
    # Performance distribution detailed
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cv_results["mean_test_score"], bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(
        best_score, color="red", linestyle="--", linewidth=2, label=f"Best: {best_score:.4f}"
    )
    ax.axvline(
        mean_score, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.4f}"
    )
    ax.set_title("Performance Distribution Across All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Test Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    analysis["plots"]["performance_dist"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # 9. PRACTICAL INTERPRETATION FOR TRADING
    print(f"\n9. PRACTICAL INTERPRETATION FOR TRADING:")
    print("-" * 50)

    # Convert F1 score to expected win rate
    expected_accuracy = best_score  # Rough approximation

    print(f"Expected Strategy Performance:")
    print(f"  • F1 Score: {best_score:.4f}")
    print(f"  • Expected Accuracy: ~{expected_accuracy*100:.1f}%")
    print(f"  • Cross-validation Consistency: {'Good' if best_std < 0.03 else 'Moderate'}")

    # Risk assessment
    if best_std > 0.04:
        print(f"\n⚠️  RISK WARNING: High variance in CV folds")
        print(f"   Strategy may perform inconsistently in live trading")
    elif best_std < 0.02:
        print(f"\n✅ LOW RISK: Excellent consistency across CV folds")
        print(f"   Strategy likely to perform similarly in live trading")

    # 10. FINAL SUMMARY
    analysis["summary"] = {
        "best_model": recommended_model,
        "best_score": best_score,
        "best_std": best_std,
        "recommended_params": recommended_params,
        "expected_accuracy": expected_accuracy,
        "risk_level": "HIGH" if best_std > 0.04 else "MEDIUM" if best_std > 0.02 else "LOW",
    }

    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return analysis


def analyze_your_results(cv_results: pd.DataFrame) -> Dict:
    """
    Custom analysis for your specific hyperparameter search results.
    """

    analysis = {}

    print("SPECIFIC INSIGHTS FROM YOUR RESULTS:")
    print("=" * 80)

    # Your specific observations
    print("\n1. KEY OBSERVATIONS:")
    print("-" * 50)

    # Top model analysis
    best_model = cv_results.sort_values(by="mean_test_score", ascending=False).iloc[0]

    # Extract parameter values safely
    max_depth = best_model.get("param_clf__max_depth", "N/A")
    n_estimators = best_model.get("param_clf__n_estimators", "N/A")

    analysis["best_model"] = best_model
    analysis["max_depth"] = max_depth
    analysis["n_estimators"] = n_estimators

    print(f"Best Model: max_depth={max_depth}, " f"n_estimators={n_estimators}")
    print(f"F1 Score: {best_model['mean_test_score']:.4f} ± {best_model['std_test_score']:.4f}")
    print(f"Training Time: {best_model['mean_fit_time']:.2f}s")

    # Compare with simpler models
    simple_models = cv_results[cv_results["param_clf__max_depth"] <= 4]
    if not simple_models.empty:
        best_simple = simple_models.nlargest(1, "mean_test_score")
        print(f"\nBest Simple Model (max_depth ≤ 4):")
        print(
            f"  max_depth={best_simple['param_clf__max_depth'].iloc[0]}, "
            f"n_estimators={best_simple['param_clf__n_estimators'].iloc[0]}"
        )
        print(
            f"  F1 Score: {best_simple['mean_test_score'].iloc[0]:.4f} "
            f"(vs best: {best_model['mean_test_score']:.4f})"
        )
        print(
            f"  Training Time: {best_simple['mean_fit_time'].iloc[0]:.2f}s "
            f"(vs best: {best_model['mean_fit_time']:.2f}s)"
        )

        analysis["best_simple_model"] = best_simple.iloc[0].to_dict()
        analysis["performance_diff"] = (
            best_model["mean_test_score"] - best_simple["mean_test_score"].iloc[0]
        )

    # Performance saturation analysis
    print(f"\n2. PERFORMANCE SATURATION:")
    print("-" * 50)

    # Check if more complex models provide diminishing returns
    depth_groups = cv_results.groupby("param_clf__max_depth")["mean_test_score"].max()
    print("Maximum performance by max_depth:")
    for depth, score in depth_groups.items():
        improvement = (
            (score - depth_groups.get(depth - 1, 0)) if depth > min(depth_groups.index) else 0
        )
        print(f"  depth={depth}: {score:.4f} (improvement: {improvement:.4f})")

    analysis["depth_groups"] = depth_groups.to_dict()

    # 3. ACTIONABLE RECOMMENDATIONS
    print(f"\n3. ACTIONABLE RECOMMENDATIONS:")
    print("-" * 50)

    # Based on your specific results
    if best_model["mean_test_score"] > 0.68:
        print("✅ Excellent performance achieved!")
        print("   Consider testing with additional features or ensemble methods")
        analysis["performance_level"] = "EXCELLENT"
    elif best_model["mean_test_score"] < 0.65:
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

    # Inference speed estimation
    avg_score_time = cv_results["mean_score_time"].mean()
    print(f"Expected Inference Speed: ~{avg_score_time*1000:.1f}ms per prediction")
    print(
        f"Training Time Range: {cv_results['mean_fit_time'].min():.2f}s to {cv_results['mean_fit_time'].max():.2f}s"
    )

    # Memory considerations
    avg_estimators = cv_results["param_clf__n_estimators"].mean()
    print(f"Average Model Size: ~{avg_estimators} trees")

    analysis["avg_inference_time"] = avg_score_time * 1000
    analysis["training_time_range"] = (
        cv_results["mean_fit_time"].min(),
        cv_results["mean_fit_time"].max(),
    )
    analysis["avg_estimators"] = avg_estimators

    # Add stability rating
    analysis["stability"] = (
        "HIGH"
        if best_model["std_test_score"] < 0.02
        else "MEDIUM" if best_model["std_test_score"] < 0.04 else "LOW"
    )

    return analysis


def generate_hyperparameter_markdown_report(
    cv_results: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    filename: Path = Path("hyperparameter_analysis_report.md"),
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
) -> Path:
    """
    Generate comprehensive markdown report from hyperparameter analysis.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame
    filename : Path
        Path to save the markdown report
    target_metric : str
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time
    stability_threshold : float
        Stability threshold

    Returns
    -------
    Path
        Path to the generated markdown file
    """

    # Ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

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
                value_str = str(value)
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

    md_content.append(f"**Total Models Evaluated**: `{len(cv_results)}`  ")
    md_content.append(
        f"**Performance Range**: `{cv_results['mean_test_score'].max():.4f}` - `{cv_results['mean_test_score'].min():.4f}`  "
    )
    md_content.append(
        f"**Average Performance**: `{cv_results['mean_test_score'].mean():.4f} ± {cv_results['mean_test_score'].std():.4f}`  "
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
                row["mean_test_score"] / row["mean_fit_time"] if row["mean_fit_time"] > 0 else 0,
            )
            md_content.append(
                f"| {i+1} | `{row['mean_test_score']:.4f}` | `{row['std_test_score']:.4f}` | `{row['mean_fit_time']:.2f}` | `{efficiency:.2f}` |"
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

            best_stable = stable_models.nlargest(1, "mean_test_score")
            if not best_stable.empty:
                md_content.append("### Best Stable Model")
                md_content.append(f"- **Score**: `{best_stable['mean_test_score'].iloc[0]:.4f}`  ")
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

    # Time statistics
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

    param_columns = [col for col in cv_results.columns if col.startswith("param_")]

    if param_columns:
        md_content.append("### Parameter Impact Analysis")
        md_content.append("")

        for param in param_columns[:3]:  # Show top 3 parameters
            param_name = param.replace("param_", "")

            param_stats = (
                cv_results.groupby(param)[["mean_test_score", "mean_fit_time"]]
                .agg({"mean_test_score": ["mean", "std", "count"], "mean_fit_time": "mean"})
                .round(4)
            )

            if not param_stats.empty:
                param_stats.columns = ["score_mean", "score_std", "count", "time_mean"]
                param_stats = param_stats.sort_values("score_mean", ascending=False)

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
                    f"**Selected Model Performance**: `{best_model.get('mean_test_score', 0):.4f} ± {best_model.get('std_test_score', 0):.4f}`  "
                )
                md_content.append(
                    f"**Training Time**: `{best_model.get('mean_fit_time', 0):.2f}s`  "
                )
                md_content.append("")

            md_content.append("### Recommended Hyperparameters")
            md_content.append("")
            md_content.append("```python")
            for key, value in recommended_params.items():
                md_content.append(f"{key} = {value}")
            md_content.append("```")
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
            md_content.append(f"- **max_depth**: `{specific_analysis.get('max_depth', 'N/A')}`  ")
            md_content.append(
                f"- **n_estimators**: `{specific_analysis.get('n_estimators', 'N/A')}`  "
            )
            md_content.append(f"- **F1 Score**: `{best_model.get('mean_test_score', 0):.4f}`  ")
            md_content.append(
                f"- **Standard Deviation**: `{best_model.get('std_test_score', 0):.4f}`  "
            )
            md_content.append(f"- **Training Time**: `{best_model.get('mean_fit_time', 0):.2f}s`  ")
            md_content.append("")

        if "best_simple_model" in specific_analysis:
            md_content.append("### Comparison with Simpler Model")
            md_content.append("")
            md_content.append(
                f"- **Performance Difference**: `{specific_analysis.get('performance_diff', 0):.4f}`  "
            )
            md_content.append(
                "- Simple models may offer better generalization and faster inference"
            )
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
    md_content.append("### Complete Results (Top 10 Models)")
    md_content.append("")

    if "top_models" in analysis_results:
        top_models_detailed = analysis_results["top_models"].head(10).copy()

        md_content.append("| Rank | Mean Score | Std Score | Fit Time | Parameters |")
        md_content.append("|------|------------|-----------|----------|------------|")

        for i, (_, row) in enumerate(top_models_detailed.iterrows()):
            params = str(row["params"])
            if len(params) > 80:
                params = params[:77] + "..."

            md_content.append(
                f"| {i+1} | `{row['mean_test_score']:.4f}` | `{row['std_test_score']:.4f}` | `{row['mean_fit_time']:.2f}s` | `{params}` |"
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
    md_content.append("1. **Cross-Validation**: 5-fold stratified CV  ")
    md_content.append("2. **Scoring Metric**: F1 Score (macro average)  ")
    md_content.append("3. **Hyperparameter Search**: GridSearch/RandomizedSearch  ")
    md_content.append("4. **Stability Analysis**: Models with std ≤ 0.03 considered stable  ")
    md_content.append("5. **Time Efficiency**: Pareto frontier analysis  ")
    md_content.append("")

    # Footer
    md_content.append("---")
    md_content.append("*Report generated by Hyperparameter Analysis Module*  ")
    md_content.append(f"*Report saved to: `{filename}`*  ")

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"✅ Markdown report generated: {filename}")

    return filename


def generate_complete_hyperparameter_report(
    cv_results: pd.DataFrame,
    strategy_config: Optional[Dict] = None,
    output_dir: Path = Path("hyperparameter_report"),
    filename: str = "hyperparameter_analysis_report.md",
    target_metric: str = "mean_test_score",
    time_constraint: Optional[float] = None,
    stability_threshold: float = 0.03,
) -> Path:
    """
    Complete hyperparameter analysis workflow with markdown report generation.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results DataFrame
    output_dir : Path
        Directory to save all outputs
    filename : str
        Name of the markdown report file
    target_metric : str
        Primary metric for optimization
    time_constraint : float, optional
        Maximum acceptable training time
    stability_threshold : float
        Stability threshold

    Returns
    -------
    Path
        Path to the generated markdown report
    """

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define report path
    report_path = output_dir / filename

    print("🔍 Running complete hyperparameter analysis workflow...")

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

    print(f"✅ Analysis complete! Files saved in: {output_dir}")
    print(f"📊 Report: {report_file}")
    print(f"📁 Raw data: {csv_path}")

    return report_file


# Example usage
if __name__ == "__main__":
    # Example of how to use the functions
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create a sample grid search (simulated)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
    }

    # For demonstration, create a sample cv_results DataFrame
    import itertools

    # Create all combinations of parameters
    all_params = list(
        itertools.product(
            param_grid["n_estimators"], param_grid["max_depth"], param_grid["min_samples_split"]
        )
    )

    # Create sample cv_results
    cv_results = pd.DataFrame(
        {
            "mean_test_score": np.random.uniform(0.6, 0.8, len(all_params)),
            "std_test_score": np.random.uniform(0.01, 0.05, len(all_params)),
            "mean_fit_time": np.random.uniform(0.5, 5.0, len(all_params)),
            "mean_score_time": np.random.uniform(0.01, 0.1, len(all_params)),
            "param_clf__n_estimators": [p[0] for p in all_params],
            "param_clf__max_depth": [p[1] for p in all_params],
            "param_clf__min_samples_split": [p[2] for p in all_params],
            "params": [
                {"clf__n_estimators": p[0], "clf__max_depth": p[1], "clf__min_samples_split": p[2]}
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

    # Generate the report with strategy config
    report_path = generate_complete_hyperparameter_report(
        cv_results=cv_results,
        strategy_config=strategy_config,  # ADD THIS LINE
        output_dir=Path("example_hyperparameter_report"),
        filename="example_analysis.md",
    )
    print(f"\n📋 Report generated at: {report_path}")
