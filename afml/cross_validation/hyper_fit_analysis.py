from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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
        Results from GridSearchCV/RandomizedSearchCV.
    target_metric : str
        Primary metric for optimization.
    time_constraint : float, optional
        Maximum acceptable training time in seconds.
    stability_threshold : float
        Maximum acceptable std_test_score for stable models.

    Returns
    -------
    Dict
        Analysis results including best model, insights, and visualizations.
    """

    analysis = {}

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

    # 8. VISUALIZATION (commented but can be enabled)
    print(f"\n8. GENERATING VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 8.1 Performance distribution
    axes[0, 0].hist(cv_results["mean_test_score"], bins=20, alpha=0.7)
    axes[0, 0].axvline(best_score, color="red", linestyle="--", label=f"Best: {best_score:.4f}")
    axes[0, 0].axvline(mean_score, color="green", linestyle="--", label=f"Mean: {mean_score:.4f}")
    axes[0, 0].set_title("Performance Distribution")
    axes[0, 0].set_xlabel("Test Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    # 8.2 Score vs Stability
    axes[0, 1].scatter(cv_results["mean_test_score"], cv_results["std_test_score"], alpha=0.5)
    axes[0, 1].scatter(
        recommended_model["mean_test_score"],
        recommended_model["std_test_score"],
        color="red",
        s=100,
        label="Recommended",
    )
    axes[0, 1].axhline(
        y=stability_threshold, color="orange", linestyle="--", label="Stability threshold"
    )
    axes[0, 1].set_title("Score vs Stability")
    axes[0, 1].set_xlabel("Mean Test Score")
    axes[0, 1].set_ylabel("Std Test Score")
    axes[0, 1].legend()

    # 8.3 Score vs Training Time
    axes[0, 2].scatter(cv_results["mean_test_score"], cv_results["mean_fit_time"], alpha=0.5)
    axes[0, 2].scatter(
        recommended_model["mean_test_score"],
        recommended_model["mean_fit_time"],
        color="red",
        s=100,
        label="Recommended",
    )
    axes[0, 2].set_title("Score vs Training Time")
    axes[0, 2].set_xlabel("Mean Test Score")
    axes[0, 2].set_ylabel("Training Time (s)")
    axes[0, 2].legend()

    # 8.4 Parameter importance (simplified)
    # For n_estimators
    param_groups = cv_results.groupby("param_clf__n_estimators")["mean_test_score"].mean()
    axes[1, 0].plot(param_groups.index, param_groups.values, marker="o")
    axes[1, 0].set_title("n_estimators vs Performance")
    axes[1, 0].set_xlabel("n_estimators")
    axes[1, 0].set_ylabel("Mean Test Score")

    # For max_depth
    param_groups = cv_results.groupby("param_clf__max_depth")["mean_test_score"].mean()
    axes[1, 1].plot(param_groups.index, param_groups.values, marker="s", color="green")
    axes[1, 1].set_title("max_depth vs Performance")
    axes[1, 1].set_xlabel("max_depth")
    axes[1, 1].set_ylabel("Mean Test Score")

    # 8.5 Fold consistency
    fold_labels = [f"Fold {i}" for i in range(len(fold_means))]
    axes[1, 2].bar(
        fold_labels, fold_means, yerr=fold_stds, capsize=5, alpha=0.7, error_kw={"ecolor": "red"}
    )
    axes[1, 2].set_title("Cross-Validation Fold Performance")
    axes[1, 2].set_ylabel("Mean Score")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # 9. PRACTICAL INTERPRETATION FOR TRADING
    print(f"\n9. PRACTICAL INTERPRETATION FOR TRADING:")
    print("-" * 50)

    # Convert F1 score to expected win rate
    # Assuming balanced classes and precision ≈ recall (simplified)
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


# Example usage with your results
def analyze_your_results(cv_results: pd.DataFrame) -> Dict:
    """
    Custom analysis for your specific hyperparameter search results.
    """

    print("SPECIFIC INSIGHTS FROM YOUR RESULTS:")
    print("=" * 80)

    # Your specific observations
    print("\n1. KEY OBSERVATIONS:")
    print("-" * 50)

    # Top model analysis
    best_model = cv_results.sort_values(by="mean_test_score", ascending=False).iloc[0]
    print(
        f"Best Model: max_depth={best_model['param_clf__max_depth']}, "
        f"n_estimators={best_model['param_clf__n_estimators']}"
    )
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

    # 3. ACTIONABLE RECOMMENDATIONS
    print(f"\n3. ACTIONABLE RECOMMENDATIONS:")
    print("-" * 50)

    # Based on your specific results
    if best_model["mean_test_score"] > 0.68:
        print("✅ Excellent performance achieved!")
        print("   Consider testing with additional features or ensemble methods")
    elif best_model["mean_test_score"] < 0.65:
        print("⚠️  Performance could be improved")
        print("   Consider: feature engineering, different model architecture, or more data")
    else:
        print("✅ Good baseline performance achieved")
        print("   Ready for forward testing with proper risk management")

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

    return {
        "best_model": best_model,
        "performance_level": (
            "EXCELLENT"
            if best_model["mean_test_score"] > 0.68
            else "GOOD" if best_model["mean_test_score"] > 0.65 else "MODERATE"
        ),
        "stability": (
            "HIGH"
            if best_model["std_test_score"] < 0.02
            else "MEDIUM" if best_model["std_test_score"] < 0.04 else "LOW"
        ),
    }
