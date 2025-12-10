"""
Comprehensive analysis and visualization suite for meta-labeling performance evaluation.
This module provides detailed analysis tools including statistical tests, visual comparisons,
and reporting functionality.
"""

from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def compare_strategies(results: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Create a side-by-side comparison of primary vs meta-labeled strategy.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance
        verbose: Print formatted comparison table

    Returns:
        DataFrame with comparative metrics
    """
    primary = results["primary_metrics"]
    meta = results["meta_metrics"]

    # Select key metrics for comparison
    metrics_to_compare = [
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "volatility",
        "win_rate",
        "profit_factor",
        "num_trades",
        "avg_win",
        "avg_loss",
        "kelly_criterion",
        "expectancy",
    ]

    comparison = pd.DataFrame(
        {
            "Primary": [primary.get(m, np.nan) for m in metrics_to_compare],
            "Meta": [meta.get(m, np.nan) for m in metrics_to_compare],
        },
        index=metrics_to_compare,
    )

    # Calculate improvement
    comparison["Improvement"] = comparison["Meta"] - comparison["Primary"]
    comparison["Improvement %"] = (comparison["Meta"] / comparison["Primary"] - 1) * 100

    # Mark which strategy is better for each metric
    comparison["Better"] = "Meta"
    for metric in ["max_drawdown", "volatility", "avg_loss", "num_trades"]:
        if metric in comparison.index:
            # For these metrics, lower is better
            comparison.loc[metric, "Better"] = np.where(
                comparison.loc[metric, "Improvement"] < 0, "Meta", "Primary"
            )

    if verbose:
        print(f"\n{'='*100}")
        print(f"STRATEGY COMPARISON: {results['strategy_name']}")
        print(f"{'='*100}\n")

        print(f"Signal Filtering:")
        print(f"  Total Signals:     {results['total_primary_signals']:,}")
        print(f"  Filtered Signals:  {results['filtered_signals']:,}")
        print(f"  Filter Rate:       {meta['signal_filter_rate']:.1%}")
        print(f"  Confidence Thresh: {meta['confidence_threshold']:.2f}\n")

        print(comparison.to_string())
        print(f"\n{'='*100}\n")

    return comparison


def calculate_risk_adjusted_metrics(results: dict) -> pd.DataFrame:
    """
    Calculate advanced risk-adjusted performance metrics.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance

    Returns:
        DataFrame with risk-adjusted metrics
    """
    primary = results["primary_metrics"]
    meta = results["meta_metrics"]

    def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio (probability weighted ratio of gains vs losses)"""
        if returns.empty:
            return 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        return gains.sum() / losses.sum() if losses.sum() > 0 else np.inf

    def tail_ratio(returns: pd.Series) -> float:
        """Ratio of 95th percentile to 5th percentile"""
        if returns.empty:
            return 0
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else np.inf

    primary_returns = results["primary_returns"]
    meta_returns = results["meta_returns"]

    metrics = pd.DataFrame(
        {
            "Primary": [
                omega_ratio(primary_returns),
                tail_ratio(primary_returns),
                (
                    primary["sharpe_ratio"] / primary["max_drawdown"]
                    if primary["max_drawdown"] > 0
                    else 0
                ),
                (
                    primary["win_rate"] * primary["avg_win"] / abs(primary["avg_loss"])
                    if primary["avg_loss"] != 0
                    else 0
                ),
                primary["expectancy"] / primary["volatility"] if primary["volatility"] > 0 else 0,
            ],
            "Meta": [
                omega_ratio(meta_returns),
                tail_ratio(meta_returns),
                meta["sharpe_ratio"] / meta["max_drawdown"] if meta["max_drawdown"] > 0 else 0,
                (
                    meta["win_rate"] * meta["avg_win"] / abs(meta["avg_loss"])
                    if meta["avg_loss"] != 0
                    else 0
                ),
                meta["expectancy"] / meta["volatility"] if meta["volatility"] > 0 else 0,
            ],
        },
        index=[
            "Omega Ratio",
            "Tail Ratio",
            "Sharpe/MaxDD",
            "Win-Loss Efficiency",
            "Expectancy/Vol",
        ],
    )

    metrics["Better"] = metrics.apply(
        lambda row: "Meta" if row["Meta"] > row["Primary"] else "Primary", axis=1
    )

    return metrics


def analyze_signal_quality(results: dict) -> Dict:
    """
    Analyze the quality and distribution of filtered signals, with both actual and normalized sizing.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance

    Returns:
        Dictionary with signal quality metrics
    """
    primary_returns = results["primary_returns"]
    meta_returns = results["meta_returns"]

    # Align returns for comparison (same timestamps)
    common_idx = primary_returns.index.intersection(meta_returns.index)
    primary_common = primary_returns.loc[common_idx]
    meta_common = meta_returns.loc[common_idx]

    # Filtered signals = those rejected by meta-labeling
    filtered_mask = ~primary_returns.index.isin(meta_returns.index)
    filtered_returns = primary_returns[filtered_mask]

    # --- Core counts ---
    total_signals = len(primary_returns)
    accepted_signals = len(meta_returns)
    rejected_signals = len(filtered_returns)

    # --- Precision/Recall framing ---
    precision = (meta_returns > 0).mean() if accepted_signals > 0 else 0
    total_winners = (primary_returns > 0).sum()
    recall = ((meta_returns > 0).sum() / total_winners) if total_winners > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # --- Quality metrics ---
    analysis = {
        "total_signals": total_signals,
        "accepted_signals": accepted_signals,
        "rejected_signals": rejected_signals,
        "filter_rate": rejected_signals / total_signals if total_signals > 0 else 0,
        # Accepted signals quality
        "accepted_win_rate": precision,
        "accepted_avg_return": meta_returns.mean() if accepted_signals > 0 else 0,
        "accepted_sharpe": (
            meta_returns.mean() / meta_returns.std() * np.sqrt(252)
            if accepted_signals > 1 and meta_returns.std() > 0
            else 0
        ),
        # Rejected signals quality
        "rejected_win_rate": (filtered_returns > 0).mean() if rejected_signals > 0 else 0,
        "rejected_avg_return": filtered_returns.mean() if rejected_signals > 0 else 0,
        "rejected_sharpe": (
            filtered_returns.mean() / filtered_returns.std() * np.sqrt(252)
            if rejected_signals > 1 and filtered_returns.std() > 0
            else 0
        ),
        # Filter effectiveness
        "avoided_losses": (
            (filtered_returns < 0).sum() / rejected_signals if rejected_signals > 0 else 0
        ),
        # Classification-style framing
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    # --- Statistical test: accepted vs rejected ---
    if rejected_signals > 0 and accepted_signals > 0:
        t_stat, p_value = stats.ttest_ind(meta_returns, filtered_returns, equal_var=False)
        analysis["ttest_pvalue"] = p_value
        analysis["significantly_better"] = p_value < 0.05
    else:
        analysis["ttest_pvalue"] = np.nan
        analysis["significantly_better"] = False

    # --- Comparative stats on aligned signals ---
    if len(common_idx) > 0:
        # Actual sizing
        analysis["aligned_primary_mean_actual"] = primary_common.mean()
        analysis["aligned_meta_mean_actual"] = meta_common.mean()
        analysis["aligned_diff_actual"] = meta_common.mean() - primary_common.mean()

        # Equal sizing normalization (force same notional per trade)
        primary_equal = (
            primary_common / primary_common.abs().mean()
            if primary_common.abs().mean() != 0
            else primary_common
        )
        meta_equal = (
            meta_common / meta_common.abs().mean() if meta_common.abs().mean() != 0 else meta_common
        )

        analysis["aligned_primary_mean_equal"] = primary_equal.mean()
        analysis["aligned_meta_mean_equal"] = meta_equal.mean()
        analysis["aligned_diff_equal"] = meta_equal.mean() - primary_equal.mean()

    return analysis


def plot_strategy_comparison(results: dict, figsize: Tuple[int, int] = (16, 10)):
    """
    Create comprehensive visualization comparing primary vs meta strategy.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance
        figsize: Figure size tuple
    """
    primary_returns = results["primary_returns"]
    meta_returns = results["meta_returns"]
    primary_metrics = results["primary_metrics"]
    meta_metrics = results["meta_metrics"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"Strategy Comparison: {results['strategy_name']}", fontsize=16, fontweight="bold")

    # 1. Cumulative Returns
    ax = axes[0, 0]
    primary_cum = (1 + primary_returns).cumprod()
    meta_cum = (1 + meta_returns).cumprod()

    ax.plot(primary_cum.index, primary_cum.values, label="Primary", alpha=0.7, linewidth=1.5)
    ax.plot(meta_cum.index, meta_cum.values, label="Meta", alpha=0.7, linewidth=1.5)
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Return Distributions
    ax = axes[0, 1]
    ax.hist(primary_returns, bins=50, alpha=0.5, label="Primary", density=True)
    ax.hist(meta_returns, bins=50, alpha=0.5, label="Meta", density=True)
    ax.set_title("Return Distributions")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Key Metrics Comparison
    ax = axes[0, 2]
    metrics_to_plot = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", "profit_factor"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    primary_vals = [primary_metrics.get(m, 0) for m in metrics_to_plot]
    meta_vals = [meta_metrics.get(m, 0) for m in metrics_to_plot]

    ax.bar(x - width / 2, primary_vals, width, label="Primary", alpha=0.7)
    ax.bar(x + width / 2, meta_vals, width, label="Meta", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot], fontsize=8)
    ax.set_title("Key Performance Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Apply log scale to compress range
    ax.set_yscale("log")
    ax.set_ylabel("Metric Value (log scale)")

    # Annotate bars with actual values
    for i, (p, m) in enumerate(zip(primary_vals, meta_vals)):
        ax.text(x[i] - width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    # 4. Drawdown Comparison
    ax = axes[1, 0]
    primary_cum = (1 + primary_returns).cumprod()
    meta_cum = (1 + meta_returns).cumprod()

    primary_dd = primary_cum / primary_cum.cummax() - 1
    meta_dd = meta_cum / meta_cum.cummax() - 1

    ax.fill_between(primary_dd.index, primary_dd.values, 0, alpha=0.5, label="Primary")
    ax.fill_between(meta_dd.index, meta_dd.values, 0, alpha=0.5, label="Meta")
    ax.set_title("Drawdown Over Time")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 5. Rolling Sharpe Ratio (if enough data)
    ax = axes[1, 1]
    window = min(252, len(primary_returns) // 5)  # ~1 year or 20% of data
    if window > 20:
        primary_rolling_sharpe = (
            primary_returns.rolling(window).mean()
            / primary_returns.rolling(window).std()
            * np.sqrt(252)
        )
        meta_rolling_sharpe = (
            meta_returns.rolling(window).mean() / meta_returns.rolling(window).std() * np.sqrt(252)
        )

        ax.plot(
            primary_rolling_sharpe.index,
            primary_rolling_sharpe.values,
            label="Primary",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.plot(
            meta_rolling_sharpe.index,
            meta_rolling_sharpe.values,
            label="Meta",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.set_title(f"Rolling Sharpe Ratio ({window} periods)")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data\nfor rolling Sharpe",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Rolling Sharpe Ratio")

    # 6. Trade Quality Analysis
    ax = axes[1, 2]
    signal_analysis = analyze_signal_quality(results)

    categories = ["Accepted\nWin Rate", "Rejected\nWin Rate", "Avoided\nLosses"]
    values = [
        signal_analysis["accepted_win_rate"],
        signal_analysis["rejected_win_rate"],
        signal_analysis["avoided_losses"],
    ]
    colors = ["green" if v > 0.5 else "red" for v in values]

    bars = ax.bar(categories, values, color=colors, alpha=0.6)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Signal Quality Analysis")
    ax.set_ylabel("Rate")
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def generate_summary_report(results: dict) -> str:
    """
    Generate a comprehensive text summary report.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance

    Returns:
        Formatted string report
    """
    primary = results["primary_metrics"]
    meta = results["meta_metrics"]
    signal_analysis = analyze_signal_quality(results)

    report = []
    report.append("=" * 80)
    report.append(f"META-LABELING PERFORMANCE REPORT: {results['strategy_name']}")
    report.append("=" * 80)
    report.append("")

    # Signal Filtering Summary
    report.append("SIGNAL FILTERING")
    report.append("-" * 40)
    report.append(f"Total Primary Signals:    {results['total_primary_signals']:,}")
    report.append(f"Accepted by Meta Model:   {results['filtered_signals']:,}")
    report.append(f"Rejected by Meta Model:   {signal_analysis['rejected_signals']:,}")
    report.append(f"Filter Rate:              {signal_analysis['filter_rate']:.1%}")
    report.append(f"Confidence Threshold:     {meta['confidence_threshold']:.2f}")
    report.append("")

    # Performance Comparison
    report.append("PERFORMANCE COMPARISON")
    report.append("-" * 40)

    metrics = [
        ("Total Return", "total_return", "{:.2%}"),
        ("Annualized Return", "annualized_return", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
        ("Calmar Ratio", "calmar_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Volatility", "volatility", "{:.2%}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Kelly Criterion", "kelly_criterion", "{:.2%}"),
    ]

    for name, key, fmt in metrics:
        p_val = primary.get(key, 0)
        m_val = meta.get(key, 0)
        change = ((m_val / p_val - 1) * 100) if p_val != 0 else 0

        better = (
            "✅"
            if (
                (key not in ["max_drawdown", "volatility"] and m_val > p_val)
                or (key in ["max_drawdown", "volatility"] and m_val < p_val)
            )
            else "❌"
        )

        report.append(
            f"{name:.<25} Primary: {fmt.format(p_val):<10} "
            f"Meta: {fmt.format(m_val):<10} ({change:+.1f}%) {better}"
        )

    report.append("")

    # Signal Quality
    report.append("SIGNAL QUALITY ANALYSIS")
    report.append("-" * 40)
    report.append(f"Accepted Signal Win Rate: {signal_analysis['accepted_win_rate']:.1%}")
    report.append(f"Rejected Signal Win Rate: {signal_analysis['rejected_win_rate']:.1%}")
    report.append(f"Filter Precision:         {signal_analysis['precision']:.1%}")
    report.append(f"Avoided Losses:           {signal_analysis['avoided_losses']:.1%}")

    if not np.isnan(signal_analysis["ttest_pvalue"]):
        report.append(f"T-Test P-Value:           {signal_analysis['ttest_pvalue']:.4f}")
        report.append(
            f"Statistically Better:     {'Yes' if signal_analysis['significantly_better'] else 'No'}"
        )

    report.append("")

    # Trade Statistics
    report.append("TRADE STATISTICS")
    report.append("-" * 40)
    report.append(
        f"Number of Trades:         Primary: {primary['num_trades']:,}  "
        f"Meta: {meta['num_trades']:,}"
    )
    report.append(
        f"Avg Trade Duration:       Primary: {primary['avg_trade_duration']}  "
        f"Meta: {meta['avg_trade_duration']}"
    )
    report.append(
        f"Best Trade:               Primary: {primary['best_trade']:.2%}  "
        f"Meta: {meta['best_trade']:.2%}"
    )
    report.append(
        f"Worst Trade:              Primary: {primary['worst_trade']:.2%}  "
        f"Meta: {meta['worst_trade']:.2%}"
    )
    report.append(
        f"Consecutive Wins:         Primary: {primary['consecutive_wins']}  "
        f"Meta: {meta['consecutive_wins']}"
    )
    report.append(
        f"Consecutive Losses:       Primary: {primary['consecutive_losses']}  "
        f"Meta: {meta['consecutive_losses']}"
    )

    report.append("")

    # Key Insights
    report.append("KEY INSIGHTS")
    report.append("-" * 40)

    insights = []

    # Check if meta-labeling improved performance
    if meta["sharpe_ratio"] > primary["sharpe_ratio"]:
        improvement = (meta["sharpe_ratio"] / primary["sharpe_ratio"] - 1) * 100
        insights.append(f"✅ Meta-labeling improved Sharpe ratio by {improvement:.1f}%")
    else:
        decline = (meta["sharpe_ratio"] / primary["sharpe_ratio"] - 1) * 100
        insights.append(f"❌ Meta-labeling decreased Sharpe ratio by {abs(decline):.1f}%")

    # Check drawdown reduction
    if meta["max_drawdown"] < primary["max_drawdown"]:
        reduction = (1 - meta["max_drawdown"] / primary["max_drawdown"]) * 100
        insights.append(f"✅ Reduced maximum drawdown by {reduction:.1f}%")

    # Check win rate improvement
    if meta["win_rate"] > primary["win_rate"]:
        improvement = (meta["win_rate"] - primary["win_rate"]) * 100
        insights.append(f"✅ Improved win rate by {improvement:.1f} percentage points")

    # Check if filtering is effective
    if signal_analysis["avoided_losses"] > 0.6:
        insights.append(
            f"✅ Effectively avoiding losses ({signal_analysis['avoided_losses']:.1%} of rejected signals)"
        )

    # Check information ratio
    if "information_ratio" in meta:
        if meta["information_ratio"] > 0:
            insights.append(f"✅ Positive information ratio ({meta['information_ratio']:.2f})")
        else:
            insights.append(f"⚠ Negative information ratio ({meta['information_ratio']:.2f})")

    for insight in insights:
        report.append(insight)

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def export_results_to_excel(results: dict, filepath: str):
    """
    Export comprehensive results to Excel with multiple sheets.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance
        filepath: Output Excel file path
    """
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Summary comparison
        comparison = compare_strategies(results, verbose=False)
        comparison.to_excel(writer, sheet_name="Comparison")

        # Risk-adjusted metrics
        risk_metrics = calculate_risk_adjusted_metrics(results)
        risk_metrics.to_excel(writer, sheet_name="Risk_Adjusted")

        # Signal quality
        signal_quality = pd.DataFrame([analyze_signal_quality(results)]).T
        signal_quality.columns = ["Value"]
        signal_quality.to_excel(writer, sheet_name="Signal_Quality")

        # Primary metrics
        pd.DataFrame([results["primary_metrics"]]).T.to_excel(writer, sheet_name="Primary_Metrics")

        # Meta metrics
        pd.DataFrame([results["meta_metrics"]]).T.to_excel(writer, sheet_name="Meta_Metrics")

        # Returns time series
        returns_df = pd.DataFrame(
            {"Primary": results["primary_returns"], "Meta": results["meta_returns"]}
        )
        returns_df.to_excel(writer, sheet_name="Returns")

    print(f"Results exported to: {filepath}")


# # Example usage
# if __name__ == "__main__":
#     # Assuming you have 'results' from evaluate_meta_labeling_performance

#     # Generate comparison
#     comparison = compare_strategies(results)

#     # Calculate risk-adjusted metrics
#     risk_metrics = calculate_risk_adjusted_metrics(results)
#     print("\nRisk-Adjusted Metrics:")
#     print(risk_metrics)

#     # Analyze signal quality
#     signal_quality = analyze_signal_quality(results)
#     print("\nSignal Quality:")
#     for key, value in signal_quality.items():
#         print(f"  {key}: {value}")

#     # Generate plots
#     fig = plot_strategy_comparison(results)
#     plt.show()

#     # Generate text report
#     report = generate_summary_report(results)
#     print(report)

#     # Export to Excel
#     # export_results_to_excel(results, 'meta_labeling_results.xlsx')
