"""
Comprehensive analysis and visualization suite for meta-labeling performance evaluation.
This module provides detailed analysis tools including statistical tests, visual comparisons,
and reporting functionality.
"""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .performance_analysis import analyze_trading_behavior


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
    ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.3)
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
    plt.style.use("dark_background")
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


def generate_meta_labeling_markdown_report(
    results_dict: Dict[
        str, Dict
    ],  # Key: bar_type, Value: results from evaluate_meta_labeling_performance
    strategy_config: Optional[Dict] = None,
    filename: Path = Path("meta_labeling_analysis_report.md"),
    include_plots: bool = True,
    plot_dir: Path = Path("meta_labeling_plots"),
) -> Path:
    """
    Generate comprehensive markdown report for meta-labeling analysis with bar type comparison.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary where keys are bar types and values are results from evaluate_meta_labeling_performance
    strategy_config : Dict, optional
        Strategy configuration dictionary
    filename : Path
        Path to save the markdown report
    include_plots : bool
        Whether to generate and embed plots
    plot_dir : Path
        Directory to save plot images

    Returns
    -------
    Path
        Path to the generated markdown file
    """

    # Create directories if needed
    filename.parent.mkdir(parents=True, exist_ok=True)
    if include_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start building markdown content
    md_content = []

    # Header
    md_content.append("# 📈 Meta-Labeling Performance Analysis Report")
    md_content.append("")
    md_content.append(f"*Generated on: {timestamp}*  ")
    md_content.append("")

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

    # Bar Type Comparison Summary
    md_content.append("## 📊 Bar Type Comparison Summary")
    md_content.append("")

    if len(results_dict) > 1:
        # Create comparison table for all bar types
        comparison_data = []

        for bar_type, results in results_dict.items():
            meta_metrics = results.get("meta_metrics", {})
            signal_analysis = analyze_signal_quality(results)

            comparison_data.append(
                {
                    "Bar Type": bar_type,
                    "Meta Sharpe": meta_metrics.get("sharpe_ratio", 0),
                    "Meta Return %": meta_metrics.get("annualized_return", 0) * 100,
                    "Max DD %": abs(meta_metrics.get("max_drawdown", 0) * 100),
                    "Win Rate %": meta_metrics.get("win_rate", 0) * 100,
                    "Profit Factor": meta_metrics.get("profit_factor", 0),
                    "Num Trades": meta_metrics.get("num_trades", 0),
                    "Filter Rate %": signal_analysis.get("filter_rate", 0) * 100,
                    "Accepted WR %": signal_analysis.get("accepted_win_rate", 0) * 100,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values("Meta Sharpe", ascending=False)

        md_content.append("### Performance by Bar Type")
        md_content.append("")
        md_content.append(comparison_df.to_markdown(index=False))
        md_content.append("")

        # Key findings
        md_content.append("### Key Findings")
        md_content.append("")

        best_bar_type = comparison_df.iloc[0]["Bar Type"]
        worst_bar_type = comparison_df.iloc[-1]["Bar Type"]

        md_content.append(f"**Best Performing Bar Type**: `{best_bar_type}`  ")
        md_content.append(f"  • Sharpe Ratio: `{comparison_df.iloc[0]['Meta Sharpe']:.2f}`  ")
        md_content.append(f"  • Annual Return: `{comparison_df.iloc[0]['Meta Return %']:.1f}%`  ")
        md_content.append("")

        md_content.append(f"**Worst Performing Bar Type**: `{worst_bar_type}`  ")
        md_content.append(f"  • Sharpe Ratio: `{comparison_df.iloc[-1]['Meta Sharpe']:.2f}`  ")
        md_content.append(f"  • Annual Return: `{comparison_df.iloc[-1]['Meta Return %']:.1f}%`  ")
        md_content.append("")

        # Add improvement statistics if primary metrics available
        if all("primary_metrics" in r for r in results_dict.values()):
            md_content.append("### Meta-Labeling Improvement Statistics")
            md_content.append("")

            improvement_data = []
            for bar_type, results in results_dict.items():
                primary = results.get("primary_metrics", {})
                meta = results.get("meta_metrics", {})

                if primary and meta:
                    sharpe_improvement = (
                        (meta.get("sharpe_ratio", 0) / primary.get("sharpe_ratio", 1)) - 1
                    ) * 100
                    dd_improvement = (
                        (meta.get("max_drawdown", 0) / primary.get("max_drawdown", 1)) - 1
                    ) * 100
                    winrate_improvement = (
                        (meta.get("win_rate", 0) / primary.get("win_rate", 1)) - 1
                    ) * 100

                    improvement_data.append(
                        {
                            "Bar Type": bar_type,
                            "Sharpe Δ%": sharpe_improvement,
                            "Max DD Δ%": dd_improvement,
                            "Win Rate Δ%": winrate_improvement,
                            "Trades Δ%": (
                                (meta.get("num_trades", 0) / primary.get("num_trades", 1)) - 1
                            )
                            * 100,
                        }
                    )

            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                md_content.append(improvement_df.to_markdown(index=False))
                md_content.append("")

    # Detailed Analysis for Each Bar Type
    md_content.append("## 🔍 Detailed Analysis by Bar Type")
    md_content.append("")

    for bar_type, results in results_dict.items():
        md_content.append(f"### {bar_type.upper()} Bars")
        md_content.append("")

        # Basic stats
        primary_metrics = results.get("primary_metrics", {})
        meta_metrics = results.get("meta_metrics", {})
        signal_analysis = analyze_signal_quality(results)

        md_content.append("#### Performance Metrics")
        md_content.append("")

        # Create metrics comparison table
        metrics_table = pd.DataFrame(
            {
                "Metric": [
                    "Total Return %",
                    "Annualized Return %",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Max Drawdown %",
                    "Volatility %",
                    "Win Rate %",
                    "Profit Factor",
                    "Number of Trades",
                    "Avg Trade Duration",
                ],
                "Primary": [
                    primary_metrics.get("total_return", 0) * 100,
                    primary_metrics.get("annualized_return", 0) * 100,
                    primary_metrics.get("sharpe_ratio", 0),
                    primary_metrics.get("sortino_ratio", 0),
                    primary_metrics.get("calmar_ratio", 0),
                    abs(primary_metrics.get("max_drawdown", 0)) * 100,
                    primary_metrics.get("volatility", 0) * 100,
                    primary_metrics.get("win_rate", 0) * 100,
                    primary_metrics.get("profit_factor", 0),
                    primary_metrics.get("num_trades", 0),
                    str(primary_metrics.get("avg_trade_duration", "N/A")),
                ],
                "Meta": [
                    meta_metrics.get("total_return", 0) * 100,
                    meta_metrics.get("annualized_return", 0) * 100,
                    meta_metrics.get("sharpe_ratio", 0),
                    meta_metrics.get("sortino_ratio", 0),
                    meta_metrics.get("calmar_ratio", 0),
                    abs(meta_metrics.get("max_drawdown", 0)) * 100,
                    meta_metrics.get("volatility", 0) * 100,
                    meta_metrics.get("win_rate", 0) * 100,
                    meta_metrics.get("profit_factor", 0),
                    meta_metrics.get("num_trades", 0),
                    str(meta_metrics.get("avg_trade_duration", "N/A")),
                ],
                "Improvement": [
                    (
                        (
                            meta_metrics.get("total_return", 0)
                            / max(primary_metrics.get("total_return", 1), 0.001)
                        )
                        - 1
                    )
                    * 100,
                    (
                        (
                            meta_metrics.get("annualized_return", 0)
                            / max(primary_metrics.get("annualized_return", 1), 0.001)
                        )
                        - 1
                    )
                    * 100,
                    meta_metrics.get("sharpe_ratio", 0) - primary_metrics.get("sharpe_ratio", 0),
                    meta_metrics.get("sortino_ratio", 0) - primary_metrics.get("sortino_ratio", 0),
                    meta_metrics.get("calmar_ratio", 0) - primary_metrics.get("calmar_ratio", 0),
                    (
                        (
                            meta_metrics.get("max_drawdown", 0)
                            / max(primary_metrics.get("max_drawdown", 1), 0.001)
                        )
                        - 1
                    )
                    * 100,
                    (
                        (
                            meta_metrics.get("volatility", 0)
                            / max(primary_metrics.get("volatility", 1), 0.001)
                        )
                        - 1
                    )
                    * 100,
                    (meta_metrics.get("win_rate", 0) - primary_metrics.get("win_rate", 0)) * 100,
                    meta_metrics.get("profit_factor", 0) - primary_metrics.get("profit_factor", 0),
                    meta_metrics.get("num_trades", 0) - primary_metrics.get("num_trades", 0),
                    "N/A",
                ],
            }
        )

        # Format the table
        for i, row in metrics_table.iterrows():
            improvement = row["Improvement"]
            if isinstance(improvement, (int, float)):
                if i in [5, 6]:  # For drawdown and volatility, negative improvement is good
                    color = "🟢" if improvement < 0 else "🔴"
                else:  # For other metrics, positive improvement is good
                    color = "🟢" if improvement > 0 else "🔴"

                if isinstance(improvement, float):
                    metrics_table.at[i, "Improvement"] = f"{color} {improvement:+.2f}"

        md_content.append(metrics_table.to_markdown(index=False))
        md_content.append("")

        # Signal Quality Analysis
        md_content.append("#### Signal Quality Analysis")
        md_content.append("")

        signal_table = pd.DataFrame(
            {
                "Metric": [
                    "Total Signals",
                    "Accepted Signals",
                    "Rejected Signals",
                    "Filter Rate %",
                    "Accepted Win Rate %",
                    "Rejected Win Rate %",
                    "Avoided Losses %",
                    "Precision %",
                    "Recall %",
                    "F1 Score",
                ],
                "Value": [
                    signal_analysis.get("total_signals", 0),
                    signal_analysis.get("accepted_signals", 0),
                    signal_analysis.get("rejected_signals", 0),
                    signal_analysis.get("filter_rate", 0) * 100,
                    signal_analysis.get("accepted_win_rate", 0) * 100,
                    signal_analysis.get("rejected_win_rate", 0) * 100,
                    signal_analysis.get("avoided_losses", 0) * 100,
                    signal_analysis.get("precision", 0) * 100,
                    signal_analysis.get("recall", 0) * 100,
                    signal_analysis.get("f1_score", 0),
                ],
            }
        )

        md_content.append(signal_table.to_markdown(index=False))
        md_content.append("")

        # Statistical Significance
        if "ttest_pvalue" in signal_analysis:
            md_content.append("#### Statistical Significance")
            md_content.append("")
            md_content.append(f"T-Test P-Value: `{signal_analysis['ttest_pvalue']:.4f}`  ")
            if signal_analysis["ttest_pvalue"] < 0.05:
                md_content.append("✅ **Statistically Significant Improvement** (p < 0.05)  ")
            else:
                md_content.append("⚠️ **Not Statistically Significant** (p ≥ 0.05)  ")
            md_content.append("")

        # Add separator between bar types
        if bar_type != list(results_dict.keys())[-1]:
            md_content.append("---")
            md_content.append("")

    # Generate and Embed Plots
    if include_plots and results_dict:
        md_content.append("## 📊 Visual Analysis")
        md_content.append("")

        # Generate comparison plots
        if len(results_dict) > 1:
            # Bar type comparison plot
            comparison_plot = _generate_bar_type_comparison_plot(results_dict)
            comparison_b64 = _plot_to_base64(comparison_plot)
            md_content.append("### Bar Type Performance Comparison")
            md_content.append("")
            md_content.append(
                f'<img src="data:image/png;base64,{comparison_b64}" style="width: 100%; max-width: 1400px;">'
            )
            md_content.append("")

            # Sharpe ratio evolution plot
            sharpe_plot = _generate_sharpe_evolution_plot(results_dict)
            sharpe_b64 = _plot_to_base64(sharpe_plot)
            md_content.append("### Sharpe Ratio Evolution Comparison")
            md_content.append("")
            md_content.append(
                f'<img src="data:image/png;base64,{sharpe_b64}" style="width: 100%; max-width: 1400px;">'
            )
            md_content.append("")

        # Individual strategy plots for each bar type
        for bar_type, results in results_dict.items():
            strategy_plot = plot_strategy_comparison(results, figsize=(14, 8))
            strategy_b64 = _plot_to_base64(strategy_plot)

            md_content.append(f"### {bar_type.upper()} Bars - Strategy Comparison")
            md_content.append("")
            md_content.append(
                f'<img src="data:image/png;base64,{strategy_b64}" style="width: 100%; max-width: 1400px;">'
            )
            md_content.append("")

            # Signal quality plot
            signal_plot = _generate_signal_quality_plot(results)
            signal_b64 = _plot_to_base64(signal_plot)

            md_content.append(f"### {bar_type.upper()} Bars - Signal Quality Distribution")
            md_content.append("")
            md_content.append(
                f'<img src="data:image/png;base64,{signal_b64}" style="width: 100%; max-width: 1400px;">'
            )
            md_content.append("")

    # Risk-Adjusted Metrics Comparison
    md_content.append("## 🛡️ Risk-Adjusted Metrics Comparison")
    md_content.append("")

    risk_metrics_data = []
    for bar_type, results in results_dict.items():
        risk_metrics = calculate_risk_adjusted_metrics(results)

        risk_metrics_data.append(
            {
                "Bar Type": bar_type,
                "Omega Ratio": risk_metrics.loc["Omega Ratio", "Meta"],
                "Tail Ratio": risk_metrics.loc["Tail Ratio", "Meta"],
                "Sharpe/MaxDD": risk_metrics.loc["Sharpe/MaxDD", "Meta"],
                "Win-Loss Efficiency": risk_metrics.loc["Win-Loss Efficiency", "Meta"],
                "Expectancy/Vol": risk_metrics.loc["Expectancy/Vol", "Meta"],
            }
        )

    if risk_metrics_data:
        risk_metrics_df = pd.DataFrame(risk_metrics_data)
        md_content.append(risk_metrics_df.to_markdown(index=False))
        md_content.append("")

        # Identify best risk-adjusted bar type
        best_omega = risk_metrics_df.loc[risk_metrics_df["Omega Ratio"].idxmax(), "Bar Type"]
        best_sharpe_dd = risk_metrics_df.loc[risk_metrics_df["Sharpe/MaxDD"].idxmax(), "Bar Type"]

        md_content.append("#### Best Risk-Adjusted Performers")
        md_content.append("")
        md_content.append(f"**Best Omega Ratio**: `{best_omega}`  ")
        md_content.append(f"**Best Sharpe/MaxDD Ratio**: `{best_sharpe_dd}`  ")
        md_content.append("")

    # Trading Behavior Analysis
    md_content.append("## 📈 Trading Behavior Analysis")
    md_content.append("")

    behavior_data = []
    for bar_type, results in results_dict.items():
        primary_returns = results.get("primary_returns", pd.Series())
        meta_returns = results.get("meta_returns", pd.Series())

        if len(primary_returns) > 0 and len(meta_returns) > 0:
            primary_positions = pd.Series(
                1, index=primary_returns.index
            )  # Simplified position series
            meta_positions = pd.Series(1, index=meta_returns.index)

            primary_behavior = analyze_trading_behavior(primary_positions, primary_returns)
            meta_behavior = analyze_trading_behavior(meta_positions, meta_returns)

            behavior_data.append(
                {
                    "Bar Type": bar_type,
                    "Total Signals": primary_behavior.get("total_critical_points", 0),
                    "Meta Signals": meta_behavior.get("total_critical_points", 0),
                    "Flip Count": meta_behavior.get("flip_count", 0),
                    "Flattening Count": meta_behavior.get("flattening_count", 0),
                    "Signal Reduction %": (
                        (
                            primary_behavior.get("total_critical_points", 1)
                            - meta_behavior.get("total_critical_points", 0)
                        )
                        / primary_behavior.get("total_critical_points", 1)
                    )
                    * 100,
                }
            )

    if behavior_data:
        behavior_df = pd.DataFrame(behavior_data)
        md_content.append(behavior_df.to_markdown(index=False))
        md_content.append("")

    # Conclusions and Recommendations
    md_content.append("## 🎯 Conclusions and Recommendations")
    md_content.append("")

    if len(results_dict) > 1:
        # Find overall best performer
        best_performer = None
        best_sharpe = -np.inf

        for bar_type, results in results_dict.items():
            sharpe = results.get("meta_metrics", {}).get("sharpe_ratio", -np.inf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_performer = bar_type

        md_content.append(f"### 🏆 Recommended Bar Type: `{best_performer}`")
        md_content.append("")
        md_content.append(f"**Rationale**: Highest Sharpe Ratio (`{best_sharpe:.2f}`)  ")
        md_content.append("")

        # Key recommendations
        md_content.append("### Key Recommendations")
        md_content.append("")

        recommendations = []

        for bar_type, results in results_dict.items():
            meta_metrics = results.get("meta_metrics", {})
            signal_analysis = analyze_signal_quality(results)

            if signal_analysis.get("accepted_win_rate", 0) > 0.6:
                recommendations.append(
                    f"✅ `{bar_type}`: High accepted win rate ({signal_analysis['accepted_win_rate']:.1%}) - Consider increasing position size"
                )
            elif signal_analysis.get("accepted_win_rate", 0) < 0.45:
                recommendations.append(
                    f"⚠️ `{bar_type}`: Low accepted win rate ({signal_analysis['accepted_win_rate']:.1%}) - Consider adjusting confidence threshold"
                )

            if meta_metrics.get("profit_factor", 0) > 2.0:
                recommendations.append(
                    f"✅ `{bar_type}`: Excellent profit factor ({meta_metrics['profit_factor']:.2f})"
                )
            elif meta_metrics.get("profit_factor", 0) < 1.2:
                recommendations.append(
                    f"❌ `{bar_type}`: Poor profit factor ({meta_metrics['profit_factor']:.2f}) - Review strategy logic"
                )

        for rec in recommendations:
            md_content.append(f"- {rec}  ")

        md_content.append("")

        # Production considerations
        md_content.append("### Production Considerations")
        md_content.append("")

        for bar_type, results in results_dict.items():
            meta_metrics = results.get("meta_metrics", {})
            num_trades = meta_metrics.get("num_trades", 0)
            avg_duration = meta_metrics.get("avg_trade_duration", pd.Timedelta(0))

            if num_trades > 100:
                md_content.append(
                    f"**`{bar_type}`**: Sufficient trade sample ({num_trades} trades) for reliable statistics  "
                )
            else:
                md_content.append(
                    f"**`{bar_type}`**: Limited trade sample ({num_trades} trades) - results may not be statistically significant  "
                )

            if isinstance(avg_duration, pd.Timedelta):
                if avg_duration.days > 7:
                    md_content.append(
                        f"  • Long average trade duration ({avg_duration.days} days) - suitable for swing trading  "
                    )
                else:
                    md_content.append(
                        f"  • Short average trade duration ({avg_duration.days} days) - suitable for day trading  "
                    )

        md_content.append("")

    # Appendix
    md_content.append("## 📚 Appendix")
    md_content.append("")

    md_content.append("### A. Glossary of Metrics")
    md_content.append("")
    md_content.append("- **Sharpe Ratio**: Risk-adjusted return (higher is better)  ")
    md_content.append("- **Sortino Ratio**: Risk-adjusted return focusing on downside risk  ")
    md_content.append("- **Calmar Ratio**: Return relative to maximum drawdown  ")
    md_content.append("- **Omega Ratio**: Probability-weighted ratio of gains vs losses  ")
    md_content.append("- **Profit Factor**: Gross profit divided by gross loss  ")
    md_content.append("- **Win Rate**: Percentage of profitable trades  ")
    md_content.append("- **Max Drawdown**: Maximum peak-to-trough decline  ")
    md_content.append("- **Filter Rate**: Percentage of signals rejected by meta-model  ")
    md_content.append("- **Precision**: Percentage of accepted signals that were profitable  ")
    md_content.append("- **Recall**: Percentage of profitable signals that were accepted  ")
    md_content.append("")

    md_content.append("### B. Bar Type Characteristics")
    md_content.append("")
    md_content.append("| Bar Type | Pros | Cons | Best For |")
    md_content.append("|----------|------|------|----------|")
    md_content.append(
        "| **Tick Bars** | Volume-based, reduces time gaps | Complex implementation | High-frequency, volume-based strategies |"
    )
    md_content.append(
        "| **Time Bars** | Simple, standardized | Ignores volume | Traditional time-based strategies |"
    )
    md_content.append(
        "| **Dollar Bars** | Equal dollar volume, reduces noise | Requires tick data | Dollar-based strategies, institutional |"
    )
    md_content.append(
        "| **Volume Bars** | Volume-based, natural flow | Market-dependent | Volume-based strategies |"
    )
    md_content.append("")

    # Footer
    md_content.append("---")
    md_content.append(f"*Report generated by Meta-Labeling Analysis Module*  ")
    md_content.append(f"*Report saved to: `{filename}`*  ")
    if include_plots:
        md_content.append(f"*Plots saved in: `{plot_dir}`*  ")

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    # Save plots to directory if needed
    if include_plots:
        _save_all_plots(results_dict, plot_dir)

    print(f"✅ Meta-labeling report generated: {filename}")
    if include_plots:
        print(f"📊 Plots saved in: {plot_dir}")

    return filename


def _plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _generate_bar_type_comparison_plot(results_dict: Dict[str, Dict]) -> plt.Figure:
    """Generate comparison plot for different bar types."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Bar Type Performance Comparison", fontsize=16, fontweight="bold")

    metrics_to_plot = [
        ("sharpe_ratio", "Sharpe Ratio", "meta_metrics"),
        ("annualized_return", "Annual Return", "meta_metrics"),
        ("max_drawdown", "Max Drawdown", "meta_metrics"),
        ("win_rate", "Win Rate", "meta_metrics"),
        ("profit_factor", "Profit Factor", "meta_metrics"),
        ("num_trades", "Number of Trades", "meta_metrics"),
    ]

    colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))
    bar_types = list(results_dict.keys())

    for idx, (metric_key, title, metric_source) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        values = []
        for bar_type in bar_types:
            results = results_dict[bar_type]
            source_dict = results.get(metric_source, {})
            value = source_dict.get(metric_key, 0)
            if metric_key == "max_drawdown":
                value = abs(value)  # Make drawdown positive for comparison
            values.append(value)

        bars = ax.bar(bar_types, values, color=colors[: len(bar_types)])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(bar_types)))
        ax.set_xticklabels(bar_types, rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


def _generate_sharpe_evolution_plot(results_dict: Dict[str, Dict]) -> plt.Figure:
    """Generate cumulative returns and rolling Sharpe plot for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Strategy Evolution by Bar Type", fontsize=16, fontweight="bold")

    # Colors for different bar types
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    # Plot 1: Cumulative Returns
    ax1 = axes[0, 0]
    for (bar_type, results), color in zip(results_dict.items(), colors):
        meta_returns = results.get("meta_returns", pd.Series())
        if len(meta_returns) > 0:
            cum_returns = (1 + meta_returns).cumprod()
            ax1.plot(
                cum_returns.index,
                cum_returns.values,
                label=bar_type,
                color=color,
                alpha=0.7,
                linewidth=2,
            )

    ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Rolling Sharpe (63-day window)
    ax2 = axes[0, 1]
    for (bar_type, results), color in zip(results_dict.items(), colors):
        meta_returns = results.get("meta_returns", pd.Series())
        if len(meta_returns) > 63:
            rolling_sharpe = (
                meta_returns.rolling(63).mean() / meta_returns.rolling(63).std() * np.sqrt(252)
            )
            ax2.plot(
                rolling_sharpe.index,
                rolling_sharpe.values,
                label=bar_type,
                color=color,
                alpha=0.7,
                linewidth=1.5,
            )

    ax2.set_title("63-Day Rolling Sharpe Ratio", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(alpha=0.3)

    # Plot 3: Drawdown Comparison
    ax3 = axes[1, 0]
    for (bar_type, results), color in zip(results_dict.items(), colors):
        meta_returns = results.get("meta_returns", pd.Series())
        if len(meta_returns) > 0:
            cum_returns = (1 + meta_returns).cumprod()
            drawdown = cum_returns / cum_returns.cummax() - 1
            ax3.fill_between(
                drawdown.index, drawdown.values, 0, alpha=0.3, label=bar_type, color=color
            )

    ax3.set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Drawdown")
    ax3.grid(alpha=0.3)

    # Plot 4: Monthly Returns Heatmap (Fixed Version)
    ax4 = axes[1, 1]
    monthly_data = []
    bar_type_labels = []
    all_dates = []

    # Collect all monthly returns and dates
    for bar_type, results in results_dict.items():
        meta_returns = results.get("meta_returns", pd.Series())
        if len(meta_returns) > 0:
            # Use 'ME' instead of deprecated 'M'
            monthly = meta_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            monthly_data.append(monthly)
            bar_type_labels.append(bar_type)
            all_dates.extend(monthly.index)

    if monthly_data:
        # Find the common date range
        all_dates = pd.DatetimeIndex(all_dates)
        if len(all_dates) > 0:
            start_date = all_dates.min()
            end_date = all_dates.max()
            date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

            # Reindex all series to the common date range
            aligned_data = []
            for monthly in monthly_data:
                # Reindex to common date range, forward fill missing values
                aligned = monthly.reindex(date_range)
                aligned_data.append(aligned.fillna(0).values)

            # Create the heatmap matrix
            monthly_matrix = np.array(aligned_data)

            # Plot heatmap
            im = ax4.imshow(monthly_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.1)
            ax4.set_title("Monthly Returns Heatmap", fontsize=12, fontweight="bold")
            ax4.set_yticks(range(len(bar_type_labels)))
            ax4.set_yticklabels(bar_type_labels)
            ax4.set_xlabel("Months")

            # Set x-ticks for better readability
            num_months = len(date_range)
            if num_months > 12:
                # Show every 3rd month label
                tick_positions = range(0, num_months, 3)
                tick_labels = [date_range[i].strftime("%Y-%m") for i in tick_positions]
            else:
                tick_positions = range(num_months)
                tick_labels = [date_range[i].strftime("%Y-%m") for i in tick_positions]

            ax4.set_xticks(tick_positions)
            ax4.set_xticklabels(tick_labels, rotation=45, ha="right")
            plt.colorbar(im, ax=ax4, label="Return")
        else:
            ax4.text(
                0.5,
                0.5,
                "No monthly data available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
    else:
        ax4.text(
            0.5, 0.5, "No monthly data available", ha="center", va="center", transform=ax4.transAxes
        )
    plt.tight_layout()
    return fig


def _generate_signal_quality_plot(results: Dict) -> plt.Figure:
    """Generate signal quality distribution plot."""
    signal_analysis = analyze_signal_quality(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Signal distribution pie chart
    ax1 = axes[0]
    labels = ["Accepted", "Rejected"]
    sizes = [signal_analysis.get("accepted_signals", 0), signal_analysis.get("rejected_signals", 0)]
    colors = ["#4CAF50", "#F44336"]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Signal Distribution", fontsize=12, fontweight="bold")

    # Plot 2: Win rate comparison
    ax2 = axes[1]
    categories = ["Accepted", "Rejected", "Primary"]
    values = [
        signal_analysis.get("accepted_win_rate", 0),
        signal_analysis.get("rejected_win_rate", 0),
        results.get("primary_metrics", {}).get("win_rate", 0),
    ]
    colors = ["green" if v > 0.5 else "red" for v in values]

    bars = ax2.bar(categories, values, color=colors, alpha=0.6)
    ax2.axhline(y=0.5, color="white", linestyle="--", alpha=0.3)
    ax2.set_title("Win Rate Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim([0, 1])

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.style.use("dark_background")
    plt.tight_layout()
    return fig


def _save_all_plots(results_dict: Dict[str, Dict], plot_dir: Path):
    """Save all generated plots to directory."""
    # Bar type comparison plot
    comparison_plot = _generate_bar_type_comparison_plot(results_dict)
    comparison_plot.savefig(plot_dir / "bar_type_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(comparison_plot)

    # Sharpe evolution plot
    sharpe_plot = _generate_sharpe_evolution_plot(results_dict)
    sharpe_plot.savefig(plot_dir / "sharpe_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(sharpe_plot)

    # Individual strategy plots
    for bar_type, results in results_dict.items():
        # Strategy comparison plot
        strategy_plot = plot_strategy_comparison(results, figsize=(14, 8))
        strategy_plot.savefig(
            plot_dir / f"{bar_type}_strategy_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close(strategy_plot)

        # Signal quality plot
        signal_plot = _generate_signal_quality_plot(results)
        signal_plot.savefig(
            plot_dir / f"{bar_type}_signal_quality.png", dpi=150, bbox_inches="tight"
        )
        plt.close(signal_plot)


def generate_complete_meta_labeling_report(
    results_dict: Dict[str, Dict],
    strategy_config: Optional[Dict] = None,
    output_dir: Path = Path("meta_labeling_reports"),
    filename: str = "meta_labeling_analysis_report.md",
) -> Path:
    """
    Complete meta-labeling analysis workflow with markdown report generation.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary of results for different bar types
    strategy_config : Dict, optional
        Strategy configuration
    output_dir : Path
        Directory to save all outputs
    filename : str
        Name of the markdown report file

    Returns
    -------
    Path
        Path to the generated markdown report
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    # Define report path
    report_path = output_dir / filename

    print("🔍 Generating meta-labeling analysis report...")

    # Generate markdown report
    report_file = generate_meta_labeling_markdown_report(
        results_dict=results_dict,
        strategy_config=strategy_config,
        filename=report_path,
        include_plots=True,
        plot_dir=plot_dir,
    )

    # Also save raw comparison data as CSV
    comparison_data = []
    for bar_type, results in results_dict.items():
        meta_metrics = results.get("meta_metrics", {})
        signal_analysis = analyze_signal_quality(results)

        row = {"Bar Type": bar_type}
        row.update({f"meta_{k}": v for k, v in meta_metrics.items()})
        row.update({f"signal_{k}": v for k, v in signal_analysis.items()})
        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / "meta_labeling_comparison.csv", index=False)

    print(f"✅ Analysis complete! Files saved in: {output_dir}")
    print(f"📊 Report: {report_file}")
    print(f"📈 Plots: {plot_dir}")
    print(f"📁 Raw data: {output_dir / 'meta_labeling_comparison.csv'}")

    return report_file


# Example usage
if __name__ == "__main__":
    # Example with simulated data
    from datetime import datetime, timedelta

    # Create sample results for different bar types
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    sample_results = {}

    for bar_type in ["tick", "time", "dollar"]:
        # Simulate returns
        n_trades = np.random.randint(50, 200)
        trade_dates = np.random.choice(dates, n_trades, replace=False)
        trade_dates.sort()

        primary_returns = pd.Series(np.random.normal(0.001, 0.02, n_trades), index=trade_dates)

        meta_returns = pd.Series(
            np.random.normal(0.0015, 0.015, n_trades - 20), index=trade_dates[: n_trades - 20]
        )

        # Simulate metrics
        primary_metrics = {
            "total_return": 0.15,
            "annualized_return": 0.18,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 1.8,
            "max_drawdown": -0.12,
            "volatility": 0.15,
            "win_rate": 0.52,
            "profit_factor": 1.4,
            "num_trades": n_trades,
            "avg_trade_duration": pd.Timedelta(days=3),
            "expectancy": 0.005,
            "kelly_criterion": 0.12,
            "best_trade": 0.08,
            "worst_trade": -0.06,
            "consecutive_wins": 5,
            "consecutive_losses": 3,
        }

        meta_metrics = {
            "total_return": 0.18,
            "annualized_return": 0.22,
            "sharpe_ratio": 1.5 + np.random.uniform(-0.2, 0.2),
            "sortino_ratio": 1.8,
            "calmar_ratio": 2.1,
            "max_drawdown": -0.09,
            "volatility": 0.12,
            "win_rate": 0.58,
            "profit_factor": 1.8,
            "num_trades": n_trades - 20,
            "avg_trade_duration": pd.Timedelta(days=4),
            "signal_filter_rate": 0.25,
            "confidence_threshold": 0.6,
            "expectancy": 0.008,
            "kelly_criterion": 0.15,
            "best_trade": 0.09,
            "worst_trade": -0.05,
            "consecutive_wins": 6,
            "consecutive_losses": 2,
            "trades_per_year": int((n_trades - 20) * 1.2),  # Added this key
        }

        sample_results[bar_type] = {
            "strategy_name": f"Bollinger_{bar_type}",  # This key is REQUIRED
            "primary_metrics": primary_metrics,
            "meta_metrics": meta_metrics,
            "primary_returns": primary_returns,
            "meta_returns": meta_returns,
            "total_primary_signals": n_trades,
            "filtered_signals": n_trades - 20,
        }
