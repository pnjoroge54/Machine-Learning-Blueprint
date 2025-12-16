import warnings
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def performance_attribution_analysis(
    labels_df: pd.DataFrame,
    returns: pd.Series,
    predictions: pd.Series = None,
    signal_strength_bins: Union[int, list] = 5,
    regime_lookback: int = 252,
    min_observations: int = 10,
    plot_results: bool = True,
    figsize: tuple = (15, 12),
) -> Dict:
    """
    Decompose trading performance by signal strength and market regime.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Output from trend_scanning_labels with columns: ['t1', 'window', 'slope',
        't_value', 'rsquared', 'ret', 'bin']
    returns : pd.Series
        Price returns series (should align with labels_df index)
    predictions : pd.Series, optional
        Model predictions (-1, 0, 1). If None, uses labels_df['bin']
    signal_strength_bins : int or list, default=5
        Number of signal strength bins or explicit bin edges
    regime_lookback : int, default=252
        Lookback period for market regime classification (trading days)
    min_observations : int, default=10
        Minimum observations required per bin for statistical validity
    plot_results : bool, default=True
        Whether to generate visualization plots
    figsize : tuple, default=(15, 12)
        Figure size for plots

    Returns
    -------
    dict
        Comprehensive attribution results with the following keys:
        - 'signal_strength_analysis': Performance by t-value magnitude
        - 'market_regime_analysis': Performance by market conditions
        - 'combined_analysis': Two-dimensional attribution
        - 'summary_stats': Overall performance metrics
        - 'raw_data': Underlying data for further analysis
    """

    # Validate inputs
    required_cols = ["t_value", "ret", "bin"]
    missing_cols = [col for col in required_cols if col not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in labels_df: {missing_cols}")

    # Prepare data
    df = labels_df.copy()
    if predictions is not None:
        df["prediction"] = predictions.reindex(df.index)
        df["correct_direction"] = (df["bin"] * df["prediction"]) > 0
    else:
        df["prediction"] = df["bin"]
        df["correct_direction"] = True  # Using actual labels as predictions

    # Convert direction-agnostic returns to directional trading returns
    # Your trend_scanning_labels returns are holding period returns regardless of direction
    # For trading analysis, we need to multiply by the position direction
    df["trading_return"] = df["ret"] * df["prediction"]  # Long (+1) or Short (-1)
    df["abs_return"] = df["ret"].abs()  # Original direction-agnostic return magnitude

    # Add market regime features
    df = _add_market_regimes(df, returns, regime_lookback)

    # Add signal strength bins
    df = _add_signal_strength_bins(df, signal_strength_bins)

    # Filter for actual trades (non-zero predictions)
    trades_df = df[df["prediction"] != 0].copy()

    if len(trades_df) < min_observations:
        warnings.warn(
            f"Insufficient trades ({len(trades_df)}) for meaningful attribution"
        )

    # 1. Signal Strength Analysis
    signal_analysis = _analyze_by_signal_strength(trades_df, min_observations)

    # 2. Market Regime Analysis
    regime_analysis = _analyze_by_market_regime(trades_df, min_observations)

    # 3. Combined Analysis
    combined_analysis = _analyze_combined_attribution(trades_df, min_observations)

    # 4. Summary Statistics
    summary_stats = _calculate_summary_stats(trades_df)

    # 5. Visualization
    if plot_results:
        _create_attribution_plots(
            signal_analysis, regime_analysis, combined_analysis, trades_df, figsize
        )

    return {
        "signal_strength_analysis": signal_analysis,
        "market_regime_analysis": regime_analysis,
        "combined_analysis": combined_analysis,
        "summary_stats": summary_stats,
        "raw_data": trades_df,
    }


def _add_market_regimes(
    df: pd.DataFrame, returns: pd.Series, lookback: int
) -> pd.DataFrame:
    """Add market regime classifications to the dataframe."""
    df = df.copy()

    # Align returns with df index
    aligned_returns = returns.reindex(df.index, method="ffill")

    # Calculate rolling market metrics
    rolling_ret = aligned_returns.rolling(lookback, min_periods=lookback // 2)
    rolling_vol = rolling_ret.std() * np.sqrt(252)  # Annualized volatility
    rolling_trend = rolling_ret.mean() * 252  # Annualized return

    # Define regime thresholds (can be customized)
    vol_low, vol_high = rolling_vol.quantile([0.33, 0.67])
    trend_low, trend_high = rolling_trend.quantile([0.33, 0.67])

    # Market regimes
    conditions = [
        (rolling_vol <= vol_low) & (rolling_trend <= trend_low),  # Low Vol, Bear
        (rolling_vol <= vol_low) & (rolling_trend > trend_high),  # Low Vol, Bull
        (rolling_vol > vol_high) & (rolling_trend <= trend_low),  # High Vol, Bear
        (rolling_vol > vol_high) & (rolling_trend > trend_high),  # High Vol, Bull
    ]

    choices = ["Low_Vol_Bear", "Low_Vol_Bull", "High_Vol_Bear", "High_Vol_Bull"]

    df["market_regime"] = np.select(conditions, choices, default="Medium_Vol")
    df["market_volatility"] = rolling_vol
    df["market_trend"] = rolling_trend

    return df


def _add_signal_strength_bins(df: pd.DataFrame, bins: Union[int, list]) -> pd.DataFrame:
    """Add signal strength bins based on absolute t-values."""
    df = df.copy()

    abs_t_values = np.abs(df["t_value"])

    if isinstance(bins, int):
        # Create quantile-based bins, handling duplicates
        try:
            # First try with duplicates='drop' to get actual bin edges
            _, bin_edges = pd.qcut(
                abs_t_values, q=bins, retbins=True, duplicates="drop"
            )
            actual_bins = len(bin_edges) - 1

            # Create labels matching actual number of bins
            labels = [f"Weak_{i + 1}" for i in range(actual_bins)]

            df["signal_strength_bin"] = pd.qcut(
                abs_t_values, q=bins, labels=labels, duplicates="drop"
            )
        except ValueError:
            # Fallback: use pd.cut with percentile-based edges
            percentiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(abs_t_values, percentiles)
            # Remove duplicates manually
            bin_edges = np.unique(bin_edges)
            actual_bins = len(bin_edges) - 1

            labels = [f"Weak_{i + 1}" for i in range(actual_bins)]
            df["signal_strength_bin"] = pd.cut(
                abs_t_values, bins=bin_edges, labels=labels, include_lowest=True
            )
    else:
        # Use provided bin edges
        df["signal_strength_bin"] = pd.cut(
            abs_t_values,
            bins=bins,
            labels=[f"Bin_{i + 1}" for i in range(len(bins) - 1)],
        )
        bin_edges = bins

    df["abs_t_value"] = abs_t_values
    df["signal_strength_numeric"] = abs_t_values

    return df


def _analyze_by_signal_strength(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Analyze performance by signal strength bins."""
    if "signal_strength_bin" not in df.columns:
        return pd.DataFrame()

    analysis = (
        df.groupby("signal_strength_bin")
        .agg(
            {
                "trading_return": [
                    "count",
                    "mean",
                    "std",
                    "sum",
                ],  # Use directional returns
                "abs_return": "mean",  # Average magnitude of price moves
                "abs_t_value": ["mean", "min", "max"],
                "correct_direction": "mean",
                "prediction": lambda x: (x != 0).sum(),  # Number of trades
            }
        )
        .round(4)
    )

    # Flatten column names
    analysis.columns = ["_".join(col).strip() for col in analysis.columns]

    # Add derived metrics
    analysis["sharpe_ratio"] = (
        analysis["trading_return_mean"] / analysis["trading_return_std"]
    ).fillna(0) * np.sqrt(252)

    analysis["hit_rate"] = analysis["correct_direction_mean"]
    analysis["total_return"] = analysis["trading_return_sum"]

    # Filter for statistical significance
    analysis = analysis[analysis["trading_return_count"] >= min_obs]

    return analysis.sort_index()


def _analyze_by_market_regime(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Analyze performance by market regime."""
    if "market_regime" not in df.columns:
        return pd.DataFrame()

    analysis = (
        df.groupby("market_regime")
        .agg(
            {
                "trading_return": [
                    "count",
                    "mean",
                    "std",
                    "sum",
                ],  # Use directional returns
                "abs_return": "mean",  # Average magnitude of price moves
                "market_volatility": "mean",
                "market_trend": "mean",
                "correct_direction": "mean",
                "abs_t_value": "mean",
            }
        )
        .round(4)
    )

    # Flatten column names
    analysis.columns = ["_".join(col).strip() for col in analysis.columns]

    # Add derived metrics
    analysis["sharpe_ratio"] = (
        analysis["trading_return_mean"] / analysis["trading_return_std"]
    ).fillna(0) * np.sqrt(252)

    analysis["hit_rate"] = analysis["correct_direction_mean"]
    analysis["total_return"] = analysis["trading_return_sum"]

    # Filter for statistical significance
    analysis = analysis[analysis["trading_return_count"] >= min_obs]

    return analysis


def _analyze_combined_attribution(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Two-dimensional analysis: signal strength x market regime."""
    if "signal_strength_bin" not in df.columns or "market_regime" not in df.columns:
        return pd.DataFrame()

    analysis = (
        df.groupby(["signal_strength_bin", "market_regime"])
        .agg(
            {
                "trading_return": [
                    "count",
                    "mean",
                    "std",
                    "sum",
                ],  # Use directional returns
                "abs_return": "mean",  # Average magnitude of price moves
                "correct_direction": "mean",
                "abs_t_value": "mean",
            }
        )
        .round(4)
    )

    # Flatten column names
    analysis.columns = ["_".join(col).strip() for col in analysis.columns]

    # Add derived metrics
    analysis["sharpe_ratio"] = (
        analysis["trading_return_mean"] / analysis["trading_return_std"]
    ).fillna(0) * np.sqrt(252)

    analysis["hit_rate"] = analysis["correct_direction_mean"]
    analysis["total_return"] = analysis["trading_return_sum"]

    # Filter for statistical significance
    analysis = analysis[analysis["trading_return_count"] >= min_obs]

    return analysis.reset_index()


def _calculate_summary_stats(df: pd.DataFrame) -> Dict:
    """Calculate overall performance summary statistics."""
    if len(df) == 0:
        return {}

    trading_returns = df["trading_return"].dropna()  # Use directional returns
    abs_returns = (
        df["abs_return"].dropna() if "abs_return" in df else df["ret"].abs().dropna()
    )

    return {
        "total_trades": len(df),
        "total_return": trading_returns.sum(),
        "mean_return": trading_returns.mean(),
        "return_volatility": trading_returns.std(),
        "sharpe_ratio": (
            (trading_returns.mean() / trading_returns.std()) * np.sqrt(252)
            if trading_returns.std() > 0
            else 0
        ),
        "hit_rate": df["correct_direction"].mean()
        if "correct_direction" in df
        else np.nan,
        "max_drawdown": _calculate_max_drawdown(trading_returns.cumsum()),
        "skewness": stats.skew(trading_returns),
        "kurtosis": stats.kurtosis(trading_returns),
        "best_trade": trading_returns.max(),
        "worst_trade": trading_returns.min(),
        "avg_signal_strength": df["abs_t_value"].mean(),
        "avg_price_move_magnitude": abs_returns.mean(),  # Average absolute price movement
    }


def _calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return 0.0

    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    return drawdown.min()


def _create_attribution_plots(
    signal_analysis, regime_analysis, combined_analysis, trades_df, figsize
):
    """Create comprehensive visualization of attribution analysis."""

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Performance Attribution Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Returns by Signal Strength
    if not signal_analysis.empty:
        ax = axes[0, 0]
        signal_analysis["trading_return_mean"].plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Mean Trading Return by Signal Strength")
        ax.set_xlabel("Signal Strength Bin")
        ax.set_ylabel("Mean Trading Return")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Plot 2: Sharpe Ratio by Signal Strength
    if not signal_analysis.empty:
        ax = axes[0, 1]
        signal_analysis["sharpe_ratio"].plot(kind="bar", ax=ax, color="lightcoral")
        ax.set_title("Sharpe Ratio by Signal Strength")
        ax.set_xlabel("Signal Strength Bin")
        ax.set_ylabel("Sharpe Ratio")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Plot 3: Hit Rate by Signal Strength
    if not signal_analysis.empty:
        ax = axes[0, 2]
        signal_analysis["hit_rate"].plot(kind="bar", ax=ax, color="lightgreen")
        ax.set_title("Hit Rate by Signal Strength")
        ax.set_xlabel("Signal Strength Bin")
        ax.set_ylabel("Hit Rate")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random")
        ax.legend()

    # Plot 4: Returns by Market Regime
    if not regime_analysis.empty:
        ax = axes[1, 0]
        regime_analysis["trading_return_mean"].plot(kind="bar", ax=ax, color="gold")
        ax.set_title("Mean Trading Return by Market Regime")
        ax.set_xlabel("Market Regime")
        ax.set_ylabel("Mean Trading Return")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Plot 5: Trade Distribution
    ax = axes[1, 1]
    trades_df["prediction"].value_counts().plot(
        kind="bar", ax=ax, color="purple", alpha=0.7
    )
    ax.set_title("Trade Distribution by Direction")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Number of Trades")

    # Plot 6: Trading Return Distribution
    ax = axes[1, 2]
    trades_df["trading_return"].hist(bins=50, ax=ax, color="orange", alpha=0.7)
    ax.set_title("Trading Return Distribution")
    ax.set_xlabel("Trading Return")
    ax.set_ylabel("Frequency")
    ax.axvline(
        x=trades_df["trading_return"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {trades_df['trading_return'].mean():.4f}",
    )
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Additional heatmap for combined analysis if data exists
    if not combined_analysis.empty and len(combined_analysis) > 1:
        plt.figure(figsize=(12, 8))

        pivot_data = combined_analysis.pivot(
            index="signal_strength_bin",
            columns="market_regime",
            values="trading_return_mean",
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".4f",
            cmap="RdYlBu_r",
            center=0,
            cbar_kws={"label": "Mean Return"},
        )
        plt.title("Mean Return: Signal Strength × Market Regime")
        plt.xlabel("Market Regime")
        plt.ylabel("Signal Strength")
        plt.tight_layout()
        plt.show()


# Example usage function
def example_usage():
    """
    Example of how to use the performance attribution analysis.
    """
    # Assuming you have your trend scanning labels and returns
    # labels_df = trend_scanning_labels(close_prices, ...)
    # returns = close_prices.pct_change()
    # predictions = your_model.predict(features)  # Optional

    # results = performance_attribution_analysis(
    #     labels_df=labels_df,
    #     returns=returns,
    #     predictions=predictions,  # Optional - if None, uses labels_df['bin']
    #     signal_strength_bins=5,   # or [0, 1, 2, 3, 5, 10] for custom bins
    #     regime_lookback=252,      # 1 year for market regime classification
    #     plot_results=True
    # )

    # # Access results
    # print("Signal Strength Analysis:")
    # print(results['signal_strength_analysis'])
    #
    # print("\nMarket Regime Analysis:")
    # print(results['market_regime_analysis'])
    #
    # print("\nSummary Stats:")
    # for key, value in results['summary_stats'].items():
    #     print(f"{key}: {value:.4f}")

    pass
