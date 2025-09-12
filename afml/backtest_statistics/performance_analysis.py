import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from numba import njit
from scipy.stats import kurtosis, skew

from .statistics import (
    all_bets_concentration,
    average_holding_period,
    drawdown_and_time_under_water,
    information_ratio,
    timing_of_flattening_and_flips,
)

# Metrics where a lower value is preferable (e.g., risk, losses).
lower_is_better = [
    "max_drawdown",
    "avg_drawdown",
    "volatility",
    "downside_volatility",
    "avg_loss",
    "worst_trade",
    "consecutive_losses",
    "var_95",
    "cvar_95",
    "ulcer_index",
]


# --- Helper Functions for Trade Analysis ---


def get_trades(returns: pd.Series, positions: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies individual trades from a positions series and calculates their
    compounded returns and holding durations.

    This function iterates through trade events, which are defined by the points
    where a position is either fully closed (flattened) or reversed (flipped).
    It uses `timing_of_flattening_and_flips` to detect these event timestamps.

    Args:
        returns: A Series of asset returns, indexed by timestamp.
        positions: A Series of position sizes (e.g., -1 for short, 1 for long,
                   0 for flat), indexed by timestamp.

    Returns:
        A tuple containing two lists:
        - A list of compounded returns for each closed trade.
        - A series of durations (as Timedelta objects) for each closed trade.
    """
    if positions.empty or returns.empty:
        return [], []

    # Get the timestamps where trades are closed or reversed.
    trade_end_times = timing_of_flattening_and_flips(positions)
    if trade_end_times.empty:
        return [], []
    else:
        trade_end_times = returns.index.get_indexer(trade_end_times)
        trade_returns, trade_durations = _get_trades_numba_core(
            returns.to_numpy(),
            positions.to_numpy(),
            trade_end_times,
        )
        trade_durations = [returns.index[x[1]] - returns.index[x[0]] for x in trade_durations]

        return pd.Series(trade_returns), pd.Series(trade_durations)


@njit(cache=True)
def _get_trades_numba_core(returns, positions, trade_end_times):
    trade_returns = []
    trade_durations = []
    last_trade_start_idx = 0

    # Iterate through each trade's end time to define a trade period.
    for end_time in trade_end_times:
        # Locate the integer index for the end of the current trade.
        trade_end_idx_loc = end_time

        # Slice the positions series to isolate the current trade.
        trade_slice = positions[last_trade_start_idx : trade_end_idx_loc + 1]

        # Ignore periods where no position was held (all zeros).
        trade_pos_series = trade_slice[trade_slice != 0]
        if len(trade_pos_series) == 0:
            last_trade_start_idx = trade_end_idx_loc + 1
            continue

        # The direction of the trade (long or short) is determined by the first non-zero position.
        trade_direction = np.sign(trade_pos_series[0])

        # Get the returns corresponding to the trade's holding period.
        trade_period_returns = returns[last_trade_start_idx : trade_end_idx_loc + 1]

        # Calculate the final return for the trade by compounding the period returns.
        # The returns are multiplied by the trade direction (1 for long, -1 for short).
        directed_returns = trade_period_returns * trade_direction
        compounded_return = (1 + directed_returns).prod() - 1
        trade_returns.append(compounded_return)

        # Calculate the duration of the trade.
        if len(trade_period_returns) > 1:
            trade_durations.append((last_trade_start_idx, trade_end_idx_loc))
        else:
            # For single-period trades, duration is considered zero.
            trade_durations.append((0, 0))

        # The next trade starts immediately after the current one ends.
        last_trade_start_idx = trade_end_idx_loc + 1

    return trade_returns, trade_durations


def _calculate_trade_stats(
    returns: pd.Series,
    periods_per_year: int,
    total_periods: int,
) -> dict:
    """
    Calculates a dictionary of common trade-based performance statistics.

    This function takes lists of returns and durations for individual trades
    and computes metrics like win rate, profit factor, and expectancy.

    Args:
        returns: A Series of returns from each individual trade.
        periods_per_year: The number of periods in a trading year.
        total_periods: The total number of periods in the backtest horizon.

    Returns:
        A dictionary containing key trade statistics.
    """

    if returns.empty:
        return {}

    num_trades = len(returns)
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]

    # Calculate basic win/loss metrics.
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

    # The profit factor is the gross profit divided by the gross loss.
    gross_profit = winning_trades.sum()
    gross_loss = abs(losing_trades.sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    # Expectancy is the average amount you expect to win or lose per trade.
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # The Kelly Criterion helps determine the optimal fraction of capital to allocate.
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_criterion = win_rate - ((1 - win_rate) / win_loss_ratio)
    else:
        # If there are no losses, the Kelly fraction is infinite (or undefined).
        kelly_criterion = 0 if avg_win == 0 else float("inf")

    def max_consecutive(arr: np.ndarray) -> Tuple[int, int]:
        """Helper to find the longest streaks of wins and losses."""
        is_positive = arr > 0
        is_negative = arr < 0

        # Calculate max consecutive wins
        max_wins = 0
        current_wins = 0
        for x in is_positive:
            current_wins = current_wins + 1 if x else 0
            max_wins = max(max_wins, current_wins)

        # Calculate max consecutive losses
        max_losses = 0
        current_losses = 0
        for x in is_negative:
            current_losses = current_losses + 1 if x else 0
            max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    consecutive_wins, consecutive_losses = max_consecutive(returns.values)

    return {
        "num_trades": num_trades,
        "trades_per_year": int(
            num_trades * (periods_per_year / total_periods) if total_periods > 0 else 0
        ),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": returns.max() if num_trades > 0 else 0,
        "worst_trade": returns.min() if num_trades > 0 else 0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "kelly_criterion": kelly_criterion,
        "consecutive_wins": consecutive_wins,
        "consecutive_losses": consecutive_losses,
    }


# --- Helper Functions for General Metrics ---


def get_annualization_factors(
    timeframe: Optional[str] = None,
    data_index: Optional[pd.DatetimeIndex] = None,
    trading_days_per_year: int = 252,
    trading_hours_per_day: float = 6.5,
) -> Tuple[float, int]:
    """
    Calculates the annualization factor (sqrt(N)) and periods per year (N).

    This function operates in one of two modes:
    1.  **Explicit Calculation** (if `timeframe` is provided): For standard,
        time-based bars like 'D1', 'H4', 'M15'.
    2.  **Empirical Inference** (if `data_index` is provided): For non-standard
        bars (e.g., tick, volume, dollar) by analyzing the timestamps of the data.

    Args:
        timeframe: The time-based frame string (e.g., 'M15', 'H4', 'D1').
        data_index: The DatetimeIndex of the returns series.
        trading_days_per_year: Number of trading days in a year.
        trading_hours_per_day: Trading hours in a day.

    Returns:
        A tuple containing (annualization_factor, periods_per_year).

    Raises:
        ValueError: If neither or both `timeframe` and `data_index` are provided.
    """
    if timeframe and data_index is not None:
        raise ValueError("Provide either 'timeframe' or 'data_index', not both.")
    if not timeframe and data_index is None:
        raise ValueError("Either 'timeframe' or 'data_index' must be provided.")

    # Mode 1: Explicit calculation for standard time-based bars.
    if timeframe:
        timeframe = timeframe.upper()
        # Extract the numeric part of the timeframe string, defaulting to 1 if none.
        numeric_val = int("".join(filter(str.isdigit, timeframe)) or 1)

        # Define the number of periods within a full trading year for each base unit.
        periods_in_year = {
            "W": 52,
            "D": trading_days_per_year,
            "H": trading_days_per_year * trading_hours_per_day,
            "M": trading_days_per_year * trading_hours_per_day * 60,
        }

        time_unit = "".join(filter(str.isalpha, timeframe))
        if time_unit not in periods_in_year:
            raise ValueError(f"Invalid timeframe unit '{time_unit}'. Must be W, D, H, or M.")

        # Calculate periods per year by dividing the base by the timeframe's numeric value.
        periods_per_year = periods_in_year[time_unit] / numeric_val
        return np.sqrt(periods_per_year), int(round(periods_per_year))

    # Mode 2: Empirical inference for non-standard bars from the data's index.
    if data_index is not None:
        if not isinstance(data_index, pd.DatetimeIndex) or len(data_index) < 2:
            warnings.warn(
                "data_index is too short. Falling back to default daily.",
                UserWarning,
                stacklevel=2,
            )
            return np.sqrt(trading_days_per_year), trading_days_per_year

        # Estimate the average time between bars.
        avg_delta = (data_index[-1] - data_index[0]) / (len(data_index) - 1)
        if avg_delta.total_seconds() == 0:
            warnings.warn(
                "Cannot infer frequency from index. Falling back to default daily.",
                UserWarning,
                stacklevel=2,
            )
            return np.sqrt(trading_days_per_year), trading_days_per_year

        # Calculate periods per year based on the average bar frequency.
        periods_per_year = pd.Timedelta(days=365.25) / avg_delta
        return np.sqrt(periods_per_year), int(round(periods_per_year))

    # This line should not be reached due to the initial checks.
    return np.sqrt(trading_days_per_year), trading_days_per_year


# --- Main Orchestrator Function ---


def calculate_performance_metrics(
    returns: pd.Series,
    data_index: pd.Index,
    positions: Optional[pd.Series] = None,
    trading_days_per_year: int = 252,
    trading_hours_per_day: float = 24,
) -> dict:
    """
    Calculates a comprehensive set of performance metrics for a return series.

    This function serves as the main orchestrator, calling helper functions
    to compute portfolio-level stats, risk/return ratios, drawdown metrics,
    and trade-based statistics.

    Args:
        returns: A Series of returns from individual trades.
        data_index: The index from the full data used.
        positions: An optional Series of positions (same index as returns)
            to enable trade-based analysis.
        trading_days_per_year: The number of trading days in a year.
        trading_hours_per_day: The number of trading hours in a day.

    Returns:
        A dictionary containing a wide range of performance metrics.
    """
    # --- Input Validation and Sanitization ---
    if not returns.index.isin(data_index).all():
        raise ValueError("Returns index must be fully contained within data index.")
    if positions.index != returns.index:
        raise ValueError("Positions must have the same index as returns.")

    # If data is too short or has no variance, return a dict of zeros.
    if len(returns) < 2 or returns.std() == 0:
        metric_keys = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "avg_drawdown",
            "drawdown_duration",
            "volatility",
            "downside_volatility",
            "num_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "expectancy",
            "kelly_criterion",
            "var_95",
            "cvar_95",
            "skewness",
            "kurtosis",
            "best_trade",
            "worst_trade",
            "consecutive_wins",
            "consecutive_losses",
            "avg_trade_duration",
            "trades_per_year",
            "ulcer_index",
        ]
        zero_metrics = {metric: 0 for metric in metric_keys}
        zero_metrics["avg_trade_duration"] = pd.Timedelta(0)
        return zero_metrics

    # --- Annualization ---
    annualization_factor, periods_per_year = get_annualization_factors(
        data_index=data_index,
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )

    # --- Overall Return & Risk ---
    total_return = (1 + returns).prod() - 1
    years = (data_index[-1] - data_index[0]).days / 365.25 if periods_per_year > 0 else 0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = returns.std()
    annualized_volatility = volatility * annualization_factor

    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std(ddof=0) if len(downside_returns) > 0 else 0
    annualized_downside_volatility = downside_volatility * annualization_factor

    # --- Ratios & Distribution ---
    sharpe_ratio = (returns.mean() / volatility) * annualization_factor if volatility != 0 else 0.0
    sortino_ratio = (
        (returns.mean() / downside_volatility) * annualization_factor
        if downside_volatility != 0
        else 0.0
    )
    skewness = skew(returns) if len(returns) > 3 else 0
    kurtosis_ = kurtosis(returns) if len(returns) > 3 else 0
    positive_concentration, negative_concentration, time_concentration = all_bets_concentration(
        returns
    )

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": annualized_volatility,
        "downside_volatility": annualized_downside_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        # Value at Risk (VaR) at 95% confidence level.
        "var_95": np.percentile(returns, 5),
        # Conditional VaR (CVaR) or Expected Shortfall at 95% confidence.
        "cvar_95": (
            returns[returns <= np.percentile(returns, 5)].mean()
            if any(returns <= np.percentile(returns, 5))
            else 0
        ),
        "skewness": skewness,
        "kurtosis": kurtosis_,
        "pos_concentration": positive_concentration,
        "neg_concentration": negative_concentration,
        "time_concentration": time_concentration,
    }

    # --- Drawdown Stats ---
    drawdown, time_under_water = drawdown_and_time_under_water((1 + returns).cumprod())
    drawdown_stats = {
        "max_drawdown": drawdown.max() if not drawdown.empty else 0,
        "avg_drawdown": drawdown.mean() if not drawdown.empty else 0,
        "drawdown_duration": (
            pd.Timedelta(days=time_under_water.mean()).round("1s")
            if not time_under_water.empty
            else pd.Timedelta(0)
        ),
        # The Ulcer Index measures the depth and duration of drawdowns.
        "ulcer_index": np.sqrt((drawdown**2).mean()) if not drawdown.empty else 0,
    }
    metrics.update(drawdown_stats)

    # The Calmar ratio is the annualized return over the maximum drawdown.
    metrics["calmar_ratio"] = (
        annualized_return / metrics["max_drawdown"] if metrics["max_drawdown"] != 0 else 0.0
    )

    # --- Trade Stats (if positions are provided) ---
    total_periods = len(data_index)

    if positions:
        metrics["avg_trade_duration"] = average_holding_period(positions)
        bet_frequency = timing_of_flattening_and_flips(positions).shape[0]
        metrics["bet_frequency"] = bet_frequency
        metrics["bets_per_year"] = int(
            bet_frequency * (periods_per_year / total_periods) if total_periods > 0 else 0
        )

    trade_stats = _calculate_trade_stats(returns, periods_per_year, total_periods)

    side = positions.loc[returns.index]
    longs = (side > 0).sum()
    shorts = (side < 0).sum()
    trade_stats["ratio_of_longs"] = longs / (longs + shorts)

    metrics.update(trade_stats)

    return metrics


def analyze_trading_behavior(positions: pd.Series, returns: pd.Series) -> dict:
    """
    Analyze trading behavior using flattening and flip points.

    Parameters:
    positions: Series of target positions
    returns: Series of returns

    Returns:
    Dictionary with trading behavior metrics
    """
    # Get critical points
    critical_points = timing_of_flattening_and_flips(positions)

    # Classify points as flips or flattenings
    flip_mask = positions.reindex(critical_points) != 0
    flips = critical_points[flip_mask]
    flattenings = critical_points[~flip_mask]

    # Calculate returns around these points
    pre_returns = []
    post_returns = []

    for point in critical_points:
        if point in returns.index:
            # Look at returns 5 periods before and after
            pre_period = returns[returns.index < point].tail(5)
            post_period = returns[returns.index > point].head(5)

            if not pre_period.empty:
                pre_returns.append(pre_period.mean())
            if not post_period.empty:
                post_returns.append(post_period.mean())

    return {
        "total_critical_points": len(critical_points),
        "flip_count": len(flips),
        "flattening_count": len(flattenings),
        "avg_pre_critical_return": np.mean(pre_returns) if pre_returns else 0,
        "avg_post_critical_return": np.mean(post_returns) if post_returns else 0,
        "critical_points": critical_points,
    }


# --- Helper Functions for Meta-Labelling Analysis ---


def evaluate_meta_labeling_performance(
    primary_signals: pd.Series,
    meta_probabilities: pd.Series,
    returns: pd.Series,
    confidence_threshold: float = 0.5,
    trading_days_per_year: int = 252,
    trading_hours_per_day: int = 24,
    strategy_name: str = "Strategy",
) -> dict:
    """
    Evaluates and compares the performance of a primary strategy against a
    meta-labeled version of that strategy.

    This function simulates two strategies:
    1.  The primary strategy, which takes all signals.
    2.  The meta-labeled strategy, which filters trades based on a confidence
        threshold and sizes them according to the meta-model's probability.

    Args:
        primary_signals: A Series of trading signals from the primary model.
        meta_probabilities: A Series or array of probabilities from the meta-model.
        returns: A Series of the underlying asset's returns.
        confidence_threshold: The minimum probability required to take a trade.
        trading_days_per_year: The number of trading days in a year.
        trading_hours_per_day: The number of trading hours per day.
        strategy_name: The name of the strategy for reporting.

    Returns:
        A dictionary containing the performance metrics for both strategies,
        their return series, and other comparison metadata.
    """
    # Align returns data with the signal index for accurate backtesting.
    aligned_returns = returns.reindex(primary_signals.index, fill_value=0)

    # --- 1. Primary Strategy Performance ---
    primary_positions = primary_signals.copy()
    # Calculate returns by multiplying the previous period's position by the current period's return.
    primary_returns = (primary_positions.shift(1) * aligned_returns).dropna()

    # --- 2. Meta-Labeled Strategy Performance ---
    meta_positions = primary_signals.copy()

    # Ensure probabilities are a Series aligned with the primary signals index.
    aligned_probs = meta_probabilities.reindex(primary_signals.index, fill_value=0.5)

    # Clip probabilities to avoid issues with log(0) or division by zero in z-score.
    adjusted_meta_prob = aligned_probs.clip(lower=1e-6, upper=1 - 1e-6)

    # Filter trades: Set position to 0 for trades below the confidence threshold.
    confident_trades = aligned_probs > confidence_threshold
    meta_positions[~confident_trades] = 0

    # --- Bet Sizing Logic ---
    # Convert the model's probability into a z-score.
    z_scores = (adjusted_meta_prob - 0.5) / (adjusted_meta_prob * (1 - adjusted_meta_prob)) ** 0.5
    # Map the z-score to a bet size from -1 to 1 using the normal CDF, creating a sigmoid-like sizing.
    bet_sizes_raw = 2 * stats.norm.cdf(z_scores) - 1

    # Apply the calculated bet size to the signals that were not filtered out.
    active_signal_indices = meta_positions != 0
    meta_positions.loc[active_signal_indices] *= np.abs(bet_sizes_raw[active_signal_indices])

    # Calculate meta-strategy returns with the new, dynamically sized positions.
    meta_returns = (meta_positions.shift(1) * aligned_returns).dropna()

    # --- 3. Performance Calculation ---
    primary_metrics = calculate_performance_metrics(
        returns=primary_returns,
        data_index=returns.index,
        positions=primary_positions,
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )
    meta_metrics = calculate_performance_metrics(
        returns=meta_returns,
        data_index=returns.index,
        positions=meta_positions,
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )

    # --- 4. Meta-Specific Metrics ---
    total_signals = len(primary_signals[primary_signals != 0])
    filtered_signals = len(meta_positions[meta_positions != 0])
    # The percentage of trades that were filtered out by the meta-model.
    meta_metrics["signal_filter_rate"] = (
        1 - (filtered_signals / total_signals) if total_signals > 0 else 0
    )
    meta_metrics["confidence_threshold"] = confidence_threshold

    # Information Ratio: Measures the meta-model's ability to generate excess return per unit of tracking error.
    excess_returns = meta_returns - primary_returns.reindex(meta_returns.index, fill_value=0)
    _, periods_per_year = get_annualization_factors(
        data_index=excess_returns.index,
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )
    meta_metrics["information_ratio"] = information_ratio(
        returns=excess_returns, benchmark=0, entries_per_year=periods_per_year
    )

    return {
        "strategy_name": strategy_name,
        "primary_metrics": primary_metrics,
        "meta_metrics": meta_metrics,
        "primary_returns": primary_returns,
        "meta_returns": meta_returns,
        "total_primary_signals": total_signals,
        "filtered_signals": filtered_signals,
    }


def print_meta_labeling_comparison(results: dict, save_path: str = None):
    """
    Prints and optionally saves a comprehensive comparison of the primary strategy
    versus the meta-labeled strategy performance.

    Args:
        results: Output dictionary from `evaluate_meta_labeling_performance`
        save_path: If provided, saves the output to this file path
    """
    import io
    from contextlib import redirect_stdout

    # Capture output in a string buffer
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        strategy_name = results["strategy_name"]
        primary_metrics = results["primary_metrics"]
        meta_metrics = results["meta_metrics"]

        print(f"\n{'='*100}")
        print(f"Meta-Labeling Performance Analysis: {strategy_name}")
        print(f"{'='*100}")

        # --- Signal Filtering Summary ---
        print(f"\nSignal Filtering Summary:")
        print(f"  Total Primary Signals: {results['total_primary_signals']:,}")
        print(f"  Filtered Signals: {results['filtered_signals']:,}")
        print(f"  Filter Rate: {meta_metrics['signal_filter_rate']:,.2%}")
        print(f"  Confidence Threshold: {meta_metrics['confidence_threshold']}")

        # --- Core Performance Metrics Table ---
        print(
            f"\n{'CORE PERFORMANCE METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}"
        )
        print("=" * 75)
        core_metrics = [
            ("Total Return", "total_return", "%"),
            ("Annualized Return", "annualized_return", "%"),
            ("Sharpe Ratio", "sharpe_ratio", "4f"),
            ("Sortino Ratio", "sortino_ratio", "4f"),
            ("Calmar Ratio", "calmar_ratio", "4f"),
            ("Information Ratio", "information_ratio", "4f"),
        ]
        for display_name, metric_key, fmt in core_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics.get(metric_key, 0)
                meta_val = meta_metrics.get(metric_key, 0)
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.2%}" if fmt == "%" else f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.2%}" if fmt == "%" else f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Risk Metrics Table ---
        print(f"\n{'RISK METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}")
        print("=" * 75)
        risk_metrics = [
            ("Max Drawdown", "max_drawdown", "%"),
            ("Avg Drawdown", "avg_drawdown", "%"),
            ("Volatility (Ann.)", "volatility", "4f"),
            ("Downside Volatility", "downside_volatility", "4f"),
            ("Ulcer Index", "ulcer_index", "4f"),
            ("VaR (95%)", "var_95", "%"),
            ("CVaR (95%)", "cvar_95", "%"),
        ]
        for display_name, metric_key, fmt in risk_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics.get(metric_key, 0)
                meta_val = meta_metrics.get(metric_key, 0)
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.2%}" if fmt == "%" else f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.2%}" if fmt == "%" else f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Trading Metrics Table ---
        print(f"\n{'TRADING METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}")
        print("=" * 75)
        trading_metrics = [
            ("Number of Trades", "num_trades", "0f"),
            ("Trades per Year", "trades_per_year", "1f"),
            ("Win Rate", "win_rate", "%"),
            ("Avg Win", "avg_win", "%"),
            ("Avg Loss", "avg_loss", "%"),
            ("Best Trade", "best_trade", "%"),
            ("Worst Trade", "worst_trade", "%"),
            ("Profit Factor", "profit_factor", "2f"),
            ("Expectancy", "expectancy", "%"),
            ("Kelly Criterion", "kelly_criterion", "4f"),
            ("Max Consecutive Wins", "consecutive_wins", "0f"),
            ("Max Consecutive Losses", "consecutive_losses", "0f"),
        ]
        for display_name, metric_key, fmt in trading_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics[metric_key]
                meta_val = meta_metrics[metric_key]
                improvement = calculate_improvement(primary_val, meta_val, metric_key)

                if fmt == "%":
                    primary_str, meta_str = f"{primary_val:,.2%}", f"{meta_val:,.2%}"
                elif fmt == "0f":
                    primary_str, meta_str = f"{primary_val:,.0f}", f"{meta_val:,.0f}"
                elif fmt == "1f":
                    primary_str, meta_str = f"{primary_val:,.1f}", f"{meta_val:,.1f}"
                elif fmt == "2f":
                    primary_str, meta_str = f"{primary_val:,.2f}", f"{meta_val:,.2f}"
                else:
                    primary_str, meta_str = f"{primary_val:,.4f}", f"{meta_val:,.4f}"

                improvement_str = f"{improvement:+,.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Distribution Metrics Table ---
        print(
            f"\n{'DISTRIBUTION METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}"
        )
        print("=" * 75)
        dist_metrics = [("Skewness", "skewness", "4f"), ("Kurtosis", "kurtosis", "4f")]
        for display_name, metric_key, fmt in dist_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics[metric_key]
                meta_val = meta_metrics[metric_key]
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Summary Assessment ---
        print(f"\n{'SUMMARY ASSESSMENT'}")
        print("=" * 50)
        key_improvements = []
        if "sharpe_ratio" in meta_metrics and "sharpe_ratio" in primary_metrics:
            sharpe_imp = calculate_improvement(
                primary_metrics["sharpe_ratio"],
                meta_metrics["sharpe_ratio"],
                "sharpe_ratio",
            )
            key_improvements.append(("Sharpe Ratio", sharpe_imp))
        if "total_return" in meta_metrics and "total_return" in primary_metrics:
            return_imp = calculate_improvement(
                primary_metrics["total_return"],
                meta_metrics["total_return"],
                "total_return",
            )
            key_improvements.append(("Total Return", return_imp))
        if "max_drawdown" in meta_metrics and "max_drawdown" in primary_metrics:
            dd_imp = calculate_improvement(
                primary_metrics["max_drawdown"],
                meta_metrics["max_drawdown"],
                "max_drawdown",
            )
            key_improvements.append(("Max Drawdown", dd_imp))

        avg_improvement = np.mean([imp for _, imp in key_improvements if imp != float("inf")])
        if avg_improvement > 10:
            assessment = "✅ Meta-labeling shows SIGNIFICANT improvement"
        elif avg_improvement > 5:
            assessment = "✅ Meta-labeling shows GOOD improvement"
        elif avg_improvement > 0:
            assessment = "⚠️  Meta-labeling shows MODEST improvement"
        else:
            assessment = "❌ Meta-labeling DOES NOT improve performance"

        print(f"  {assessment}")
        for metric_name, improvement in key_improvements:
            if improvement != float("inf"):
                print(f"  {metric_name} Change: {improvement:+.1f}%")

        if "sharpe_ratio" in meta_metrics and meta_metrics["sharpe_ratio"] > primary_metrics.get(
            "sharpe_ratio", 0
        ):
            print(f"\n✅ Meta-labeling improves risk-adjusted returns")
        if "signal_filter_rate" in meta_metrics:
            print(
                f"\n📊 Signal filtering removed {meta_metrics['signal_filter_rate']:,.1%} of trades"
            )
            if meta_metrics["signal_filter_rate"] > 0.3:
                print(f"   High filtering rate suggests meta-model is selective")

    # Get the captured output
    output_text = output_buffer.getvalue()

    # Print to console
    print(output_text)

    # Save to file if path provided
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\nOutput saved to: {save_path}")


def calculate_improvement(primary_val: float, meta_val: float, metric_key: str) -> float:
    """
    Calculates the percentage improvement of a metric from primary to meta.

    This function correctly handles the direction of improvement: for some
    metrics, higher is better (e.g., Sharpe Ratio), while for others,
    lower is better (e.g., Max Drawdown).

    Args:
        primary_val: The metric value for the primary strategy.
        meta_val: The metric value for the meta-labeled strategy.
        metric_key: The name of the metric, used to determine if lower is better.

    Returns:
        The percentage improvement.
    """
    if primary_val == 0:
        return 0 if meta_val == 0 else float("inf")

    # For metrics where a lower value is preferable (e.g., risk, losses).
    lower_is_better = [
        "max_drawdown",
        "avg_drawdown",
        "volatility",
        "downside_volatility",
        "avg_loss",
        "worst_trade",
        "consecutive_losses",
        "var_95",
        "cvar_95",
        "ulcer_index",
    ]
    if metric_key in lower_is_better:
        return (primary_val - meta_val) / abs(primary_val) * 100

    # For metrics where a higher value is better.
    return (meta_val - primary_val) / abs(primary_val) * 100


# --- Main Orchestrator Function ---


def run_meta_labeling_analysis(
    df: pd.DataFrame,
    signals: pd.Series,
    meta_probabilities: Union[pd.Series, np.ndarray],
    confidence_threshold: float = 0.5,
    trading_days_per_year: int = 252,
    trading_hours_per_day: int = 24,
    strategy_name: str = "Strategy",
    save_path: str = None,
):
    """
    A wrapper function to run a complete meta-labeling analysis.

    This function prepares the necessary returns data before calling the
    main `evaluate_meta_labeling_performance` function and then prints the results.

    Args:
        df: The full DataFrame containing historical price data.
        signals: Trading signals from the primary model for the test period.
        meta_probabilities: Probabilities from the meta-model for the test period.
        confidence_threshold: The probability threshold for filtering trades.
        strategy_name: The name for the strategy run.
        save_path: If provided, saves the output to this file path
    """
    # Isolate returns for the test period, including one prior point for shift(1).
    test_start = signals.index[0]
    pre_test_point = df.index[df.index.get_loc(test_start) - 1]
    return_index = signals.index.union([pre_test_point])
    returns = df.loc[return_index, "close"].pct_change()

    results = evaluate_meta_labeling_performance(
        primary_signals=signals,
        meta_probabilities=meta_probabilities,
        returns=returns,
        confidence_threshold=confidence_threshold,
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
        strategy_name=strategy_name,
    )

    print_meta_labeling_comparison(results, save_path)
    return results
