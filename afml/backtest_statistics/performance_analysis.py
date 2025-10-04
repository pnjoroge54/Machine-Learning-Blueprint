import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kurtosis, skew

from ..bet_sizing.bet_sizing import (
    bet_size_budget,
    bet_size_probability,
    bet_size_reserve,
)
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
    "kurtosis",
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
            returns.values,
            positions.values,
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
        value_counts_index = data_index.diff().value_counts()
        if value_counts_index.max() > int(len(data_index) * 0.7):
            # Check if using a common timeframe despite holidays and weekends
            # t = 365.25 if value_counts_index.nunique() == 1 else 252
            avg_delta = value_counts_index.idxmax()
            periods_per_year = pd.Timedelta(days=trading_days_per_year) / avg_delta
            print(f"Inferred timeframe to be {avg_delta}.")
        else:
            median_delta = data_index.diff().median()
            print(f"Inferred timeframe from median delta as {median_delta}.")
            if median_delta.total_seconds() == 0:
                warnings.warn(
                    "Cannot infer frequency from index. Falling back to default daily.",
                    UserWarning,
                    stacklevel=2,
                )
                return np.sqrt(trading_days_per_year), trading_days_per_year

            # Calculate periods per year based on the median bar frequency.
            periods_per_year = pd.Timedelta(days=trading_days_per_year) / median_delta
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
        zero_metrics["avg_trade_duration"] = np.nan
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

    if positions is not None:
        hp = average_holding_period(positions)
        metrics["avg_trade_duration"] = (
            pd.Timedelta(days=round(hp, 3)) if hp > 0 else hp
        )  # Rounded so output doesn't include nanoseconds
        bet_frequency = timing_of_flattening_and_flips(positions).size
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


def get_positions_from_events(
    data_index: pd.DatetimeIndex,
    events_t1: pd.Series,
    sides: pd.Series,
):
    """
    Create series of target positions from event positions.
    Parameters
    ----------
    data_index : pd.DatetimeIndex
        DateTime index of data.
    events_t1 : pd.Series
        End times of events
    sides: pd.Series
        Trade direction

    Returns
    -------
    pd.Series
        Series containing the target positions for data_index.
    """
    end_times = pd.Index(events_t1).difference(events_t1.index)
    positions = sides.reindex(data_index)
    positions.loc[end_times] = 0  # End of trade
    positions = positions.ffill().fillna(0)
    return positions


def evaluate_meta_labeling_performance(
    events: pd.DataFrame,
    meta_probabilities: pd.Series,
    close: pd.Series,
    confidence_threshold: float = 0.5,
    trading_days_per_year: int = 252,
    trading_hours_per_day: int = 24,
    strategy_name: str = "Strategy",
    bet_sizing: str = None,
    **kwargs,
) -> dict:
    """
    Evaluates and compares the performance of a primary strategy against a
    meta-labeled version of that strategy.

    This function simulates two strategies:
    1.  The primary strategy, which takes all signals.
    2.  The meta-labeled strategy, which filters trades based on a confidence
        threshold and sizes them according to the meta-model's probability.

    Args:
        events: A DataFrame of trade events that contains at least the columns 't1' and 'side'.
            - index: Event start times
            - t1: Event end times, i.e., the time of first barrier touch
            - side: Trade direction
        meta_probabilities: A Series or array of probabilities from the meta-model.
        close: A Series of prices that cover the period encapsulated in events.
        confidence_threshold: The minimum probability required to take a trade.
        trading_days_per_year: The number of trading days in a year.
        trading_hours_per_day: The number of trading hours per day.
        strategy_name: The name of the strategy for reporting.

    Returns:
        A dictionary containing the performance metrics for both strategies,
        their return series, and other comparison metadata.
    """
    data_index = close.index

    # Calculate returns
    events = events.dropna(subset=["t1"])
    all_dates = events.index.union(other=events["t1"].array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")
    returns = prices.loc[events["t1"].array].array / prices.loc[events.index] - 1
    primary_returns = returns * events["side"]
    primary_positions = get_positions_from_events(data_index, events["t1"], events["side"])

    # Filter trades: Set position to 0 for trades below the confidence threshold.
    aligned_probs = meta_probabilities.reindex(events.index, fill_value=0.5)
    confident_trades = aligned_probs > confidence_threshold
    meta_prob = aligned_probs[confident_trades]
    meta_events = events[confident_trades]

    # --- Bet Sizing Logic ---
    if bet_sizing is None:
        bets = meta_events["side"].copy()
        bet_sizing = "none"
    elif bet_sizing == "probability":
        bets = bet_size_probability(
            meta_events, meta_prob, num_classes=2, pred=meta_events["side"], **kwargs
        )
    elif bet_sizing == "budget":
        bets = bet_size_budget(meta_events["t1"], meta_events["side"])
        bets = bets["bet_size"]
    elif bet_sizing == "reserve":
        bets = bet_size_reserve(meta_events["t1"], meta_events["side"], **kwargs)
        bets = bets["bet_size"]

    msg = f"Bet Sizing Method: {bet_sizing.title()} | Confidence Threshold: {confidence_threshold}"
    msg = msg + f"\n{kwargs}" if kwargs else msg
    logger.info(msg)

    # Apply the calculated bet size to the signals that were not filtered out.
    meta_positions = get_positions_from_events(data_index, meta_events["t1"], bets)
    meta_returns = (returns * meta_positions).dropna()

    # --- Performance Calculation ---

    primary_metrics = calculate_performance_metrics(
        primary_returns,
        data_index,
        primary_positions,
        trading_days_per_year,
        trading_hours_per_day,
    )
    meta_metrics = calculate_performance_metrics(
        meta_returns,
        data_index,
        meta_positions,
        trading_days_per_year,
        trading_hours_per_day,
    )

    # --- Meta-Specific Metrics ---
    total_signals = len(events)
    filtered_signals = len(meta_events)

    # The percentage of trades that were filtered out by the meta-model.
    meta_metrics["signal_filter_rate"] = (
        1 - (filtered_signals / total_signals) if total_signals > 0 else 0
    )
    meta_metrics["confidence_threshold"] = confidence_threshold

    # Information Ratio: Measures the meta-model's ability to generate excess return per unit of tracking error.
    excess_returns = meta_returns - primary_returns.reindex(meta_returns.index, fill_value=0)
    _, periods_per_year = get_annualization_factors(
        data_index=data_index,
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
