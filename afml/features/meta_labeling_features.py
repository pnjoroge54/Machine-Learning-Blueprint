import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ..cache import cacheable
from .trading_session import get_time_features


@njit(parallel=True, fastmath=True, cache=True)
def _calculate_rolling_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated calculation of confusion matrix components for rolling windows.

    Returns: (tp, fp, tn, fn, total_weight) arrays
    """
    n = len(y_true)
    tp = np.full(n, np.nan, dtype=np.float64)
    fp = np.full(n, np.nan, dtype=np.float64)
    tn = np.full(n, np.nan, dtype=np.float64)
    fn = np.full(n, np.nan, dtype=np.float64)
    total_weight = np.full(n, np.nan, dtype=np.float64)

    for i in prange(window - 1, n):
        start = i - window + 1
        tpi = fpi = tni = fni = totali = 0.0

        for j in range(start, i + 1):
            w = weights[j]
            true_val = y_true[j]
            pred_val = y_pred[j]

            totali += w

            if pred_val == 1:
                if true_val == 1:
                    tpi += w
                else:
                    fpi += w
            else:  # pred_val == 0 or -1
                if true_val == 1:
                    fni += w
                else:
                    tni += w

        tp[i] = tpi
        fp[i] = fpi
        tn[i] = tni
        fn[i] = fni
        total_weight[i] = totali

    return tp, fp, tn, fn, total_weight


@njit(parallel=True, fastmath=True, cache=True)
def _calculate_rolling_directional_metrics(
    y_true: np.ndarray, signals: np.ndarray, weights: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rolling metrics separately for long and short signals.

    Returns: (long_win_rate, short_win_rate, signal_frequency,
              long_signal_freq, short_signal_freq)
    """
    n = len(y_true)
    long_win_rate = np.full(n, np.nan, dtype=np.float64)
    short_win_rate = np.full(n, np.nan, dtype=np.float64)
    signal_frequency = np.full(n, np.nan, dtype=np.float64)
    long_signal_freq = np.full(n, np.nan, dtype=np.float64)
    short_signal_freq = np.full(n, np.nan, dtype=np.float64)

    for i in prange(window - 1, n):
        start = i - window + 1

        long_wins = 0.0
        long_total = 0.0
        short_wins = 0.0
        short_total = 0.0
        signal_count = 0

        for j in range(start, i + 1):
            signal = signals[j]
            if signal != 0:
                signal_count += 1
                w = weights[j]

                if signal == 1:  # Long signal
                    long_total += w
                    if y_true[j] == 1:
                        long_wins += w
                elif signal == -1:  # Short signal
                    short_total += w
                    if y_true[j] == 1:
                        short_wins += w

        # Calculate metrics
        signal_frequency[i] = signal_count / window

        if long_total > 0:
            long_win_rate[i] = long_wins / long_total
            long_signal_freq[i] = long_total / window
        else:
            long_win_rate[i] = np.nan
            long_signal_freq[i] = 0.0

        if short_total > 0:
            short_win_rate[i] = short_wins / short_total
            short_signal_freq[i] = short_total / window
        else:
            short_win_rate[i] = np.nan
            short_signal_freq[i] = 0.0

    return (
        long_win_rate,
        short_win_rate,
        signal_frequency,
        long_signal_freq,
        short_signal_freq,
    )


@njit(fastmath=True, cache=True)
def _calculate_rolling_volatility(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling volatility (standard deviation of returns)."""
    n = len(prices)
    volatility = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        returns = np.diff(np.log(prices[i - window : i]))
        if len(returns) > 1:
            volatility[i] = np.std(returns)
        else:
            volatility[i] = 0.0

    return volatility


@njit(fastmath=True, cache=True)
def _calculate_rolling_trend_strength(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling trend strength using linear regression slope."""
    n = len(prices)
    trend_strength = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        x = np.arange(window)
        y = prices[i - window : i]

        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator > 0:
            slope = numerator / denominator
            # Normalize by price level to get trend strength
            trend_strength[i] = slope / y_mean
        else:
            trend_strength[i] = 0.0

    return trend_strength


@cacheable(time_aware=True)
def calculate_rolling_metrics(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    sample_weight: pd.Series,
    signals: Optional[pd.Series] = None,
    window_sizes: List[int] = [20, 50, 100],
    include_trend_metrics: bool = True,
    include_volatility_metrics: bool = True,
) -> pd.DataFrame:
    """
    Enhanced rolling metrics calculation for meta-labelling.

    Incorporates suggestions from the meta-labelling guide while maintaining
    compatibility with -style triple barrier labeling.

    Parameters
    ----------
    events : pd.DataFrame
        Event labels from triple barrier method with 'bin' column.
    prices : pd.DataFrame
        Price data with at least 'close' column.
    sample_weight : pd.Series
        Sample weights for each event.
    signals : pd.Series, optional
        Primary strategy signals (-1, 0, 1). If None, assumes all events
        correspond to signals (y_pred = 1 for all).
    window_sizes : List[int]
        Window sizes for rolling calculations.
    include_trend_metrics : bool
        Whether to include trend-following context metrics.
    include_volatility_metrics : bool
        Whether to include volatility-adjusted metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling metrics as features.
    """
    warnings.filterwarnings("ignore", message="invalid value encountered")

    # Prepare arrays
    y_true = events["bin"].to_numpy().astype(np.int8)
    weights = sample_weight.to_numpy().astype(np.float32)
    price_array = prices["close"].to_numpy().astype(np.float64)

    # Handle signals - if not provided, assume all predictions are positive
    if signals is None:
        y_pred = np.ones(len(y_true), dtype=np.int8)
        signal_array = np.ones(len(y_true), dtype=np.int8)  # Assume all long
    else:
        signal_array = signals.to_numpy().astype(np.int8)
        # For classification metrics, convert to binary predictions
        y_pred = (signal_array != 0).astype(np.int8)

    n = len(y_true)
    metrics_df = pd.DataFrame(index=events.index)
    is_meta_label = set(y_true.unique()) == {0, 1}

    # Calculate basic rolling metrics for each window size
    for window in window_sizes:
        if window > n:
            continue

        # 1. Calculate confusion matrix components
        tp, fp, tn, fn, total_weight = _calculate_rolling_confusion_matrix(
            y_true, y_pred, weights, window
        )

        # 2. Calculate standard classification metrics
        with np.errstate(divide="ignore", invalid="ignore"):
            # Accuracy
            acc = (tp + tn) / total_weight
            metrics_df[f"rolling_accuracy_{window}"] = acc

            # Precision
            prec = tp / (tp + fp)
            metrics_df[f"rolling_precision_{window}"] = prec

            # Recall
            if not is_meta_label:
                # Recall is always 1 for meta-labels
                rec = tp / (tp + fn)
                metrics_df[f"rolling_recall_{window}"] = rec

            # F1 Score
            f1 = 2 * (prec * rec) / (prec + rec)
            metrics_df[f"rolling_f1_{window}"] = f1

            # Win Rate (same as accuracy for binary classification)
            if not is_meta_label:
                win_rate = (tp + tn) / total_weight
                metrics_df[f"rolling_win_rate_{window}"] = win_rate

        # 3. Calculate directional metrics if signals are available
        if signals is not None:
            long_win_rate, short_win_rate, signal_freq, long_freq, short_freq = (
                _calculate_rolling_directional_metrics(
                    y_true, signal_array, weights, window
                )
            )

            metrics_df[f"rolling_long_win_rate_{window}"] = long_win_rate
            metrics_df[f"rolling_short_win_rate_{window}"] = short_win_rate
            metrics_df[f"rolling_signal_frequency_{window}"] = signal_freq
            metrics_df[f"rolling_long_frequency_{window}"] = long_freq
            metrics_df[f"rolling_short_frequency_{window}"] = short_freq

            # Signal quality metrics
            if window >= 10:  # Require minimum signals for meaningful ratio
                with np.errstate(divide="ignore", invalid="ignore"):
                    long_short_ratio = long_freq / (short_freq + 1e-10)
                    metrics_df[f"rolling_long_short_ratio_{window}"] = long_short_ratio

        # 4. Calculate market context metrics
        if include_volatility_metrics and window >= 20:
            volatility = _calculate_rolling_volatility(price_array, window)
            metrics_df[f"rolling_volatility_{window}"] = volatility

            # Volatility-adjusted win rate
            vol_adj_win_rate = win_rate / (volatility + 1e-10)
            metrics_df[f"rolling_vol_adj_win_rate_{window}"] = vol_adj_win_rate

        # 5. Calculate trend-following context metrics
        if include_trend_metrics and window >= 30:
            trend_strength = _calculate_rolling_trend_strength(price_array, window)
            metrics_df[f"rolling_trend_strength_{window}"] = trend_strength

            # Performance during strong trends vs mean reversion
            if signals is not None:
                # Calculate win rate for signals that align with trend
                trend_aligned_win_rate = np.full(n, np.nan, dtype=np.float64)

                for i in range(window - 1, n):
                    start = i - window + 1
                    aligned_wins = 0.0
                    aligned_total = 0.0

                    for j in range(start, i + 1):
                        if signal_array[j] != 0 and window > j >= 5:
                            # Determine if signal aligns with recent trend
                            recent_trend = np.sign(price_array[j] - price_array[j - 5])
                            signal_direction = signal_array[j]

                            if recent_trend * signal_direction > 0:  # Aligned
                                aligned_total += weights[j]
                                if y_true[j] == 1:
                                    aligned_wins += weights[j]

                    if aligned_total > 0:
                        trend_aligned_win_rate[i] = aligned_wins / aligned_total

                metrics_df[f"rolling_trend_aligned_win_rate_{window}"] = (
                    trend_aligned_win_rate
                )

        # 6. Calculate drawdown and performance persistence metrics
        if window >= 30:
            # Calculate rolling Sharpe-like ratio (simplified)
            returns = np.diff(np.log(price_array[-window:]))
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10)
                metrics_df.loc[events.index[-1], f"rolling_sharpe_{window}"] = (
                    sharpe_ratio
                )

            # Maximum drawdown in the window
            if len(price_array) >= window:
                rolling_max = np.maximum.accumulate(price_array[-window:])
                drawdown = (price_array[-window:] - rolling_max) / rolling_max
                max_drawdown = np.min(drawdown)
                metrics_df.loc[events.index[-1], f"rolling_max_drawdown_{window}"] = (
                    max_drawdown
                )

    # 7. Calculate cross-window metrics (pattern recognition features)
    if len(window_sizes) >= 2:
        for i in range(len(window_sizes)):
            for j in range(i + 1, len(window_sizes)):
                w1, w2 = window_sizes[i], window_sizes[j]

                # Performance divergence between time horizons
                if (
                    f"rolling_win_rate_{w1}" in metrics_df.columns
                    and f"rolling_win_rate_{w2}" in metrics_df.columns
                ):
                    divergence = (
                        metrics_df[f"rolling_win_rate_{w1}"]
                        - metrics_df[f"rolling_win_rate_{w2}"]
                    )
                    metrics_df[f"performance_divergence_{w1}_{w2}"] = divergence

                # Signal frequency trend
                if signals is not None:
                    if (
                        f"rolling_signal_frequency_{w1}" in metrics_df.columns
                        and f"rolling_signal_frequency_{w2}" in metrics_df.columns
                    ):
                        freq_trend = (
                            metrics_df[f"rolling_signal_frequency_{w1}"]
                            - metrics_df[f"rolling_signal_frequency_{w2}"]
                        )
                        metrics_df[f"signal_freq_trend_{w1}_{w2}"] = freq_trend

    # Fill NaN values with forward fill then backward fill
    metrics_df = metrics_df.ffill().bfill()

    # Normalize metrics to [0, 1] range for better model performance
    for col in metrics_df.columns:
        col_min = metrics_df[col].min()
        col_max = metrics_df[col].max()
        if col_max > col_min:
            metrics_df[col] = (metrics_df[col] - col_min) / (col_max - col_min)

    return metrics_df


def add_meta_label_features(
    features: pd.DataFrame,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    sample_weights: pd.Series,
    config: Dict = {},
) -> pd.DataFrame:
    """
    Enhanced feature engineering for meta-labelling that incorporates
    rolling performance metrics and market context.

    Parameters
    ----------
    features : pd.DataFrame
        Existing engineered features.
    events : pd.DataFrame
        Triple barrier event labels.
    prices : pd.DataFrame
        Price data.
    sample_weights : pd.Series
        Sample weights.
    primary_strategy : BaseStrategy
        Primary strategy instance to extract signals.
    data : pd.DataFrame
        Raw bar data for additional context.
    config : Dict
        Configuration dictionary with keys:
        - window_sizes: List of window sizes for rolling metrics
        - include_trend_metrics: Whether to include trend metrics
        - include_volatility_metrics: Whether to include volatility metrics
        - min_signals_for_metrics: Minimum signals required for metrics

    Returns
    -------
    pd.DataFrame
        Features augmented with meta-label features.
    """

    # Calculate enhanced rolling metrics
    rolling_metrics = calculate_rolling_metrics(
        events=events,
        prices=prices,
        sample_weight=sample_weights,
        signals=(events["side"] if "side" in events else events["bin"]),
        window_sizes=config.get("window_sizes", [20, 50, 100]),
        include_trend_metrics=config.get("include_trend_metrics", True),
        include_volatility_metrics=config.get("include_volatility_metrics", True),
    )

    # Add market regime features
    regime_features = calculate_market_regime_features(prices)

    # Combine all features
    all_features = pd.concat(
        [features, rolling_metrics, regime_features], axis=1
    ).dropna()

    return all_features


def calculate_market_regime_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market regime features for meta-labelling context.

    Returns features that help identify market conditions where
    the primary strategy is likely to perform well or poorly.
    """
    close = prices["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=prices.index)

    # Volatility regime
    vol_20 = returns.rolling(20).std()
    vol_50 = returns.rolling(50).std()
    features["volatility_regime"] = pd.qcut(vol_20, 4, labels=False, duplicates="drop")
    features["volatility_trend"] = (vol_50 / vol_20 - 1).fillna(0)

    # Trend regime
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    features["trend_strength"] = (sma_20 / sma_50 - 1).fillna(0)
    features["trend_direction"] = np.sign(features["trend_strength"])

    # Mean reversion/extreme price position
    zscore_20 = (close - sma_20) / close.rolling(20).std()
    features["price_position"] = zscore_20.fillna(0)
    features["extreme_price"] = (abs(zscore_20) > 2).astype(int)

    # Volume features (if available)
    if "volume" in prices.columns:
        volume = prices["volume"]
        volume_ratio = volume / volume.rolling(20).mean()
        features["volume_surge"] = (volume_ratio > 1.5).astype(int)
    elif "tick_volume" in prices.columns:
        volume = prices["tick_volume"]
        volume_ratio = volume / volume.rolling(20).mean()
        features["tick_volume_surge"] = (volume_ratio > 1.5).astype(int)

    return features
