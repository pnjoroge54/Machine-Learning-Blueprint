import warnings

import numpy as np
import pandas as pd
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _calculate_rolling_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


@njit(fastmath=True, cache=True)
def _calculate_rolling_sharpe_drawdown(
    prices: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Sharpe-like ratio and maximum drawdown over a trailing window.

    For each position ``i >= window`` both metrics are computed over
    ``prices[i - window : i]``, producing a genuine per-event series rather
    than a single end-of-sample value (the previous implementation wrote only
    the last row and relied on ffill/bfill, which collapsed the column to a
    constant).
    """
    n = len(prices)
    sharpe = np.full(n, np.nan, dtype=np.float64)
    max_dd = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        win = prices[i - window : i]
        rets = np.diff(np.log(win))
        if len(rets) > 1:
            sharpe[i] = np.mean(rets) / (np.std(rets) + 1e-10)

        # Maximum drawdown over the window
        running_max = win[0]
        worst = 0.0
        for k in range(len(win)):
            running_max = max(running_max, win[k])
            dd = (win[k] - running_max) / running_max
            worst = min(worst, dd)
        max_dd[i] = worst

    return sharpe, max_dd


def calculate_rolling_metrics(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    sample_weight: pd.Series,
    signals: pd.Series | None = None,
    window_sizes: list[int] = [20, 50, 100],
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

    # Align all inputs to a common index before converting to positional arrays.
    # `sample_weight` may cover fewer rows than `events` (e.g. weights are
    # computed only on events with non-NaN features), and the rolling loops below
    # index every array positionally — a length/order mismatch would otherwise
    # raise an IndexError or silently read the wrong rows.
    common_index = events.index.intersection(sample_weight.index).intersection(prices.index)
    if signals is not None:
        common_index = common_index.intersection(signals.index)
    if len(common_index) != len(events.index):
        events = events.loc[common_index]
        prices = prices.loc[common_index]
        sample_weight = sample_weight.loc[common_index]
        if signals is not None:
            signals = signals.loc[common_index]

    # Prepare arrays
    y_true = events["bin"].to_numpy(np.int8)
    weights = sample_weight.to_numpy(np.float32)
    price_array = prices["close"].to_numpy(np.float32)

    # Handle signals - if not provided, assume all predictions are positive
    if signals is None:
        y_pred = np.ones(len(y_true), dtype=np.int8)
        signal_array = np.ones(len(y_true), dtype=np.int8)  # Assume all long
    else:
        signal_array = signals.to_numpy().astype(np.int8)
        # For classification metrics, convert to binary predictions
        y_pred = (signal_array != 0).astype(np.int8)

    # All positional arrays must now share the same length.
    assert len(weights) == len(y_true) == len(price_array) == len(signal_array), (
        "Aligned arrays disagree in length: "
        f"weights={len(weights)}, y_true={len(y_true)}, "
        f"prices={len(price_array)}, signals={len(signal_array)}"
    )

    n = len(y_true)
    metrics_df = pd.DataFrame(index=events.index)

    # In the pure meta-labelling case every event carries a primary signal, so
    # ``y_pred`` is uniformly positive (no ``signal == 0`` rows). That makes a
    # family of classification metrics degenerate, and they would only be
    # dropped downstream by DropConstant/DropDuplicate:
    #   - rolling_recall ≡ 1            (fn ≡ 0)
    #   - rolling_win_rate ≡ precision  (tn ≡ 0, so accuracy == precision)
    #   - rolling_signal_frequency ≡ 1  (every row is a signal)
    #   - signal_freq_trend ≡ 0         (difference of two constants)
    # Detect the condition and skip emitting them rather than computing,
    # storing, and then discarding constant/duplicate columns every run.
    all_events_are_signals = not np.any(y_pred == 0)

    # Calculate basic rolling metrics for each window size
    for window in window_sizes:
        if window > n:
            continue

        # 1. Confusion matrix components
        tp, fp, tn, fn, total_weight = _calculate_rolling_confusion_matrix(
            y_true, y_pred, weights, window
        )

        # 2. Standard classification metrics (always compute)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = (tp + tn) / total_weight
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)  # recall for both meta and non‑meta
            f1 = 2 * (prec * rec) / (prec + rec)
            win_rate = acc  # accuracy = win rate for binary

        metrics_df[f"rolling_accuracy_{window}"] = acc
        metrics_df[f"rolling_precision_{window}"] = prec
        metrics_df[f"rolling_f1_{window}"] = f1
        if not all_events_are_signals:
            # recall ≡ 1 and win_rate(accuracy) ≡ precision when every event
            # is a signal; only emit them when they can vary.
            metrics_df[f"rolling_recall_{window}"] = rec
            metrics_df[f"rolling_win_rate_{window}"] = win_rate

        # 3. Directional metrics if signals are available
        if signals is not None:
            long_win_rate, short_win_rate, signal_freq, long_freq, short_freq = (
                _calculate_rolling_directional_metrics(y_true, signal_array, weights, window)
            )

            metrics_df[f"rolling_long_win_rate_{window}"] = long_win_rate
            metrics_df[f"rolling_short_win_rate_{window}"] = short_win_rate
            if not all_events_are_signals:
                # signal_frequency ≡ 1 when every event carries a signal.
                metrics_df[f"rolling_signal_frequency_{window}"] = signal_freq
            metrics_df[f"rolling_long_frequency_{window}"] = long_freq
            metrics_df[f"rolling_short_frequency_{window}"] = short_freq

            # Signal quality metrics
            if window >= 10:
                with np.errstate(divide="ignore", invalid="ignore"):
                    long_short_ratio = long_freq / (short_freq + 1e-10)
                    metrics_df[f"rolling_long_short_ratio_{window}"] = long_short_ratio

        # 4. Volatility metrics
        if include_volatility_metrics and window >= 20:
            volatility = _calculate_rolling_volatility(price_array, window)
            metrics_df[f"rolling_volatility_{window}"] = volatility

            # Volatility-adjusted win rate (using accuracy)
            vol_adj_win_rate = win_rate / (volatility + 1e-10)
            metrics_df[f"rolling_vol_adj_win_rate_{window}"] = vol_adj_win_rate

        # 5. Trend-following context metrics
        #
        # Note: the plain windowed-slope feature (`rolling_trend_strength_{window}`)
        # is intentionally NOT emitted here. The base feature set already carries a
        # richer, backward-looking trend slope from `trend_scanning_labels`
        # (`trend_slope`/`trend_t_value`/`trend_rsquared`, adaptive horizon), so a
        # fixed-window slope would just be a collinear duplicate. We keep only the
        # trend-aligned win rate below, which conditions outcomes on signal/trend
        # alignment and is genuinely novel signal.
        if include_trend_metrics and window >= 30:
            # Win rate for signals aligned with recent trend
            if signals is not None:
                trend_aligned_win_rate = np.full(n, np.nan, dtype=np.float64)

                for i in range(window - 1, n):
                    start = i - window + 1
                    aligned_wins = 0.0
                    aligned_total = 0.0

                    for j in range(start, i + 1):
                        if signal_array[j] != 0 and j >= 5:
                            # Determine if signal aligns with recent trend
                            recent_trend = np.sign(price_array[j] - price_array[j - 5])
                            signal_direction = signal_array[j]

                            if recent_trend * signal_direction > 0:  # Aligned
                                aligned_total += weights[j]
                                if y_true[j] == 1:
                                    aligned_wins += weights[j]

                    if aligned_total > 0:
                        trend_aligned_win_rate[i] = aligned_wins / aligned_total

                metrics_df[f"rolling_trend_aligned_win_rate_{window}"] = trend_aligned_win_rate

        # 6. Drawdown and performance persistence (rolling, per-event)
        if window >= 30 and len(price_array) >= window:
            sharpe_series, max_dd_series = _calculate_rolling_sharpe_drawdown(price_array, window)
            metrics_df[f"rolling_sharpe_{window}"] = sharpe_series
            metrics_df[f"rolling_max_drawdown_{window}"] = max_dd_series

    # 7. Cross-window metrics (pattern recognition)
    if len(window_sizes) >= 2:
        for i in range(len(window_sizes)):
            for j in range(i + 1, len(window_sizes)):
                w1, w2 = window_sizes[i], window_sizes[j]

                # Performance divergence. Prefer win-rate; fall back to
                # precision when win_rate was suppressed as degenerate
                # (win_rate ≡ precision in the all-signal case), so this
                # informative cross-window feature is preserved either way.
                base = (
                    "rolling_win_rate"
                    if f"rolling_win_rate_{w1}" in metrics_df.columns
                    else "rolling_precision"
                )
                col1 = f"{base}_{w1}"
                col2 = f"{base}_{w2}"
                if col1 in metrics_df.columns and col2 in metrics_df.columns:
                    divergence = metrics_df[col1] - metrics_df[col2]
                    metrics_df[f"performance_divergence_{w1}_{w2}"] = divergence

                # Signal frequency trend
                if signals is not None:
                    col1 = f"rolling_signal_frequency_{w1}"
                    col2 = f"rolling_signal_frequency_{w2}"
                    if col1 in metrics_df.columns and col2 in metrics_df.columns:
                        freq_trend = metrics_df[col1] - metrics_df[col2]
                        metrics_df[f"signal_freq_trend_{w1}_{w2}"] = freq_trend

    # Fill missing values and normalize
    metrics_df = metrics_df.ffill().bfill()

    # ffill/bfill cannot fill a column that is entirely NaN. This happens for the
    # directional win-rate columns in a one-sided model: e.g. the long-only side
    # of the dual pipeline never produces short signals, so `rolling_short_win_rate_*`
    # is NaN for every row. Left unfilled, such a column would later wipe out every
    # row in the downstream `dropna()`. Neutralise to 0 so the column becomes
    # constant and is dropped cleanly by DropConstantFeatures.
    metrics_df = metrics_df.fillna(0.0)

    for col in metrics_df.columns:
        col_min = metrics_df[col].min()
        col_max = metrics_df[col].max()
        if col_max > col_min:
            metrics_df[col] = (metrics_df[col] - col_min) / (col_max - col_min)

    return metrics_df


def add_meta_label_features(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    sample_weight: pd.Series,
    config: dict = {},
) -> pd.DataFrame:
    """
    Enhanced feature engineering for meta-labelling that incorporates
    rolling performance metrics and market context.

    Parameters
    ----------
    events : pd.DataFrame
        Triple barrier event labels.
    prices : pd.DataFrame
        Price data.
    sample_weight : pd.Series
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
        sample_weight=sample_weight,
        signals=(events["side"] if "side" in events else events["bin"]),
        window_sizes=config.get("window_sizes", [5, 20]),
        include_trend_metrics=config.get("include_trend_metrics", True),
        include_volatility_metrics=config.get("include_volatility_metrics", True),
    )

    # Add market regime features
    regime_features = calculate_market_regime_features(prices)

    # Combine all features
    all_features = pd.concat([rolling_metrics, regime_features], axis=1)

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
