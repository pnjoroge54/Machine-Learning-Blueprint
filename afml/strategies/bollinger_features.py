import pandas as pd
import pandas_ta as ta

from ..cache.selective_cleaner import smart_cacheable
from ..features.fracdiff import frac_diff_ffd, fracdiff_optimal
from ..features.moving_averages import calculate_ma_differences
from ..features.returns import get_lagged_returns, rolling_autocorr_numba
from ..util.misc import optimize_dtypes
from ..util.volatility import get_period_vol, get_yang_zhang_vol


@smart_cacheable
def create_bollinger_features(df, lookback_window=10, bb_period=20, bb_std=2):
    """
    Create features for meta-labeling model
    """
    features = df[["close"]]
    features["spread"] = df["spread"] / df["close"]

    # Bollinger features
    bb_feat = df.ta.bbands(bb_period, bb_std)
    features[["bb_lower", "bb_upper"]] = bb_feat.iloc[:, :2]
    features[["bb_bandwidth", "bb_percentage"]] = bb_feat.iloc[:, -2:]

    # Fractionally_differenced prices
    d0 = fracdiff_optimal(features["close"], verbose=True)[1]
    d1 = fracdiff_optimal(features["bb_lower"], verbose=True)[1]
    d2 = fracdiff_optimal(features["bb_upper"], verbose=True)[1]
    d = max(d0, d1, d2)
    ffd_cols = ["close", "bb_lower", "bb_upper"]
    features[ffd_cols] = frac_diff_ffd(features[ffd_cols], d)

    # Price-based features
    lagged_ret = get_lagged_returns(df.close, lags=[1, 5, 10], nperiods=3)
    features = features.join(lagged_ret)

    features["vol"] = get_yang_zhang_vol(df.open, df.high, df.low, df.close, window=5)
    features[f"vol_{bb_period}"] = get_yang_zhang_vol(
        df.open, df.high, df.low, df.close, window=bb_period
    )
    for t in range(1, 6):
        features[f"vol_lag_{t}"] = features["vol"].shift(t)

    features["autocorr"] = rolling_autocorr_numba(
        features["returns"].values, lookback=lookback_window
    )
    for t in range(1, 6):
        features[f"autocorr_{t}"] = features["autocorr"].shift(t)

    for num_hours in (1, 4, 24):
        features[f"H{num_hours}_vol"] = get_period_vol(df.close, lookback=100, hours=num_hours)
    features.columns = features.columns.str.replace("H24", "D1")

    features["returns_skew"] = features["returns"].rolling(lookback_window).skew()
    features["returns_kurt"] = features["returns"].rolling(lookback_window).kurt()

    # Technical indicators
    # Volatility
    features["tr"] = df.ta.true_range()
    features["atr"] = df.ta.atr(14)

    # Moving average differences
    windows = (5, 10, 20, 50, 100, 200)
    ma_diffs = calculate_ma_differences(df.close, windows)
    ma_diffs = ma_diffs.div(features["atr"], axis=0)  # Normalize by ATR
    features = features.join(ma_diffs)

    # Momentum
    features["rsi"] = df.ta.rsi()
    features[["stoch_rsi_k", "stoch_rsi_d"]] = df.ta.stochrsi().iloc[:, :2]

    # Trend
    features[["adx", "dmp", "dmn"]] = df.ta.adx()
    features["dm_net"] = features["dmp"] - features["dmn"]
    features[["macd", "macdh"]] = df.ta.macd().iloc[:, :2]

    # Abbreviate "returns" to "ret" in columns
    features.columns = features.columns.str.replace(r"returns", "ret", regex=True)
    features = optimize_dtypes(features, verbose=False)  # Conserve memory

    return features
