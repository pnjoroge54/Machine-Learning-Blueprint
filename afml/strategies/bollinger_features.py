import pandas as pd
import pandas_ta as ta
import talib

from ..features.moving_averages import get_MA_diffs
from ..features.returns import get_lagged_returns, rolling_autocorr_numba
from ..util.volatility import get_period_vol, get_yang_zhang_vol


def create_bollinger_features(data, lookback_window=10, bb_period=20, bb_std=2):
    """
    Create features for meta-labeling model
    """
    df = data.copy()
    features = pd.DataFrame(index=df.index)

    features["rel_spread"] = df["spread"] / df["close"]

    # Bollinger features
    bb_feat = df.ta.bbands(bb_period, bb_std)
    features["bb_bandwidth"] = bb_feat.filter(regex="BBB")
    features["bb_percentage"] = bb_feat.filter(regex="BBP")

    # Price-based features
    # NOTE: Returns are lagged so no need to apply shift
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
    features["atr"] = df.ta.atr()

    # Moving average differences
    ma_diffs = get_MA_diffs(df.close, windows=(5, 20, 50, 100))
    ma_diffs = ma_diffs.div(features["atr"], axis=0)  # Normalize by ATR
    features = features.join(ma_diffs)

    # Momentum
    mom_feat = pd.concat((df.ta.mom(10), df.ta.mom(50), df.ta.mom(100)), axis=1)
    mom_feat.columns = mom_feat.columns.str.lower()
    features = features.join(mom_feat)  # Momentum indicators
    features["rsi"] = df.ta.rsi()
    stochrsi = df.ta.stochrsi()
    features["stoch_rsi_k"] = stochrsi.iloc[:, 0]  # Stochastic RSI %K
    features["stoch_rsi_d"] = stochrsi.iloc[:, 1]

    # Trend
    adx = df.ta.adx()  # ADX
    adx.columns = [
        x.split("_")[0].lower() for x in adx.columns
    ]  # Rename columns to match convention
    adx["dm_net"] = adx["dmp"] - adx["dmn"]
    features = features.join(adx)  # Concatenate ADX columns [['adx', 'dm_net']]
    features["macd"], _, features["macd_hist"] = talib.MACD(
        df.close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    return features
