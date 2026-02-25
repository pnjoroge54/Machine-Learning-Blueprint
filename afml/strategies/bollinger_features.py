import datetime
from typing import Union

import matplotlib.lines as mlines
import mplfinance as mpf
import numpy as np
import pandas as pd

from ..features.moving_averages import calculate_ma_differences, get_ma_crossovers
from ..features.returns import get_lagged_returns, rolling_autocorr_numba
from ..util.misc import optimize_dtypes
from ..util.volatility import get_garman_klass_vol, get_parkinson_vol, get_period_vol
from .signal_processing import get_entries
from .trading_strategies import BollingerStrategy


def create_bollinger_features(df: pd.DataFrame, window: int = 20, std: float = 2):
    """
    Create features for meta-labeling model
    """
    features = pd.DataFrame(index=df.index)
    features["spread"] = df["spread"] / df["close"]

    # --- 1. Returns Features ---
    # Garman Volatility
    features[f"parkinson_vol_{window}"] = get_parkinson_vol(
        df["high"], df["low"], window=window
    )

    # Hourly EWM(num_hours) Volatility
    for num_hours in (1, 4, 24):
        features[f"H{num_hours}_vol"] = get_period_vol(
            df["close"], lookback=window, hours=num_hours
        )
    features.columns = features.columns.str.replace("H24", "D1")

    # Lagged returns normalized by volatility
    lagged_ret = get_lagged_returns(df["close"], lags=[1, 5, 10], nperiods=3)
    features = features.join(
        lagged_ret.div(features[f"parkinson_vol_{window}"], axis=0)
    )  # Normalize returns

    # Distribution
    features["returns_skew"] = features["returns"].rolling(window).skew()
    features["returns_kurt"] = features["returns"].rolling(window).kurt()

    # Autocorrelations of normalized returns
    features["autocorr"] = rolling_autocorr_numba(
        features["returns"].to_numpy(), lookback=window
    )
    for t in range(1, 6):
        features[f"autocorr_{t}"] = features["autocorr"].shift(t)

    # --- 2. Technical Analysis Features ---
    # Bollinger Bands
    bbands = df.ta.bbands(window, std, std).iloc[:, -2:]  # Use BBP and BBB only

    # Volatility
    tr = df.ta.true_range()
    atr = df.ta.atr()

    # Momentum
    rsi = df.ta.rsi()
    stochrsi = df.ta.stochrsi()

    # Trend
    adx = df.ta.adx()
    adx["dm_net"] = adx.iloc[:, 1] - adx.iloc[:, 2]
    macd = df.ta.macd().iloc[:, :2]

    ta_features = [bbands, tr, atr, rsi, stochrsi, adx, macd]
    features = features.join(ta_features)

    # --- 3. Advanced Bollinger Features ---
    # Normalize Bollinger Bands
    bb_bandwidth = bbands.filter(regex="BBB")

    # Add momentum features
    features["bb_bandwidth_diff"] = bb_bandwidth.diff()  # Momentum
    features["bb_bandwidth_ma"] = bb_bandwidth.rolling(5).mean()  # Smoothing
    features["bb_bw_mom"] = bb_bandwidth.pct_change(3)
    features["bb_bw_regime"] = (bb_bandwidth > bb_bandwidth.quantile(0.75)).astype(int)

    # Calculate oscillators and volatility measures
    bb_bandwidth_diff = (
        features["bb_bandwidth_diff"].apply(np.sign).fillna(0)
    )  # Bandwidth change
    features["is_widening_bb"] = bb_bandwidth_diff.replace({-1: 0}).astype("int8")
    features["is_shrinking_bb"] = bb_bandwidth_diff.replace({1: 0}).astype("int8")

    # --- 4. Add side prediction ---
    # Previous because we shift after
    features["prev_signal"] = BollingerStrategy(window, std).generate_signals(df)

    # --- 5. Lag features ---
    features = features.shift().dropna()

    # --- 6. Formatting ---
    # Abbreviate "returns" to "ret" in columns
    features.columns = features.columns.str.lower().str.replace(
        "returns", "ret", regex=True
    )

    # --- 7. Conserve memory ---
    features = optimize_dtypes(features, verbose=False)

    return features


def plot_bbands(
    data: pd.DataFrame,
    start: Union[str, datetime.datetime],
    end: Union[str, datetime.datetime],
    window: int = 20,
    std: float = 1.5,
    width: float = 7.5,
    height: float = 5,
    linewidth: float = 1,
    markersize: int = 40,
):
    """
    Plots a financial chart with Bollinger Bands and custom trading labels.

    Args:
        data (pd.DataFrame): The DataFrame containing OHLCV data.
        start (Union[str, datetime.datetime]): The start date for the plot.
        end (Union[str, datetime.datetime]): The end date for the plot.
        window (int): The lookback period for the Bollinger Bands.
        std (float): The number of standard deviations for the first set of bands.
        width (float): The width of the plot figure in inches.
        height (float): The height of the plot figure in inches.
        linewidth (float): The line width for the bands.
    """
    df = data.loc[start:end, ["open", "high", "low", "close"]].copy()
    std = float(std)

    # Compute first set of bands
    df.ta.bbands(window, std, append=True)
    upper_col = f"BBU_{window}_{std}"
    lower_col = f"BBL_{window}_{std}"
    mid_col = f"BBM_{window}_{std}"

    # We remove the 'label' keyword as it is not supported in this version.
    upper = mpf.make_addplot(df[upper_col], color="lightgreen", width=linewidth)
    lower = mpf.make_addplot(df[lower_col], color="lightgreen", width=linewidth)
    mid = mpf.make_addplot(df[mid_col], color="orange", width=linewidth)
    bands = [upper, lower, mid]

    # --- ENTRY/EXIT SIGNALS ---
    side, t_events = get_entries(
        strategy=BollingerStrategy(window, std), data=df, on_crossover=True
    )
    entries = side.loc[t_events]

    # Long entry: close crosses below lower band
    long_entry = entries == 1

    # Short entry: close crosses above upper band
    short_entry = entries == -1

    long_entry_plot = mpf.make_addplot(
        df["close"].where(long_entry),
        type="scatter",
        markersize=markersize,
        marker="^",
        color="lime",
    )
    # exit_plot = mpf.make_addplot(
    #     df["close"].where(long_exit),
    #     type="scatter",
    #     markersize=markersize,
    #     marker="v",
    #     color="red",
    # )
    short_entry_plot = mpf.make_addplot(
        df["close"].where(short_entry),
        type="scatter",
        markersize=markersize,
        marker="v",
        color="orange",
    )
    # short_exit_plot = mpf.make_addplot(
    #     df["close"].where(short_exit),
    #     type="scatter",
    #     markersize=markersize,
    #     marker="^",
    #     color="cyan",
    # )

    bands += [long_entry_plot, short_entry_plot]

    # --- STYLE ---
    my_dark_style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212"},
        marketcolors=mpf.make_marketcolors(
            up="lime",
            down="red",
            wick={"up": "lime", "down": "red"},
            edge={"up": "lime", "down": "red"},
            volume="gray",
        ),
    )

    # --- PLOT ---
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=my_dark_style,
        addplot=bands,
        title="Price with Bollinger Bands & Signals",
        ylabel="Price",
        figsize=(width, height),
        returnfig=True,
    )

    # We first collect the handles for the lines and markers separately.
    handles = []
    labels = []
    bands_handles = axes[0].lines

    # Get handles and labels for the line plots (Bollinger Bands)
    labels.extend([f"Upper Band ({std}σ)", f"Lower Band ({std}σ)", "Middle Band"])
    handles.extend(bands_handles)

    # Create dummy line handles for the scatter markers to ensure correct order
    long_entry_handle = mlines.Line2D(
        [],
        [],
        color="lime",
        marker="^",
        linestyle="None",
        markersize=10,
        label="Long Entry",
    )
    long_exit_handle = mlines.Line2D(
        [],
        [],
        color="red",
        marker="v",
        linestyle="None",
        markersize=10,
        label="Long Exit",
    )
    short_entry_handle = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="v",
        linestyle="None",
        markersize=10,
        label="Short Entry",
    )
    short_exit_handle = mlines.Line2D(
        [],
        [],
        color="cyan",
        marker="^",
        linestyle="None",
        markersize=10,
        label="Short Exit",
    )

    # Add the dummy handles to the lists
    handles.extend(
        [long_entry_handle, long_exit_handle, short_entry_handle, short_exit_handle]
    )
    labels.extend(["Long Entry", "Long Exit", "Short Entry", "Short Exit"])

    # Create the legend with the custom handles and labels
    axes[0].legend(handles, labels, loc="best", fontsize="small")


def plot_bbands_dual_bbp_bw(
    data: pd.DataFrame,
    start: Union[str, datetime.datetime],
    end: Union[str, datetime.datetime],
    window: int = 20,
    std: float = 2.0,
    width: float = 9,
    height: float = 6,
    linewidth: float = 1,
    markersize: int = 40,
):
    df = data.loc[start:end, ["open", "high", "low", "close"]].copy()
    std = float(std)

    # Compute Bollinger Bands
    df.ta.bbands(window, std, append=True)
    upper_col = f"BBU_{window}_{std}"
    lower_col = f"BBL_{window}_{std}"
    mid_col = f"BBM_{window}_{std}"
    bbp_col = f"BBP_{window}_{std}"  # %B
    bbb_col = f"BBB_{window}_{std}"  # Bandwidth

    # --- Signal logic ---
    long_entry = (
        (df[bbp_col].shift(1) < 0.2)
        & (df[bbp_col] >= 0.2)
        & (df[bbb_col] > df[bbb_col].rolling(5).mean())
    )
    long_entry.name = "Long Entry"

    long_exit = (df[bbp_col].shift(1) > 0.8) & (df[bbp_col] <= 0.8)
    long_exit.name = "Long Exit"

    short_entry = (
        (df[bbp_col].shift(1) > 0.8)
        & (df[bbp_col] <= 0.8)
        & (df[bbb_col] > df[bbb_col].rolling(5).mean())
    )
    short_entry.name = "Short Entry"

    short_exit = (df[bbp_col].shift(1) < 0.2) & (df[bbp_col] >= 0.2)
    short_exit.name = "Short Exit"

    # --- Top panel: price + bands + markers ---
    m = 40  # markersize

    price_plots = [
        mpf.make_addplot(df[upper_col], color="green", width=linewidth, panel=0),
        mpf.make_addplot(df[lower_col], color="green", width=linewidth, panel=0),
        mpf.make_addplot(df[mid_col], color="orange", width=linewidth, panel=0),
        mpf.make_addplot(
            df["close"].where(long_entry),
            type="scatter",
            markersize=markersize,
            marker="^",
            color="lime",
            panel=0,
        ),
        mpf.make_addplot(
            df["close"].where(long_exit),
            type="scatter",
            markersize=markersize,
            marker="v",
            color="red",
            panel=0,
        ),
        mpf.make_addplot(
            df["close"].where(short_entry),
            type="scatter",
            markersize=markersize,
            marker="v",
            color="orange",
            label="Short Entry",
            panel=0,
        ),
        mpf.make_addplot(
            df["close"].where(short_exit),
            type="scatter",
            markersize=markersize,
            marker="^",
            color="cyan",
            panel=0,
        ),
    ]

    # --- Bottom panel: %B and Bandwidth ---
    indicator_plots = [
        mpf.make_addplot(df[bbp_col], color="yellow", width=1.2, panel=1, ylabel="%B"),
        mpf.make_addplot(
            df[bbb_col],
            color="magenta",
            width=1.2,
            panel=1,
            secondary_y=True,
            ylabel="Bandwidth",
        ),
    ]

    # --- Style with log y-axis ---
    my_dark_style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={
            "axes.facecolor": "#121212",
            "figure.facecolor": "#121212",
            "yscale": "log",  # log scale for price panel
        },
        marketcolors=mpf.make_marketcolors(
            up="lime",
            down="red",
            wick={"up": "lime", "down": "red"},
            edge={"up": "lime", "down": "red"},
            volume="gray",
        ),
    )

    # --- Plot ---
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=my_dark_style,
        addplot=price_plots + indicator_plots,
        title="Price (log) with BB %B/Bandwidth Signals",
        ylabel="Price",
        figsize=(width, height),
        panel_ratios=(3, 1),
        returnfig=True,
    )

    # Extract only the scatter handles for the legend
    handles = []
    labels = []
    for line in axes[0].lines:
        if line.get_linestyle() == "-":  # scatter markers
            handles.append(line)
            labels.append(line.get_label())

    axes[0].legend(handles, labels, loc="best")


# def generate_bollinger_mean_reverting_features(df, window=20, bb_dev=2, rsi_period=14):
#     """
#     Generate mean-reversion features using TA-Lib
#     Requires: 'open', 'high', 'low', 'close' columns
#     """
#     if not {"open", "high", "low", "close"}.issubset(df.columns):
#         raise ValueError("DataFrame must contain 'open', 'high', 'low', and 'close' columns.")

#     # Calculate Bollinger Bands
#     if not {"bb_upper", "bb_mid", "bb_lower"}.issubset(df.columns):
#         df = bollinger_mean_reverting_entries(df.copy(), window=window, std=bb_dev)

#     # Normalize Bollinger Bands
#     df["bb_upper_dev"] = (df["bb_upper"] / df["close"] - 1) * 100
#     df["bb_lower_dev"] = (1 - bb_lower / df["close"]) * 100

#     # Add proper relative features
#     std_dev = (df["bb_upper"] - df["bb_mid"]) / bb_dev
#     df["bb_zscore"] = (df["close"] - df["bb_mid"]) / std_dev
#     # df['bb_pctb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

#     # Add momentum features
#     df["bb_bandwidth_diff"] = df["bb_bandwidth"].diff()  # Momentum
#     df["bb_bandwidth_ma"] = df["bb_bandwidth"].rolling(5).mean()  # Smoothing
#     df["bb_bw_mom"] = df["bb_bandwidth"].pct_change(3)
#     df["bb_bw_regime"] = (df["bb_bandwidth"] > df["bb_bandwidth"].quantile(0.75)).astype("int8")

#     # Calculate candlestick patterns
#     df["CDL_HAMMER"] = talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])
#     df["CDL_INVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(
#         df["open"], df["high"], df["low"], df["close"]
#     )
#     df["CDL_ENGULFING"] = talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"])
#     df["CDL_HARAMI"] = talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"])
#     df["CDL_DOJI"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
#     df["CDL_DRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(df["open"], df["high"], df["low"], df["close"])
#     df["CDL_GRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(
#         df["open"], df["high"], df["low"], df["close"]
#     )

#     # Tweezers patterns
#     prev_close = df["close"].shift(1)
#     prev_open = df["open"].shift(1)
#     tolerance_factor = 0.001  # 0.1% of price
#     df["CDL_TWEEZER_BOT"] = (
#         (np.abs(df["low"].shift(1) - df["low"]) <= df["close"] * tolerance_factor)
#         & (prev_close < prev_open)
#         & (df["close"] > df["open"])
#     )
#     df["CDL_TWEEZER_TOP"] = (
#         (np.abs(df["high"].shift(1) - df["high"]) <= df["close"] * tolerance_factor)
#         & (prev_close > prev_open)
#         & (df["close"] < df["open"])
#     )

#     # Band proximity (with tolerance)
#     tolerance = 0.005  # 0.5% tolerance
#     df["at_upper_band"] = (df["high"] >= df["bb_upper"] * (1 - tolerance)).astype(int)
#     df["at_lower_band"] = (df["low"] <= df["bb_lower"] * (1 + tolerance)).astype(int)

#     # Calculate oscillators and volatility measures
#     df["bb_bandwidth_diff"] = df["bb_bandwidth"].diff().apply(np.sign).fillna(0)  # Bandwidth change
#     df["is_widening_bb"] = (
#         df["bb_bandwidth_diff"].replace({-1: 0}).astype(int)
#     )  # Convert -1 to 0 for consistency
#     df["is_shrinking_bb"] = (
#         df["bb_bandwidth_diff"].replace({1: 0}).astype(int)
#     )  # Convert 1 to 0 for consistency
#     df["tr"] = talib.TRANGE(df["high"], df["low"], df["close"])  # True Range for volatility
#     df["atr"] = talib.ATR(
#         df["high"], df["low"], df["close"], timeperiod=rsi_period
#     )  # ATR for volatility

#     # bb_half_life = int(window / 2)
#     df["yz_vol"] = get_yang_zhang_vol(
#         df["open"], df["high"], df["low"], df["close"], window=window
#     )  # Yang-Zhang volatility
#     df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(
#         df["close"], fastperiod=12, slowperiod=26, signalperiod=9
#     )  # MACD
#     df["rsi"] = talib.RSI(df["close"], timeperiod=rsi_period)  # RSI for overbought/oversold

#     # Stochastic Oscillator
#     df["stoch_k"], df["stoch_d"] = talib.STOCH(
#         df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3
#     )

#     adx = df.ta.adx(length=rsi_period)  # Using pandas_ta for ADX
#     adx.columns = [
#         x.split("_")[0].lower() for x in adx.columns
#     ]  # Rename columns to match convention
#     df = pd.concat([df, adx], axis=1)  # Concatenate ADX columns
#     df["is_adx_positive"] = (df["adx"] > 20).astype(int)  # ADX threshold for trend strength

#     # Create pattern-band features
#     pattern_map = {
#         "bullish_hammer": "CDL_HAMMER",
#         "bullish_inverted_hammer": "CDL_INVERTEDHAMMER",
#         "bullish_engulfing": "CDL_ENGULFING",
#         "bullish_harami": "CDL_HARAMI",
#         "dragonfly_doji": "CDL_DRAGONFLYDOJI",
#         "tweezers_bottom": "CDL_TWEEZER_BOT",
#         "bearish_inverted_hammer": "CDL_INVERTEDHAMMER",
#         "bearish_hammer": "CDL_HAMMER",
#         "bearish_engulfing": "CDL_ENGULFING",
#         "bearish_harami": "CDL_HARAMI",
#         "gravestone_doji": "CDL_GRAVESTONEDOJI",
#         "tweezers_top": "CDL_TWEEZER_TOP",
#     }

#     for feature, pattern in pattern_map.items():
#         # TA-Lib returns 100 for bullish, -100 for bearish patterns
#         multiplier = 1 if "bullish" in feature else -1
#         band_cond = df["at_lower_band"] if "bullish" in feature else df["at_upper_band"]

#         df[feature] = ((df[pattern] == (100 * multiplier)) & (band_cond == 1)).astype(int)

#     # Add confirmation labels
#     df["next_close"] = df["close"].shift(-1)
#     df["confirmation_bullish"] = (
#         (df["next_close"] > df["close"]) & (df["at_lower_band"] == 1)
#     ).astype(int)
#     df["confirmation_bearish"] = (
#         (df["next_close"] < df["close"]) & (df["at_upper_band"] == 1)
#     ).astype(int)

#     # Clean up intermediate columns
#     intermediate_cols = df.columns[
#         df.columns.str.startswith("CDL_") | df.columns.isin(["next_close"])
#     ]
#     df = df.drop(columns=intermediate_cols, errors="ignore")  # Drop intermediate columns safely

#     int_cols = df.select_dtypes(include=[int, bool]).columns
#     df[int_cols] = df[int_cols].astype("int8")  # Memory optimization

#     return df
