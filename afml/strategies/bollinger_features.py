import datetime
from typing import Union

import matplotlib.lines as mlines
import mplfinance as mpf
import pandas as pd
import pandas_ta as ta

from ..cache.selective_cleaner import smart_cacheable
from ..features.fracdiff import frac_diff_ffd, fracdiff_optimal
from ..features.moving_averages import calculate_ma_differences
from ..features.returns import get_lagged_returns, rolling_autocorr_numba
from ..util.misc import optimize_dtypes
from ..util.volatility import get_period_vol, get_yang_zhang_vol
from .signal_processing import get_entries
from .strategies import BollingerStrategy


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


def plot_bbands(
    data: pd.DataFrame,
    start: Union[str, datetime.datetime],
    end: Union[str, datetime.datetime],
    window: int = 20,
    std0: float = 1.5,
    std1: float = None,
    width: float = 7.5,
    height: float = 5,
    linewidth: float = 1,
    markersize: int = 40,
):
    """
    Plots a financial chart with Bollinger Bands and custom trading signals.

    Args:
        data (pd.DataFrame): The DataFrame containing OHLCV data.
        start (Union[str, datetime.datetime]): The start date for the plot.
        end (Union[str, datetime.datetime]): The end date for the plot.
        window (int): The lookback period for the Bollinger Bands.
        std0 (float): The number of standard deviations for the first set of bands.
        std1 (float, optional): The number of standard deviations for the second set of bands.
        width (float): The width of the plot figure in inches.
        height (float): The height of the plot figure in inches.
        linewidth (float): The line width for the bands.
    """
    df = data.loc[start:end, ["open", "high", "low", "close"]].copy()
    std0 = float(std0)

    # Compute first set of bands
    df.ta.bbands(window, std0, append=True)
    upper_col = f"BBU_{window}_{std0}"
    lower_col = f"BBL_{window}_{std0}"
    mid_col = f"BBM_{window}_{std0}"

    # We remove the 'label' keyword as it is not supported in this version.
    upper = mpf.make_addplot(df[upper_col], color="lightgreen", width=linewidth)
    lower = mpf.make_addplot(df[lower_col], color="lightgreen", width=linewidth)
    mid = mpf.make_addplot(df[mid_col], color="orange", width=linewidth)
    bands = [upper, lower, mid]

    # Optional second set of bands
    linewidth1 = linewidth * 1.5
    if std1:
        std1 = float(std1)
        df.ta.bbands(window, std1, append=True)
        # We remove the 'label' keyword from these plots as well.
        upper1 = mpf.make_addplot(df[f"BBU_{window}_{std1}"], color="blue", width=linewidth1)
        lower1 = mpf.make_addplot(df[f"BBL_{window}_{std1}"], color="blue", width=linewidth1)
        bands += [upper1, lower1]

    # --- ENTRY/EXIT SIGNALS ---
    signals, t_events = get_entries(strategy=BollingerStrategy(window, std0), data=df)
    signals = signals.loc[t_events]

    # Long entry: close crosses above lower band
    long_entry = signals[signals == 1]
    # long_entry = (df["close"].shift(1) < df[lower_col].shift(1)) & (df["close"] > df[lower_col])

    # Long exit: close crosses below upper band
    long_exit = (df["close"].shift(1) > df[upper_col].shift(1)) & (df["close"] < df[upper_col])

    # Short entry: close crosses below upper band
    # short_entry = (df["close"].shift(1) > df[upper_col].shift(1)) & (df["close"] < df[upper_col])
    short_entry = signals[signals == -1]
    # Short exit: close crosses above lower band
    short_exit = (df["close"].shift(1) < df[lower_col].shift(1)) & (df["close"] > df[lower_col])

    # Marker plots with labels for legend
    # We remove the 'label' keyword and will create the legend manually.
    entry_plot = mpf.make_addplot(
        df["close"].where(long_entry),
        type="scatter",
        markersize=markersize,
        marker="^",
        color="lime",
    )
    exit_plot = mpf.make_addplot(
        df["close"].where(long_exit),
        type="scatter",
        markersize=markersize,
        marker="v",
        color="red",
    )
    short_entry_plot = mpf.make_addplot(
        df["close"].where(short_entry),
        type="scatter",
        markersize=markersize,
        marker="v",
        color="orange",
    )
    short_exit_plot = mpf.make_addplot(
        df["close"].where(short_exit),
        type="scatter",
        markersize=markersize,
        marker="^",
        color="cyan",
    )

    bands += [entry_plot, exit_plot, short_entry_plot, short_exit_plot]

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
    if std1:
        labels.extend(
            [
                f"Upper Band ({std0}σ)",
                f"Lower Band ({std0}σ)",
                "Middle Band",
                f"Upper Band ({std1}σ)",
                f"Lower Band ({std1}σ)",
            ]
        )
        handles.extend(bands_handles)
    else:
        labels.extend([f"Upper Band ({std0}σ)", f"Lower Band ({std0}σ)", "Middle Band"])
        handles.extend(bands_handles)

    # Create dummy line handles for the scatter markers to ensure correct order
    long_entry_handle = mlines.Line2D(
        [], [], color="lime", marker="^", linestyle="None", markersize=10, label="Long Entry"
    )
    long_exit_handle = mlines.Line2D(
        [], [], color="red", marker="v", linestyle="None", markersize=10, label="Long Exit"
    )
    short_entry_handle = mlines.Line2D(
        [], [], color="orange", marker="v", linestyle="None", markersize=10, label="Short Entry"
    )
    short_exit_handle = mlines.Line2D(
        [], [], color="cyan", marker="^", linestyle="None", markersize=10, label="Short Exit"
    )

    # Add the dummy handles to the lists
    handles.extend([long_entry_handle, long_exit_handle, short_entry_handle, short_exit_handle])
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
            df[bbb_col], color="magenta", width=1.2, panel=1, secondary_y=True, ylabel="Bandwidth"
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
