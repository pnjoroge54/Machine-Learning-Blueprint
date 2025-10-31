import datetime
from typing import Union

import matplotlib.lines as mlines
import mplfinance as mpf
import pandas as pd

from ..cache.selective_cleaner import smart_cacheable
from ..features.moving_averages import calculate_ma_differences, get_ma_crossovers
from ..features.returns import get_lagged_returns, rolling_autocorr_numba
from ..util.misc import optimize_dtypes
from ..util.volatility import get_garman_klass_vol, get_period_vol
from .signal_processing import get_entries
from .strategies import BollingerStrategy


@smart_cacheable
def create_bollinger_features(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2):
    """
    Create features for meta-labeling model
    """
    features = pd.DataFrame(index=df.index)
    features["spread"] = df["spread"] / df["close"]

    # --- 1. Returns Features ---
    # Garman Volatility
    features["vol"] = get_garman_klass_vol(df.open, df.high, df.low, df.close, window=bb_period)

    # Hourly EWM(num_hours) Volatility
    for num_hours in (1, 4, 24):
        features[f"H{num_hours}_vol"] = get_period_vol(
            df.close, lookback=bb_period, hours=num_hours
        )
    features.columns = features.columns.str.replace("H24", "D1")

    # Lagged returns normalized by volatility
    lagged_ret = get_lagged_returns(df.close, lags=[1, 5, 10], nperiods=3)
    features = features.join(lagged_ret.div(features["vol"], axis=0))  # Normalize returns

    # Distribution
    features["returns_skew"] = features["returns"].rolling(bb_period).skew()
    features["returns_kurt"] = features["returns"].rolling(bb_period).kurt()

    # Autocorrelations of normalized returns
    features["autocorr"] = rolling_autocorr_numba(
        features["returns"].to_numpy(), lookback=bb_period
    )
    for t in range(1, 6):
        features[f"autocorr_{t}"] = features["autocorr"].shift(t)

    # --- 2. Technical Analysis Features ---
    # Bollinger Bands
    bbands = df.ta.bbands(bb_period, bb_std).iloc[:, -2:]  # Use BBP and BBB only

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

    # --- 3. Moving Average Features ---
    windows = (10, 20, 50, 100, 200)
    ma_diffs = calculate_ma_differences(df.close, windows, drop=True)
    ma_diffs = ma_diffs.div(atr, axis=0)  # Normalize by ATR
    ma_crossovers = get_ma_crossovers(df.close, windows, drop=True)
    features = features.join([ma_diffs, ma_crossovers])

    # --- 4. Add side prediction ---
    features["prev_signal"] = BollingerStrategy(bb_period, bb_std).generate_signals(df)
    features = features.shift().dropna()

    # --- 5. Formatting ---
    # Abbreviate "returns" to "ret" in columns
    features.columns = features.columns.str.lower().str.replace("returns", "ret", regex=True)

    # Conserve memory
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
