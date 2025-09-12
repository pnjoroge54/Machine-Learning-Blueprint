import datetime
from typing import Union

import matplotlib.lines as mlines
import mplfinance as mpf
import pandas as pd
import pandas_ta as ta


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
    # Long entry: close crosses above lower band
    long_entry = (df["close"].shift(1) < df[lower_col].shift(1)) & (df["close"] > df[lower_col])

    # Long exit: close crosses below upper band
    long_exit = (df["close"].shift(1) > df[upper_col].shift(1)) & (df["close"] < df[upper_col])

    # Short entry: close crosses below upper band
    short_entry = (df["close"].shift(1) > df[upper_col].shift(1)) & (df["close"] < df[upper_col])

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
