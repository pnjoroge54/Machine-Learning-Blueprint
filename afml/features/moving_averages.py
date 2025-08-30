from itertools import combinations
from typing import List, Tuple, Union

import pandas as pd
import seaborn as sns
from feature_engine.selection import DropCorrelatedFeatures
from loguru import logger
from matplotlib import pyplot as plt


def calculate_ma_differences(
    close: pd.Series, windows: Union[List, Tuple], threshold: float = 0.8, verbose: bool = False
):
    """
    Moving average differences.

    param close: (pd.Series) Close prices
    param windows: (Union[List, Tuple]) Windows to create differences for, e.g. (10, 20, 50)
    param threshold: (float)
    param verbose: (bool) If True, display heatmap of correlation matrix
    returns df: (pd.DataFrame)
    """
    df = pd.DataFrame(index=close.index)
    sma = {window: close.rolling(window, closed="left").mean() for window in windows}

    # Create differences of all unique combinations of windows
    for win in combinations(windows, 2):
        fast_window, slow_window = sorted(win)
        df[f"sma_diff_{fast_window}_{slow_window}"] = sma[fast_window] - sma[slow_window]

    dcf = DropCorrelatedFeatures(threshold=threshold)
    out = dcf.fit_transform(df)
    dropped = df.columns.difference(out.columns).to_list()
    if len(dropped) > 0:
        logger.info(
            f"\nDropped features with correlation > {threshold}: \n\t{dropped}"
            f"\nKept features: \n\t{out.columns.to_list()}"
        )
        if verbose:
            corr_matrix = df.corr()
            # Set the figure size for better readability
            plt.figure(figsize=(12, 4))

            # Create the heatmap with the mask
            sns.heatmap(
                corr_matrix,
                cmap="coolwarm",  # Choose a colormap
                linewidths=0.5,  # Add lines to separate the cells
                annot=True,  # Annotate with the correlation values
                fmt=".2f",  # Format the annotations to two decimal places
                cbar_kws={"shrink": 0.8},  # Shrink the color bar
            )

            plt.title("Correlation Matrix")
            plt.show()

    return out
