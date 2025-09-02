import pandas as pd
from statsmodels.tsa.stattools import adfuller


def is_stationary(df: pd.DataFrame, alpha: float = 0.05, verbose: bool = True):
    not_stationary = []
    for col in df:
        adf = adfuller(df[col], maxlag=1, regression="c", autolag=None)
        if not (adf[0] < adf[4][f"{alpha:.0%}"] and adf[1] < alpha):
            not_stationary.append(col)

    if verbose:
        if not_stationary:
            print("These features are not stationary:")
            for x in not_stationary:
                print(x)
        else:
            print("All features are stationary.")

    return not_stationary
