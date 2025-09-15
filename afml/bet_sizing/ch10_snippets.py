"""
This module contains the code snippets found in Chapter 10 of "Advances in Financial Machine Learning" by
Marcos López de Prado. The code has been amended for readability, to conform to PEP8 rules, to keep the snippets as
manageable single-units of functionality, as well as to account for deprecation of functions originally used, but is
otherwise unaltered.
"""

import warnings

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import norm


def get_signal(prob, num_classes, pred=None):
    """
    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE
    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this
    representation, the probability will always be between 1/num_classes and 1.0.

    :param prob: (pd.Series) The probability of the predicted bet side.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :return: (pd.Series) The bet size.
    """
    # Get signals from predictions.
    if prob.shape[0] == 0:
        return pd.Series(dtype="float64")

    # Clip probabilities to avoid issues with log(0) or division by zero in z-score.
    prob = prob.clip(lower=1e-6, upper=1 - 1e-6)

    # Generate signals from multinomial classification (one-vs-rest).
    bet_sizes = (prob - 1 / num_classes) / (prob * (1 - prob)) ** 0.5

    # Allow for bet size to be returned with or without side.
    if not isinstance(pred, type(None)):
        # signal = side * size
        bet_sizes = pred * (2 * norm.cdf(bet_sizes) - 1)
    else:
        # signal = size only
        bet_sizes = pd.Series(2 * norm.cdf(bet_sizes) - 1, index=prob.index)

    # Note 1: In the book, this function contains a conditional statement checking for a column named 'side',
    # then executes what is essentially the above line. This has been removed as it appears to be redundant
    # and simplifies the function.

    # Note 2: In the book, this function includes the averaging and discretization steps, which are omitted here.
    # The functions for performing these are included in this file, and can be applied as options in the user-level
    # functions in bet_sizing.py.

    return bet_sizes


def avg_active_signals(signals):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.

    :param signals: (pandas.DataFrame) Contains at least the following columns:
    - **signal** - the bet size
    - **t1** - the closing time of the bet
        And the index must be datetime format
    :param num_threads: (int) Number of threads to use in multiprocessing.
        If None, Numba is used, else `mp_pandas_obj`.
    :return: (pandas.Series) The averaged bet sizes
    """
    # Convert datetimes to nanoseconds
    signal_times = signals.index.view(np.int64)
    signal_values = signals["signal"].values

    # Convert end times, handling NaT as maximum time
    MAX_TIME = np.iinfo(np.int64).max
    end_times = np.where(signals["t1"].isna(), MAX_TIME, signals["t1"].view(np.int64))

    # Get all unique evaluation times
    eval_times = np.unique(np.concatenate((signal_times, signals["t1"].dropna().view(np.int64))))

    # Calculate results
    results = _calculate_active_signals(signal_times, end_times, signal_values, eval_times)

    # Convert back to datetime index
    return pd.Series(results, index=pd.to_datetime(eval_times), name="signal")


# Numba-optimized function to calculate average active signals
@njit(parallel=True, cache=True)
def _calculate_active_signals(signal_times, end_times, signal_values, eval_times):
    """
    Calculate average active signals at evaluation times using parallel processing.

    :param signal_times: Start times of signals (int64 nanoseconds)
    :param end_times: End times of signals (int64 nanoseconds, MAX_TIME for NaT)
    :param signal_values: Bet size values
    :param eval_times: Times to evaluate active signals (int64 nanoseconds)
    :return: Array of averaged signal values
    """
    results = np.zeros(len(eval_times), dtype=np.float64)

    for i in prange(len(eval_times)):
        t = eval_times[i]
        total = 0.0
        count = 0

        for j in range(len(signal_times)):
            # Check if signal is active at time t
            if signal_times[j] <= t and t < end_times[j]:
                total += signal_values[j]
                count += 1

        results[i] = total / count if count > 0 else 0.0

    return results


def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING
    Discretizes the bet size signal based on the step size given.

    :param signal0: (pandas.Series) The signal to discretize.
    :param step_size: (float) Step size.
    :return: (pandas.Series) The discretized signal.
    """
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor
    return signal1


# ==============================================================================
# SNIPPET 10.4 - DYNAMIC POSITION SIZE AND LIMIT PRICE
# The below functions are part of or derived from the functions
# in snippet 10.4.
# ==============================================================================
# Bet size calculations based on a sigmoid function.
def bet_size_sigmoid(w_param, price_div):
    """
    Part of SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, forecast price - market price.
    :return: (float) The bet size.
    """
    return price_div * ((w_param + price_div**2) ** (-0.5))


def get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :return: (int) Target position.
    """
    return int(bet_size_sigmoid(w_param, forecast_price - market_price) * max_pos)


def inv_price_sigmoid(forecast_price, w_param, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a sigmoid function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    return forecast_price - m_bet_size * (w_param / (1 - m_bet_size**2)) ** 0.5


def limit_price_sigmoid(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the limit price.
    Based on a sigmoid function for a bet size algorithm.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :return: (float) Limit price.
    """
    if target_pos == pos:
        # Return NaN if the current and target positions are the same to avoid divide-by-zero error.
        return np.nan

    sgn = np.sign(target_pos - pos)
    l_p = 0
    for j in range(abs(pos + sgn), abs(target_pos + 1)):
        l_p += inv_price_sigmoid(forecast_price, w_param, j / float(max_pos))

    l_p = l_p / abs(target_pos - pos)
    return l_p


def get_w_sigmoid(price_div, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    Based on a sigmoid function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """
    return (price_div**2) * ((m_bet_size ** (-2)) - 1)


# ==============================================================================
# Bet size calculations based on a power function.
def bet_size_power(w_param, price_div):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price, must be between -1 and 1, inclusive.
    :return: (float) The bet size.
    """
    if not (-1 <= price_div <= 1):
        raise ValueError(
            f"Price divergence must be between -1 and 1, inclusive. Found price divergence value:"
            f" {price_div}"
        )
    if price_div == 0.0:
        return 0.0

    return np.sign(price_div) * abs(price_div) ** w_param


def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Target position.
    """
    return int(bet_size_power(w_param, forecast_price - market_price) * max_pos)


def inv_price_power(forecast_price, w_param, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a power function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    if m_bet_size == 0.0:
        return forecast_price
    return forecast_price - np.sign(m_bet_size) * abs(m_bet_size) ** (1 / w_param)


def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. Based on a power function for a bet size algorithm.

    :param target_pos: (float) Target position.
    :param pos: (float) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Limit price.
    """
    sgn = np.sign(target_pos - pos)
    l_p = 0
    for j in range(abs(pos + sgn), abs(target_pos + 1)):
        l_p += inv_price_power(forecast_price, w_param, j / float(max_pos))

    l_p = l_p / abs(target_pos - pos)
    return l_p


def get_w_power(price_div, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'w' coefficient must be greater than or equal to zero.
    Based on a power function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """
    if not -1 <= price_div <= 1:
        raise ValueError(
            "Price divergence argument 'x' must be between -1 and 1,"
            " inclusive when using function 'power'."
        )

    w_calc = np.log(m_bet_size / np.sign(price_div)) / np.log(abs(price_div))
    if w_calc < 0:
        warnings.warn("'w' parameter evaluates to less than zero. Zero is returned.", UserWarning)

    return max(0, w_calc)


# ==============================================================================
# Bet size calculation functions, power and sigmoid packaged together.
# This is useful as more bet sizing function options are added.
def bet_size(w_param, price_div, func):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) The bet size.
    """
    return {"sigmoid": bet_size_sigmoid, "power": bet_size_power}[func](w_param, price_div)


def get_target_pos(w_param, forecast_price, market_price, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (int) Target position.
    """
    return {"sigmoid": get_target_pos_sigmoid, "power": get_target_pos_power}[func](
        w_param, forecast_price, market_price, max_pos
    )


def inv_price(forecast_price, w_param, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    return {"sigmoid": inv_price_sigmoid, "power": inv_price_power}[func](
        forecast_price, w_param, m_bet_size
    )


def limit_price(target_pos, pos, forecast_price, w_param, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. The 'func' argument allows the user to choose between bet sizing functions.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Limit price.
    """
    return {"sigmoid": limit_price_sigmoid, "power": limit_price_power}[func](
        int(target_pos), int(pos), forecast_price, w_param, max_pos
    )


def get_w(price_div, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """
    return {"sigmoid": get_w_sigmoid, "power": get_w_power}[func](price_div, m_bet_size)
