"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from collections import Counter

import numpy as np
from numba import njit


def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """
    counts = Counter(message)
    total = len(message)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    i, lib = 1, {message[0]}
    while i < len(message):
        for j in range(i, len(message)):
            phrase = message[i : j + 1]
            if phrase not in lib:
                lib.add(phrase)
                break
        i = j + 1
    return len(lib) / len(message)


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    total = len(message) - word_length
    cnt = Counter(message[i - word_length : i] for i in range(word_length, len(message)))
    return {word: count / total for word, count in cnt.items()}


def get_plug_in_entropy(message: str, word_length: int = 1) -> float:
    """
    Vectorized plug-in entropy estimator.

    :param message: (str) Encoded message (ASCII characters only)
    :param word_length: (int) Block length for joint entropy estimation
    :return: (float) Plug-in entropy (bits per symbol)
    """
    n = len(message)
    if n <= word_length:
        return 0.0

    # Total number of blocks (original uses n - word_length windows)
    n_windows = n - word_length

    # Convert string to array of bytes (fast, zero-copy)
    # message is assumed to be ASCII (e.g. 'a','b','c'), so 1 byte per char
    arr = np.frombuffer(message.encode(), dtype=np.uint8)

    # Keep only the first n-1 characters to match the original's window set
    arr = arr[: n - 1]  # truncate the last character

    # Extract all sliding windows of length word_length
    # This creates a view, shape = (n_windows, word_length)
    windows = np.lib.stride_tricks.sliding_window_view(arr, word_length)

    # Count unique rows (each row is a block)
    # `return_counts=True` gives the counts directly
    _, counts = np.unique(windows, axis=0, return_counts=True)

    # Compute probabilities and plug-in entropy
    probs = counts / n_windows
    entropy = -np.sum(probs * np.log2(probs)) / word_length
    return entropy


@njit(cache=True)
def _match_length(
    message: str, start_index: int, window: int
) -> int | str:  # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = ""
    for length in range(window):
        msg1 = message[start_index : start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j : j + length + 1]
            if len(msg1) != len(msg0):
                continue
            if msg1 == msg0:
                sub_str = msg1
                break  # Search for higher l.
    return len(sub_str) + 1, sub_str  # Matched length + 1


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    out = {"h": 0, "r": 0, "num": 0, "sum": 0, "sub_str": []}
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = _match_length(message, i, i)
            out["sum"] += np.log2(i + 1) / length  # To avoid Doeblin condition
        else:
            length, msg_ = _match_length(message, i, window)
            out["sum"] += np.log2(window + 1) / length  # To avoid Doeblin condition
        out["sub_str"].append(msg_)
        out["num"] += 1
    try:
        out["h"] = out["sum"] / out["num"]
    except ZeroDivisionError:
        out["h"] = 0
    out["r"] = 1 - out["h"] / (
        np.log2(len(message)) if np.log2(len(message)) > 0 else 1
    )  # Redundancy, 0<=r<=1
    return out["h"]
