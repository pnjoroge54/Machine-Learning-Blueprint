"""
Various functions for message encoding (quantile)
"""

import numpy as np
from numba import jit


def encode_tick_rule_array(tick_rule_array):
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) Tick rules
    :return: (str) Encoded message
    """
    arr = np.asarray(tick_rule_array, dtype=np.int8)
    # map -1 -> 'b', 0 -> 'c', 1 -> 'a'
    chars = np.array(["b", "c", "a"])[arr + 1]
    return chars.tobytes().decode()  # or ''.join(chars)


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) ASCII symbols
    """
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) Values to split on quantiles
    :param num_letters: (int) Number of letters(quantiles) to encode
    :return: (dict) Dict of quantile-symbol
    """
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for quant, letter in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, quant)] = letter
    return encoding_dict


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) Values to split on quantiles
    :param step: (float) Discretization step (sigma)
    :return: (dict) Dict of value-symbol
    """
    i = 0
    ascii_table = _get_ascii_table()
    encoding_dict = {}
    encoding_steps = np.arange(min(array), max(array), step)
    for element in encoding_steps:
        try:
            encoding_dict[element] = ascii_table[i]
        except IndexError:
            raise ValueError(
                f"Length of dictionary ceil((max(arr) - min(arr)) / step = {len(encoding_steps)} is more than ASCII table lenght)"
            )
        i += 1
    return encoding_dict


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) Values
    :param value: (float) Value for which the nearest element needs to be found
    :return: (float) The nearest to the value element in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int) Value to use
    :param encoding_dict: (dict) Used dictionary
    :return: (str) Letter from encoding dict
    """
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]


@jit(cache=True, forceobj=True)
def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) Values to encode
    :param encoding_dict: (dict) Dict of quantile-symbol
    :return: (str) Encoded message
    """
    # Get sorted thresholds and matching letters
    thresholds = np.array(sorted(encoding_dict.keys()))
    letters = [encoding_dict[t] for t in thresholds]
    # digitize returns bin indices (1-based for right side, but careful)
    # We'll use searchsorted on raw values
    arr = np.asarray(array)
    bin_indices = np.searchsorted(thresholds, arr, side="right") - 1
    # Clip to valid range (values below first threshold map to last bin? depends on semantics)
    # Actually your original _find_nearest grabs the nearest, not the first above.
    # For quantile mapping the boundaries are quantiles, so nearest is fine.
    # We can use digitize with nearest via custom logic, but simplest: compute distances.
    # For a robust nearest-neighbor mapping:
    idx = np.abs(arr[:, None] - thresholds[None, :]).argmin(axis=1)
    # Map to letters:
    chars = np.array(letters)[idx]
    return chars.tobytes().decode()
