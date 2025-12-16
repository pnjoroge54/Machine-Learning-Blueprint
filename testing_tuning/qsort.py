import numpy as np
from typing import List, Tuple, Union, Optional


def qsortd(first: int, last: int, data: np.ndarray) -> None:
    """Quick sort for double array"""
    if first >= last:
        return

    lower, upper = first, last
    split = data[(first + last) // 2]

    while lower <= upper:
        while split > data[lower]:
            lower += 1
        while split < data[upper]:
            upper -= 1

        if lower == upper:
            lower += 1
            upper -= 1
        elif lower < upper:
            # Swap elements
            data[lower], data[upper] = data[upper], data[lower]
            lower += 1
            upper -= 1

    if first < upper:
        qsortd(first, upper, data)
    if lower < last:
        qsortd(lower, last, data)


def qsortds(first: int, last: int, data: np.ndarray, slave: np.ndarray) -> None:
    """Quick sort for double array with one slave array"""
    if first >= last:
        return

    lower, upper = first, last
    split = data[(first + last) // 2]

    while lower <= upper:
        while split > data[lower]:
            lower += 1
        while split < data[upper]:
            upper -= 1

        if lower == upper:
            lower += 1
            upper -= 1
        elif lower < upper:
            # Swap both main and slave arrays
            data[lower], data[upper] = data[upper], data[lower]
            slave[lower], slave[upper] = slave[upper], slave[lower]
            lower += 1
            upper -= 1

    if first < upper:
        qsortds(first, upper, data, slave)
    if lower < last:
        qsortds(lower, last, data, slave)


def qsortds_multiple(
    first: int, last: int, data: np.ndarray, *slaves: np.ndarray
) -> None:
    """Generalized quick sort with multiple slave arrays"""
    if first >= last:
        return

    lower, upper = first, last
    split = data[(first + last) // 2]

    while lower <= upper:
        while split > data[lower]:
            lower += 1
        while split < data[upper]:
            upper -= 1

        if lower == upper:
            lower += 1
            upper -= 1
        elif lower < upper:
            # Swap main array
            data[lower], data[upper] = data[upper], data[lower]
            # Swap all slave arrays
            for slave in slaves:
                slave[lower], slave[upper] = slave[upper], slave[lower]
            lower += 1
            upper -= 1

    if first < upper:
        qsortds_multiple(first, upper, data, *slaves)
    if lower < last:
        qsortds_multiple(lower, last, data, *slaves)


# For integer arrays
def qsorti(first: int, last: int, data: np.ndarray) -> None:
    """Quick sort for integer array"""
    if first >= last:
        return

    lower, upper = first, last
    split = data[(first + last) // 2]

    while lower <= upper:
        while split > data[lower]:
            lower += 1
        while split < data[upper]:
            upper -= 1

        if lower == upper:
            lower += 1
            upper -= 1
        elif lower < upper:
            data[lower], data[upper] = data[upper], data[lower]
            lower += 1
            upper -= 1

    if first < upper:
        qsorti(first, upper, data)
    if lower < last:
        qsorti(lower, last, data)
