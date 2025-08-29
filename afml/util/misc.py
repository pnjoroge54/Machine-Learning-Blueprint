"""
Various useful functions
"""

import functools
import io
import os
import re
import time
from datetime import datetime as dt
from datetime import timedelta
from typing import Callable, Literal, Optional, Tuple, Union

import nbformat as nbf
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from numba import njit


def crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int):
    # pylint: disable=invalid-name
    """
    Splits df into chunks of chunksize

    :param df: (pd.DataFrame) Dataframe to split
    :param chunksize: (int) Number of rows in chunk
    :return: (list) Chunks (pd.DataFrames)
    """
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object


def indices_to_mask(indices, length):
    """
    Convert an array of indices into a boolean mask of given length.

    Parameters
    ----------
    indices : array-like of integers
        The indices to be marked as True.
    length : int
        The desired length of the output boolean mask.

    Returns
    -------
    mask : np.ndarray
        A boolean array with True at positions given by indices and False elsewhere.
    """
    mask = np.zeros(length, dtype=bool)
    mask[indices] = True
    return mask


@njit(parallel=True, cache=True)
def _count_max_decimals_numba(values: np.ndarray, max_places: int = 10) -> int:
    max_dec = 0
    for val in values:
        for i in range(max_places + 1):
            if np.isclose(val, round(val, i)):
                if i > max_dec:
                    max_dec = i
                break
    return max_dec


def count_max_decimals(
    values: Union[pd.Series, np.ndarray], max_places: int = 10
) -> int:
    """
    Determine the maximum number of decimal places in a numeric array or pandas Series
    without using string-based operations.

    :param values: Input array or Series of floating-point values. NaNs are ignored.
    :type values: Union[pd.Series, np.ndarray]
    :param max_places: Maximum number of decimal places to test for. Defaults to 10.
    :type max_places: int
    :return: The largest number of decimal places required to accurately represent
             any value in the input.
    :rtype: int
    """
    arr = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
    arr = arr[~np.isnan(arr)]
    return _count_max_decimals_numba(arr, max_places)


# ---  Pandas Utilities ---


class DataFrameFormatter:
    """
    A collection of reusable formatting utilities for pandas DataFrames.

    This class provides static methods that return formatting callables, suitable for
    use with both `DataFrame.apply()` and `DataFrame.style.format()`. These formatters
    are designed to support both human-readable string representations and native types
    for further processing or analysis.

    :Example:

    >>> formatter = DataFrameFormatter()

    >>> # Convert seconds to hh:mm:ss as string and as timedelta
    >>> df["duration_str"] = df["duration_sec"].apply(formatter.to_timecode("string"))
    >>> df["duration_td"] = df["duration_sec"].apply(formatter.to_timecode("object"))

    >>> # Format columns for display in styled DataFrame
    >>> df.style.format({
    ...     "sales": formatter.with_commas(),
    ...     "profit_margin": formatter.percentage(2),
    ...     "revenue": formatter.currency("€"),
    ... })

    Methods are stateless and safe to reuse across projects or report pipelines.
    """

    @staticmethod
    def with_commas():
        """Returns a formatter that adds thousands separators to numbers."""
        return lambda x: f"{x:,}"

    @staticmethod
    def to_timecode(
        mode: Literal["string", "object"] = "string",
    ) -> Callable[[Union[int, float]], Union[str, timedelta]]:
        """
        Converts seconds to hh:mm:ss format or timedelta objects.

        :param mode: 'string' to return formatted timecode, 'object' for timedelta.
        :type mode: Literal["string", "object"]
        :return: A callable that formats seconds accordingly.
        :rtype: Callable[[int | float], str | timedelta]

        :Example:

        >>> formatter = DataFrameFormatter()
        >>> time_str = formatter.to_timecode("string")(3661)
        >>> print(time_str)
        1:01:01

        >>> time_obj = formatter.to_timecode("object")(3661)
        >>> print(time_obj)
        1:01:01
        """
        if mode == "string":
            return lambda x: str(timedelta(seconds=int(x)))
        elif mode == "object":
            return lambda x: timedelta(seconds=int(x))
        else:
            raise ValueError("mode must be either 'string' or 'object'")

    @staticmethod
    def percentage(decimal_places=2):
        """Returns a formatter that formats a float as a percentage."""
        return lambda x: f"{x * 100:.{decimal_places}f}%"

    @staticmethod
    def currency(symbol="$"):
        """Returns a formatter for currency with commas and a symbol."""
        return lambda x: f"{symbol}{x:,.2f}"


def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize the dtypes of a DataFrame by downcasting numeric types and converting
    object columns with low cardinality to categoricals.

    :param df: The input DataFrame to optimize.
    :type df: pd.DataFrame
    :param verbose: Whether to print memory usage stats before and after optimization.
    :type verbose: bool
    :return: A new DataFrame with optimized dtypes.
    :rtype: pd.DataFrame
    """
    optimized_df = df.copy()
    start_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2

    for col in optimized_df.columns:
        col_dtype = optimized_df[col].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            if pd.api.types.is_integer_dtype(col_dtype):
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(col_dtype):
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast="float")
        elif pd.api.types.is_object_dtype(col_dtype):
            num_unique_values = optimized_df[col].nunique()
            num_total_values = len(optimized_df[col])
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype("category")

    end_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        reduction_pct = 100 * (start_mem - end_mem) / start_mem
        print(
            f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({reduction_pct:.1f}% reduction)"
        )

    return optimized_df


def log_column_changes(func):
    """
    Decorator that logs column name changes made by a DataFrame transformation function.

    Captures the original column headers and their transformed versions to aid reproducibility
    and debugging in data preprocessing pipelines.

    :param func: A function that returns a DataFrame with potentially renamed or flattened columns.
    :type func: Callable[[pd.DataFrame], pd.DataFrame]
    :return: Wrapped function with logging.
    :rtype: Callable
    """

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        old_cols = df.columns.tolist()
        result = func(df, *args, **kwargs)
        new_cols = result.columns.tolist()
        if old_cols != new_cols:
            logger.info("Column names changed:")
            for old, new in zip(old_cols, new_cols):
                if old != new:
                    logger.info(f"  '{old}' -> '{new}'")
        else:
            logger.info("No column name changes detected.")
        return result

    return wrapper


@log_column_changes
def flatten_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the DataFrame with flattened column names.

    This flattens MultiIndex column names by joining tuple elements with underscores,
    and returns a new DataFrame with updated column headers.

    :param df: Input DataFrame with potentially MultiIndex columns.
    :type df: pd.DataFrame
    :return: A copy of the DataFrame with flattened column names.
    :rtype: pd.DataFrame

    :Example:

    >>> df_grouped = df.groupby("key").agg({"value": ["mean", "sum"]})
    >>> df_flat = flatten_column_names(df_grouped)
    >>> df_flat.columns
    Index(['value_mean', 'value_sum'], dtype='object')
    """
    df_copy = df.copy()
    df_copy.columns = [
        "_".join(map(str, col)).strip() if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df_copy


def value_counts_data(
    series: pd.Series, verbose: bool = False, as_percentage: bool = True
) -> pd.DataFrame:
    """
    Returns a DataFrame showing raw and relative value counts of a Series.

    :param series: The input Series to analyze.
    :type series: pd.Series
    :param verbose: If True, prints the result; otherwise returns it as a DataFrame.
    :type verbose: bool
    :param as_percentage: Whether to include proportion column.
    :type as_percentage: bool
    :return: A DataFrame with counts and optional percentage column.
    :rtype: pd.DataFrame

    :Example:

    >>> value_counts_data(df["status"], verbose=True)

    status value counts:
           count  proportion
    active  1,240    0.62
    closed    760    0.38
    """
    counts = series.value_counts()
    formatted = counts.apply(lambda x: f"{x:,}")
    df = pd.DataFrame({"count": formatted})
    if as_percentage:
        df["proportion"] = series.value_counts(normalize=True)
    if verbose:
        print(f"\n{series.name} value counts:\n{df}\n")
    return df


# ---  Logging Utilities ---


def log_performance(func):
    """
    Decorator that logs the memory usage and execution time of a function.

    This utility tracks the resident memory footprint before and after a function
    call and reports the delta in megabytes along with the runtime duration.

    :param func: The function to wrap and monitor.
    :type func: Callable
    :return: A wrapped function that logs performance metrics to the configured logger.
    :rtype: Callable

    :Example:

    >>> @log_performance
    ... def heavy_function():
    ...     return np.zeros((1000, 1000))
    >>> heavy_function()
    'heavy_function' - Time: 0:00:00.002345. Memory increment: 7.63 MB (123.45 MB -> 131.08 MB)
    """

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # memory in MB
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start_time
        elapsed = timedelta(seconds=elapsed)
        mem_after = process.memory_info().rss / 1024**2
        logger.info(
            f"'{func.__name__}' - Time: {elapsed}. Memory increment: {mem_after - mem_before:.2f} MB ({mem_before:.2f} MB -> {mem_after:.2f} MB)"
        )
        return result

    return wrapper


def log_df_info(df: pd.DataFrame):
    """
    Logs the output of `df.info()` for a given DataFrame.

    This is useful for capturing column types, non-null counts, and memory usage
    in structured logs during preprocessing or debugging.

    :param df: The DataFrame whose structure will be logged.
    :type df: pd.DataFrame

    :Example:

    >>> log_df_info(df)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 4 columns):
     ...
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    buffer.close()
    logger.info("\n" + info_str)  # Log the captured output


# --- Time Helpers ---


def set_resampling_freq(timeframe: str):
    """
    Convert an MT5 timeframe string to a valid pandas resampling frequency.

    This utility interprets MetaTrader 5 (MT5) timeframe codes and maps them
    to pandas-compatible resampling frequencies. Useful for aligning time-series
    data to business days, calendar weeks, or custom intervals.

    :param timeframe: MT5-style timeframe code (e.g., 'D1', 'W1', 'H1', 'M15', 'S30').
    :type timeframe: str
    :return: A string representing the pandas frequency alias (e.g., 'B', 'W-FRI', '15min').
    :rtype: str

    :raises ValueError: If the timeframe is not recognized or supported.

    :Example:

    >>> set_resampling_freq("D1")
    'B'
    >>> set_resampling_freq("H4")
    '4h'
    >>> set_resampling_freq("M15")
    '15min'
    >>> set_resampling_freq("S30")
    '30s'
    >>> set_resampling_freq("W1")
    'W-FRI'
    """
    timeframe = timeframe.upper()
    nums = (x for x in list(timeframe) if x.isnumeric())  # list of numbers in timeframe
    x = int("".join(nums))

    if timeframe.startswith("W"):
        freq = "W-FRI"
    elif timeframe.startswith("D"):
        freq = "B"
    elif timeframe.startswith("H"):
        freq = f"{x}h"
    elif timeframe.startswith("M"):
        freq = f"{x}min"
    elif timeframe.startswith("S"):
        freq = f"{x}s"
    else:
        raise ValueError(
            """
                         Valid timeframe arguments:
                         W1: weekly
                         D1: daily
                         H(x): resample x hours, e.g. H1, H4
                         M(x): resample x minutes, e.g. M1, M5
                         S(x): resample x seconds, e.g. S15, S30
                         """
        )
    return freq


def date_conversion(
    start_date: Union[str, dt, pd.Timestamp],
    end_date: Union[str, dt, pd.Timestamp],
    default_tz: str = "UTC",
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Validates start and end dates, ensuring they are timezone-aware and in correct order.

    Args:
        start_date (Union[str, dt, pd.Timestamp]): The start date of the period.
        end_date (Union[str, dt, pd.Timestamp]): The end date of the period.
        default_tz (str): The default timezone to localize dates if they are naive.
    Returns:
        Optional[Tuple[pd.Timestamp, pd.Timestamp]]: A tuple of (start_date, end_date) as
        timezone-aware pandas Timestamps, or None if validation fails.
    """

    def convert_single_date(date_val, date_name):
        if date_val is None:
            raise ValueError(f"{date_name} cannot be None")

        try:
            # Handle various input types
            if isinstance(date_val, str):
                if not date_val.strip():
                    raise ValueError(f"{date_name} cannot be empty string")
                ts = pd.to_datetime(date_val)
            elif isinstance(date_val, (int, float)):
                # Handle Unix timestamps
                ts = pd.to_datetime(date_val, unit="s")
            else:
                ts = pd.to_datetime(date_val)

            # Ensure timezone awareness
            if ts.tzinfo is None:
                ts = ts.tz_localize(default_tz)
            elif str(ts.tzinfo) != default_tz:
                # Convert to desired timezone if different
                ts = ts.tz_convert(default_tz)

            return ts

        except Exception as e:
            raise ValueError(f"Cannot parse {date_name} '{date_val}': {str(e)}")

    # Convert both dates
    start_ts = convert_single_date(start_date, "start_date")
    end_ts = convert_single_date(end_date, "end_date")

    # Validate date order
    if start_ts >= end_ts:
        raise ValueError(f"Start date ({start_ts}) must be before end date ({end_ts})")

    return start_ts, end_ts


# --- Convert Files ---


def markdown_to_notebook(markdown_content, output_filename="AFML Experiments.ipynb"):
    """
    Convert markdown content (either as string or file path) to Jupyter notebook.

    Args:
        markdown_content (str): Can be either markdown content string or file path
        output_filename (str): Output notebook filename

    Returns:
        str: Output notebook filename
    """
    # Read content if input is a valid file path
    if isinstance(markdown_content, str) and os.path.isfile(markdown_content):
        with open(markdown_content, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = markdown_content

    # Create new notebook and pattern for code blocks
    nb = nbf.v4.new_notebook()
    pattern = r"```python\n(.*?)\n```"
    current_pos = 0
    cells = []

    # Process all code blocks
    for match in re.finditer(pattern, content, re.DOTALL):
        # Add preceding markdown
        markdown_segment = content[current_pos : match.start()].strip()
        if markdown_segment:
            clean_md = re.sub(r"\n{3,}", "\n\n", markdown_segment)
            cells.append(nbf.v4.new_markdown_cell(clean_md))

        # Add code block
        code_block = match.group(1).strip()
        if code_block:
            cells.append(nbf.v4.new_code_cell(code_block))

        current_pos = match.end()

    # Add remaining markdown after last code block
    trailing_markdown = content[current_pos:].strip()
    if trailing_markdown:
        clean_trailing = re.sub(r"\n{3,}", "\n\n", trailing_markdown)
        cells.append(nbf.v4.new_markdown_cell(clean_trailing))

    # Handle case with no code blocks
    if not cells:
        clean_content = re.sub(r"\n{3,}", "\n\n", content.strip())
        cells.append(nbf.v4.new_markdown_cell(clean_content))

    # Add cells to notebook and save
    nb.cells = cells
    with open(output_filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook saved as {output_filename}")
    return output_filename
