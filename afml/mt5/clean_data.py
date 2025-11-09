"""
Clean Raw MetaTrader 5 Data and Save.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def clean_tick_data(
    df: pd.DataFrame, timezone: str = "UTC", min_spread: float = 1e-5
) -> Optional[pd.DataFrame]:
    """
    Clean and validate Forex tick data with comprehensive quality checks.

    Args:
        df: DataFrame containing tick data with bid/ask prices and timestamp index
        timezone: Timezone to localize/convert timestamps to (default: UTC)
        min_spread: Minimum valid spread (bid-ask difference) in price units

    Returns:
        Cleaned DataFrame or None if empty after cleaning
    """
    if df.empty:
        return None

    # 1. Ensure proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.isnull()]  # Remove NaT timestamps
        except Exception as e:
            raise ValueError(f"Failed to parse index: {e}")

    # 2. Timezone handling
    if df.index.tz is None:
        df = df.tz_localize(timezone)
    else:
        df = df.tz_convert(timezone)

    # 3. Price validity checks (FIXED: removed incorrect parentheses)
    price_filter = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] > df["bid"])  # Spread must be positive
        & ((df["ask"] - df["bid"]) >= min_spread)  # Minimum spread filter
    )
    df = df[price_filter]

    if df.isna().any().sum() > 0:
        print(f"Dropped NA values: \n{df.isna().sum()}")
        df.dropna(inplace=True)

    # 4. Microsecond handling (preserve even if 0)
    if not df.index.microsecond.any():
        print("Warning: No timestamps with microsecond precision found")

    # 5. Advanced duplicate handling
    duplicate_mask = df.index.duplicated(keep="last")
    dup_count = duplicate_mask.sum()
    if dup_count > 0:
        print(f"Removed {dup_count:,} duplicate timestamps")
        df = df[~duplicate_mask]

    # 6. Chronological order with efficient sorting
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    # 7. Final validation
    if df.empty:
        print("Warning: DataFrame empty after cleaning")
        return None

    return df


def _save_cleaned_with_structure(df_cleaned: pd.DataFrame, cleaned_data_path: Path, symbol: str):
    """
    Save cleaned data preserving the original directory structure

    Args:
        df_cleaned: Cleaned DataFrame with datetime index
        cleaned_data_path: Root path for cleaned data
        symbol: Symbol name for directory structure
    """
    # Ensure the index is datetime and in UTC
    if not isinstance(df_cleaned.index, pd.DatetimeIndex):
        df_cleaned.index = pd.to_datetime(df_cleaned.index, utc=True)
    elif df_cleaned.index.tz is None:
        df_cleaned.index = df_cleaned.index.tz_localize("UTC")
    else:
        df_cleaned.index = df_cleaned.index.tz_convert("UTC")

    # Extract year and month from the index
    df_temp = df_cleaned.copy()
    df_temp["year"] = df_temp.index.year
    df_temp["month"] = df_temp.index.month

    # Group by year and month to replicate original structure
    for (year, month), group_df in df_temp.groupby(["year", "month"]):
        if group_df.empty:
            continue

        # Recreate identical directory structure: path/symbol/year/month.parquet
        output_dir = cleaned_data_path / symbol / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"month-{month:02d}.parquet"

        # Remove helper columns and ensure proper index
        final_df = group_df.drop(["year", "month"], axis=1)

        # Save with same compression settings as original data
        final_df.to_parquet(output_file, engine="pyarrow", compression="zstd", index=True)

        logger.debug(
            f"Saved {len(final_df):,} rows to {output_file.relative_to(cleaned_data_path)}"
        )
