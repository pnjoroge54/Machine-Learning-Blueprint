# -*- coding: utf-8 -*-
"""
This script provides functionalities to download financial market data
and save it in a structured, partitioned Parquet format. Data is fetched
from MetaTrader 5 (MT5) for specified symbols and date ranges, with options
to handle existing files intelligently. It includes a user-friendly GUI prompt
to manage overwriting existing data, robust logging with 'loguru', and secure
credential management using environment variables. The script is designed to be
run as a standalone application for data acquisition, ensuring that the account
used for downloading matches the account used for loading data later.

FUntionality is provided for loading downloaded files into a DataFrame for analysis,
with options to specify columns and date ranges. The script is optimized for memory
usage and provides a clear directory structure for easy data management.

Features:
- Downloads data for a given list of symbols and a date range.
- Organizes saved data into a clear directory structure: path/symbol/year/month.parquet
- Implements a user-friendly, timed GUI prompt to ask for overwriting existing files.
- Uses 'loguru' for robust logging to both console and a file.
- Secures credentials using environment variables.
- Verifies account consistency for both downloading and loading data.
- Designed to be run as a standalone script for data acquisition.
"""

import json
import os
import tkinter as tk
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dotenv import load_dotenv
from loguru import logger

from ..util.misc import date_conversion, log_df_info

# --- Credential and Login Management ---


def get_credentials_from_env(account):
    """
    Retrieves MT5 credentials from environment variables.

    Args:
        account (str): The account name (e.g., 'MyAccount').

    Returns:
        tuple: (login, password, server) or (None, None, None) if not found.
    """
    load_dotenv()
    prefix = f"MT5_ACCOUNT_{account.upper()}"
    login = os.environ.get(f"{prefix}_LOGIN")
    password = os.environ.get(f"{prefix}_PASSWORD")
    server = os.environ.get(f"{prefix}_SERVER")

    if not all([login, password, server]):
        logger.error(f"Missing one or more environment variables for account '{account}'.")
        logger.error(f"Please set {prefix}_LOGIN, {prefix}_PASSWORD, and {prefix}_SERVER.")
        return None, None, None

    if login.isnumeric():
        login = int(login)

    return login, password, server


def login_mt5(account, timeout=20000, verbose=True):
    """
    Logs in to a MetaTrader5 account using credentials from environment variables.

    Args:
        account (str): Account name to log in to.
        timeout (int): Connection timeout in milliseconds.
        verbose (bool): Whether to print detailed connection information.

    Returns:
        str: The account name if login is successful, otherwise None.
    """
    logger.info(f"Attempting to log in to MT5 with account: {account}")
    login, password, server = get_credentials_from_env(account)

    if not login:
        return None

    if not mt5.initialize(login=login, password=password, server=server, timeout=timeout):
        logger.error(f"MT5 initialize() failed for account {account}. Error: {mt5.last_error()}")
        mt5.shutdown()
        return None

    logger.success(f"Successfully logged in to MT5 as {account}.")
    if verbose:
        logger.info(f"MT5 Version: {mt5.version()}")
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(f"Connected to {terminal_info.name} at {terminal_info.path}")
        else:
            logger.warning("Could not retrieve terminal info.")

    return account


# --- Timed GUI MessageBox ---


class TimedMessageBox:
    """
    A non-blocking, timed GUI message box for user prompts.
    """

    def __init__(self, title, text, timeout_ms=5000):
        self.root = None
        self.result = "timeout"
        self.title = title
        self.text = text
        self.timeout_ms = timeout_ms

    def _create_window(self):
        self.root = tk.Tk()
        self.root.withdraw()

        msg_box = tk.Toplevel(self.root)
        msg_box.title(self.title)
        msg_box.attributes("-topmost", True)

        win_width, win_height = 350, 130
        screen_width, screen_height = (
            msg_box.winfo_screenwidth(),
            msg_box.winfo_screenheight(),
        )
        x = (screen_width // 2) - (win_width // 2)
        y = (screen_height // 2) - (win_height // 2)
        msg_box.geometry(f"{win_width}x{win_height}+{x}+{y}")

        tk.Label(msg_box, text=self.text, wraplength=320, justify="center").pack(pady=15)
        btn_frame = tk.Frame(msg_box)
        btn_frame.pack(pady=5)

        btn_yes = tk.Button(btn_frame, text="Yes", width=12, command=lambda: self._on_click("yes"))
        btn_no = tk.Button(btn_frame, text="No", width=12, command=lambda: self._on_click("no"))
        btn_yes.pack(side=tk.LEFT, padx=12)
        btn_no.pack(side=tk.LEFT, padx=12)

        self.root.after(self.timeout_ms, lambda: self._on_click("timeout"))
        self.root.mainloop()

    def _on_click(self, choice):
        self.result = choice
        if self.root:
            self.root.quit()
            self.root.destroy()

    def show(self):
        self._create_window()
        return self.result


# --- Data Validation and Verification ---


def verify_or_create_account_info(data_path, current_account_name):
    """
    Checks if the data directory is associated with the correct account.
    If no account info exists, it creates it.

    Args:
        data_path (Path): The root path of the data directory.
        current_account_name (str): The name of the account currently in use.

    Returns:
        bool: True if the account is verified, False otherwise.
    """
    account_info_file = data_path / "account_info.json"

    if account_info_file.exists():
        try:
            with open(account_info_file, "r") as f:
                stored_info = json.load(f)
                stored_name = stored_info.get("account_name")

            if stored_name and stored_name != current_account_name:
                logger.error(
                    f"Account Mismatch! This directory ('{data_path.name}') is for account '{stored_name}'."
                )
                logger.error(
                    f"Current operation is for account '{current_account_name}'. Aborting to prevent data errors."
                )
                return False
            elif not stored_name:
                # File exists but is malformed, so we fix it.
                logger.warning("Account info file is malformed. Overwriting with current account.")
                with open(account_info_file, "w") as f:
                    json.dump({"account_name": current_account_name}, f, indent=4)

        except json.JSONDecodeError:
            logger.warning(
                f"Could not read account info file. Overwriting with current account: '{current_account_name}'."
            )
            with open(account_info_file, "w") as f:
                json.dump({"account_name": current_account_name}, f, indent=4)
    else:
        logger.info(
            f"First time use for this directory. Associating it with account '{current_account_name}'."
        )
        with open(account_info_file, "w") as f:
            json.dump({"account_name": current_account_name}, f, indent=4)

    return True


# --- Data Fetching and Saving ---


def get_ticks(symbol, start_date, end_date, datetime_index=True, verbose=True):
    """
    Downloads tick data from the MT5 terminal for a given period.

    Args:
        symbol (str): The financial instrument symbol (e.g., 'EURUSD').
        start_date (pd.Timestamp): The timezone-aware start date for the data range.
        end_date (pd.Timestamp): The timezone-aware end date for the data range.
        datetime_index (bool): Set 'time' column to DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame containing the tick data, or an empty DataFrame if
                      no data is found or an error occurs.
    """
    if not mt5.terminal_info():  # Check if connection is still active
        logger.error("MT5 connection lost. Cannot download data.")
        return pd.DataFrame()

    try:
        start_date, end_date = date_conversion(start_date, end_date)
        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            logger.warning(
                f"No tick data returned for {symbol} from {start_date.date()} to {end_date.date()}."
            )
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df.drop(columns=["time_msc"], inplace=True)

        if datetime_index:
            df.set_index("time", inplace=True)

        # Keep only columns with meaningful data
        df = df.loc[:, df.any()]

        # Optimize memory usage
        for col in ["bid", "ask"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        if verbose:
            log_df_info(df)

        return df

    except Exception as e:
        logger.error(f"An error occurred while getting ticks for {symbol}: {e}")
        return pd.DataFrame()


def save_data_to_parquet(path, symbols, start_date, end_date, account_name):
    """
    Downloads and saves tick data to a partitioned Parquet structure.

    Args:
        path (Union[str, Path]): The root folder where data will be saved.
        symbols (Union[str, list, tuple]): A single symbol or a collection of symbols to download.
        start_date (Union[str, dt, pd.Timestamp]): The start date for the data range.
        end_date (Union[str, dt, pd.Timestamp]): The end date for the data range.
        account_name (str): The name of the account used for the download.
    """
    root_path = Path(path)
    root_path.mkdir(parents=True, exist_ok=True)

    if not verify_or_create_account_info(root_path, account_name):
        return

    date_range = date_conversion(start_date, end_date)
    if not date_range:
        return
    start_dt, end_dt = date_range

    if isinstance(symbols, str):
        symbols = [symbols]

    # --- Check for existing files once ---
    overwrite_response = "skip"
    dates_to_check = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    for symbol in symbols:
        for d in dates_to_check:
            file_path = root_path / symbol / str(d.year) / f"month-{d.month:02d}.parquet"
            if file_path.exists():
                overwrite_response = TimedMessageBox(
                    title="Overwrite Existing Files?",
                    text="Some data files already exist in the destination. Do you want to overwrite them?",
                ).show()
                break
        if overwrite_response != "skip":
            break

    if overwrite_response == "yes":
        logger.info("User chose to OVERWRITE existing files.")
    elif overwrite_response in ("no", "timeout"):
        logger.info("User chose to SKIP existing files. Operation will only fill in missing data.")

    # --- Main Download Loop ---
    missing_data = {}
    dates_from = pd.date_range(start=start_dt, end=end_dt, freq="MS", tz="UTC")
    dates_to = pd.date_range(start=start_dt, end=end_dt, freq="ME", tz="UTC") + pd.Timedelta(days=1)

    for i, symbol in enumerate(symbols):
        logger.info(f"Processing symbol: {symbol} [{i+1}/{len(symbols)}]")
        symbol_path = root_path / symbol
        symbol_path.mkdir(parents=True, exist_ok=True)

        for j, (start, end) in enumerate(zip(dates_from, dates_to), 1):
            year_path = symbol_path / str(start.year)
            year_path.mkdir(parents=True, exist_ok=True)
            file = year_path / f"month-{start.month:02d}.parquet"

            log_msg_prefix = f"  -> Month {start.strftime('%Y-%m')}..."

            # Display symbol info every 10 lines
            if j % 10 == 0:
                log_msg_prefix = f"{symbol} {log_msg_prefix}"

            if file.exists() and overwrite_response in ("no", "timeout"):
                logger.info(f"{log_msg_prefix} Exists, skipping.")
                continue

            df = get_ticks(symbol, start, end, verbose=False)

            if df.empty:
                logger.warning(f"{log_msg_prefix} No data found.")
                missing_data.setdefault(symbol, []).append(start.strftime("%Y-%m"))
                continue

            df.to_parquet(
                file,
                engine="pyarrow",
                compression="zstd",
            )
            logger.success(f"{log_msg_prefix} Saved {len(df):,} rows.")

    logger.info("Download process finished.")
    if missing_data:
        logger.warning("Missing data summary:")
        for symbol, months in missing_data.items():
            logger.warning(f"  - {symbol}: {', '.join(months)}")

    logger.success("All operations complete.")


# --- Loading Data from Files ---


def load_tick_data(
    path,
    symbol,
    start_date,
    end_date,
    account_name,
    columns=None,
    compress=True,
    verbose=True,
):
    """
    Loads tick data from a partitioned Parquet structure after verifying account.

    Args:
        path (Union[str, Path]): The root folder where the data is stored.
        symbol (str): The financial instrument symbol to load.
        start_date (Union[str, dt, pd.Timestamp]): The start date of the desired data range.
        end_date (Union[str, dt, pd.Timestamp]): The end date of the desired data range.
        account_name (str): The account name to verify against the data directory.
        columns (Optional[list]): A list of specific columns to load. Loads all if None.
        compress (bool): If True, save memory by optimizing dtypes.
        verbose (bool): If True, logs detailed DataFrame info upon successful load.

    Returns:
        pd.DataFrame: A DataFrame with the requested tick data, or an empty DataFrame
                      if the account verification fails, dates are invalid, or an error occurs.
    """
    root_path = Path(path)
    if not verify_or_create_account_info(root_path, account_name):
        return pd.DataFrame()

    date_range = date_conversion(start_date, end_date)
    if not date_range:
        return pd.DataFrame()
    start_dt, end_dt = date_range

    fname = root_path / symbol

    try:
        filters = [("time", ">=", start_dt), ("time", "<=", end_dt)]

        ddf = dd.read_parquet(
            fname,
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )
        df = ddf.compute()

        to_drop = []
        for col in df.columns:
            # Drop columns
            if any(np.isnan(df[col].unique())):
                to_drop.append(col)
            # Optimise dtype of flags column for memory
            if col == "flags" and compress:
                mem = df.memory_usage(deep=True).sum()  # memory before downcasting
                dtype_orig = df["flags"].dtype
                limit = df["flags"].max()
                for x in (8, 16, 32):
                    dtype = f"uint{x}"
                    if dtype_orig != dtype and np.iinfo(dtype).max >= limit:
                        df = df.astype({"flags": dtype})
                        mem = (mem - df.memory_usage(deep=True).sum()) / 1024**2
                        logger.info(
                            f"Converted flags from {dtype_orig} to {df['flags'].dtype} saving {mem:,.1f} MB."
                        )
                        break

        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            logger.info(f"Dropped empty columns {to_drop}.")

        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        logger.success(f"Loaded {len(df):,} rows of {symbol} tick data for account {account_name}.")
        if verbose:
            log_df_info(df)

        return df

    except Exception as e:
        logger.error(f"Failed to load data for {symbol}. Error: {e}")
        return pd.DataFrame()


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

    # 3. Price validity checks (more comprehensive)
    price_filter = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] > df["bid"])(  # Spread must be positive
            df["ask"] - df["bid"] >= min_spread
        )  # Minimum spread filter
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


# --- Main Execution Block ---
if __name__ == "__main__":
    MAJORS = [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "XAUUSD",
    ]

    CRYPTO = [
        "ADAUSD",
        "BTCUSD",
        "DOGUSD",
        "ETHUSD",
        "LNKUSD",
        "LTCUSD",
        "XLMUSD",
        "XMRUSD",
        "XRPUSD",
    ]

    # --- 1. User Configuration ---
    CONFIG = {
        "save_path": Path.home() / "tick_data_parquet",
        "symbols_to_download": MAJORS + CRYPTO,
        "account_to_use": "FundedNext_STLR2_6K",  # This name MUST match the one used in your environment variables
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "verbose_login": True,
    }

    # --- 2. Setup Logging ---
    # Configure logger to output to console and a file for persistent records.
    log_path = CONFIG["save_path"] / "data_download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # The default logger is console-only. Add a file sink.
    logger.add(
        log_path,
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.info("--- Starting New Data Download Session ---")

    # --- 3. Login to MT5 ---
    logged_in_account = login_mt5(account=CONFIG["account_to_use"], verbose=CONFIG["verbose_login"])

    # --- 4. Run Downloader ---
    if logged_in_account:
        save_data_to_parquet(
            path=CONFIG["save_path"],
            symbols=CONFIG["symbols_to_download"],
            start_date=CONFIG["start_date"],
            end_date=CONFIG["end_date"],
            account_name=logged_in_account,
        )

        # --- 6. Shutdown MT5 Connection ---
        mt5.shutdown()
        logger.info("--- MT5 Connection Closed. Session End ---")

    else:
        logger.critical("Could not log in to MetaTrader 5. Aborting all operations.")
