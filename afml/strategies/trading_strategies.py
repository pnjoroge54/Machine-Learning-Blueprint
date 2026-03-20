from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import talib
from loguru import logger
from datetime import datetime
from pathlib import Path


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    # Base replacements applied to every strategy name
    DEFAULT_REPLACEMENTS = {"version": "v"}
    # Subclasses can override/extend this dictionary
    PARAM_REPLACEMENTS = {}

    def __init__(self, version: int = 1):
        self.version = version
        self.info = None  # will hold version description if added

    def get_strategy_name(self) -> str:
        """Returns a string identifying the class, its parameters, version, and feature count."""
        class_name = self.__class__.__name__.replace("Strategy", "")
        ignored = {'objective', 'info'}  # ignore fields not part of parameters
        params = {k: v for k, v in vars(self).items() if k not in ignored}
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])

        # Combine base replacements with subclass-specific ones
        replacements = {**self.__class__.DEFAULT_REPLACEMENTS, **self.__class__.PARAM_REPLACEMENTS}
        # Sort keys by length descending to avoid substring issues
        for key in sorted(replacements.keys(), key=len, reverse=True):
            param_str = param_str.replace(key, replacements[key])

        return f"{class_name}_{param_str}"

    def get_objective(self) -> str:
        """Return strategy objective from {'mean_reversion', 'trend-following', 'momentum', 'pairs'}"""
        return self.objective

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1 for long, -1 for short, 0 for no position)"""
        pass

    def add_info(self, info: str, mode: Optional[str] = None) -> None:
        """
        Add descriptive information about this strategy version.

        If `mode` is 'w' (write), the existing info is overwritten.
        If `mode` is 'a' (append), the new info is appended to the existing info.
        If `mode` is None and there is existing info, the user is prompted to choose
        between appending, overwriting, or cancelling.
        If `mode` is invalid, the user is informed and the prompt is shown again.
        """
        if self.info is None:
            self.info = info
            return

        # If mode is explicitly given, use it
        if mode == 'a':
            self.info += "\n" + info
        elif mode == 'w':
            self.info = info
        elif mode is None:
            # Prompt the user
            print("\nExisting info:")
            print(self.info)
            print("\nNew info to add:")
            print(info)
            choice = input("\nDo you want to (A)ppend, (O)verwrite, or (C)ancel? [A/O/C]: ").strip().upper()
            if choice == 'A':
                self.info += "\n" + info
            elif choice == 'O':
                self.info = info
            elif choice == 'C':
                print("Operation cancelled.")
            else:
                print("Invalid choice. No changes made.")
        else:
            # Invalid mode – inform user and restart prompt
            print(f"Invalid mode: {mode}. Use 'a', 'overwrite', or None.")
            self.add_info(info, mode=None)


class BollingerStrategy(BaseStrategy):
    """
    BollingerStrategy implements a mean reversion trading strategy using Bollinger Bands.
    Attributes:
        window (int): The lookback period for calculating Bollinger Bands.
        std (float): The number of standard deviations for the bands.
        objective (str): The strategy objective, default is "mean_reversion".
        version (int): Version number of the strategy.
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on Bollinger Bands. Returns a Series where
            1 indicates a buy signal (price below lower band), -1 indicates a sell signal
            (price above upper band), and 0 indicates no signal.
        add_info(info, mode=None): Adds descriptive info about the strategy version.
    """

    # Only need to specify strategy‑specific replacements
    PARAM_REPLACEMENTS = {"window": "w"}

    def __init__(
        self,
        window: int = 20,
        std: float = 2.0,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.window = window
        self.std = std
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals using Bollinger Bands"""
        close = data["close"]

        # Calculate Bollinger Bands
        upper_band, _, lower_band = talib.BBANDS(
            close, timeperiod=self.window, nbdevup=self.std, nbdevdn=self.std
        )

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(close >= upper_band)] = -1  # Sell signal (mean reversion)
        signals[(close <= lower_band)] = 1  # Buy signal (mean reversion)
        return signals


class MACrossoverStrategy(BaseStrategy):
    """
    MACrossoverStrategy implements a moving average crossover trend-following strategy.
    Attributes:
        fast_window (int): Window size for the fast moving average.
        slow_window (int): Window size for the slow moving average.
        objective (str): The objective of the strategy (default: "trend").
        version (int): Version number of the strategy.
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on the crossover of fast and slow moving averages.
            Returns a Series with values: 1 for long, -1 for short, and 0 for neutral.
        add_info(info, mode=None): Adds descriptive info about the strategy version.
    """

    # Replace fast_window and slow_window with empty strings (effectively remove them)
    PARAM_REPLACEMENTS = {"fast_window": "", "slow_window": ""}

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        objective: str = "trend",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals based on MA crossover"""
        close = data["close"]

        # Calculate moving averages
        fast_ma = talib.MA(close, self.fast_window)
        slow_ma = talib.MA(close, self.slow_window)

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(fast_ma > slow_ma)] = 1  # Long signal when fast MA crosses above slow MA
        signals[(fast_ma < slow_ma)] = -1  # Short signal when fast MA crosses below slow MA
        return signals
