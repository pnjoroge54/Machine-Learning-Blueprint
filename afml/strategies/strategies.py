from abc import ABC, abstractmethod

import pandas as pd
import talib


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1 for long, -1 for short, 0 for no position)"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        pass

    @abstractmethod
    def get_objective(self) -> str:
        """Return strategy objective"""
        pass


class BollingerStrategy(BaseStrategy):
    """
    BollingerStrategy implements a mean reversion trading strategy using Bollinger Bands.
    Attributes:
        window (int): The lookback period for calculating Bollinger Bands.
        num_std (float): The number of standard deviations for the bands.
        objective (str): The strategy objective, default is "mean_reversion".
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on Bollinger Bands. Returns a Series where
            1 indicates a buy signal (price below lower band), -1 indicates a sell signal
            (price above upper band), and 0 indicates no signal.
        get_strategy_name() -> str:
            Returns the name of the strategy including window and standard deviation parameters.
        get_objective() -> str:
            Returns the objective of the strategy.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, objective: str = "mean_reversion"):
        self.window = window
        self.num_std = num_std
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals using Bollinger Bands"""
        close = data["close"]

        # Calculate Bollinger Bands
        upper_band, _, lower_band = talib.BBANDS(
            close, timeperiod=self.window, nbdevup=self.num_std, nbdevdn=self.num_std
        )

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="side")
        signals[(close >= upper_band)] = -1  # Sell signal (mean reversion)
        signals[(close <= lower_band)] = 1  # Buy signal (mean reversion)
        return signals

    def get_strategy_name(self) -> str:
        return f"Bollinger_w{self.window}_std{self.num_std}"

    def get_objective(self) -> str:
        return self.objective


class MACrossoverStrategy(BaseStrategy):
    """
    MACrossoverStrategy implements a moving average crossover trend-following strategy.
    Attributes:
        fast_window (int): Window size for the fast moving average.
        slow_window (int): Window size for the slow moving average.
        objective (str): The objective of the strategy (default: "trend_following").
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on the crossover of fast and slow moving averages.
            Returns a Series with values: 1 for long, -1 for short, and 0 for neutral.
        get_strategy_name() -> str:
            Returns the name of the strategy, including the fast and slow window sizes.
        get_objective() -> str:
            Returns the objective of the strategy.
    """

    def __init__(
        self, fast_window: int = 10, slow_window: int = 30, objective: str = "trend_following"
    ):
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
        signals = pd.Series(0, index=data.index, dtype="int8", name="side")
        signals[(fast_ma > slow_ma)] = 1  # Long signal when fast MA crosses above slow MA
        signals[(fast_ma < slow_ma)] = -1  # Short signal when fast MA crosses below slow MA
        return signals

    def get_strategy_name(self) -> str:
        return f"MACrossover_{self.fast_window}_{self.slow_window}"

    def get_objective(self) -> str:
        return self.objective
