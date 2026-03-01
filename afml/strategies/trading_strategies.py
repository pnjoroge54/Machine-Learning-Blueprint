from abc import ABC, abstractmethod

import cloudpickle
import numpy as np
import pandas as pd
import talib
import yaml
from loguru import logger
from datetime import datetime
from pathlib import Path


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies with automated feature versioning.

    This class acts as a central hub for strategy logic and data preprocessing. 
    It features a built-in tracker that prevents duplicate feature engineering 
    logic from being added to the pipeline by comparing function bytecode.

    Attributes:
        features (dict): A registry of unique feature functions. 
            Format: {"0": {"func": callable, "name": str, "added_at": str}, ...}
        _warned (bool): Internal flag to prevent redundant logging of feature-less warnings.

    Example:
        >>> # 1. Define a strategy inheriting from BaseStrategy
        >>> class MyStrategy(BaseStrategy):
        ...     def __init__(self, period=14):
        ...         super().__init__()
        ...         self.period = period
        ...     def generate_signals(self, data):
        ...         data = self.apply_features(data)
        ...         return (data['rsi'] < 30).astype(int)
        ...     def get_objective(self): return "momentum"
        ...
        >>> # 2. Instantiate and register features
        >>> strat = MyStrategy(period=21)
        >>> 
        >>> @strat.register_feature
        ... def rsi(df):
        ...     import talib
        ...     return talib.RSI(df['close'], timeperiod=21)
        ...
        >>> # 3. Attempting to register the same logic again will trigger a skip
        >>> @strat.register_feature
        ... def duplicate_rsi(df):
        ...     import talib
        ...     return talib.RSI(df['close'], timeperiod=21) 
        >>> # Output: DEBUG | Skipped duplicate: duplicate_rsi
    """
    def __init__(self):
        self.features = {}
        self._warned = False
        # Automatically determine the path of the file where the class is defined
        self.base_path = Path.home() / "strategy_configs"

    def export_strategy(self):
        """Saves metadata and logic using the smart naming convention."""
        # Use class name (minus 'Strategy') as a category folder
        category = self.__class__.__name__.replace("Strategy", "")
        export_dir = self.base_path / category / self.get_strategy_name()
        export_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "strategy": self.get_strategy_name(),
            "params": {k: v for k, v in vars(self).items() if k not in {'features', '_warned', 'base_path'}},
            "audit": {v: info["name"] for v, info in self.features.items()}
        }

        with open(export_dir / "config.yaml", 'w') as f:
            yaml.dump(metadata, f)

        with open(export_dir / "logic.pkl", 'wb') as f:
            cloudpickle.dump(self.features, f)
            
        logger.info(f"💾 Strategy exported to: {export_dir}")

    def load_logic(self, tag: str = None):
        """
        Reloads logic from the automated export path.
        
        Args:
            tag (str): The folder name (e.g., 'BollingerStrategy_window:20_f2').
                      If None, defaults to the current strategy's auto-generated name.
        """
        # 1. Determine the category (e.g., 'Bollinger')
        category = self.__class__.__name__.replace("Strategy", "")
        
        # 2. Determine the specific folder to look in
        target_folder = tag if tag else self.get_strategy_name()
        
        # 3. Construct the path
        logic_file = self.base_path / category / target_folder / "logic.pkl"
        
        if logic_file.exists():
            with open(logic_file, 'rb') as f:
                self.features = cloudpickle.load(f)
            logger.info(f"📂 Logic restored from: {target_folder}")
        else:
            logger.error(f"❌ No logic file found at {logic_file}")
            
    def _get_test_data(self):
        """Generates dummy data that includes all currently registered features."""
        # Start with base OHLC
        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 210, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Apply existing features so the new one can "see" them
        for info in self.features.values():
            try:
                df[info["name"]] = info["func"](df)
            except Exception:
                continue # Skip if test data causes an error in existing features
        return df
        
    def _validate_feature(self, func):
        """Validates that the function works with the current feature set."""
        test_df = self._get_test_data()
        
        try:
            result = func(test_df)
            
            # Check for column existence (KeyError prevention)
            if not isinstance(result, pd.Series):
                raise TypeError(f"Must return pd.Series")
            
            if len(result) != len(test_df):
                raise ValueError(f"Length mismatch")
                
            return True
        except KeyError as e:
            logger.error(f"Dependency Missing: Feature '{func.__name__}' requires column {e}")
            return False
        except Exception as e:
            logger.error(f"Validation failed for '{func.__name__}': {e}")
            return False
            
    def register_feature(self, func):
        """
        Decorator to register a unique feature function in the strategy's dictionary AND data validation.
        
        Args:
            func (callable): A function that takes a DataFrame and returns a Series.
            
        Returns:
            callable: The original function, allowing it to be used normally.
        """
        # Check logic first
        new_code = func.__code__.co_code
        if any(f["func"].__code__.co_code == new_code for f in self.features.values()):
            logger.debug(f"Skipped duplicate: {func.__name__}")
            return func

        # Validate data integrity
        if self._validate_feature(func):
            version_key = str(len(self.features))
            self.features[version_key] = {
                "func": func,
                "name": func.__name__,
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"✅ Registered & Validated v{version_key}: {func.__name__}")
        
        return func

    def apply_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies all registered functions to the provided DataFrame."""
        if not self.features and not self._warned:
            logger.warning(f"[{self.__class__.__name__}] running with 0 features.")
            self._warned = True
            
        df = data.copy()
        for version, info in self.features.items():
            df[info["name"]] = info["func"](df)
        return df

    def get_strategy_name(self) -> str:
        """Returns a string identifying the class, its parameters, and feature count."""
        class_name = self.__class__.__name__
        ignored = {'features', '_warned'}
        params = {k: v for k, v in vars(self).items() if k not in ignored}
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        return f"{class_name}({param_str})_f{len(self.features)}"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Abstract: Must be implemented to define entry/exit logic."""
        pass

    @abstractmethod
    def get_objective(self) -> str:
        """Abstract: Must return 'mean_reversion', 'trend', 'momentum', or 'pairs'."""
        pass
        

class BollingerStrategy(BaseStrategy):
    """
    BollingerStrategy implements a mean reversion trading strategy using Bollinger Bands.
    Attributes:
        window (int): The lookback period for calculating Bollinger Bands.
        std (float): The number of standard deviations for the bands.
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

    def __init__(
        self, window: int = 20, std: float = 2.0, objective: str = "mean_reversion"
    ):
        super().__init__()
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

    def get_objective(self) -> str:
        return self.objective


class MACrossoverStrategy(BaseStrategy):
    """
    MACrossoverStrategy implements a moving average crossover trend-following strategy.
    Attributes:
        fast_window (int): Window size for the fast moving average.
        slow_window (int): Window size for the slow moving average.
        objective (str): The objective of the strategy (default: "trend").
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
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        objective: str = "trend",
    ):
        super().__init__()
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
        signals[(fast_ma > slow_ma)] = (
            1  # Long signal when fast MA crosses above slow MA
        )
        signals[
            (fast_ma < slow_ma)
        ] = -1  # Short signal when fast MA crosses below slow MA
        return signals

    def get_objective(self) -> str:
        return self.objective
