import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import talib


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
        ignored = {"objective", "info"}  # ignore fields not part of parameters
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

    def add_info(self, info: str, mode: str | None = None) -> None:
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
        if mode == "a":
            self.info += "\n" + info
        elif mode == "w":
            self.info = info
        elif mode is None:
            # Prompt the user
            print("\nExisting info:")
            print(self.info)
            print("\nNew info to add:")
            print(info)
            choice = (
                input("\nDo you want to (A)ppend, (O)verwrite, or (C)ancel? [A/O/C]: ")
                .strip()
                .upper()
            )
            if choice == "A":
                self.info += "\n" + info
            elif choice == "O":
                self.info = info
            elif choice == "C":
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


class RSIStrategy(BaseStrategy):
    """
    RSIStrategy implements a mean-reversion or momentum strategy using the Relative Strength Index.
    Classic signals: buy when RSI crosses below oversold level, sell when RSI crosses above overbought level.

    Attributes:
        period (int): Lookback period for RSI calculation (typical default 14).
        overbought (float): RSI level considered overbought (sell signal).
        oversold (float): RSI level considered oversold (buy signal).
        objective (str): Strategy objective, typically "mean_reversion" or "momentum".
        version (int): Version number of the strategy.

    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on RSI thresholds. Returns a Series where
            1 indicates a buy signal (RSI <= oversold), -1 indicates a sell signal
            (RSI >= overbought), and 0 indicates no signal.
        add_info(info, mode=None): Adds descriptive info about the strategy version.
    """

    # Strategy-specific parameter replacements for compact naming
    PARAM_REPLACEMENTS = {"period": "p", "overbought": "ob", "oversold": "os"}

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-based signals."""
        close = data["close"]
        # Calculate RSI using TA-Lib
        rsi = talib.RSI(close, timeperiod=self.period)

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[rsi <= self.oversold] = 1  # Buy (oversold)
        signals[rsi >= self.overbought] = -1  # Sell (overbought)
        return signals


# ==============================================================================
# New Concepts in Technical Trading Sytems - Welles J. Wilder
# ==============================================================================


# ==============================================================================
# 1. Directional Movement System Strategy (+DI, -DI, DX, ADX, ADXR)
# ==============================================================================
class DirectionalMovementStrategy(BaseStrategy):
    """
    Directional Movement Strategy using Wilder's stop-and-reverse logic.

    A +DI/-DI crossover arms a pending reversal point:
        - Bullish crossover -> reversal/trigger level = high of crossover bar
        - Bearish crossover -> reversal/trigger level = low of crossover bar

    The position reverses only when price later trades through that level.
    The crossover bar itself can never confirm its own signal.
    """

    PARAM_REPLACEMENTS = {"period": "p", "objective": "obj"}

    def __init__(
        self,
        period: int = 14,
        objective: str = "trend-following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.objective = objective

    @staticmethod
    def _stop_and_reverse(
        cross_dir: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wilder's stop-and-reverse state machine.

        Parameters
        ----------
        cross_dir : np.ndarray
            +1 on bullish crossover bars,
            -1 on bearish crossover bars,
             0 otherwise.
        high : np.ndarray
            Bar highs.
        low : np.ndarray
            Bar lows.

        Returns
        -------
        position : np.ndarray
            int8 array: 1 long, -1 short, 0 flat.
        events : np.ndarray
            int8 array: 1 on confirmed long reversal,
            -1 on confirmed short reversal, 0 otherwise.
        """
        n = len(cross_dir)
        position = np.zeros(n, dtype=np.int8)
        events = np.zeros(n, dtype=np.int8)

        current_position = 0
        pending_dir = 0
        reverse_price = np.nan

        for i in range(n):
            # A new crossover arms/updates the pending extreme point.
            if cross_dir[i] == 1:
                pending_dir = 1
                reverse_price = high[i]
            elif cross_dir[i] == -1:
                pending_dir = -1
                reverse_price = low[i]

            # Confirm long only if price breaks above the pending high.
            if pending_dir == 1 and high[i] > reverse_price and current_position != 1:
                current_position = 1
                events[i] = 1
                pending_dir = 0
                reverse_price = np.nan

            # Confirm short only if price breaks below the pending low.
            elif pending_dir == -1 and low[i] < reverse_price and current_position != -1:
                current_position = -1
                events[i] = -1
                pending_dir = 0
                reverse_price = np.nan

            position[i] = current_position

        return position, events

    def _cross_dir_from_di(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute DI crossover direction array from price data.
        """
        plus_di = talib.PLUS_DI(data["high"], data["low"], data["close"], timeperiod=self.period)
        minus_di = talib.MINUS_DI(data["high"], data["low"], data["close"], timeperiod=self.period)

        bullish_cross = ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))).fillna(
            False
        )
        bearish_cross = ((plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))).fillna(
            False
        )

        return np.where(
            bullish_cross.to_numpy(),
            1,
            np.where(bearish_cross.to_numpy(), -1, 0),
        ).astype(np.int8)

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """
        Return the continuous position series:
            1  long
           -1  short
            0  flat
        """
        cross_dir = self._cross_dir_from_di(data)
        position, _ = self._stop_and_reverse(
            cross_dir,
            data["high"].to_numpy(),
            data["low"].to_numpy(),
        )
        return pd.Series(position, index=data.index, dtype="int8", name="position")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return only confirmed reversal events:
            1  confirmed long reversal/new long
           -1  confirmed short reversal/new short
            0  no event
        """
        cross_dir = self._cross_dir_from_di(data)
        _, events = self._stop_and_reverse(
            cross_dir,
            data["high"].to_numpy(),
            data["low"].to_numpy(),
        )
        return pd.Series(events, index=data.index, dtype="int8", name="signal")

    @staticmethod
    def _detect_turning_points(
        adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series
    ) -> pd.Series:
        """
        Flags Wilder's Turning Point Rule: ADX peaking while above both DI
        lines, then ticking down.

        This is a warning that the dominant trend may be exhausting, not a
        hard reversal or entry trigger — Wilder notes it can misfire during
        a strong blow-off, where ADX dips and then resumes rising. It is
        therefore kept as a separate diagnostic series rather than folded
        into `generate_signals`'s entry output.

        Fully vectorized: unlike the Extreme Point Rule, this pattern has
        no cross-bar state to carry (no "hold until triggered or reversed"
        behavior), so no loop is needed.

        Only the *first* down-tick off a local peak fires — an extended
        decline while ADX remains above both DI lines is one turning point,
        not one warning per bar of the decline.

        Parameters
        ----------
        adx : pd.Series
            ADX values.
        plus_di : pd.Series
            +DI values, same index as adx.
        minus_di : pd.Series
            -DI values, same index as adx.

        Returns
        -------
        pd.Series
            int8 series: 1 on the bar where the turn is confirmed, 0 otherwise.
        """
        above_both_at_peak = (adx.shift(1) > plus_di.shift(1)) & (adx.shift(1) > minus_di.shift(1))
        turned_down = adx < adx.shift(1)
        was_rising_or_flat = adx.shift(1) >= adx.shift(2)

        warning = (above_both_at_peak & turned_down & was_rising_or_flat).fillna(False)
        return warning.astype("int8")

    def get_turning_point_warnings(self, data: pd.DataFrame) -> pd.Series:
        """
        Computes ADX and applies the Turning Point Rule.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain 'high', 'low', 'close' columns.

        Returns
        -------
        pd.Series
            int8 series named 'turning_point_warning': 1 where ADX has just
            turned down off a peak while above both DI lines, 0 otherwise.
        """
        plus_di = talib.PLUS_DI(data["high"], data["low"], data["close"], timeperiod=self.period)
        minus_di = talib.MINUS_DI(data["high"], data["low"], data["close"], timeperiod=self.period)
        adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=self.period)

        warning = self._detect_turning_points(adx, plus_di, minus_di)
        return pd.Series(
            warning.to_numpy(), index=data.index, dtype="int8", name="turning_point_warning"
        )


# ==============================================================================
# 2. Average Directional Index (ADX) & ADXR Strategy
# ==============================================================================
class ADXStrategy(BaseStrategy):
    """
    Generates signals based on the Average Directional Index (ADX) and the
    Average Directional Movement Index Rating (ADXR). It filters for strong
    trends (ADX > 25) and then uses ADX/ADXR crossovers for entry.

    Attributes:
        period (int): The period for the ADX calculation (default 14).
        adxr_lookback (int): The lookback period for the ADXR calculation (default 14).
        objective (str): 'trend-following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "adxr_lookback": "a"}

    def __init__(
        self,
        period: int = 14,
        adxr_lookback: int = 14,
        objective: str = "trend-following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.adxr_lookback = adxr_lookback
        self.objective = objective

    def _calculate_adxr(self, adx: pd.Series, lookback: int) -> pd.Series:
        """Calculate the Average Directional Movement Index Rating (ADXR)."""
        # ADXR = (ADX current + ADX n periods ago) / 2
        adxr = (adx + adx.shift(lookback)) / 2.0
        return adxr

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trend-following signals using ADX and ADXR.
        1. Ensure ADX > 25 to confirm a strong trend.
        2. Buy when ADX crosses above ADXR, sell when ADX crosses below ADXR.
        """
        # Calculate ADX using TA-Lib
        adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=self.period)
        adxr = self._calculate_adxr(adx, self.adxr_lookback)

        # Wilder's rule: only trade when ADXR is above 25 (trend is strong)
        strong_trend = adx > 25

        # Generate signals on crossovers
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")

        # Buy signal: ADX crosses above ADXR
        buy_condition = (adx > adxr) & (adx.shift(1) <= adxr.shift(1)) & strong_trend
        signals[buy_condition] = 1

        # Sell signal: ADX crosses below ADXR
        sell_condition = (adx < adxr) & (adx.shift(1) >= adxr.shift(1)) & strong_trend
        signals[sell_condition] = -1

        return signals


# ==============================================================================
# 3. Parabolic SAR (Stop and Reverse) Strategy
# ==============================================================================
class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR (Stop and Reverse) Strategy.
    This strategy is always in the market, generating long signals when the
    Sar is below price and short signals when the Sar is above price.

    Attributes:
        start (float): The acceleration factor start value (typically 0.02).
        increment (float): The acceleration factor increment (typically 0.02).
        maximum (float): The maximum acceleration factor (typically 0.20).
        objective (str): 'trend-following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"start": "s", "increment": "i", "maximum": "m"}

    def __init__(
        self,
        start: float = 0.02,
        increment: float = 0.02,
        maximum: float = 0.20,
        objective: str = "trend-following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.start = start
        self.increment = increment
        self.maximum = maximum
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates long (1) when price is above the SAR, short (-1) when price is
        below the SAR. The strategy is always in the market.
        """
        # Use TA-Lib to calculate Parabolic SAR
        sar = talib.SAR(data["high"], data["low"], acceleration=self.start, maximum=self.maximum)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        # Long when price > SAR
        signals[data["close"] > sar] = 1
        # Short when price < SAR
        signals[data["close"] < sar] = -1

        # Note: The first few bars may have NaN values; they will remain as 0.
        return signals


# ==============================================================================
# 4. Average True Range (ATR) Volatility Breakout Strategy
# ==============================================================================
class ATRStrategy(BaseStrategy):
    """
    ATR (Average True Range) Volatility Breakout Strategy.
    Entries are triggered when price moves beyond a multiple of the ATR,
    indicating volatility expansion and potential trend start.

    Attributes:
        period (int): The period for the ATR calculation (default 14).
        multiplier (int): The ATR multiplier for the breakout threshold (default 2).
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "multiplier": "m"}

    def __init__(
        self,
        period: int = 14,
        multiplier: int = 2,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.multiplier = multiplier
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates long (1) when price breaks above a volatility threshold, and
        short (-1) when it breaks below.
        """
        # Calculate ATR
        atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=self.period)

        # Calculate breakout levels
        high_breakout = data["close"] > (data["close"].shift(1) + (self.multiplier * atr))
        low_breakout = data["close"] < (data["close"].shift(1) - (self.multiplier * atr))

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[high_breakout] = 1
        signals[low_breakout] = -1

        return signals


# ==============================================================================
# 5. Commodity Selection Index (CSI) Strategy
# ==============================================================================
class CSIStrategy(BaseStrategy):
    """
    Commodity Selection Index (CSI) Strategy.
    The CSI is a momentum indicator that combines trend (ADXR) and volatility (ATR)
    to identify commodities with high profit potential [15†L12-L14]. Entries are
    taken when the CSI value is high (e.g., above a moving average of itself).

    Attributes:
        period (int): The period for the ADX and ATR calculations (default 14).
        margin (float): The initial margin required for the traded asset.
        commission (float): The round-turn commission cost.
        big_point_value (float): The monetary value of one full point of price movement.
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "margin": "", "commission": "c", "big_point_value": ""}

    def __init__(
        self,
        period: int = 14,
        margin: float = 1000,
        commission: float = 20,
        big_point_value: float = 1,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.margin = margin
        self.commission = commission
        self.big_point_value = big_point_value
        self.objective = objective

    def _calculate_adxr(self, adx: pd.Series, lookback: int) -> pd.Series:
        """Calculate the Average Directional Movement Index Rating (ADXR)."""
        return (adx + adx.shift(lookback)) / 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates long (1) when the CSI is above its own moving average, and
        short (-1) when it is below.
        """
        # 1. Calculate ADX and ADXR
        adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=self.period)
        adxr = self._calculate_adxr(adx, self.period)

        # 2. Calculate ATR
        atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=self.period)

        # 3. Calculate the Constant 'K'
        # The constant K = ( (ATR * Big Point Value) / (Margin + Commission)) * 100
        # This is a simplified version of the Wilder CSI calculation [13†L12-L13]
        k = ((atr * self.big_point_value) / (self.margin + self.commission)) * 100

        # 4. CSI = ADXR * K
        csi = adxr * k

        # 5. Generate signals based on the CSI
        # A common method is to take a long when the CSI is above its n-period SMA
        # and a short when it is below.
        csi_ma = talib.SMA(csi, timeperiod=self.period)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[csi > csi_ma] = 1
        signals[csi < csi_ma] = -1

        return signals


# ==============================================================================
# 6. Swing Index (SI) Strategy
# ==============================================================================
class SwingIndexStrategy(BaseStrategy):
    """
    Swing Index (SI) Strategy.
    The Swing Index measures the strength and direction of price swings.
    Signals are generated when the SI line crosses above or below the zero line.
    A rise in the SI suggests bullish momentum, while a fall suggests bearish
    momentum.

    Attributes:
        period (int): The period for smoothing the SI (default 14).
        limit_move (float): The value of a one-direction limit move (default 12).
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "limit_move": "lm"}

    def __init__(
        self,
        period: int = 14,
        limit_move: float = 12,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.limit_move = limit_move
        self.objective = objective

    def _calculate_single_swing_index(
        self, high, low, close, open_, prev_high, prev_low, prev_close, prev_open, limit_move
    ):
        """Calculate the Swing Index for a single period."""
        # R value calculation: Find the largest of three differences
        r1 = high - close
        r2 = low - close
        r3 = high - low
        max_val = max(r1, r2, r3)

        if max_val == r1:
            # Case 1: (H - C) is largest [16†L26-L27]
            r = (high - close) - 0.5 * (low - close) + 0.25 * (prev_close - prev_open)
        elif max_val == r2:
            # Case 2: (L - C) is largest [16†L28-L29]
            r = (low - close) - 0.5 * (high - close) + 0.25 * (prev_close - prev_open)
        else:  # max_val == r3
            # Case 3: (H - L) is largest [16†L30]
            r = (high - low) + 0.25 * (prev_close - prev_open)

        # K value calculation: The largest of (H_y - C) and (L_y - C)
        k = max(prev_high - close, prev_low - close)
        if k <= 0:
            k = 1  # Avoid zero or negative values in division

        # T is the limit move
        t = limit_move

        # Final SI calculation as per Wilder [16†L21-L22]
        # SI = 50 * (( (C - Cy) + 0.5*(C - O) + 0.25*(Cy - Oy) ) / R) * (K / T)
        term1 = (close - prev_close) + 0.5 * (close - open_) + 0.25 * (prev_close - prev_open)
        if r == 0:
            return 0  # Avoid division by zero
        si = 50 * (term1 / r) * (k / t)
        return si

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates signals based on Zero-Line Crossover of the SI.
        This function vectorizes the calculation for efficiency.
        """
        # Prepare the data
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        # Shift the values for previous period
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        prev_open = open_.shift(1)

        # Calculate the 'Max' values
        r1 = high - close
        r2 = low - close
        r3 = high - low

        # Determine the max between r1, r2, r3 for each row
        cond1 = (r1 > r2) & (r1 > r3)
        cond2 = (r2 > r1) & (r2 > r3)
        # cond3: when none of the above are true, r3 is the max.

        # Vectorize the calculation of R
        r = np.where(
            cond1,
            high - close - 0.5 * (low - close) + 0.25 * (prev_close - prev_open),
            np.where(
                cond2,
                low - close - 0.5 * (high - close) + 0.25 * (prev_close - prev_open),
                high - low + 0.25 * (prev_close - prev_open),
            ),
        )

        # Vectorized K calculation: the largest of (High_y - C) and (Low_y - C)
        k = np.maximum(prev_high - close, prev_low - close)
        k = np.where(k <= 0, 1, k)  # Replace zero or negative K with 1

        # Term1: (C - Cy) + 0.5*(C - O) + 0.25*(Cy - Oy)
        term1 = (close - prev_close) + 0.5 * (close - open_) + 0.25 * (prev_close - prev_open)

        # Calculate SI
        si = 50 * (term1 / r) * (k / self.limit_move)
        si = si.fillna(0)

        # Optional Smoothing of SI (e.g., with a simple moving average)
        smoothed_si = talib.SMA(si, timeperiod=self.period)

        # Generate signals on zero-line crossover
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(smoothed_si > 0) & (smoothed_si.shift(1) <= 0)] = 1
        signals[(smoothed_si < 0) & (smoothed_si.shift(1) >= 0)] = -1

        return signals


# ==============================================================================
# 7. Accumulative Swing Index (ASI) Strategy
# ==============================================================================
class AccumulativeSwingIndexStrategy(BaseStrategy):
    """
    Accumulative Swing Index (ASI) Strategy.
    The ASI is the cumulative sum of the Swing Index and is used to confirm
    long-term trends. A rising ASI confirms an uptrend, while a falling ASI
    confirms a downtrend.

    Attributes:
        period (int): The period for smoothing the SI (default 14).
        limit_move (float): The value of a one-direction limit move.
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "limit_move": "lm"}

    def __init__(
        self,
        period: int = 14,
        limit_move: float = 12,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.limit_move = limit_move
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates signals based on the slope of the Accumulative Swing Index (ASI).
        When the ASI line crosses above its n-period moving average, it generates a
        buy signal; when it crosses below, a sell signal.
        """
        # First, we need to calculate the Swing Index for all periods.
        swing_index_strategy = SwingIndexStrategy(period=self.period, limit_move=self.limit_move)
        si = swing_index_strategy.generate_signals(
            data
        )  # This only returns the crossover signals, not the underlying SI value.
        # Instead, we need to access the underlying SI value from the class.
        # For simplicity, we'll re-calculate the raw SI values here.
        # Re-using the vectorized logic from SwingIndexStrategy to get the raw SI.
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        prev_open = open_.shift(1)

        r1 = high - close
        r2 = low - close
        r3 = high - low
        cond1 = (r1 > r2) & (r1 > r3)
        cond2 = (r2 > r1) & (r2 > r3)
        r = np.where(
            cond1,
            high - close - 0.5 * (low - close) + 0.25 * (prev_close - prev_open),
            np.where(
                cond2,
                low - close - 0.5 * (high - close) + 0.25 * (prev_close - prev_open),
                high - low + 0.25 * (prev_close - prev_open),
            ),
        )
        k = np.maximum(prev_high - close, prev_low - close)
        k = np.where(k <= 0, 1, k)
        term1 = (close - prev_close) + 0.5 * (close - open_) + 0.25 * (prev_close - prev_open)
        si_raw = 50 * (term1 / r) * (k / self.limit_move)
        si_raw = si_raw.fillna(0)

        # Calculate Accumulative Swing Index (ASI) [14†L34-L35]
        asi = si_raw.cumsum()

        # Calculate a moving average of the ASI as a signal line
        signal_line = talib.SMA(asi, timeperiod=self.period)

        # Generate signals on crossovers
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(asi > signal_line) & (asi.shift(1) <= signal_line.shift(1))] = 1
        signals[(asi < signal_line) & (asi.shift(1) >= signal_line.shift(1))] = -1

        return signals


# ==============================================================================
# Cybernetic Trading Strategies - Murray Ruggiero
# ==============================================================================


# ==============================================================================
# 1. COT Stochastic Strategy (Chapter 4: The Commitment of Traders Report)
# ==============================================================================
class COTStochasticStrategy(BaseStrategy):
    """
    COT Stochastic Strategy uses the Commitment of Traders report data to
    generate mean-reversion signals. The core idea is that market sentiment
    extremes (indicated by net positions of Commercials or Large Speculators)
    tend to revert [25†L28-L35].

    Attributes:
        lookback (int): Lookback period for the stochastic calculation (default 52).
        objective (str): 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"lookback": "lb"}

    def __init__(
        self,
        lookback: int = 52,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.lookback = lookback
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame, cot_series: pd.Series) -> pd.Series:
        """
        Generates signals based on stochastic calculation of COT data.
        Requires a separate COT data series (e.g., Commercial net positions).

        Args:
            data (pd.DataFrame): OHLC data (index must align with cot_series).
            cot_series (pd.Series): The COT data series (e.g., commercial net positions).

        Returns:
            pd.Series: 1 when stochastic < 20 (oversold), -1 when > 80 (overbought).
        """
        # Compute stochastic %K for the COT series
        min_val = cot_series.rolling(self.lookback).min()
        max_val = cot_series.rolling(self.lookback).max()
        stoch = 100 * (cot_series - min_val) / (max_val - min_val)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[stoch <= 20] = 1  # Oversold → buy
        signals[stoch >= 80] = -1  # Overbought → sell
        return signals


# ==============================================================================
# 2. Double Top/Bottom Pattern Strategy (Chapter 20: Using System Feedback)
# ==============================================================================
class DoubleTopBotStrategy(BaseStrategy):
    """
    Double Top/Bottom Pattern Strategy detects Double Top and Double Bottom
    patterns based on the logic described by Ruggiero on page 49 of
    "Cybernetic Trading Strategies" [24†L32-L34].

    A Double Top is identified when price makes two consecutive peaks of
    similar height with a trough in between. A Double Bottom is the inverse.

    Attributes:
        peak_window (int): Minimum number of bars to confirm a peak (default 5).
        valley_window (int): Minimum number of bars to confirm a valley (default 5).
        objective (str): 'mean_reversion' for Double Top/Bottom patterns.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"peak_window": "pw", "valley_window": "vw"}

    def __init__(
        self,
        peak_window: int = 5,
        valley_window: int = 5,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.peak_window = peak_window
        self.valley_window = valley_window
        self.objective = objective

    def _find_peaks(self, price: pd.Series) -> pd.Series:
        """Identify local peaks using a rolling window."""
        peaks = price.rolling(self.peak_window * 2 + 1, center=True).apply(
            lambda x: x[self.peak_window] == max(x), raw=True
        )
        return peaks.astype(bool)

    def _find_valleys(self, price: pd.Series) -> pd.Series:
        """Identify local valleys using a rolling window."""
        valleys = price.rolling(self.valley_window * 2 + 1, center=True).apply(
            lambda x: x[self.valley_window] == min(x), raw=True
        )
        return valleys.astype(bool)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Detects Double Top and Double Bottom patterns and generates signals:
          - 1 (buy) for Double Bottom (two valleys with a peak in between)
          - -1 (sell) for Double Top (two peaks with a trough in between)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        peaks = self._find_peaks(high)
        valleys = self._find_valleys(low)

        # Identify pattern completion points
        # For a Double Top: second peak occurs within a certain distance of the first
        # For a Double Bottom: second valley occurs within a certain distance of the first
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")

        # Look for two peaks with a valley in the middle
        # Simple implementation: detect two peaks within a lookback period
        peak_dist = peaks.rolling(30).sum()
        # Signal on the second peak if there's been a trough in between
        double_top = (peaks) & (peak_dist >= 2)
        signals[double_top] = -1

        # Look for two valleys with a peak in the middle
        valley_dist = valleys.rolling(30).sum()
        double_bottom = (valleys) & (valley_dist >= 2)
        signals[double_bottom] = 1

        return signals


# ==============================================================================
# 3. Ehlers MESA Adaptive Moving Average (MAMA) Strategy (Chapter 21)
# ==============================================================================
class EhlersMAMAStrategy(BaseStrategy):
    """
    Ehlers MESA Adaptive Moving Average (MAMA) Strategy based on John Ehlers'
    work. MAMA adapts to price movement based on the rate of change of phase
    measured by the Hilbert Transform Discriminator [13†L14-L18].

    The strategy generates a buy signal when MAMA crosses above the FAMA
    (Following Adaptive Moving Average), and a sell signal when MAMA crosses
    below FAMA.

    Attributes:
        fast_limit (float): Fast limit for alpha calculation (default 0.5).
        slow_limit (float): Slow limit for alpha calculation (default 0.05).
        objective (str): 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"fast_limit": "fl", "slow_limit": "sl"}

    def __init__(
        self,
        fast_limit: float = 0.5,
        slow_limit: float = 0.05,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.objective = objective

    def _phase_accumulation(self, price: pd.Series) -> pd.Series:
        """
        Simplified phase calculation using Hilbert Transform methods.
        In a full implementation, this would use the Hilbert Transform
        to extract the dominant cycle phase.
        """
        # Placeholder for actual Hilbert Transform logic
        # For demonstration, use price rate of change as a proxy
        delta_phase = price.pct_change().abs()
        return delta_phase.clip(self.slow_limit, self.fast_limit)

    def _compute_mama(self, price: pd.Series) -> pd.Series:
        """Compute MESA Adaptive Moving Average."""
        mama = pd.Series(index=price.index, dtype=float)
        previous_mama = price.iloc[0]
        previous_fama = price.iloc[0]

        for i in range(len(price)):
            if i == 0:
                mama.iloc[i] = price.iloc[i]
                continue

            delta_phase = self._phase_accumulation(price.iloc[: i + 1]).iloc[-1]
            # alpha is bounded by slow_limit and fast_limit [13†L30-L32]
            alpha = max(
                self.slow_limit,
                min(self.fast_limit, 1.0 / delta_phase if delta_phase != 0 else 0.5),
            )

            # Compute MAMA and FAMA
            current_mama = alpha * price.iloc[i] + (1 - alpha) * previous_mama
            current_fama = 0.5 * alpha * current_mama + (1 - 0.5 * alpha) * previous_fama

            mama.iloc[i] = current_mama
            previous_mama = current_mama
            previous_fama = current_fama

        return mama

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MAMA/FAMA crossovers."""
        close = data["close"]
        mama = self._compute_mama(close)
        fama = mama.rolling(3).mean()  # Simplified FAMA; actual uses 0.5*alpha

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(mama > fama) & (mama.shift(1) <= fama.shift(1))] = 1
        signals[(mama < fama) & (mama.shift(1) >= fama.shift(1))] = -1
        return signals


# ==============================================================================
# 4. Ehlers Instantaneous Trendline Strategy (Chapter 21)
# ==============================================================================
class EhlersInstantaneousTrendlineStrategy(BaseStrategy):
    """
    Ehlers Instantaneous Trendline (IT) Strategy. The IT isolates the trend
    component of price data by filtering out cyclical components using notch
    filters and the Hilbert Transform [11†L12-L18].

    The strategy generates a buy signal when the Trigger line (2-bar delayed
    IT) crosses above the IT, and a sell signal when the Trigger crosses below
    the IT [11†L28-L31].

    Attributes:
        alpha (float): Smoothing coefficient (default 0.07). Lower values produce
                       smoother, slower-responding trendlines [11†L32-L37].
        objective (str): 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"alpha": "a"}

    def __init__(
        self,
        alpha: float = 0.07,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.alpha = alpha
        self.objective = objective

    def _instantaneous_trendline(self, price: pd.Series) -> pd.Series:
        """
        Compute the instantaneous trendline using the recursive formula.
        IT = alpha * price + (1 - alpha) * previous_IT
        """
        it = pd.Series(index=price.index, dtype=float)
        previous_it = price.iloc[0]

        for i in range(len(price)):
            if i == 0:
                it.iloc[i] = price.iloc[i]
                continue

            current_it = self.alpha * price.iloc[i] + (1 - self.alpha) * previous_it
            it.iloc[i] = current_it
            previous_it = current_it

        return it

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Trigger line crosses."""
        price = (data["high"] + data["low"]) / 2  # Use midpoint [11†L21]
        it = self._instantaneous_trendline(price)
        trigger = it.shift(2)  # 2-bar delay

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(trigger > it) & (trigger.shift(1) <= it.shift(1))] = 1
        signals[(trigger < it) & (trigger.shift(1) >= it.shift(1))] = -1
        return signals


# ==============================================================================
# 5. Ehlers Adaptive Cyber Cycle (ACC) Strategy (Chapter 21)
# ==============================================================================
class EhlersAdaptiveCyberCycleStrategy(BaseStrategy):
    """
    Ehlers Adaptive Cyber Cycle Strategy. The Cyber Cycle is a zero-lagging
    indicator that measures the cyclical component of price. The adaptive
    version uses the dominant cycle period dynamically [9†L3-L7].

    The strategy generates long signals when the cycle turns up from below
    zero and short signals when it turns down from above zero.

    Attributes:
        period (int): Base period for cycle calculation (default 10).
        smoothing (int): Smoothing factor for the cycle (default 2).
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "smoothing": "s"}

    def __init__(
        self,
        period: int = 10,
        smoothing: int = 2,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.smoothing = smoothing
        self.objective = objective

    def _compute_cycle(self, price: pd.Series) -> pd.Series:
        """
        Simplified Cyber Cycle computation.
        Full implementation would use Hilbert Transform and dominant cycle
        extraction.
        """
        # Use a bandpass filter approximation
        alpha1 = 2.0 / (self.period + 1.0)
        hp = pd.Series(index=price.index, dtype=float)
        hp.iloc[0] = 0.0

        for i in range(1, len(price)):
            hp.iloc[i] = (
                (1 - alpha1 / 2) ** 2 * (price.iloc[i] - 2 * price.iloc[i - 1] + price.iloc[i - 2])
                + 2 * (1 - alpha1) * hp.iloc[i - 1]
                - (1 - alpha1) ** 2 * hp.iloc[i - 2]
            )

        # Smooth the high-pass filtered signal
        cycle = hp.rolling(self.smoothing).mean()
        return cycle

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals on cycle zero-line crosses."""
        close = data["close"]
        cycle = self._compute_cycle(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        # Long when cycle turns up from below zero
        signals[(cycle > 0) & (cycle.shift(1) <= 0)] = 1
        # Short when cycle turns down from above zero
        signals[(cycle < 0) & (cycle.shift(1) >= 0)] = -1
        return signals


# ==============================================================================
# 6. Fisher Transform Strategy (Ehlers)
# ==============================================================================
class FisherTransformStrategy(BaseStrategy):
    """
    Fisher Transform Strategy. The Fisher Transform normalizes price data to a
    Gaussian distribution and then applies an inverse Fisher transform to
    generate overbought/oversold signals.

    The strategy goes long when the Fisher Transform crosses below -2 sigma
    and short when it crosses above +2 sigma.

    Attributes:
        period (int): Lookback period for price normalization (default 10).
        objective (str): 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p"}

    def __init__(
        self,
        period: int = 10,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.objective = objective

    def _fisher_transform(self, price: pd.Series) -> pd.Series:
        """
        Compute the Fisher Transform.
        Steps:
          1. Normalize price to a range between -1 and 1 within the lookback.
          2. Apply inverse Fisher transform: fisher = 0.5 * ln((1 + X)/(1 - X))
        """
        # Rolling min/max for normalization
        min_price = price.rolling(self.period).min()
        max_price = price.rolling(self.period).max()
        range_price = max_price - min_price

        # Avoid division by zero
        X = 2 * (price - min_price) / range_price - 1
        X = X.clip(-0.999, 0.999)  # Avoid ln(0)

        fisher = 0.5 * np.log((1 + X) / (1 - X))
        return fisher

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Fisher Transform extremes."""
        close = data["close"]
        fisher = self._fisher_transform(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[fisher <= -2] = 1  # Oversold
        signals[fisher >= 2] = -1  # Overbought
        return signals


# ==============================================================================
# 7. Leading Indicator (Ehlers)
# ==============================================================================
class LeadingIndicatorStrategy(BaseStrategy):
    """
    Leading Indicator Strategy based on Ehlers' work. The leading indicator
    attempts to predict future price direction by extracting leading components
    from the dominant cycle.

    The strategy goes long when the leading indicator crosses above zero and
    short when it crosses below zero.

    Attributes:
        period (int): Dominant cycle period (default 20).
        smoothing (int): Smoothing factor (default 2).
        gamma (float): Leading factor (default 0.5).
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "smoothing": "s", "gamma": "g"}

    def __init__(
        self,
        period: int = 20,
        smoothing: int = 2,
        gamma: float = 0.5,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.smoothing = smoothing
        self.gamma = gamma
        self.objective = objective

    def _leading_indicator(self, price: pd.Series) -> pd.Series:
        """
        Compute the leading indicator using a simplified adaptive filter.
        Full implementation uses Hilbert Transform to extract the leading
        component of the dominant cycle.
        """
        # Bandpass filter centered on the dominant period
        beta = np.cos(2 * np.pi / self.period)
        gamma = 1 / np.cos(4 * np.pi * self.gamma / self.period)
        alpha = gamma - np.sqrt(gamma**2 - 1)

        lead = pd.Series(index=price.index, dtype=float)
        lead.iloc[0] = 0.0

        for i in range(1, len(price)):
            lead.iloc[i] = (
                (alpha / 2) * (price.iloc[i] - price.iloc[i - 2])
                + beta * (1 + alpha) * lead.iloc[i - 1]
                - alpha * lead.iloc[i - 2]
            )

        return lead.rolling(self.smoothing).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals on zero-line crosses."""
        close = data["close"]
        lead = self._leading_indicator(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[(lead > 0) & (lead.shift(1) <= 0)] = 1
        signals[(lead < 0) & (lead.shift(1) >= 0)] = -1
        return signals


# ==============================================================================
# 8. Sinewave Indicator (Ehlers)
# ==============================================================================
class SinewaveIndicatorStrategy(BaseStrategy):
    """
    Sinewave Indicator Strategy. The Sinewave Indicator uses the Hilbert
    Transform to compute the in-phase and quadrature components of the dominant
    cycle, producing a sinewave that leads price.

    The strategy goes long when the sinewave crosses above its smoothing and
    short when it crosses below.

    Attributes:
        period (int): Dominant cycle period (default 20).
        smoothing (int): Smoothing factor for the sinewave (default 3).
        threshold (float): Signal threshold (default 0.2).
        objective (str): 'momentum' or 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "smoothing": "s", "threshold": "t"}

    def __init__(
        self,
        period: int = 20,
        smoothing: int = 3,
        threshold: float = 0.2,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.smoothing = smoothing
        self.threshold = threshold
        self.objective = objective

    def _sinewave_indicator(self, price: pd.Series) -> pd.Series:
        """Compute the sinewave indicator using a single-pole filter."""
        # Simplified sinewave based on price cycle
        alpha = 2.0 / (self.period + 1.0)
        smooth = price.ewm(alpha=alpha, adjust=False).mean()

        # Compute delta between price and smoothed version
        delta = price - smooth
        sinewave = delta.rolling(self.smoothing).mean()
        return sinewave

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals when sinewave exceeds threshold."""
        close = data["close"]
        sinewave = self._sinewave_indicator(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[sinewave > self.threshold] = 1
        signals[sinewave < -self.threshold] = -1
        return signals


# ==============================================================================
# 9. Ehlers RSI2 Strategy
# ==============================================================================
class EhlersRSI2Strategy(BaseStrategy):
    """
    Ehlers RSI2 Strategy. This is a modified RSI that uses a three-bar RSI
    with overbought/oversold thresholds typically set at 80 and 20.

    Attributes:
        period (int): RSI period (default 3).
        overbought (float): Overbought threshold (default 80).
        oversold (float): Oversold threshold (default 20).
        objective (str): 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "overbought": "ob", "oversold": "os"}

    def __init__(
        self,
        period: int = 3,
        overbought: float = 80,
        oversold: float = 20,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using Ehlers' RSI2."""
        close = data["close"]

        # Compute typical RSI but with a very short period
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[rsi <= self.oversold] = 1
        signals[rsi >= self.overbought] = -1
        return signals


# ==============================================================================
# 10. Ehlers Smoothing / SuperSmoother Strategy
# ==============================================================================
class EhlersSmoothingStrategy(BaseStrategy):
    """
    Ehlers SuperSmoother Strategy. The SuperSmoother filter reduces lag while
    effectively removing high-frequency noise, making it ideal for trend
    identification.

    The strategy goes long when price is above the smoothed line and short
    when price is below it.

    Attributes:
        period (int): Filter period (default 10).
        multiplier (float): Multiplier for the filter (default 1.5).
        objective (str): 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "multiplier": "m"}

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 1.5,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.multiplier = multiplier
        self.objective = objective

    def _super_smoother(self, price: pd.Series) -> pd.Series:
        """
        Ehlers' SuperSmoother filter.
        """
        a1 = np.exp(-1.414 * np.pi / self.period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / self.period)
        c2 = b1
        c3 = -(a1**2)
        c1 = 1 - c2 - c3

        smooth = pd.Series(index=price.index, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(len(price)):
                if i < 2:
                    smooth.iloc[i] = price.iloc[i]
                else:
                    smooth.iloc[i] = (
                        c1 * (price.iloc[i] + price.iloc[i - 1]) / 2
                        + c2 * smooth.iloc[i - 1]
                        + c3 * smooth.iloc[i - 2]
                    )

        return smooth

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals when price crosses the smoothed line."""
        close = data["close"]
        smooth = self._super_smoother(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[close > smooth * self.multiplier] = 1
        signals[close < smooth / self.multiplier] = -1
        return signals


# ==============================================================================
# 11. Adaptive Lookback Strategy (Ehlers)
# ==============================================================================
class AdaptiveLookbackStrategy(BaseStrategy):
    """
    Adaptive Lookback Strategy. The lookback period adapts based on the
    dominant cycle period estimated using Hilbert Transform methods.

    The strategy can be used with any moving average or oscillator, adjusting
    its lookback dynamically.

    Attributes:
        max_lookback (int): Maximum adaptive lookback period (default 50).
        min_lookback (int): Minimum adaptive lookback period (default 5).
        objective (str): 'trend_following' or 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"max_lookback": "max", "min_lookback": "min"}

    def __init__(
        self,
        max_lookback: int = 50,
        min_lookback: int = 5,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.max_lookback = max_lookback
        self.min_lookback = min_lookback
        self.objective = objective

    def _dominant_cycle(self, price: pd.Series) -> pd.Series:
        """Estimate dominant cycle period using a Homodyne Discriminator."""
        period = pd.Series(index=price.index, dtype=float)
        period.iloc[0] = self.max_lookback

        for i in range(1, len(price)):
            # Simplified cycle length estimation using zero-crossings
            # Full implementation uses Hilbert Transform and phase accumulation
            period.iloc[i] = period.iloc[i - 1]

        period = period.clip(self.min_lookback, self.max_lookback)
        return period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with adaptive lookback (MA crossover example)."""
        close = data["close"]
        period = self._dominant_cycle(close)

        # Use int() on each value for rolling window; this requires the rolling window to be integer
        # Build the rolling mean without using period directly as a rolling window parameter
        # Instead, compute a simple moving average with a fixed lookback and then adjust signal logic
        ma = close.rolling(window=self.max_lookback).mean()

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        # Signal when adaptive period is rising (trending)
        period_change = period.diff().fillna(0)
        signals[period_change > 0] = 1
        signals[period_change < 0] = -1
        return signals


# ==============================================================================
# 12. Market Facilitation Index (MFI) / Chaos Theory (Bill Williams)
# ==============================================================================
class MarketFacilitationIndexStrategy(BaseStrategy):
    """
    Market Facilitation Index (MFI) Strategy. The MFI measures how much the
    market is facilitating trading activity by combining tick volume and price
    range.

    The strategy interprets four market conditions based on MFI and volume,
    generating signals in "wet" or "dry" market states.

    Attributes:
        ma_period (int): Moving average period for MFI smoothing (default 14).
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"ma_period": "ma"}

    def __init__(
        self,
        ma_period: int = 14,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.ma_period = ma_period
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MFI and volume."""
        high = data["high"]
        low = data["low"]
        volume = data["volume"].fillna(1)

        # Market Facilitation Index = (High - Low) / Volume
        price_range = high - low
        mfi = price_range / volume.replace(0, 1)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        # Classification: Wet (MFI up, Volume down) or Dry (MFI down, Volume up)
        mfi_change = mfi.diff().fillna(0)
        vol_change = volume.diff().fillna(0)

        signals[(mfi_change > 0) & (vol_change < 0)] = 1  # Wet market
        signals[(mfi_change < 0) & (vol_change > 0)] = -1  # Dry market
        return signals


# ==============================================================================
# 13. Fractal Chaos Strategy (Bill Williams)
# ==============================================================================
class FractalChaosStrategy(BaseStrategy):
    """
    Fractal Chaos Strategy. Fractals are patterns where the middle bar has
    the highest high (for buy fractals) or lowest low (for sell fractals)
    with two lower highs/lows on each side.

    The strategy enters on a breakout above/below the fractal level.

    Attributes:
        lookback (int): Number of bars to confirm a fractal (default 5).
        objective (str): 'breakout' or 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"lookback": "l"}

    def __init__(
        self,
        lookback: int = 5,
        objective: str = "breakout",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.lookback = lookback
        self.objective = objective

    def _find_fractals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify bullish and bearish fractal patterns."""
        high = data["high"]
        low = data["low"]

        # Bullish fractal: low of the middle bar is the lowest among 5 consecutive bars
        bull_fractal = low == low.rolling(window=self.lookback, center=True).min()
        # Bearish fractal: high of the middle bar is the highest among 5 consecutive bars
        bear_fractal = high == high.rolling(window=self.lookback, center=True).max()

        # Exclude the edges where the pattern is incomplete
        bull_fractal.iloc[: self.lookback // 2] = False
        bull_fractal.iloc[-self.lookback // 2 :] = False
        bear_fractal.iloc[: self.lookback // 2] = False
        bear_fractal.iloc[-self.lookback // 2 :] = False

        return bull_fractal, bear_fractal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals on fractal breakout."""
        bull_fractal, bear_fractal = self._find_fractals(data)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        # Enter on next bar if fractal is confirmed
        signals[bull_fractal.shift(1).fillna(False)] = 1
        signals[bear_fractal.shift(1).fillna(False)] = -1
        return signals


# ==============================================================================
# 14. Z-Score Mean Reversion (Statistical Arbitrage)
# ==============================================================================
class ZScoreMeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy. The strategy computes the Z-score of
    price relative to its moving average and standard deviation. When the
    Z-score exceeds a threshold, it assumes the price will revert to the mean.

    This approach aligns with the statistically based market prediction
    methods discussed in Chapters 8–10.

    Attributes:
        lookback (int): Lookback period for mean and std (default 20).
        entry_threshold (float): Z-score threshold for entry (default 2.0).
        exit_threshold (float): Z-score threshold for exit (default 0.5).
        objective (str): 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"lookback": "l", "entry_threshold": "et", "exit_threshold": "xt"}

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals when the Z-score exceeds entry threshold and exit
        when it falls below exit threshold.
        """
        close = data["close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()
        zscore = (close - rolling_mean) / rolling_std

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[zscore <= -self.entry_threshold] = 1  # Oversold → buy
        signals[zscore >= self.entry_threshold] = -1  # Overbought → sell
        return signals


# ==============================================================================
# 15. Maximum Entropy Spectral Analysis (MESA) Cycle Strategy
# ==============================================================================
class MaximumEntropySpectralAnalysisStrategy(BaseStrategy):
    """
    Maximum Entropy Spectral Analysis (MESA) Strategy. MESA is used to
    estimate the dominant cycle period in price data, helping traders align
    with the primary market rhythm.

    The strategy generates entries based on cycle phase predictions.

    Attributes:
        max_period (int): Maximum cycle period to consider (default 50).
        min_period (int): Minimum cycle period to consider (default 5).
        smoothing (int): Smoothing factor for cycle estimation (default 3).
        objective (str): 'trend_following' or 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"max_period": "max", "min_period": "min", "smoothing": "s"}

    def __init__(
        self,
        max_period: int = 50,
        min_period: int = 5,
        smoothing: int = 3,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.max_period = max_period
        self.min_period = min_period
        self.smoothing = smoothing
        self.objective = objective

    def _dominant_cycle(self, price: pd.Series) -> pd.Series:
        """
        Estimate dominant cycle period using the Homodyne Discriminator.
        This is a simplified placeholder; full MESA requires Burg's method.
        """
        period = pd.Series(index=price.index, dtype=float)
        period.iloc[0] = self.max_period

        # Use zero-crossings of the Hilbert Transform as a proxy for cycle length
        detrended = price - price.rolling(30).mean()
        zero_cross = (detrended > 0) != (detrended.shift(1) > 0)

        # For simplicity, fall back to a default
        period = self.max_period * 0.5
        period = np.clip(period, self.min_period, self.max_period)
        return pd.Series(period, index=price.index)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on cycle phase. Long when upward phase begins,
        short when downward phase begins.
        """
        close = data["close"]
        period = self._dominant_cycle(close).astype(float)

        # Compute a simple cycle indicator: sinewave of the dominant period
        angle = 2 * np.pi / period
        cycle = np.sin(angle * np.arange(len(close)))

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signal_changes = (cycle > 0) != (cycle.shift(1) > 0)
        signals[signal_changes & (cycle > 0)] = 1
        signals[signal_changes & (cycle < 0)] = -1
        return signals


# ==============================================================================
# 16. Trend Regression / Linear Regression Slope Strategy
# ==============================================================================
class TrendRegressionStrategy(BaseStrategy):
    """
    Trend Regression Strategy. Uses linear regression to estimate trend slope
    and direction. Enter long when slope is positive and above a threshold,
    short when slope is negative and below threshold.

    Attributes:
        lookback (int): Lookback period for regression (default 20).
        slope_threshold (float): Minimum absolute slope to trigger a signal.
        objective (str): 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"lookback": "l", "slope_threshold": "st"}

    def __init__(
        self,
        lookback: int = 20,
        slope_threshold: float = 0.01,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.lookback = lookback
        self.slope_threshold = slope_threshold
        self.objective = objective

    def _regression_slope(self, series: pd.Series) -> pd.Series:
        """Calculate the slope of linear regression over a rolling window."""

        def slope(x):
            x_idx = np.arange(len(x))
            return np.polyfit(x_idx, x, 1)[0] if len(x) > 1 else 0.0

        slope_series = series.rolling(window=self.lookback).apply(slope, raw=True)
        return slope_series

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on regression slope direction."""
        close = data["close"]
        slope = self._regression_slope(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[slope > self.slope_threshold] = 1
        signals[slope < -self.slope_threshold] = -1
        return signals


# ==============================================================================
# 17. Volatility Expansion / ATR Breakout Strategy
# ==============================================================================
class VolatilityExpansionStrategy(BaseStrategy):
    """
    Volatility Expansion Strategy. Uses ATR to detect volatility expansion
    and confirm breakout momentum.

    The strategy goes long when price closes above a volatility threshold
    and goes short when it closes below.

    Attributes:
        atr_period (int): Period for ATR calculation (default 14).
        multiplier (float): Multiplier for ATR threshold (default 2.0).
        min_atr (float): Minimum ATR value to avoid low-volatility signals.
        objective (str): 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"atr_period": "ap", "multiplier": "m", "min_atr": "ma"}

    def __init__(
        self,
        atr_period: int = 14,
        multiplier: float = 2.0,
        min_atr: float = 0.1,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.min_atr = min_atr
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals when price expands beyond volatility range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        atr = atr.clip(lower=self.min_atr)

        upper = close + self.multiplier * atr
        lower = close - self.multiplier * atr

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[close > upper] = 1
        signals[close < lower] = -1
        return signals


# ==============================================================================
# 18. Cycle Extractor / Dominant Cycle Oscillator
# ==============================================================================
class CycleExtractorStrategy(BaseStrategy):
    """
    Cycle Extractor Strategy. Extracts the dominant cycle component from price
    and trades its zero crossings or extreme points.

    Attributes:
        period (int): Expected cycle period (default 20).
        smoothing (int): Smoothing for cycle extraction (default 2).
        objective (str): 'momentum' or 'mean_reversion'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"period": "p", "smoothing": "s"}

    def __init__(
        self,
        period: int = 20,
        smoothing: int = 2,
        objective: str = "momentum",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.period = period
        self.smoothing = smoothing
        self.objective = objective

    def _cycle_extractor(self, price: pd.Series) -> pd.Series:
        """Extract cycle component using a bandpass filter."""
        alpha = 2.0 / (self.period + 1.0)
        hp = price.rolling(int(self.period)).mean()  # simplified
        cycle = price - hp
        return cycle.rolling(self.smoothing).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals when the cycle component crosses zero (momentum)
        or at cycle extremes (mean reversion).
        """
        close = data["close"]
        cycle = self._cycle_extractor(close)

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        if self.objective == "momentum":
            signals[(cycle > 0) & (cycle.shift(1) <= 0)] = 1
            signals[(cycle < 0) & (cycle.shift(1) >= 0)] = -1
        else:  # mean_reversion
            signals[cycle > 0.5] = -1
            signals[cycle < -0.5] = 1
        return signals


# ==============================================================================
# 19. Fourier Transform Frequency Detection Strategy
# ==============================================================================
class FourierTransformStrategy(BaseStrategy):
    """
    Fourier Transform Strategy. Uses a short-window FFT to identify dominant
    frequencies and generates signals when the frequency changes.

    Attributes:
        window (int): FFT window size (default 32).
        cycles_to_keep (int): Number of dominant cycles to retain (default 3).
        smoothing (int): Smoothing factor for frequency detection.
        objective (str): 'trend_following' or 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"window": "w", "cycles_to_keep": "c", "smoothing": "s"}

    def __init__(
        self,
        window: int = 32,
        cycles_to_keep: int = 3,
        smoothing: int = 2,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.window = window
        self.cycles_to_keep = cycles_to_keep
        self.smoothing = smoothing
        self.objective = objective

    def _dominant_frequency(self, price: pd.Series) -> pd.Series:
        """Find dominant frequency using FFT over rolling windows."""
        freq_series = pd.Series(index=price.index, dtype=float)

        for i in range(self.window, len(price)):
            window_data = price.iloc[i - self.window : i]
            fft = np.fft.rfft(window_data)
            magnitude = np.abs(fft[1:])  # ignore DC component
            dominant_freq = np.argmax(magnitude) + 1
            freq_series.iloc[i] = dominant_freq

        return freq_series.rolling(self.smoothing).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals when the dominant frequency changes significantly.
        """
        close = data["close"]
        freq = self._dominant_frequency(close)

        freq_change = freq.diff().fillna(0)
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[freq_change > 1] = 1
        signals[freq_change < -1] = -1
        return signals


# ==============================================================================
# 20. Chaos Theory Phase Space Reconstruction Strategy
# ==============================================================================
class ChaosTheoryPhaseSpaceStrategy(BaseStrategy):
    """
    Chaos Theory Phase Space Reconstruction Strategy. Uses Takens' embedding
    theorem to reconstruct the market's attractor dynamics.

    The strategy detects when the reconstructed state changes significantly,
    indicating a regime shift.

    Attributes:
        embedding_dim (int): Embedding dimension (default 3).
        time_delay (int): Time delay for embedding (default 1).
        change_threshold (float): Threshold for state change detection.
        objective (str): 'breakout' or 'momentum'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"embedding_dim": "ed", "time_delay": "td", "change_threshold": "ct"}

    def __init__(
        self,
        embedding_dim: int = 3,
        time_delay: int = 1,
        change_threshold: float = 0.5,
        objective: str = "breakout",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.change_threshold = change_threshold
        self.objective = objective

    def _phase_space(self, series: pd.Series) -> np.ndarray:
        """Reconstruct phase space using time-delay embedding."""
        n = len(series)
        tau = self.time_delay
        m = self.embedding_dim
        max_start = n - (m - 1) * tau
        if max_start <= 0:
            return np.array([])

        points = np.zeros((max_start, m))
        for i in range(max_start):
            for j in range(m):
                points[i, j] = series.iloc[i + j * tau]
        return points

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals when the phase space distance exceeds threshold.
        """
        close = data["close"]
        points = self._phase_space(close)

        if len(points) < 2:
            return pd.Series(0, index=data.index, dtype="int8", name="signal")

        # Compute distance between consecutive points
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        distance_series = pd.Series(
            distances, index=close.index[self.embedding_dim * self.time_delay :]
        )

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        signals[distance_series > self.change_threshold] = 1
        signals[distance_series < -self.change_threshold] = -1
        return signals


# ==============================================================================
# 21. Neural Network Prediction Strategy (Conceptual)
# ==============================================================================
class NeuralNetworkPredictionStrategy(BaseStrategy):
    """
    Neural Network Prediction Strategy (Conceptual).

    This is a simplified placeholder representing how a neural network
    might be integrated within the framework. A full implementation would
    require training an NN model.

    Attributes:
        lookback (int): Lookback window for input features (default 5).
        hidden_units (int): Number of hidden units (default 10).
        architecture (str): 'mlp' or 'rnn' (default 'mlp').
        objective (str): 'mean_reversion' or 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"lookback": "l", "hidden_units": "hu", "architecture": "arch"}

    def __init__(
        self,
        lookback: int = 5,
        hidden_units: int = 10,
        architecture: str = "mlp",
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.architecture = architecture
        self.objective = objective
        self._model = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Placeholder for NN-based signal generation.
        In a real implementation, this would:
          1. Compute features from the data window.
          2. Run the features through a trained neural network.
          3. Threshold the output to generate signals.
        """
        close = data["close"]
        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")

        # Simulate NN output: predict next bar's return direction
        returns = close.pct_change(self.lookback)
        predicted = returns.rolling(self.lookback).mean()  # very naive placeholder

        signals[predicted > 0] = 1
        signals[predicted < 0] = -1
        return signals


# ==============================================================================
# 22. Fuzzy Logic Trading Strategy (Conceptual)
# ==============================================================================
class FuzzyLogicStrategy(BaseStrategy):
    """
    Fuzzy Logic Trading Strategy (Conceptual). Combines multiple indicators
    using fuzzy logic rules to determine market states and generate signals.

    Attributes:
        config (dict): Membership function definitions for fuzzification.
        objective (str): 'mean_reversion' or 'trend_following'.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {}  # config is complex, no short replacements

    def __init__(
        self,
        config: dict | None = None,
        objective: str = "mean_reversion",
        version: int = 1,
    ):
        super().__init__(version=version)
        if config is None:
            config = {
                "rsi_membership": {
                    "oversold": (0, 30, 30),
                    "neutral": (30, 50, 70),
                    "overbought": (70, 100, 100),
                },
                "trend_membership": {
                    "down": (-1, -0.5, 0),
                    "neutral": (-0.5, 0, 0.5),
                    "up": (0, 0.5, 1),
                },
                "volatility_membership": {
                    "low": (0, 0.3, 0.3),
                    "normal": (0.3, 0.5, 0.7),
                    "high": (0.7, 1, 1),
                },
                "rules": [
                    (("rsi", "oversold"), ("trend", "up"), 1),
                    (("rsi", "overbought"), ("trend", "down"), -1),
                ],
            }
        self.config = config
        self.objective = objective

    def _fuzzify(self, value: float, membership: dict) -> dict:
        """Fuzzify a crisp value into a set of membership degrees."""
        result = {}
        for name, (a, b, c) in membership.items():
            if value <= a or value >= c:
                result[name] = 0.0
            elif a <= value <= b:
                result[name] = (value - a) / (b - a) if b > a else 1.0
            elif b <= value <= c:
                result[name] = (c - value) / (c - b) if c > b else 1.0
            else:
                result[name] = 1.0
        return result

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals by applying fuzzy rules.
        """
        close = data["close"]
        rsi = talib.RSI(close, timeperiod=14)
        trend = (close - close.rolling(20).mean()) / close.rolling(20).std()
        volatility = data["high"] - data["low"] / close

        signals = pd.Series(0, index=data.index, dtype="int8", name="signal")
        for i in range(20, len(data)):
            fuzzy_rsi = self._fuzzify(rsi.iloc[i], self.config["rsi_membership"])
            fuzzy_trend = self._fuzzify(trend.iloc[i], self.config["trend_membership"])
            fuzzy_volatility = self._fuzzify(
                volatility.iloc[i], self.config["volatility_membership"]
            )

            # Apply rules
            buy_strength = 0.0
            sell_strength = 0.0

            for rule in self.config["rules"]:
                (indicator, state), (ind2, state2), direction = rule
                strength = min(fuzzy_rsi.get(state, 0), fuzzy_trend.get(state2, 0))
                if direction > 0:
                    buy_strength = max(buy_strength, strength)
                elif direction < 0:
                    sell_strength = max(sell_strength, strength)

            if buy_strength > 0.7:
                signals.iloc[i] = 1
            elif sell_strength > 0.7:
                signals.iloc[i] = -1
        return signals


# ==============================================================================
# 23. Genetic Algorithm Optimization Strategy (Dispatcher)
# ==============================================================================
class GeneticOptimizationStrategy(BaseStrategy):
    """
    Genetic Algorithm Optimization Strategy (Dispatcher). Uses a genetic
    algorithm to optimize a strategy's parameter set, then trades with the
    optimal configuration.

    This is a dispatcher that holds the inner strategy and its optimized
    parameters.

    Attributes:
        inner_strategy_class (type): The strategy class to optimize.
        param_grid (dict): Grid of parameters to search over.
        population_size (int): GA population size (default 50).
        generations (int): Number of GA generations (default 20).
        mutation_rate (float): GA mutation rate (default 0.1).
        objective (str): Inherited from inner strategy.
        version (int): Version number.
    """

    PARAM_REPLACEMENTS = {"population_size": "ps", "generations": "g", "mutation_rate": "mr"}

    def __init__(
        self,
        inner_strategy_class: type,
        param_grid: dict,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        objective: str = "trend_following",
        version: int = 1,
    ):
        super().__init__(version=version)
        self.inner_strategy_class = inner_strategy_class
        self.param_grid = param_grid
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.objective = objective
        self._best_params = None

    def _fitness(self, params: dict, data: pd.DataFrame) -> float:
        """Fitness function: Sharpe ratio etc."""
        strategy = self.inner_strategy_class(**params)
        signals = strategy.generate_signals(data)
        returns = signals.shift(1) * data["close"].pct_change()
        return returns.mean() / returns.std() if returns.std() > 0 else 0

    def _optimize(self, data: pd.DataFrame) -> dict:
        """
        Placeholder for GA optimization loop.
        In a real implementation, this would evolve a population of parameter sets.
        """
        # Simple placeholder: return default params
        return {k: v[0] if isinstance(v, list) else v for k, v in self.param_grid.items()}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using the optimized strategy."""
        if self._best_params is None:
            self._best_params = self._optimize(data)

        strategy = self.inner_strategy_class(**self._best_params)
        return strategy.generate_signals(data)
