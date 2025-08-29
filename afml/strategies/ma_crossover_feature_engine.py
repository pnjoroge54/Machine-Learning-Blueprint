from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import talib

from ..labeling.trend_scanning import get_bins_from_trend


class ForexFeatureEngine:
    """
    Feature engineering specifically designed for forex MA crossover strategies
    Focus on currency-specific patterns, central bank influences, and 24-hour market dynamics
    """

    def __init__(self, pair_name: str = "EURUSD"):
        self.pair_name = pair_name
        self.base_currency = pair_name[:3]
        self.quote_currency = pair_name[3:]

    def calculate_all_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None,
        additional_pairs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all forex-relevant features for MA crossover strategy

        Parameters:
        price_data: DataFrame with OHLC columns
        volume_data: Series with volume data (if available, often limited in forex)
        additional_pairs: Dict of related currency pair OHLC data for correlation features

        Returns:
        DataFrame with all calculated features
        """

        features = pd.DataFrame(index=price_data.index)
        close = price_data["close"]
        high = price_data["high"]
        low = price_data["low"]
        open_price = price_data["open"]

        # Core MA Features
        ma_features = self._calculate_ma_features(close)
        features = pd.concat([features, ma_features], axis=1)

        # Volatility & Range Features (Critical for Forex)
        vol_features = self._calculate_volatility_features(price_data)
        features = pd.concat([features, vol_features], axis=1)

        # Trend Strength Features
        trend_features = self._calculate_trend_features(price_data)
        features = pd.concat([features, trend_features], axis=1)

        # Session-Based Features (Forex-Specific)
        session_features = self._calculate_session_features(price_data)
        features = pd.concat([features, session_features], axis=1)

        # Currency Strength Features
        if additional_pairs:
            strength_features = self._calculate_currency_strength_features(close, additional_pairs)
            features = pd.concat([features, strength_features], axis=1)

        # Risk Environment Features
        risk_features = self._calculate_risk_environment_features(price_data)
        features = pd.concat([features, risk_features], axis=1)

        # Time-Based Features (Important for 24h markets)
        time_features = self._calculate_time_features(price_data.index)
        features = pd.concat([features, time_features], axis=1)

        # Market Structure Features
        structure_features = self._calculate_market_structure_features(price_data)
        features = pd.concat([features, structure_features], axis=1)

        return features.fillna(method="ffill").fillna(0)

    def _calculate_ma_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Core moving average features optimized for forex"""
        close = price_data["close"]
        features = pd.DataFrame(index=close.index)

        # Multiple MA periods suitable for forex
        ma_periods = [10, 20, 50, 100, 200]

        # Calculate MAs
        mas = {}
        for period in ma_periods:
            mas[period] = close.rolling(period).mean()
            features[f"ma_{period}"] = mas[period]

        # MA Crossover Signals
        features["ma_10_20_cross"] = np.where(mas[10] > mas[20], 1, -1)
        features["ma_20_50_cross"] = np.where(mas[20] > mas[50], 1, -1)
        features["ma_50_200_cross"] = np.where(mas[50] > mas[200], 1, -1)

        # MA Spreads (normalized by ATR)
        atr = self._calculate_atr(price_data)
        features["ma_spread_10_20"] = (mas[10] - mas[20]) / (atr + 1e-8)
        features["ma_spread_20_50"] = (mas[20] - mas[50]) / (atr + 1e-8)
        features["ma_spread_50_200"] = (mas[50] - mas[200]) / (atr + 1e-8)

        # MA Slopes (trend strength)
        for period in [20, 50]:
            features[f"ma_{period}_slope"] = mas[period].pct_change(5)

        # Price position relative to MAs
        features["price_above_ma_20"] = (close > mas[20]).astype(int)
        features["price_above_ma_50"] = (close > mas[50]).astype(int)

        # MA Ribbon (all MAs aligned)
        ma_alignment = 1
        for i in range(len(ma_periods) - 1):
            ma_alignment *= np.where(mas[ma_periods[i]] > mas[ma_periods[i + 1]], 1, -1)
        features["ma_ribbon_aligned"] = ma_alignment

        return features

    def _calculate_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Volatility features critical for forex risk management"""
        features = pd.DataFrame(index=price_data.index)

        close = price_data["close"]
        high = price_data["high"]
        low = price_data["low"]

        # ATR (multiple periods)
        features["atr_14"] = self._calculate_atr(price_data, 14)
        features["atr_21"] = self._calculate_atr(price_data, 21)

        # ATR Regime
        atr_ma = features["atr_14"].rolling(50).mean()
        features["atr_regime"] = features["atr_14"] / (atr_ma + 1e-8)

        # Realized Volatility
        returns = close.pct_change()
        features["realized_vol_10"] = returns.rolling(10).std() * np.sqrt(252)
        features["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(252)

        # Volatility of Volatility
        features["vol_of_vol"] = features["realized_vol_20"].rolling(20).std()

        # high-low Range Features
        features["hl_range"] = (high - low) / close
        features["hl_range_ma"] = features["hl_range"].rolling(20).mean()
        features["hl_range_regime"] = features["hl_range"] / (features["hl_range_ma"] + 1e-8)

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        features["bb_upper"] = bb_ma + (bb_std_val * bb_std)
        features["bb_lower"] = bb_ma - (bb_std_val * bb_std)
        features["bb_position"] = (close - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"] + 1e-8
        )
        features["bb_squeeze"] = bb_std_val / (bb_std_val.rolling(50).mean() + 1e-8)

        return features

    def _calculate_trend_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Trend strength and quality features"""
        features = pd.DataFrame(index=price_data.index)

        close = price_data["close"]
        high = price_data["high"]
        low = price_data["low"]

        # ADX (Average Directional Index)
        adx_features = self._calculate_adx(high, low, close)
        features = pd.concat([features, adx_features], axis=1)

        # Efficiency Ratio (trending vs ranging)
        features["efficiency_ratio_14"] = self._calculate_efficiency_ratio(close, 14)
        features["efficiency_ratio_30"] = self._calculate_efficiency_ratio(close, 30)

        # Linear Regression Features
        for period in [20, 50]:
            lr_features = self._calculate_linear_regression_features(close, period)
            for key, value in lr_features.items():
                features[f"{key}_{period}"] = value

        # Momentum Features
        features["roc_10"] = close.pct_change(10)
        features["roc_20"] = close.pct_change(20)
        features["momentum_14"] = close / close.shift(14) - 1

        # Higher Highs, Lower Lows
        features["hh_ll_20"] = self._calculate_hh_ll_count(high, low, 20)

        # Trend Persistence
        returns = close.pct_change()
        features["trend_persistence"] = returns.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )

        return features

    def _calculate_session_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Forex session-specific features"""
        features = pd.DataFrame(index=price_data.index)

        # Assuming UTC timestamps
        hour = price_data.index.hour

        # Session Identification
        features["asian_session"] = ((hour >= 23) | (hour < 8)).astype(int)
        features["london_session"] = ((hour >= 7) & (hour < 16)).astype(int)
        features["ny_session"] = ((hour >= 13) & (hour < 22)).astype(int)
        features["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)  # London-NY overlap

        # Session-based volatility
        close = price_data["close"]
        returns = close.pct_change()

        # Calculate session-specific volatility patterns
        for session in ["asian_session", "london_session", "ny_session"]:
            session_mask = features[session] == 1
            if session_mask.sum() > 0:
                session_vol = returns[session_mask].rolling(20, min_periods=1).std()
                features[f"{session}_vol"] = session_vol.reindex(price_data.index, method="ffill")

        # Current session relative volatility
        current_vol = returns.rolling(10).std()
        features["session_vol_ratio"] = current_vol / (returns.rolling(50).std() + 1e-8)

        return features

    def _calculate_currency_strength_features(
        self, close: pd.Series, additional_pairs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Currency strength analysis using multiple pairs"""
        features = pd.DataFrame(index=close.index)

        # Calculate individual currency strength
        base_strength = 0
        quote_strength = 0
        pair_count = 0

        for pair_name, pair_data in additional_pairs.items():
            if len(pair_data) == 0:
                continue

            pair_close = pair_data["close"].reindex(close.index, method="ffill")
            pair_returns = pair_close.pct_change(10)

            if self.base_currency in pair_name[:3]:
                if pair_name[:3] == self.base_currency:
                    base_strength += pair_returns
                else:
                    base_strength -= pair_returns
                pair_count += 1

            if self.quote_currency in pair_name:
                if pair_name[3:] == self.quote_currency:
                    quote_strength -= pair_returns
                else:
                    quote_strength += pair_returns

        if pair_count > 0:
            features["base_currency_strength"] = base_strength / pair_count
            features["quote_currency_strength"] = quote_strength / pair_count
            features["currency_strength_diff"] = (
                features["base_currency_strength"] - features["quote_currency_strength"]
            )

        # Correlation with major pairs (if available)
        if "EURUSD" in additional_pairs and self.pair_name != "EURUSD":
            eurusd_returns = additional_pairs["EURUSD"]["close"].pct_change()
            pair_returns = close.pct_change()
            features["correlation_eurusd"] = pair_returns.rolling(50).corr(eurusd_returns)

        return features

    def _calculate_risk_environment_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Risk environment and market stress indicators"""
        features = pd.DataFrame(index=price_data.index)

        close = price_data["close"]
        returns = close.pct_change()

        # Risk-on/Risk-off proxy using return patterns
        features["return_skew_20"] = returns.rolling(20).skew()
        features["return_kurtosis_20"] = returns.rolling(20).kurt()

        # Tail risk measures
        features["var_95"] = returns.rolling(50).quantile(0.05)  # 5% VaR
        features["cvar_95"] = returns[returns <= features["var_95"]].rolling(50).mean()

        # Market stress proxy (high volatility + negative skew)
        vol_20 = returns.rolling(20).std()
        skew_20 = returns.rolling(20).skew()
        features["market_stress"] = (vol_20 > vol_20.rolling(100).quantile(0.8)) & (skew_20 < -0.5)
        features["market_stress"] = features["market_stress"].astype(int)

        # Drawdown features
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        features["current_drawdown"] = (cum_returns - rolling_max) / rolling_max
        features["days_since_high"] = (
            (cum_returns < rolling_max)
            .astype(int)
            .groupby((cum_returns == rolling_max).cumsum())
            .cumsum()
        )

        return features

    def _calculate_time_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Time-based features for 24-hour forex markets"""
        features = pd.DataFrame(index=index)

        # Basic time features
        features["hour"] = index.hour
        features["day_of_week"] = index.dayofweek
        features["day_of_month"] = index.day
        features["month"] = index.month

        # Cyclical encoding for continuous time features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        # Key forex timing patterns
        features["friday_ny_close"] = (
            (features["day_of_week"] == 4) & (features["hour"] >= 21)
        ).astype(int)
        features["sunday_open"] = ((features["day_of_week"] == 6) & (features["hour"] <= 2)).astype(
            int
        )
        features["month_end"] = (features["day_of_month"] >= 28).astype(int)
        features["quarter_end"] = (
            (features["month"] % 3 == 0) & (features["day_of_month"] >= 28)
        ).astype(int)

        return features

    def _calculate_market_structure_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure and pattern features"""
        features = pd.DataFrame(index=price_data.index)

        close = price_data["close"]
        high = price_data["high"]
        low = price_data["low"]
        open_price = price_data["open"]

        # Candlestick patterns
        features["doji"] = (abs(close - open_price) <= (high - low) * 0.1).astype(int)
        features["hammer"] = (
            (close > open_price)
            & ((close - open_price) > 2 * (high - close))
            & ((open_price - low) > 2 * (close - open_price))
        ).astype(int)

        # Price action features
        features["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        features["outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)

        # Support/Resistance levels (simplified)
        features["near_recent_high"] = (close >= high.rolling(20).max() * 0.999).astype(int)
        features["near_recent_low"] = (close <= low.rolling(20).min() * 1.001).astype(int)

        # Fractal patterns
        fractal_features = self._calculate_enhanced_fractal_features(price_data)
        features = features.join(fractal_features)

        return features

    # Helper functions
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range calculation"""
        return price_data.ta.atr(period)

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.DataFrame:
        """ADX calculation"""
        features = pd.DataFrame(index=close.index)

        features["adx"] = talib.ADX(high, low, close, period)
        features["plus_di"] = talib.PLUS_DI(high, low, close, period)
        features["minus_di"] = talib.MINUS_DI(high, low, close, period)

        return features

    def _calculate_efficiency_ratio(self, close: pd.Series, period: int) -> pd.Series:
        """Efficiency Ratio (directional movement / total movement)"""
        direction = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(period).sum()
        return direction / (volatility + 1e-8)

    def _calculate_linear_regression_features(
        self, close: pd.Series, period: int
    ) -> Dict[str, pd.Series]:
        lr_results = get_bins_from_trend(
            close, span=[period], volatility_threshold=0.0, lookforward=False
        )
        if lr_results.shape[1] > 0:
            lr_results = lr_results[["slope", "rsquared", "tval"]]
        else:
            lr_results = pd.DataFrame(0, index=close.index, columns=["slope", "rsquared", "tval"])
        lr_results.columns = [f"lr_{col}" for col in lr_results.columns]
        return lr_results.to_dict(orient="series")

    def _calculate_hh_ll_count(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Count of higher highs and lower lows"""
        hh = (high > high.shift(1)).rolling(period).sum()
        ll = (low < low.shift(1)).rolling(period).sum()
        return hh - ll  # Positive = more higher highs, Negative = more lower lows

    def _calculate_fractals(self, price: pd.Series, type_: str, n: int = 2) -> pd.Series:
        """Simple fractal calculation"""
        if type_ == "high":
            fractals = (price == price.rolling(2 * n + 1, center=True).max()).astype(int)
        else:  # low
            fractals = (price == price.rolling(2 * n + 1, center=True).min()).astype(int)
        return fractals

        # In your feature engineering

    def _calculate_enhanced_fractal_features(self, price_data: pd.DataFrame):
        high = price_data["high"]
        low = price_data["low"]

        fractal_features = calculate_enhanced_fractals(high, low)

        # Fractal-based trend confirmation
        close = price_data["close"]
        features = pd.DataFrame(index=price_data.index)

        features["fractal_trend_confirmation"] = np.where(
            fractal_features["valid_fractal_high"] == 1,
            1,  # Bullish confirmation
            np.where(
                fractal_features["valid_fractal_low"] == 1,
                -1,  # Bearish confirmation
                0,  # No clear fractal confirmation
            ),
        )

        # Distance to nearest fractal level (for risk management)
        recent_high_fractal = high[fractal_features["valid_fractal_high"] == 1].rolling(10).max()
        recent_low_fractal = low[fractal_features["valid_fractal_low"] == 1].rolling(10).min()

        features["distance_to_fractal_resistance"] = (recent_high_fractal - close).reindex(
            price_data.index, method="ffill"
        )

        features["distance_to_fractal_support"] = (close - recent_low_fractal).reindex(
            price_data.index, method="ffill"
        )

        return features


def calculate_enhanced_fractals(high: pd.Series, low: pd.Series, n: int = 2) -> pd.DataFrame:
    """
    Enhanced fractal detection with strength measurement
    """
    # Basic fractal patterns
    fractal_high = (high == high.rolling(2 * n + 1, center=True).max()).astype(int)
    fractal_low = (low == low.rolling(2 * n + 1, center=True).min()).astype(int)

    # Fractal strength (how much higher/lower than surrounding fractals)
    fractal_high_strength = np.where(
        fractal_high == 1, high / high.rolling(2 * n + 1, center=True).mean() - 1, 0
    )

    fractal_low_strength = np.where(
        fractal_low == 1, 1 - low / low.rolling(2 * n + 1, center=True).mean(), 0
    )

    # Fractal validation (volume/volatility confirmation could be added)
    valid_fractal_high = (fractal_high == 1) & (fractal_high_strength > 0.002)  # 0.2% threshold
    valid_fractal_low = (fractal_low == 1) & (fractal_low_strength > 0.002)

    return pd.DataFrame(
        {
            "fractal_high": fractal_high,
            "fractal_low": fractal_low,
            "fractal_high_strength": fractal_high_strength,
            "fractal_low_strength": fractal_low_strength,
            "valid_fractal_high": valid_fractal_high.astype(int),
            "valid_fractal_low": valid_fractal_low.astype(int),
        },
        index=high.index,
    )


# Example usage
if __name__ == "__main__":
    # Generate sample forex data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="H")

    # Simulate realistic forex price movements
    returns = np.random.normal(0, 0.0005, 1000)  # Small forex-like returns
    price = 1.1000 + np.cumsum(returns)

    # Create realistic OHLC
    price_df = pd.DataFrame(index=dates)
    price_df["close"] = price
    price_df["open"] = price_df["close"].shift(1).fillna(price[0])
    price_df["high"] = price_df[["open", "close"]].max(axis=1) + np.random.uniform(0, 0.0002, 1000)
    price_df["low"] = price_df[["open", "close"]].min(axis=1) - np.random.uniform(0, 0.0002, 1000)

    # Initialize feature engine
    engine = ForexFeatureEngine("EURUSD")

    # Calculate features
    features = engine.calculate_all_features(price_df)

    print(f"Generated {features.shape[1]} features for {features.shape[0]} observations")
    print(f"\nFeature columns:")
    for i, col in enumerate(features.columns):
        print(f"{i+1:2d}. {col}")

    print(f"\nSample of first 5 rows and 10 columns:")
    print(features.iloc[:5, :10].round(4))
