from datetime import timedelta
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..features.fractals import (
    calculate_enhanced_fractals,
    calculate_fractal_trend_features,
)
from ..features.time import get_time_features
from ..labeling.trend_scanning import trend_scanning_labels
from ..util.misc import optimize_dtypes, set_resampling_freq


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
        timeframe: str,
        lr_period: Tuple[int] = (5, 20),
        volume_data: Optional[pd.Series] = None,
        additional_pairs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all forex-relevant features for MA crossover strategy

        Parameters:
        price_data: DataFrame with OHLC columns
        timeframe: Timeframe string (e.g., 'M1', 'M5', 'M15', 'H1')
        lr_period: Range of periods scanned for linear regression trend features
        volume_data: Series with volume data (if available, often limited in forex)
        additional_pairs: Dict of related currency pair OHLC data for correlation features

        Returns:
        DataFrame with all calculated features
        """

        features = pd.DataFrame(index=price_data.index)
        all_features = []

        close = price_data["close"]
        self.returns = np.log(close).diff().copy()

        # Core MA Features
        ma_features = self._calculate_ma_features(price_data)
        all_features += [ma_features]

        # Volatility & Range Features (Critical for Forex)
        vol_features = self._calculate_volatility_features(price_data)
        all_features += [vol_features]

        # Trend Strength Features
        trend_features = self._calculate_trend_features(price_data, lr_period)
        all_features += [trend_features]

        # Currency Strength Features
        if additional_pairs:
            strength_features = self._calculate_currency_strength_features(close, additional_pairs)
            all_features += [strength_features]

        # Risk Environment Features
        risk_features = self._calculate_risk_environment_features(price_data)
        all_features += [risk_features]

        # Time-Based Features (Important for 24h markets)
        time_features = get_time_features(price_data, timeframe)
        all_features += [time_features]

        # Market Structure Features
        structure_features = self._calculate_market_structure_features(price_data)
        all_features += [structure_features]

        # Higher Timeframe Features
        # freq = set_resampling_freq(timeframe)
        # tf_minutes = int(pd.to_timedelta(freq).total_seconds() / 60)
        # higher_timeframe_features = self._add_higher_timeframe_features(price_data, tf_minutes)
        # all_features += [higher_timeframe_features]

        # Combine all features
        features = features.join(all_features)
        features = optimize_dtypes(features)

        return features.ffill().shift().replace([np.inf, -np.inf], np.nan).fillna(0)

    def _calculate_ma_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Core moving average features optimized for forex"""
        close = price_data["close"]
        features = pd.DataFrame(index=close.index)

        # Multiple MA periods suitable for forex
        ma_periods = [10, 20, 50, 100, 200]

        # Calculate MAs
        mas = {}
        for period in ma_periods:
            mas[period] = price_data.ta.sma(period)
            features[f"ma_{period}"] = mas[period]

        # MA Crossover Signals
        features["ma_10_20_cross"] = np.where(mas[10] > mas[20], 1, -1)
        features["ma_20_50_cross"] = np.where(mas[20] > mas[50], 1, -1)
        features["ma_50_200_cross"] = np.where(mas[50] > mas[200], 1, -1)

        # MA Spreads (normalized by ATR)
        atr = price_data.ta.atr(14)
        features["ma_spread_10_20"] = (mas[10] - mas[20]) / atr
        features["ma_spread_20_50"] = (mas[20] - mas[50]) / atr
        features["ma_spread_50_200"] = (mas[50] - mas[200]) / atr

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

        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # ATR (multiple periods)
        features["atr_14"] = price_data.ta.atr(14)
        features["atr_21"] = price_data.ta.atr(21)

        # ATR Regime
        atr_ma = features["atr_14"].rolling(50).mean()
        features["atr_regime"] = features["atr_14"] / atr_ma

        # Realized Volatility (annualized)
        returns = self.returns

        # Use trading days that account for weekends
        features["realized_vol_10"] = returns.rolling(10).std() * np.sqrt(252)
        features["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(252)
        features["realized_vol_50"] = returns.rolling(50).std() * np.sqrt(252)

        # Volatility of Volatility
        features["vol_of_vol"] = features["realized_vol_20"].rolling(20).std()

        # high-low Range Features
        features["hl_range"] = (high - low) / close
        features["hl_range_ma"] = features["hl_range"].rolling(20).mean()
        features["hl_range_regime"] = features["hl_range"] / features["hl_range_ma"]

        # Bollinger Bands
        bb = price_data.ta.bbands(length=20, std=2)
        features = features.assign(
            bb_upper=bb["BBU_20_2.0"],
            bb_lower=bb["BBL_20_2.0"],
            bb_percent=bb["BBP_20_2.0"],
            bb_bandwidth=bb["BBB_20_2.0"],
        )
        bb_std_val = bb["BBM_20_2.0"]
        features["bb_squeeze"] = bb_std_val / bb_std_val.rolling(50).mean()

        return features

    def _calculate_trend_features(
        self, price_data: pd.DataFrame, lr_period: Tuple[int]
    ) -> pd.DataFrame:
        """Trend strength and quality features"""
        features = pd.DataFrame(index=price_data.index)

        close = price_data["close"]
        high = price_data["high"]
        low = price_data["low"]

        # Efficiency Ratio (trending vs ranging)
        features["efficiency_ratio_14"] = self._calculate_efficiency_ratio(close, 14)
        features["efficiency_ratio_30"] = self._calculate_efficiency_ratio(close, 30)

        # ADX (Average Directional Index)
        adx_features = price_data.ta.adx(14)
        adx_features.columns = adx_features.columns.str.lower()
        adx_features["adx_trend_strength"] = np.where(adx_features["adx_14"] > 25, 1, 0)
        adx_features["adx_trend_direction"] = np.where(
            adx_features["dmp_14"] > adx_features["dmn_14"], 1, -1
        )

        # Linear Regression Features
        lr_features = self._calculate_linear_regression_features(close, lr_period)
        features = features.join([adx_features, lr_features])

        # Momentum Features
        features["roc_10"] = close.pct_change(10)
        features["roc_20"] = close.pct_change(20)
        features["momentum_14"] = close / close.shift(14) - 1

        # Higher Highs, Lower Lows
        features["hh_ll_20"] = self._calculate_hh_ll_count(high, low, 20)

        # Trend Persistence
        features["trend_persistence"] = (
            self.returns.gt(0).astype(float).rolling(20).mean().fillna(0.5)
        )

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
            pair_returns = self.returns
            features["correlation_eurusd"] = pair_returns.rolling(50).corr(eurusd_returns)

        return features

    def _calculate_risk_environment_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Risk environment and market stress indicators"""
        features = pd.DataFrame(index=price_data.index)
        returns = self.returns

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
    def _calculate_efficiency_ratio(self, close: pd.Series, period: int) -> pd.Series:
        """Efficiency Ratio (directional movement / total movement)"""
        direction = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(period).sum()
        return direction / volatility

    def _calculate_linear_regression_features(
        self, close: pd.Series, period: Tuple[int]
    ) -> pd.DataFrame:
        lr_results = trend_scanning_labels(close, span=period, lookforward=False).drop(
            columns=["t1", "bin"]
        )
        lr_results.columns = [f"trend_{col}" for col in lr_results.columns]
        return lr_results

    def _calculate_hh_ll_count(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Count of higher highs and lower lows"""
        hh = (high > high.shift(1)).rolling(period).sum()
        ll = (low < low.shift(1)).rolling(period).sum()
        return hh - ll  # Positive = more higher highs, Negative = more lower lows

    def _calculate_enhanced_fractal_features(self, price_data: pd.DataFrame):
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # Fractal-based trend confirmation
        features = pd.DataFrame(index=price_data.index)
        fractal_features = calculate_enhanced_fractals(high, low, close)
        fractal_trend_features = calculate_fractal_trend_features(
            close,
            fractal_features["fractal_breakout_up"],
            fractal_features["fractal_breakout_down"],
        )
        features = features.join(pd.DataFrame(fractal_trend_features))

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

    def _add_higher_timeframe_features(
        self, price_data: pd.DataFrame, base_tf_minutes: int
    ) -> pd.DataFrame:
        """
        Add higher timeframe MA features to any minute-based data

        :param price_data: DataFrame with OHLC data
        :param base_tf_minutes: Base timeframe in minutes (e.g., 1, 5, 15, 60)
        :return: DataFrame with HTF features added
        """
        features = pd.DataFrame(index=price_data.index)

        # Define higher timeframes as multiples of base timeframe
        htf_multipliers = {
            "4h": timedelta(hours=4)
            // timedelta(minutes=base_tf_minutes),  # 4 hours in base timeframe units
            "1d": timedelta(days=1)
            // timedelta(minutes=base_tf_minutes),  # 1 day in base timeframe units
            "1w": timedelta(days=7)
            // timedelta(minutes=base_tf_minutes),  # 1 week in base timeframe units
        }

        # Only add HTF features where multiplier makes sense (>= 2)
        for htf_name, multiplier in htf_multipliers.items():
            if multiplier >= 2:  # Must be at least 2x the base timeframe

                # Simulate HTF moving averages
                features[f"htf_{htf_name}_ma_20"] = price_data.ta.sma(20 * multiplier)
                features[f"htf_{htf_name}_ma_50"] = price_data.ta.sma(50 * multiplier)

                # HTF trend direction
                features[f"htf_{htf_name}_trend"] = np.where(
                    features[f"htf_{htf_name}_ma_20"] > features[f"htf_{htf_name}_ma_50"], 1, -1
                )

                # HTF trend strength (ribbon alignment)
                htf_ma_10 = price_data.ta.sma(10 * multiplier)
                htf_ma_100 = price_data.ta.sma(100 * multiplier)

                htf_alignment = np.where(htf_ma_10 > features[f"htf_{htf_name}_ma_20"], 1, -1)
                htf_alignment *= np.where(
                    features[f"htf_{htf_name}_ma_20"] > features[f"htf_{htf_name}_ma_50"], 1, -1
                )
                htf_alignment *= np.where(features[f"htf_{htf_name}_ma_50"] > htf_ma_100, 1, -1)

                features[f"htf_{htf_name}_ribbon_aligned"] = htf_alignment

                # Price position relative to HTF MAs
                close = price_data["close"]
                features[f"price_above_htf_{htf_name}_ma_20"] = (
                    close > features[f"htf_{htf_name}_ma_20"]
                ).astype(int)
                features[f"price_above_htf_{htf_name}_ma_50"] = (
                    close > features[f"htf_{htf_name}_ma_50"]
                ).astype(int)

        return features
