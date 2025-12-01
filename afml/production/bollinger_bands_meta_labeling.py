"""
Correct Architecture: Bollinger Bands → Meta-Labeling
No intermediate ML model needed.

This implements the proper separation:
- Bollinger Bands: Primary model (side prediction)
- Meta-Model: Secondary model (size/confidence prediction)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


class BollingerBandsMetaLabeling:
    """
    Complete meta-labeling system with Bollinger Bands as primary model.

    Key Design Decisions:
    1. BB generates sides directly (no ML primary model)
    2. Rolling metrics track BB performance (not some other model)
    3. Meta-model learns WHEN to trust BB signals
    4. Cold start: No predictions until N signals have been observed
    """

    def __init__(self, bb_window=20, bb_std=2, min_signals_for_metrics=20):
        """
        Args:
            bb_window: Bollinger Bands SMA window
            bb_std: Number of standard deviations for bands
            min_signals_for_metrics: Minimum signals before calculating metrics (cold start)
        """
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.min_signals_for_metrics = min_signals_for_metrics

        # Track BB signal history for rolling metrics
        self.signal_history = []  # List of (signal, actual_outcome) tuples
        self.meta_model = None

    def generate_bb_signals(self, prices):
        """
        Primary Model: Generate Bollinger Bands trading signals.
        Returns: Series of {-1, 0, +1}
        """
        signals = pd.Series(0, index=prices.index)

        # Calculate Bollinger Bands
        sma = prices.rolling(window=self.bb_window).mean()
        std = prices.rolling(window=self.bb_window).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)

        # Mean reversion strategy
        # Long when price crosses below lower band
        long_signal = (prices < lower_band) & (prices.shift(1) >= lower_band.shift(1))
        # Short when price crosses above upper band
        short_signal = (prices > upper_band) & (prices.shift(1) <= upper_band.shift(1))

        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals, sma, upper_band, lower_band

    def calculate_meta_labels(
        self, prices, signals, holding_period=5, profit_target=0.02, stop_loss=0.01
    ):
        """
        Calculate meta-labels using triple barrier method.
        Only calculates for observations where signals != 0.

        Returns: Series of meta-labels (1 = profitable, 0 = not profitable)
        """
        meta_labels = pd.Series(np.nan, index=prices.index)

        for i in range(len(prices) - holding_period):
            if signals.iloc[i] == 0:
                continue

            entry_price = prices.iloc[i]
            side = signals.iloc[i]
            future_prices = prices.iloc[i + 1 : i + holding_period + 1]

            if side == 1:  # Long position
                returns = (future_prices - entry_price) / entry_price
                hit_profit = (returns >= profit_target).any()
                hit_stop = (returns <= -stop_loss).any()

                if hit_profit and not hit_stop:
                    meta_labels.iloc[i] = 1
                elif hit_stop:
                    meta_labels.iloc[i] = 0
                else:
                    final_return = (future_prices.iloc[-1] - entry_price) / entry_price
                    meta_labels.iloc[i] = 1 if final_return > 0 else 0

            elif side == -1:  # Short position
                returns = (entry_price - future_prices) / entry_price
                hit_profit = (returns >= profit_target).any()
                hit_stop = (returns <= -stop_loss).any()

                if hit_profit and not hit_stop:
                    meta_labels.iloc[i] = 1
                elif hit_stop:
                    meta_labels.iloc[i] = 0
                else:
                    final_return = (entry_price - future_prices.iloc[-1]) / entry_price
                    meta_labels.iloc[i] = 1 if final_return > 0 else 0

        return meta_labels

    def calculate_rolling_metrics(self, window_sizes=[20, 50]):
        """
        Calculate rolling performance metrics of BB signals.

        Returns: Dictionary of rolling metrics or None if insufficient history
        """
        if len(self.signal_history) < self.min_signals_for_metrics:
            return None  # Cold start - not enough signal history

        metrics = {}

        for window in window_sizes:
            if len(self.signal_history) < window:
                metrics[f"rolling_accuracy_{window}"] = np.nan
                metrics[f"rolling_precision_{window}"] = np.nan
                metrics[f"rolling_recall_{window}"] = np.nan
                metrics[f"rolling_f1_{window}"] = np.nan
                continue

            # Get last N signals and their outcomes
            recent = self.signal_history[-window:]
            y_pred = [1 for _ in recent]  # All signals were "predicted positive"
            y_true = [outcome for _, outcome in recent]

            metrics[f"rolling_accuracy_{window}"] = accuracy_score(y_true, y_pred)
            metrics[f"rolling_precision_{window}"] = precision_score(
                y_true, y_pred, zero_division=0
            )
            metrics[f"rolling_recall_{window}"] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f"rolling_f1_{window}"] = f1_score(y_true, y_pred, zero_division=0)

        return metrics

    def prepare_features(self, prices, signals, index):
        """
        Prepare feature vector for a specific signal at given index.

        Features include:
        - Signal characteristics (direction, strength)
        - Market characteristics (volatility, momentum)
        - Rolling performance metrics of BB strategy

        Returns: Dictionary of features or None if in cold start period
        """
        # Get rolling metrics
        rolling_metrics = self.calculate_rolling_metrics()

        if rolling_metrics is None:
            return None  # Still in cold start period

        # Basic signal features
        features = {
            "signal": signals.iloc[index],
        }

        # Market context features
        if index >= 20:
            features["volatility_20"] = prices.iloc[index - 20 : index].std()
            features["price_position"] = (
                prices.iloc[index] - prices.iloc[index - 20 : index].mean()
            ) / prices.iloc[index - 20 : index].std()
        else:
            features["volatility_20"] = np.nan
            features["price_position"] = np.nan

        if index >= 5:
            features["momentum_5"] = prices.iloc[index] / prices.iloc[index - 5] - 1
        else:
            features["momentum_5"] = np.nan

        if index >= 10:
            features["momentum_10"] = prices.iloc[index] / prices.iloc[index - 10] - 1
        else:
            features["momentum_10"] = np.nan

        # Add rolling performance metrics
        features.update(rolling_metrics)

        return features

    def train(self, prices, validation_split=0.3):
        """
        Complete training pipeline:
        1. Generate BB signals
        2. Calculate meta-labels
        3. Build features (including rolling metrics)
        4. Train meta-model

        Note: First N signals used only for rolling metric calculation (cold start)
        """
        print("=" * 80)
        print("Training Bollinger Bands Meta-Labeling System")
        print("=" * 80)

        # Step 1: Generate BB signals (Primary Model)
        signals, sma, upper_band, lower_band = self.generate_bb_signals(prices)
        n_signals = (signals != 0).sum()
        print(f"\n1. Bollinger Bands generated {n_signals} signals")
        print(f"   Long signals: {(signals == 1).sum()}")
        print(f"   Short signals: {(signals == -1).sum()}")

        # Step 2: Calculate meta-labels
        meta_labels = self.calculate_meta_labels(prices, signals)
        bb_precision = meta_labels[signals != 0].mean()
        print(f"\n2. Meta-labels calculated")
        print(f"   Raw BB strategy precision: {bb_precision:.2%}")

        # Step 3: Build training dataset with rolling metrics
        feature_list = []
        target_list = []

        for i in range(len(prices)):
            if signals.iloc[i] == 0:
                continue  # Skip non-signals

            if pd.isna(meta_labels.iloc[i]):
                continue  # Skip if meta-label couldn't be calculated

            # Update signal history BEFORE calculating features
            # This ensures rolling metrics are based on PAST signals only
            if len(self.signal_history) > 0:
                features = self.prepare_features(prices, signals, i)

                if features is not None:  # Past cold start period
                    feature_list.append(features)
                    target_list.append(meta_labels.iloc[i])

            # Now add this signal to history for future rolling calculations
            self.signal_history.append((signals.iloc[i], meta_labels.iloc[i]))

        print(f"\n3. Features prepared")
        print(f"   Cold start period: {self.min_signals_for_metrics} signals")
        print(f"   Training observations after cold start: {len(feature_list)}")

        # Convert to DataFrame
        X = pd.DataFrame(feature_list)
        y = pd.Series(target_list)

        # Remove any remaining NaN values
        valid_mask = ~X.isnull().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"   Final training samples: {len(X)}")

        # Step 4: Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n4. Train/Validation split")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")

        # Step 5: Train meta-model
        self.meta_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10, random_state=42
        )

        self.meta_model.fit(X_train, y_train)

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": self.meta_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"\n5. Meta-model trained")
        print(f"\n   Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"     {row['feature']:30s}: {row['importance']:.4f}")

        # Step 6: Evaluate
        y_val_pred_proba = self.meta_model.predict_proba(X_val)[:, 1]

        baseline_precision = y_val.mean()

        print(f"\n6. Validation Performance")
        print(f"\n   Baseline (accept all BB signals):")
        print(f"     Precision: {baseline_precision:.2%}")
        print(f"     Number of trades: {len(y_val)}")

        for threshold in [0.5, 0.6, 0.7]:
            filtered_mask = y_val_pred_proba >= threshold
            if filtered_mask.sum() == 0:
                continue

            filtered_precision = y_val[filtered_mask].mean()
            n_trades = filtered_mask.sum()

            print(f"\n   Meta-model @ threshold {threshold:.1f}:")
            print(f"     Precision: {filtered_precision:.2%}")
            print(f"     Number of trades: {n_trades} ({n_trades/len(y_val):.1%} of signals)")
            print(f"     Improvement: {filtered_precision - baseline_precision:+.2%}")

        print("\n" + "=" * 80)

        return {
            "feature_importance": feature_importance,
            "validation_metrics": {
                "baseline_precision": baseline_precision,
                "y_val": y_val,
                "y_val_pred_proba": y_val_pred_proba,
            },
        }

    def predict(self, prices, signals, index):
        """
        Make prediction for a new signal at given index.

        Returns:
            - None if in cold start period
            - Probability of success otherwise
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = self.prepare_features(prices, signals, index)

        if features is None:
            return None  # Still in cold start

        # Convert to DataFrame with same column order as training
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.meta_model.feature_names_in_]

        return self.meta_model.predict_proba(feature_df)[0, 1]


def run_example():
    """
    Complete example showing the correct architecture.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range("2020-01-01", periods=n_points, freq="D")
    returns = np.random.normal(0.0005, 0.02, n_points)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    # Initialize and train system
    system = BollingerBandsMetaLabeling(
        bb_window=20, bb_std=2, min_signals_for_metrics=20  # Cold start period
    )

    results = system.train(prices, validation_split=0.3)

    print("\n" + "=" * 80)
    print("KEY ARCHITECTURE POINTS:")
    print("=" * 80)
    print("\n1. Bollinger Bands IS the primary model (no ML layer needed)")
    print("2. Rolling metrics track BB performance over last N signals")
    print("3. Meta-model only makes predictions after cold start period")
    print("4. Meta-model learns WHEN to trust BB signals (not how to generate them)")
    print("\n" + "=" * 80)

    return system, results


if __name__ == "__main__":
    system, results = run_example()
