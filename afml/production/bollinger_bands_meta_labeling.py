"""
Correct Architecture: Bollinger Bands → Meta-Labeling
No intermediate ML model needed.

This implements the proper separation:
- Bollinger Bands: Primary model (side prediction)
- Meta-Model: Secondary model (size/confidence prediction)
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from afml.cache.unified_cache_system import cacheable
from afml.cross_validation.cross_validation import (
    PurgedKFold,
    PurgedSplit,
    ml_cross_val_score,
)
from afml.labeling.triple_barrier import (
    add_vertical_barrier,
    get_event_weights,
    triple_barrier_labels,
)
from afml.sample_weights.optimized_attribution import (
    get_weights_by_time_decay_optimized,
)
from afml.strategies.bollinger_features import create_bollinger_features
from afml.strategies.signal_processing import get_entries
from afml.strategies.signals import BaseStrategy, BollingerStrategy
from afml.util.volatility import get_daily_vol

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

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2,
        min_signals_for_metrics: int = 20,
        target_lookback: int = 100,
        profit_target: float = 1,
        stop_loss: float = 2,
        max_holding_period: Dict[str, int] = dict(days=1),
        min_ret: float = 0.0,
        vertical_barrier_zero: bool = True,
        filter_as_series: bool = True,
    ):
        """
        Parameters
        ----------
            bb_window : int
                Bollinger Bands SMA window
            bb_std : float
                Number of standard deviations for bands
            min_signals_for_metrics : int
                Minimum signals before calculating metrics (cold start)
            target_lookback : int
                Lookback window for volatility estimation with EWM daily volatility.
            profit_target : float, default=1
                Profit-taking threshold multiplier.
            stop_loss : float, default=1
                Stop-loss threshold multiplier.
            max_holding_period : dict, default={'days': 1}
                Maximum holding period for vertical barrier.
            min_ret : float, default=0.0
                Minimum return threshold.
            vertical_barrier_zero : bool, default=True
                Set label to zero if vertical barrier is reached.
            filter_as_series : bool, default=True
                Pass volatility threshold as series instead of scalar.
        """
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.min_signals_for_metrics = min_signals_for_metrics
        self.strategy = BollingerStrategy(bb_window, bb_std)
        self.signals = pd.Series()
        self.target_lookback = target_lookback
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.min_ret = min_ret
        self.vertical_barrier_zero = vertical_barrier_zero
        self.filter_as_series = filter_as_series

        # Track BB signal history for rolling metrics
        self.signal_history = []  # List of (signal, actual_outcome) tuples
        self.meta_model = None

    @cacheable()
    def calculate_meta_labels(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate trading events using the triple-barrier method.

        Parameters
        ----------
        data : pd.DataFrame
            Price bars with 'close' column.

        Returns
        -------
        pd.DataFrame
            Event labels with columns:
            - 'bin' : {-1, 0, 1} classification
            - 't1'  : vertical barrier timestamps
            - 'w'   : sample weights
            - 'tW'  : uniqueness weights
        """
        close = data["close"]

        # Compute signals and CUSUM-filtered trade events
        target = get_daily_vol(close, self.target_lookback)  # target volatility for barriers
        threshold = (
            target if self.filter_as_series else target.mean()
        )  # target threshold for CUSUM-filter
        side, t_events = get_entries(self.strategy, data, threshold)

        # Compute barriers
        vb = add_vertical_barrier(t_events, close, **self.max_holding_period)
        events = triple_barrier_labels(
            close,
            target,
            t_events,
            vertical_barrier_times=vb,
            side_prediction=side,
            pt_sl=[self.profit_target, self.stop_loss],
            min_ret=self.min_ret,
            min_pct=0.05,
            vertical_barrier_zero=self.vertical_barrier_zero,
            drop=True,
            verbose=False,
        )
        events = get_event_weights(events, close)
        self.signals = side.reindex(events.index)
        return events

    def compute_sample_weights(
        self,
        events: pd.DataFrame,
        data: pd.DataFrame,
        X_train: pd.DataFrame,
        model: ClassifierMixin,
    ) -> pd.Series:
        """
        Compute sample weights with time decay.

        Parameters
        ----------
        events : pd.DataFrame
            Event labels with uniqueness weights.
        data: pd.DataFrame
            Price data.
        X_train: pd.DataFrame
            Training features
        model: ClassifierMixin
            Classifier model.

        Returns
        -------
        pd.Series
            Sample weights.

        Notes
        -----
        - First run: ~5s; cached: ~0.1s (≈50x speedup).
        """
        valid_index = X_train.index.intersection(events.index)
        weighting_schemes = {
            "unweighted": pd.Series(1.0, index=valid_index),
            "uniqueness": events.loc[valid_index, "tW"],
            "return": events.loc[valid_index, "w"],
        }
        X = X_train.loc[valid_index]
        y = events.loc[valid_index, "bin"]
        cv_gen = PurgedKFold(n_splits=5, t1=events.loc[valid_index, "t1"], pct_embargo=0.01)
        best_scheme = None

        def get_best_weighting_scheme(weight, scheme, best_score):
            nonlocal best_scheme
            cv_scores = ml_cross_val_score(
                clone(model).set_params(n_estimators=None),
                X,
                y,
                cv_gen,
                sample_weight_train=weight,
                sample_weight_score=weight,
                scoring="f1",
            )
            score = cv_scores.mean()
            best_score = max(score, best_score)
            if not best_scheme or score == best_score:
                best_scheme = scheme
            return best_scheme, best_score

        best_score = 0
        for scheme, weight in weighting_schemes.items():
            best_scheme, best_score = get_best_weighting_scheme(weight, scheme, best_score)

        decay_factors = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]
        best_weighting_scheme = best_scheme
        best_weight = weighting_schemes[best_scheme]
        for time_decay in reversed(decay_factors):
            for linear in (0, 1):
                decay_w = get_weights_by_time_decay_optimized(
                    triple_barrier_events=events.loc[valid_index],
                    close_index=data.index,
                    last_weight=time_decay,
                    linear=linear,
                    av_uniqueness=events.loc[valid_index, "tW"],
                )
                weight = best_weight * decay_w
                scheme = f"{best_weighting_scheme}_{('linear' if linear else 'exponential')}"
                weighting_schemes[f"{scheme}_decay_{time_decay}"] = weight
                best_scheme, best_score = get_best_weighting_scheme(weight, scheme, best_score)

        print("Best Weighting Scheme:", " ".join(best_scheme.split("_")).title())

        return weighting_schemes[best_scheme]

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

    def prepare_features(self, data, signals, index):
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
        idx = data.index.get_loc(signals.index[index])
        prices = data["close"]

        # Market context features
        if index >= 20:
            features["volatility_20"] = prices.iloc[idx - 20 : idx].std()
            features["price_position"] = (
                prices.iloc[idx] - prices.iloc[idx - 20 : idx].mean()
            ) / prices.iloc[idx - 20 : idx].std()
        else:
            features["volatility_20"] = np.nan
            features["price_position"] = np.nan

        if index >= 5:
            features["momentum_5"] = prices.iloc[idx] / prices.iloc[idx - 5] - 1
        else:
            features["momentum_5"] = np.nan

        if index >= 10:
            features["momentum_10"] = prices.iloc[idx] / prices.iloc[idx - 10] - 1
        else:
            features["momentum_10"] = np.nan

        # Add rolling performance metrics
        features.update(rolling_metrics)

        return features

    def get_features(self, data, meta_features):
        features = create_bollinger_features(data, self.bb_window, self.bb_std)
        return features.join(meta_features, how="inner").dropna()

    def train(self, data, test_split=0.3):
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

        # Step 1: Calculate meta-labels
        events = self.calculate_meta_labels(data)
        bb_precision = events["bin"].mean()
        print(f"\n1. Meta-labels calculated")
        print(f"   Raw BB strategy precision: {bb_precision:.2%}")

        # Step 2: Build training dataset with rolling metrics
        meta_features = {}

        for i in range(len(events)):
            # Update signal history BEFORE calculating features
            # This ensures rolling metrics are based on PAST signals only
            if len(self.signal_history) > 0:
                features = self.prepare_features(data, self.signals, i)

                if features is not None:  # Past cold start period
                    meta_features[events.index[i]] = features

            # Now add this signal to history for future rolling calculations
            self.signal_history.append((self.signals.iloc[i], events["bin"].iloc[i]))
        meta_features = pd.DataFrame.from_dict(meta_features, orient="index")

        print(f"\n2. Features prepared")
        print(f"   Cold start period: {self.min_signals_for_metrics} signals")
        print(f"   Training observations after cold start: {len(meta_features)}")

        # Convert to DataFrame
        features = self.get_features(data, meta_features)

        # Remove any remaining NaN values
        valid_index = events.index.intersection(features.index)
        cont = events.loc[valid_index]
        X = features.loc[valid_index]
        y = cont["bin"]

        print(f"   Final training samples: {len(X)}")

        # Step 4: Train/validation split
        train_idx, test_idx = PurgedSplit(t1=cont["t1"], test_size_pct=test_split).split(X)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n4. Train/Validation split")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_test)} samples")

        # Step 5: Train meta-model
        av_uniqueness = cont["tW"].iloc[train_idx].mean()
        self.meta_model = RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            class_weight="balanced_subsample",
            max_depth=4,
            max_samples=av_uniqueness,
            min_weight_fraction_leaf=0.05,
            random_state=42,
            n_jobs=-1,
        )

        w_train = self.compute_sample_weights(cont, data, X_train, self.meta_model)
        self.meta_model.fit(X_train, y_train, sample_weight=w_train)

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": self.meta_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"\n5. Meta-model trained")
        print(f"\n   Top 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"     {row['feature']:30s}: {row['importance']:.4f}")

        # Step 6: Evaluate
        y_pred_proba = self.meta_model.predict_proba(X_test)[:, 1]

        baseline_precision = y_test.mean()

        print(f"\n6. Validation Performance")
        print(f"\n   Baseline (accept all BB signals):")
        print(f"     Precision: {baseline_precision:.2%}")
        print(f"     Number of trades: {len(y_test)}")

        for threshold in [0.5, 0.6, 0.7]:
            filtered_mask = y_pred_proba >= threshold
            if filtered_mask.sum() == 0:
                continue

            filtered_precision = y_test[filtered_mask].mean()
            n_trades = filtered_mask.sum()

            print(f"\n   Meta-model @ threshold {threshold:.1f}:")
            print(f"     Precision: {filtered_precision:.2%}")
            print(f"     Number of trades: {n_trades} ({n_trades/len(y_test):.1%} of signals)")
            print(f"     Improvement: {filtered_precision - baseline_precision:+.2%}")

        print("\n" + "=" * 80)

        return {
            "feature_importance": feature_importance,
            "validation_metrics": {
                "baseline_precision": baseline_precision,
                "y_test": y_test,
                "y_pred_proba": y_pred_proba,
            },
        }

    def predict(self, data, signals, index):
        """
        Make prediction for a new signal at given index.

        Returns:
            - None if in cold start period
            - Probability of success otherwise
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")

        meta_features = self.prepare_features(data, signals, index)
        meta_features = pd.DataFrame(meta_features, index=signals.index[index])

        if features is None:
            return None  # Still in cold start

        # Convert to DataFrame with same column order as training
        features = self.get_features(data, meta_features)
        features = features[self.meta_model.feature_names_in_]

        return self.meta_model.predict_proba(features)[0, 1]


def run_example():
    """
    Complete example showing the correct architecture.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range("2020-01-01", periods=n_points, freq="D")
    returns = np.random.normal(0.0005, 0.02, n_points)
    data = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    # Initialize and train system
    system = BollingerBandsMetaLabeling(
        bb_window=20, bb_std=2, min_signals_for_metrics=20  # Cold start period
    )

    results = system.train(data, validation_split=0.3)

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
