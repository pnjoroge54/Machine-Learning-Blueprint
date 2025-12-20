"""
Example: Train a model for MQL5 integration
Demonstrates how to prepare a model for the ML Bridge Server.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def generate_synthetic_features(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic trading features for demonstration."""
    np.random.seed(42)

    # Feature 1-3: Moving averages (simulated)
    ma_10 = np.random.randn(n_samples) * 0.01 + 1.1000
    ma_20 = np.random.randn(n_samples) * 0.01 + 1.1005
    ma_50 = np.random.randn(n_samples) * 0.01 + 1.1010

    # Feature 4-5: RSI variations
    rsi_14 = np.random.uniform(20, 80, n_samples)
    rsi_28 = np.random.uniform(20, 80, n_samples)

    # Feature 6-7: Volatility measures
    atr = np.random.uniform(0.0010, 0.0050, n_samples)
    std_dev = np.random.uniform(0.0005, 0.0030, n_samples)

    # Feature 8-9: Volume indicators
    vol_ma = np.random.uniform(1000, 5000, n_samples)
    vol_ratio = np.random.uniform(0.5, 2.0, n_samples)

    # Feature 10: Momentum
    momentum = np.random.randn(n_samples) * 0.005

    df = pd.DataFrame(
        {
            "ma_10": ma_10,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "rsi_14": rsi_14,
            "rsi_28": rsi_28,
            "atr": atr,
            "std_dev": std_dev,
            "vol_ma": vol_ma,
            "vol_ratio": vol_ratio,
            "momentum": momentum,
        }
    )

    return df


def generate_labels(features: pd.DataFrame) -> np.ndarray:
    """Generate synthetic labels based on features."""
    # Simple rule: BUY (1) if momentum > 0 and RSI < 70, else SELL (0)
    labels = np.where(
        (features["momentum"] > 0) & (features["rsi_14"] < 70), 1, 0
    )  # BUY  # SELL

    return labels


def train_and_save_model():
    """Train a RandomForest model and save for ML Bridge Server."""
    print("=" * 70)
    print("TRAINING ML MODEL FOR MQL5 INTEGRATION")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic training data...")
    features = generate_synthetic_features(n_samples=10000)
    labels = generate_labels(features)

    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(
        f"   Class distribution: BUY={np.sum(labels)}, SELL={len(labels) - np.sum(labels)}"
    )

    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train model
    print("\n3. Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("   ✓ Training complete")

    # Evaluate
    print("\n4. Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"   Train accuracy: {train_score:.4f}")
    print(f"   Test accuracy: {test_score:.4f}")

    y_pred = model.predict(X_test)
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

    print("\n   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    print("\n5. Feature Importance:")
    feature_importance = pd.DataFrame(
        {"feature": features.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']:15s}: {row['importance']:.4f}")

    # Save model
    print("\n6. Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Save model
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {model_dir / 'model.pkl'}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "version": "1.0",
        "feature_names": features.columns.tolist(),
        "n_features": len(features.columns),
        "train_accuracy": float(train_score),
        "test_accuracy": float(test_score),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "created": pd.Timestamp.now().isoformat(),
    }

    import json

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)
    print(f"   ✓ Metadata saved to {model_dir / 'metadata.json'}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start ML Bridge Server:")
    print("   python ml_bridge_server.py --model random_forest --port 9090")
    print("\n2. Configure MQL5 EA to connect to localhost:9090")
    print("\n3. Monitor logs and cache performance")
    print("=" * 70)


if __name__ == "__main__":
    train_and_save_model()
