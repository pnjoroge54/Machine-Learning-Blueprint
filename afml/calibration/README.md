```markdown
# Probability Calibration Toolkit for Financial Machine Learning

This module provides a comprehensive set of tools for calibrating classifier probabilities and evaluating calibration quality, with special attention to the challenges of financial time series data. It integrates seamlessly with purged cross‑validation techniques described in *Advances in Financial Machine Learning* (López de Prado, 2018) and includes both standard calibration methods and custom‑designed cross‑validated isotonic regression.

## Key Features

- **Calibration Metrics** – Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE).
- **Reliability Diagrams** – Plot calibration curves with optional probability distribution histograms.
- **Bootstrap Confidence Intervals** – Add confidence bands to reliability curves.
- **Calibration Methods** – Platt scaling (logistic regression) and isotonic regression.
- **Cross‑Validation Integration** – `CVIsotonicCalibrator` fits isotonic regression on out‑of‑fold predictions using purged cross‑validation, avoiding temporal leakage.
- **Comprehensive Cross‑Validation Analysis** – Compare raw vs. calibrated performance (PWA, Brier, accuracy, etc.) across folds.
- **Designed for Financial Time Series** – Respects temporal order and embargo periods.

## Installation

If you are using this module as part of a larger project, simply place the `calibration.py` file in your project directory. To install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

If you plan to use the scoring functions (e.g., probability_weighted_accuracy), ensure that the scoring.py module is also available in your project. The example assumes that probability_weighted_accuracy is defined in a package‑relative import (from .scoring import probability_weighted_accuracy). Adjust the import path accordingly.

Dependencies

· Python ≥ 3.7
· numpy
· pandas
· scikit‑learn ≥ 0.24
· matplotlib

Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import the toolkit
from calibration import (
    CVIsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    plot_reliability,
    calibration_report
)
from cross_validation import PurgedKFold  # assuming you have it

# Generate synthetic financial data with timestamps
X, y = make_classification(n_samples=1000, random_state=42)
t1 = pd.Series(
    index=pd.date_range('2020-01-01', periods=1000, freq='D'),
    data=pd.date_range('2020-01-15', periods=1000, freq='D')
)

# Set up purged cross‑validation
cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

# Train a classifier and get out‑of‑fold probabilities (example function)
# Here we assume you have a function that returns OOF probabilities, e.g.:
# from calibration import oof_predict_proba
# oof_probs = oof_predict_proba(clf, X, y, cv=cv)
# For the sake of this example, we'll simulate them:
clf = RandomForestClassifier()
oof_probs = np.random.rand(1000)  # dummy

# Evaluate calibration
ece = expected_calibration_error(y, oof_probs)
brier = brier_score(y, oof_probs)
print(f"ECE: {ece:.4f}, Brier: {brier:.4f}")

# Generate a calibration report
report = calibration_report(y, oof_probs)
print(report)

# Plot reliability diagram
plot_reliability(y, oof_probs, title="Before Calibration")
plt.show()
```

Main Components

CVIsotonicCalibrator

A scikit‑learn‑compatible calibrator that fits an isotonic regression on out‑of‑fold predictions from purged cross‑validation. It prevents data leakage and returns calibrated probabilities.

```python
calibrator = CVIsotonicCalibrator(estimator=RandomForestClassifier(), cv=cv)
calibrator.fit(X, y)
calibrated_probs = calibrator.predict_proba(X)[:, 1]
```

analyze_calibrated_cross_val_scores

Performs a full cross‑validation comparison between the raw estimator and the calibrated version. Returns per‑fold scores, a summary DataFrame, and confusion matrices.

```python
ret_scores, scores_df, confusion_matrices = analyze_calibrated_cross_val_scores(
    base_estimator=RandomForestClassifier(),
    X=X, y=y,
    cv_gen=cv,                # purged cross‑validator
    sample_weight_train=None, # optional weights
    calibrator_cv=cv          # same CV for calibration
)
print(scores_df)
```

Calibration Metrics

· brier_score(y_true, p_pred) – Mean squared error between predictions and outcomes.
· expected_calibration_error(y_true, p_pred, n_bins=10, strategy='uniform') – Average absolute deviation between predicted and observed frequencies.
· maximum_calibration_error(y_true, p_pred, n_bins=10, strategy='uniform') – Worst‑case deviation across bins.

Reliability Diagrams

· plot_reliability(y_true, p_pred, ...) – Simple calibration curve.
· plot_reliability_with_ci(y_true, p_pred, n_bootstraps=1000, ...) – Adds bootstrap confidence bands.

Calibration Report

· calibration_report(y_true, p_pred, p_calibrated=None, n_bins=10) – Returns a DataFrame with Brier, ECE, MCE, and improvements if calibrated probabilities are provided.

Advanced Usage

Calibrating with Platt Scaling

```python
from calibration import fit_platt_scaling

platt = fit_platt_scaling(y_calib, scores_calib)
calibrated = platt.predict_proba(scores_test.reshape(-1, 1))[:, 1]
```

Custom Cross‑Validation for Calibration

The CVIsotonicCalibrator expects a cross‑validator that supports the split method and returns train/test indices. For financial applications, PurgedKFold is recommended, but any scikit‑learn compatible CV can be used.

References

· López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons. (Chapter 7 – Probability Calibration)
· Niculescu‑Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.

License

This module is provided under the MIT License. See the LICENSE file for details.

---

For questions or contributions, please refer to the project repository.

```
