# Probability Calibration Toolkit for Financial Machine Learning

This module provides a comprehensive set of tools for calibrating classifier probabilities and evaluating calibration quality, with special attention to the challenges of **financial time series data**. It integrates seamlessly with purged cross-validation techniques described in *Advances in Financial Machine Learning* (López de Prado, 2018) and includes both standard calibration methods and a custom-designed cross-validated isotonic regression.

## Key Features

- **Calibration Metrics** — Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- **Reliability Diagrams** — Plot calibration curves with optional probability distribution histograms
- **Bootstrap Confidence Intervals** — Add confidence bands to reliability curves
- **Calibration Methods** — Platt scaling (sigmoid) and isotonic regression
- **Cross-Validation Integration** — `CVIsotonicCalibrator` fits calibration on out-of-fold predictions using purged cross-validation, avoiding temporal leakage
- **Comprehensive Analysis** — Compare raw vs. calibrated performance (PWA, Brier, accuracy, etc.) across folds
- **Financial Time Series Focus** — Full support for sample weights and temporal order preservation

## Dependencies

- Python ≥ 3.7
- numpy
- pandas
- scikit-learn ≥ 0.24
- matplotlib

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Import the toolkit
from calibration import (
    CVIsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    plot_reliability,
    calibration_report
)
from cross_validation import PurgedKFold   # your purged CV implementation

# Generate synthetic financial data
X, y = make_classification(n_samples=1000, random_state=42)
t1 = pd.Series(
    index=pd.date_range('2020-01-01', periods=1000, freq='D'),
    data=pd.date_range('2020-01-15', periods=1000, freq='D')
)

# Purged cross-validation
cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

# Create and fit calibrator (supports isotonic or Platt scaling)
calibrator = CVIsotonicCalibrator(
    estimator=RandomForestClassifier(random_state=42),
    cv=cv,
    method="isotonic"          # "isotonic" (default) or "sigmoid"
)

calibrator.fit(X, y)

# Get calibrated probabilities
calibrated_probs = calibrator.predict_proba(X)[:, 1]

# Evaluate calibration quality
print(f"Brier score: {brier_score(y, calibrated_probs):.4f}")
print(f"ECE: {expected_calibration_error(y, calibrated_probs):.4f}")

# Generate report and plot
report = calibration_report(y, calibrated_probs)
print(report)

plot_reliability(y, calibrated_probs, title="After Calibration")
plt.show()
```

## Main Components

### CVIsotonicCalibrator

A scikit-learn compatible calibrator that performs cross-validated calibration using out-of-fold predictions from purged cross-validation. Supports both isotonic regression and Platt scaling.

```python
calibrator = CVIsotonicCalibrator(
    estimator=RandomForestClassifier(),
    cv=cv,
    method="isotonic"   # or "sigmoid"
)
calibrator.fit(X, y, sample_weight=weights)   # supports sample weights
calibrated_probs = calibrator.predict_proba(X)[:, 1]
```

**Available scoring methods:**
- `.score()` → Probability-Weighted Accuracy (PWA, higher is better)
- `.brier_score()` → Brier score (lower is better)

### analyze_calibrated_cross_val_scores

Performs a full cross-validation comparison between the raw estimator and the calibrated version.

```python
ret_scores, scores_df, confusion_matrices = analyze_calibrated_cross_val_scores(
    base_estimator=RandomForestClassifier(),
    X=X,
    y=y,
    cv_gen=cv,
    sample_weight_train=weights,
    sample_weight_score=weights,
    calibrator_cv=cv
)
print(scores_df)
```

### Calibration Metrics

- `brier_score(y_true, p_pred)` — Mean squared error between predictions and outcomes (lower is better)
- `expected_calibration_error(y_true, p_pred, n_bins=10, strategy='uniform')` — Average absolute deviation between predicted probability and observed frequency
- `maximum_calibration_error(y_true, p_pred, n_bins=10, strategy='uniform')` — Worst-case deviation across bins

### Reliability Diagrams

- `plot_reliability(y_true, p_pred, ...)` — Simple calibration curve with optional histogram
- `plot_reliability_with_ci(y_true, p_pred, n_bootstraps=1000, ...)` — With bootstrap 95% confidence intervals

### Calibration Report

```python
report = calibration_report(y_true, p_pred, p_calibrated=None, n_bins=10)
```

Returns a DataFrame with Brier, ECE, MCE, and improvement metrics when calibrated probabilities are provided.

## Advanced Usage

### Using Platt Scaling Directly

```python
from calibration import fit_platt_scaling, apply_calibration

platt = fit_platt_scaling(y_calib, scores_calib, sample_weight=weights)
calibrated = apply_calibration(platt, scores_test)
```

### Custom Cross-Validation

`CVIsotonicCalibrator` works with any cross-validator that implements `.split(X, y)`. For financial applications, `PurgedKFold` (or `PurgedWalkForwardCV`) is strongly recommended.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. (Chapter 7)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.

## License

This module is provided under the **MIT License**.

---

For questions or contributions, please refer to the project repository.
```

