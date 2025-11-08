# Probability Calibration Toolkit User Guide

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Calibration Metrics](#calibration-metrics)
- [Visualization](#visualization)
- [Calibration Methods](#calibration-methods)
- [Cross-Validation Integration](#cross-validation-integration)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

The Probability Calibration Toolkit provides comprehensive tools for calibrating classifier probabilities and evaluating calibration quality, with special considerations for financial time series data. Proper calibration is crucial in financial applications where probability estimates directly impact position sizing, risk management, and trading decisions.

### Key Features

- **Calibration Metrics**: Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- **Reliability Analysis**: Calibration curves with confidence intervals
- **Multiple Calibration Methods**: Platt scaling and isotonic regression
- **Financial Data Integration**: Works with purged and combinatorial cross-validation
- **Bootstrap Confidence Intervals**: Statistical significance testing for calibration

## Installation and Setup

### Prerequisites

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
```

### Importing the Module

```python
from calibration import (
    brier_score, expected_calibration_error, maximum_calibration_error,
    compute_reliability, plot_reliability, plot_reliability_with_ci,
    fit_platt_scaling, fit_isotonic_calibration, apply_calibration,
    oof_predict_proba, calibrate_with_oof, calibrated_cross_val_predict,
    calibration_report
)
```

## Core Concepts

### What is Probability Calibration?

Probability calibration ensures that predicted probabilities reflect true event frequencies. For example, when a model predicts 70% probability, the event should occur approximately 70% of the time.

### Why Calibration Matters in Finance

- **Risk Management**: Accurate probabilities are essential for VaR calculations
- **Position Sizing**: Kelly criterion and other sizing methods rely on well-calibrated probabilities
- **Model Selection**: Helps distinguish between lucky and skillful models
- **Regulatory Compliance**: Well-calibrated models are often required for regulatory capital calculations

## Quick Start

### Basic Calibration Workflow

```python
# Generate sample financial data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
t1 = pd.Series(
    index=pd.date_range('2020-01-01', periods=1000, freq='D'),
    data=pd.date_range('2020-01-15', periods=1000, freq='D')
)

# Create classifier and get predictions
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
probabilities = clf.predict_proba(X)[:, 1]

# Evaluate calibration
brier = brier_score(y, probabilities)
ece = expected_calibration_error(y, probabilities)
print(f"Brier Score: {brier:.4f}, ECE: {ece:.4f}")

# Plot reliability diagram
plot_reliability(y, probabilities, title="Initial Calibration")
plt.show()
```

### Integration with Purged Cross-Validation

```python
from cross_validation import PurgedKFold

# Create purged CV for financial data
cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

# Get out-of-fold probabilities for proper calibration assessment
oof_probs = oof_predict_proba(clf, X, y, cv=cv)

# Fit calibration mapping
calibrator, calibrated_probs = calibrate_with_oof(
    clf, X, y, cv=cv, method='isotonic'
)

# Compare before and after calibration
report = calibration_report(y, oof_probs, calibrated_probs)
print(report)
```

## Calibration Metrics

### Brier Score

The Brier score measures the mean squared difference between predicted probabilities and actual outcomes.

```python
# Calculate Brier score
y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
score = brier_score(y_true, y_pred)
print(f"Brier Score: {score:.4f}")  # Lower is better
```

### Expected Calibration Error (ECE)

ECE measures the average absolute difference between predicted probabilities and observed frequencies across probability bins.

```python
ece_uniform = expected_calibration_error(y_true, y_pred, strategy='uniform')
ece_quantile = expected_calibration_error(y_true, y_pred, strategy='quantile')
print(f"ECE (uniform): {ece_uniform:.4f}")
print(f"ECE (quantile): {ece_quantile:.4f}")
```

### Maximum Calibration Error (MCE)

MCE identifies the worst-case calibration error across all probability bins.

```python
mce = maximum_calibration_error(y_true, y_pred)
print(f"Maximum Calibration Error: {mce:.4f}")
```

## Visualization

### Basic Reliability Diagram

```python
# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred = np.clip(y_true + np.random.normal(0, 0.3, 1000), 0.05, 0.95)

# Plot reliability diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

plot_reliability(y_true, y_pred, ax=ax1, title="Uniform Binning")
plot_reliability(y_true, y_pred, ax=ax2, strategy='quantile', title="Quantile Binning")

plt.tight_layout()
plt.show()
```

### Reliability Diagram with Confidence Intervals

```python
# Plot with bootstrap confidence intervals
plot_reliability_with_ci(
    y_true, y_pred, 
    n_bins=10, 
    n_bootstraps=1000,
    title="Reliability Diagram with 95% CI"
)
plt.show()
```

### Comprehensive Calibration Assessment

```python
def comprehensive_calibration_analysis(y_true, p_pred, model_name=""):
    """Generate comprehensive calibration analysis report."""
    
    # Compute metrics
    brier_val = brier_score(y_true, p_pred)
    ece_val = expected_calibration_error(y_true, p_pred)
    mce_val = maximum_calibration_error(y_true, p_pred)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Reliability diagram
    plot_reliability(y_true, p_pred, ax=axes[0, 0], title=f"{model_name} - Reliability")
    
    # Reliability with CI
    plot_reliability_with_ci(y_true, p_pred, ax=axes[0, 1], 
                           title=f"{model_name} - Reliability with CI")
    
    # Probability distribution
    axes[1, 0].hist(p_pred, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Predicted Probability Distribution')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    
    # Metrics summary
    axes[1, 1].axis('off')
    text = f"""Calibration Metrics:
    Brier Score: {brier_val:.4f}
    ECE: {ece_val:.4f}
    MCE: {mce_val:.4f}
    
    Interpretation:
    Brier < 0.01: Excellent
    Brier < 0.05: Good  
    Brier < 0.10: Reasonable
    Brier > 0.15: Poor"""
    
    axes[1, 1].text(0.1, 0.9, text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return {'brier': brier_val, 'ece': ece_val, 'mce': mce_val}

# Usage
metrics = comprehensive_calibration_analysis(y_true, y_pred, "Random Forest")
```

## Calibration Methods

### Platt Scaling (Sigmoid)

Platt scaling fits a logistic regression to map scores to probabilities.

```python
# Generate calibration data
y_calib = np.random.randint(0, 2, 500)
scores_calib = np.random.rand(500)

# Fit Platt scaling
platt_calibrator = fit_platt_scaling(y_calib, scores_calib)

# Apply calibration
new_scores = np.random.rand(100)
calibrated_probs = apply_calibration(platt_calibrator, new_scores)
```

### Isotonic Regression

Isotonic regression fits a non-decreasing step function to map scores to probabilities.

```python
# Fit isotonic regression
isotonic_calibrator = fit_isotonic_calibration(y_calib, scores_calib)

# Apply calibration
calibrated_probs = apply_calibration(isotonic_calibrator, new_scores)
```

### Method Comparison

```python
def compare_calibration_methods(y_true, scores, test_scores):
    """Compare different calibration methods."""
    
    methods = {
        'sigmoid': fit_platt_scaling(y_true, scores),
        'isotonic': fit_isotonic_calibration(y_true, scores)
    }
    
    results = {}
    for method_name, calibrator in methods.items():
        calibrated = apply_calibration(calibrator, test_scores)
        results[method_name] = calibrated
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for method_name, calibrated in results.items():
        plt.hist(calibrated, bins=30, alpha=0.5, label=method_name)
    
    plt.xlabel('Calibrated Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Calibration Methods Comparison')
    plt.show()
    
    return results
```

## Cross-Validation Integration

### Out-of-Fold Probabilities

```python
from cross_validation import PurgedKFold

# Create financial time series cross-validation
cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

# Get out-of-fold probabilities
oof_probs = oof_predict_proba(clf, X, y, cv=cv)

# Evaluate calibration on OOF probabilities
ece = expected_calibration_error(y, oof_probs)
print(f"Out-of-Fold ECE: {ece:.4f}")
```

### Full Calibration Workflow

```python
def full_calibration_pipeline(estimator, X, y, t1, calibration_method='isotonic'):
    """Complete calibration pipeline for financial data."""
    
    # Create purged cross-validation
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    
    # Step 1: Get out-of-fold probabilities
    print("Step 1: Generating out-of-fold probabilities...")
    oof_probs = oof_predict_proba(estimator, X, y, cv=cv)
    
    # Step 2: Evaluate pre-calibration
    print("Step 2: Evaluating pre-calibration performance...")
    pre_cal_metrics = {
        'brier': brier_score(y, oof_probs),
        'ece': expected_calibration_error(y, oof_probs),
        'mce': maximum_calibration_error(y, oof_probs)
    }
    
    # Step 3: Fit calibrator
    print("Step 3: Fitting calibration mapping...")
    calibrator, calibrated_probs = calibrate_with_oof(
        estimator, X, y, cv=cv, method=calibration_method
    )
    
    # Step 4: Evaluate post-calibration
    print("Step 4: Evaluating post-calibration performance...")
    post_cal_metrics = {
        'brier': brier_score(y, calibrated_probs),
        'ece': expected_calibration_error(y, calibrated_probs),
        'mce': maximum_calibration_error(y, calibrated_probs)
    }
    
    # Step 5: Generate report
    print("Step 5: Generating calibration report...")
    report = calibration_report(y, oof_probs, calibrated_probs)
    
    # Step 6: Visualize results
    print("Step 6: Creating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_reliability(y, oof_probs, ax=ax1, title="Pre-Calibration")
    plot_reliability(y, calibrated_probs, ax=ax2, title="Post-Calibration")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'calibrator': calibrator,
        'pre_calibration': pre_cal_metrics,
        'post_calibration': post_cal_metrics,
        'report': report,
        'oof_probs': oof_probs,
        'calibrated_probs': calibrated_probs
    }
```

### Combinatorial Purged CV Integration

```python
from combinatorial import CombinatorialPurgedKFold

def combinatorial_calibration(estimator, X, y, samples_info_sets):
    """Calibration with combinatorial purged cross-validation."""
    
    # Create combinatorial CV
    comb_cv = CombinatorialPurgedKFold(
        n_splits=5,
        n_test_splits=2,
        samples_info_sets=samples_info_sets,
        embargo=1
    )
    
    # Get calibrated probabilities using nested CV
    calibrated_probs = calibrated_cross_val_predict(
        estimator, X, y, cv=comb_cv, calibration_method='isotonic'
    )
    
    # Evaluate
    ece = expected_calibration_error(y, calibrated_probs)
    brier = brier_score(y, calibrated_probs)
    
    print(f"Combinatorial CV Calibration - ECE: {ece:.4f}, Brier: {brier:.4f}")
    
    return calibrated_probs
```

## Advanced Usage

### Meta-Labeling Calibration

```python
def meta_labeling_calibration(primary_model, meta_model, X, y_primary, y_meta, t1):
    """Calibrate both primary and meta-labeling models."""
    
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    
    # Calibrate primary model
    primary_calibrator, primary_probs = calibrate_with_oof(
        primary_model, X, y_primary, cv=cv, method='isotonic'
    )
    
    # Create meta-features including primary probabilities
    X_meta = X.copy()
    X_meta['primary_prob'] = primary_probs
    
    # Calibrate meta-model
    meta_calibrator, meta_probs = calibrate_with_oof(
        meta_model, X_meta, y_meta, cv=cv, method='isotonic'
    )
    
    return {
        'primary_calibrator': primary_calibrator,
        'meta_calibrator': meta_calibrator,
        'primary_probs': primary_probs,
        'meta_probs': meta_probs
    }
```

### Time Series Calibration Monitoring

```python
def calibration_drift_monitoring(estimator, X, y, t1, window_size=252):
    """Monitor calibration drift over time."""
    
    dates = X.index if hasattr(X, 'index') else pd.RangeIndex(len(X))
    ece_values = []
    brier_values = []
    time_points = []
    
    for end_idx in range(window_size, len(X), 30):  # Monthly evaluation
        start_idx = end_idx - window_size
        
        # Extract window
        X_window = X.iloc[start_idx:end_idx]
        y_window = y.iloc[start_idx:end_idx]
        t1_window = t1.iloc[start_idx:end_idx]
        
        # Create CV for this window
        cv = PurgedKFold(n_splits=3, t1=t1_window, pct_embargo=0.01)
        
        try:
            # Get OOF probabilities and evaluate calibration
            oof_probs = oof_predict_proba(estimator, X_window, y_window, cv=cv)
            ece = expected_calibration_error(y_window, oof_probs)
            brier = brier_score(y_window, oof_probs)
            
            ece_values.append(ece)
            brier_values.append(brier)
            time_points.append(dates[end_idx])
        except Exception as e:
            print(f"Error at window ending {dates[end_idx]}: {e}")
            continue
    
    # Plot drift
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time_points, ece_values, marker='o', color='red')
    ax1.set_title('Expected Calibration Error Over Time')
    ax1.set_ylabel('ECE')
    ax1.grid(True)
    
    ax2.plot(time_points, brier_values, marker='o', color='blue')
    ax2.set_title('Brier Score Over Time')
    ax2.set_ylabel('Brier Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame({
        'date': time_points,
        'ece': ece_values,
        'brier': brier_values
    })
```

### Ensemble Calibration

```python
def ensemble_calibration(models, X, y, t1, calibration_method='isotonic'):
    """Calibrate an ensemble of models."""
    
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    calibrated_models = []
    
    for i, model in enumerate(models):
        print(f"Calibrating model {i+1}/{len(models)}...")
        
        # Calibrate individual model
        calibrator, oof_probs = calibrate_with_oof(
            model, X, y, cv=cv, method=calibration_method
        )
        
        calibrated_models.append({
            'model': model,
            'calibrator': calibrator,
            'oof_probs': oof_probs
        })
    
    # Create ensemble probabilities
    ensemble_probs = np.mean([cm['oof_probs'] for cm in calibrated_models], axis=0)
    
    # Calibrate ensemble
    ensemble_calibrator = fit_isotonic_calibration(y, ensemble_probs)
    
    return {
        'calibrated_models': calibrated_models,
        'ensemble_calibrator': ensemble_calibrator,
        'ensemble_probs': ensemble_probs
    }
```

## Best Practices

### 1. Data Splitting Strategy

```python
def proper_calibration_setup(X, y, t1, test_size=0.2):
    """Proper setup for calibration with time series data."""
    
    # Split data temporally
    split_point = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    t1_train = t1.iloc[:split_point]
    
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]
    t1_test = t1.iloc[split_point:]
    
    # Create CV for training period only
    cv = PurgedKFold(n_splits=5, t1=t1_train, pct_embargo=0.01)
    
    return (X_train, y_train, t1_train, cv), (X_test, y_test, t1_test)
```

### 2. Probability Threshold Selection

```python
def find_optimal_threshold(y_true, p_pred, metric='f1'):
    """Find optimal probability threshold for classification."""
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.linspace(0.1, 0.9, 50)
    scores = []
    
    for threshold in thresholds:
        y_pred = (p_pred >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError("Metric must be 'f1', 'precision', or 'recall'")
        
        scores.append(score)
    
    optimal_threshold = thresholds[np.argmax(scores)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.title(f'Optimal Threshold Selection ({metric.upper()})')
    plt.show()
    
    return optimal_threshold
```

### 3. Calibration Validation

```python
def validate_calibration(calibrator, X_val, y_val, X_test, y_test):
    """Validate calibration on separate validation and test sets."""
    
    # Get base model predictions
    base_probs_val = calibrator.base_estimator.predict_proba(X_val)[:, 1]
    base_probs_test = calibrator.base_estimator.predict_proba(X_test)[:, 1]
    
    # Apply calibration
    cal_probs_val = calibrator.predict_proba(X_val)[:, 1]
    cal_probs_test = calibrator.predict_proba(X_test)[:, 1]
    
    # Compare metrics
    val_metrics = {
        'base_brier': brier_score(y_val, base_probs_val),
        'cal_brier': brier_score(y_val, cal_probs_val),
        'base_ece': expected_calibration_error(y_val, base_probs_val),
        'cal_ece': expected_calibration_error(y_val, cal_probs_val)
    }
    
    test_metrics = {
        'base_brier': brier_score(y_test, base_probs_test),
        'cal_brier': brier_score(y_test, cal_probs_test),
        'base_ece': expected_calibration_error(y_test, base_probs_test),
        'cal_ece': expected_calibration_error(y_test, cal_probs_test)
    }
    
    print("Validation Set Metrics:")
    print(f"  Base Brier: {val_metrics['base_brier']:.4f} -> Calibrated: {val_metrics['cal_brier']:.4f}")
    print(f"  Base ECE: {val_metrics['base_ece']:.4f} -> Calibrated: {val_metrics['cal_ece']:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  Base Brier: {test_metrics['base_brier']:.4f} -> Calibrated: {test_metrics['cal_brier']:.4f}")
    print(f"  Base ECE: {test_metrics['base_ece']:.4f} -> Calibrated: {test_metrics['cal_ece']:.4f}")
    
    return val_metrics, test_metrics
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Poor Calibration Performance

```python
def diagnose_calibration_issues(y_true, p_pred):
    """Diagnose common calibration issues."""
    
    # Check probability range
    print(f"Probability range: [{p_pred.min():.3f}, {p_pred.max():.3f}]")
    
    # Check for extreme probabilities
    extreme_low = np.sum(p_pred < 0.01) / len(p_pred)
    extreme_high = np.sum(p_pred > 0.99) / len(p_pred)
    print(f"Extreme probabilities (<0.01): {extreme_low:.1%}")
    print(f"Extreme probabilities (>0.99): {extreme_high:.1%}")
    
    # Check class distribution
    print(f"Positive class frequency: {y_true.mean():.1%}")
    
    # Check reliability by bins
    df_reliability = compute_reliability(y_true, p_pred)
    print("\nReliability by bins:")
    print(df_reliability[['bin_center', 'pred_mean', 'true_frac', 'count']])
    
    # Recommendations
    if extreme_low + extreme_high > 0.1:
        print("\nRecommendation: Consider clipping extreme probabilities")
    if df_reliability['count'].min() < len(y_true) * 0.01:
        print("Recommendation: Use fewer bins or quantile binning")
```

#### 2. Handling Small Datasets

```python
def small_dataset_calibration(estimator, X, y, t1, min_samples=100):
    """Calibration strategies for small datasets."""
    
    if len(X) < min_samples:
        print(f"Small dataset detected ({len(X)} samples). Using adapted strategy.")
        
        # Use leave-one-out or small number of folds
        n_splits = max(2, min(5, len(X) // 20))
        cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=0)
        
        # Use Platt scaling (fewer parameters than isotonic)
        calibrator, oof_probs = calibrate_with_oof(
            estimator, X, y, cv=cv, method='sigmoid'
        )
    else:
        # Use standard approach
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        calibrator, oof_probs = calibrate_with_oof(
            estimator, X, y, cv=cv, method='isotonic'
        )
    
    return calibrator, oof_probs
```

#### 3. Memory Optimization

```python
def memory_efficient_calibration(estimator, X, y, t1, chunk_size=10000):
    """Memory-efficient calibration for large datasets."""
    
    if len(X) > chunk_size:
        print("Large dataset detected. Using chunked processing.")
        
        # Process in chunks
        n_chunks = len(X) // chunk_size + 1
        all_oof_probs = np.zeros(len(X))
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X))
            
            print(f"Processing chunk {i+1}/{n_chunks}...")
            
            X_chunk = X.iloc[start_idx:end_idx]
            y_chunk = y.iloc[start_idx:end_idx]
            t1_chunk = t1.iloc[start_idx:end_idx]
            
            cv = PurgedKFold(n_splits=3, t1=t1_chunk, pct_embargo=0.01)
            chunk_oof_probs = oof_predict_proba(estimator, X_chunk, y_chunk, cv=cv)
            all_oof_probs[start_idx:end_idx] = chunk_oof_probs
        
        # Fit calibrator on full OOF probabilities
        calibrator = fit_isotonic_calibration(y, all_oof_probs)
        
        return calibrator, all_oof_probs
    else:
        # Standard approach
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        return calibrate_with_oof(estimator, X, y, cv=cv, method='isotonic')
```

This user guide provides comprehensive coverage of the Probability Calibration Toolkit. For additional support or to report issues, please refer to the module documentation or contact the development team.
