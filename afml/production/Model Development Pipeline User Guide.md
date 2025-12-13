# **Model Development Pipeline User Guide**

## _Enhanced Logging & Intelligent Parameter Search_

---

## **Table of Contents**

1. [Quick Start](#quick-start)
2. [Enhanced Logging System](#enhanced-logging-system)
3. [Intelligent Parameter Search](#intelligent-parameter-search)
4. [Integration with Existing Pipelines](#integration-with-existing-pipelines)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## **1. Quick Start**

### **Basic Integration (5 Minutes)**

```python
# Replace your existing pipeline call with:
from model_development import develop_production_model_with_enhanced_logging

# Your existing configuration
config = {
    'symbol': 'EURUSD',
    'train_start': '2023-01-01',
    'train_end': '2023-06-30',
    'strategy': your_strategy,
    'data_config': {...},
    'feature_config': {...},
    'label_config': {...},
    'model_params': {...}
}

# Enhanced pipeline with logging and smart search
model, features, metrics, config = develop_production_model_with_enhanced_logging(
    **config,
    enable_advanced_search=True,  # Enable intelligent parameter search
    log_level="INFO",             # Logging level: DEBUG, INFO, WARNING, ERROR
    save_logs=True                # Save logs to file
)
```

### **Minimal Configuration**

```python
# Just add these two lines to your existing code:
from model_development import LoggingModelDevelopmentPipeline

# Replace: pipeline = ModelDevelopmentPipeline(...)
pipeline = LoggingModelDevelopmentPipeline(...)  # That's it!
```

---

## **2. Enhanced Logging System**

### **2.1 What Gets Logged**

| **Category**          | **What's Logged**                                  | **Example Output**                                                         |
| --------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| **Pipeline Steps**    | Start/end times, duration, success/failure         | `{"step": "feature_engineering", "duration": 45.2, "status": "completed"}` |
| **Data Metrics**      | Sample counts, feature dimensions, missing values  | `{"samples": 10000, "features": 50, "missing_pct": 0.5}`                   |
| **Model Training**    | CV scores, parameters, training time               | `{"cv_score": 0.85, "best_params": {...}, "training_time": 120}`           |
| **Parameter Search**  | Search progress, best scores, parameter importance | `{"iteration": 15, "best_score": 0.87, "top_param": "max_depth"}`          |
| **Errors & Warnings** | Full stack traces, context information             | `{"error": "MemoryError", "step": "model_training", "trace": "..."}`       |

### **2.2 Configuration Options**

```python
import structlog
from model_development import ModelDevelopmentLogger

# Create logger with custom configuration
logger = ModelDevelopmentLogger(
    name="my_trading_model",          # Logger name
    level="INFO",                     # Log level
    output_format="json",             # JSON or plain text
    enable_file_logging=True,         # Save to file
    file_path="logs/model_dev.log",   # Log file location
    enable_console=True               # Print to console
)

# Available log levels (in order of severity):
# DEBUG → INFO → WARNING → ERROR → CRITICAL
```

### **2.3 Log File Structure**

```
models/
├── EURUSD/
│   ├── 1min/
│   │   └── 20240101_20240630/
│   │       ├── logs/
│   │       │   ├── pipeline.log          # Human-readable logs
│   │       │   ├── pipeline.jsonl        # Structured JSON logs
│   │       │   └── errors.log           # Error-only logs
│   │       └── ...
```

### **2.4 Querying and Analyzing Logs**

```python
# Load and analyze logs
from model_development import LogAnalyzer

analyzer = LogAnalyzer("logs/pipeline.jsonl")

# Get pipeline summary
summary = analyzer.get_pipeline_summary()
# Returns: {'total_steps': 7, 'successful': 6, 'failed': 1, 'total_time': 325.5}

# Find bottlenecks
bottlenecks = analyzer.identify_bottlenecks()
# Returns: [{'step': 'feature_engineering', 'duration': 145.2, 'percent_of_total': 44.6}]

# Search logs
errors = analyzer.search_logs(level="ERROR", step="model_training")

# Generate report
analyzer.generate_report(output_file="logs/analysis_report.html")
```

### **2.5 Real-time Monitoring**

```python
# Monitor pipeline progress in real-time
from model_development import PipelineMonitor

monitor = PipelineMonitor(
    pipeline_id="EURUSD_1min_model",
    update_interval=30,  # Update every 30 seconds
    metrics_to_track=['accuracy', 'f1_score', 'training_time']
)

# Start monitoring (runs in background)
monitor.start()

# Check status anytime
status = monitor.get_status()
# Returns: {'current_step': 'model_training', 'progress': 65, 'eta': '12:45'}

# Get live metrics dashboard URL
dashboard_url = monitor.get_dashboard_url()
```

---

## **3. Intelligent Parameter Search**

### **3.1 Search Strategies**

#### **A. Adaptive Random Search (Default)**

```python
# Automatically chooses optimal strategy
search_config = {
    'strategy': 'adaptive',  # Let system choose best approach
    'n_iterations': 50,      # Total iterations
    'time_budget': 3600,     # Stop after 1 hour
    'early_stopping': True   # Stop if no improvement
}
```

#### **B. Bayesian Optimization (Best for small budgets)**

```python
search_config = {
    'strategy': 'bayesian',
    'n_iterations': 30,
    'acquisition_function': 'EI',  # Expected Improvement
    'random_starts': 10            # Random searches before Bayesian
}
```

#### **C. Hyperband (Best for large spaces)**

```python
search_config = {
    'strategy': 'hyperband',
    'max_iterations': 81,          # 3^4 parameter combinations
    'eta': 3,                      # Aggressiveness of elimination
    'min_budget': 1,               # Minimum resources per configuration
    'max_budget': 27               # Maximum resources per configuration
}
```

### **3.2 Search Space Definition**

```python
# Traditional approach (still works)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Enhanced approach with distributions
from scipy.stats import loguniform, randint

param_distributions = {
    'n_estimators': randint(50, 500),           # Uniform between 50-500
    'max_depth': randint(3, 20),                # Uniform between 3-20
    'min_samples_split': randint(2, 20),        # Uniform between 2-20
    'max_features': loguniform(0.1, 1.0),       # Log-uniform for percentages
    'learning_rate': [0.01, 0.05, 0.1, 0.2]     # Discrete values still work
}

# Conditional parameters
conditional_params = {
    'boosting_type': ['gbdt', 'dart'],
    'drop_rate': {                     # Only relevant if boosting_type='dart'
        'condition': lambda params: params['boosting_type'] == 'dart',
        'distribution': uniform(0.1, 0.5)
    }
}
```

### **3.3 Multi-Objective Optimization**

```python
# Optimize for multiple objectives simultaneously
objectives = [
    {
        'name': 'f1_score',
        'direction': 'maximize',
        'weight': 0.6
    },
    {
        'name': 'inference_speed',
        'direction': 'minimize',  # Faster is better
        'weight': 0.2,
        'threshold': 100          # Must be under 100ms
    },
    {
        'name': 'model_size',
        'direction': 'minimize',
        'weight': 0.1
    },
    {
        'name': 'calibration',
        'direction': 'maximize',
        'weight': 0.1
    }
]

# Get Pareto-optimal solutions
pareto_front = search.get_pareto_front()
# Returns multiple solutions trading off different objectives
```

### **3.4 Market Regime-Specific Optimization**

```python
# Optimize parameters for different market conditions
regime_configs = {
    'high_volatility': {
        'timeframe': '1min',
        'parameters': {
            'max_depth': (3, 7),      # Shallow trees for noise
            'n_estimators': (200, 500) # Many trees
        }
    },
    'low_volatility': {
        'timeframe': '1hour',
        'parameters': {
            'max_depth': (10, 20),    # Deeper patterns
            'n_estimators': (50, 150) # Fewer trees
        }
    },
    'trending': {
        'parameters': {
            'max_features': (0.8, 1.0)  # Use most features
        }
    }
}

# Automatically detect regime and use appropriate parameters
adaptive_search = RegimeAwareParameterSearch(
    regime_configs=regime_configs,
    regime_detector=your_regime_detector
)
```

### **3.5 Visualization and Analysis**

```python
# Generate search visualizations
search.visualize(
    save_path="models/search_visualizations/",
    include=[
        'convergence_plot',      # How score improved over time
        'parameter_importance',  # Which parameters mattered most
        'parallel_coordinates',  # Relationships between parameters
        'contour_plots',         # 2D parameter interactions
        'pareto_front'           # Trade-offs between objectives
    ]
)

# Analyze search results
analysis = search.analyze_results()

print(f"Best score: {analysis.best_score}")
print(f"Parameter importance: {analysis.parameter_importance}")
print(f"Suggested next experiments: {analysis.suggestions}")
```

---

## **4. Integration with Existing Pipelines**

### **4.1 Minimal Changes Required**

#### **Option A: Replace Function Call**

```python
# OLD:
from model_development import develop_production_model

# NEW:
from model_development import develop_production_model_with_enhanced_logging

# Same parameters, plus new options
result = develop_production_model_with_enhanced_logging(
    # Your existing parameters...
    # Plus new optional parameters:
    enable_advanced_search=True,
    log_level="INFO",
    search_time_budget=7200,  # 2 hours max
    save_search_artifacts=True
)
```

#### **Option B: Use Enhanced Pipeline Class**

```python
# OLD:
pipeline = ModelDevelopmentPipeline(...)

# NEW:
pipeline = LoggingModelDevelopmentPipeline(
    # Existing parameters
    symbol=symbol,
    train_start=train_start,
    train_end=train_end,
    strategy=strategy,
    data_config=data_config,
    feature_config=feature_config,
    label_config=label_config,
    model_params=model_params,

    # New optional parameters
    logger_config={
        'level': 'INFO',
        'format': 'json',
        'output_dir': 'logs/'
    },
    search_config={
        'strategy': 'adaptive',
        'n_iterations': 50,
        'early_stopping': True
    }
)
```

### **4.2 Configuration Migration**

Your existing configuration files work as-is. Add new sections for enhanced features:

```yaml
# config.yaml - Existing structure
symbol: EURUSD
train_start: 2023-01-01
train_end: 2023-06-30
data_config:
  bar_type: tick
  bar_size: 1min
  price: mid

# Add these new sections:
enhanced_features:
  logging:
    enabled: true
    level: INFO
    output_format: json
    save_to_file: true

  parameter_search:
    enabled: true
    strategy: adaptive
    time_budget: 3600 # seconds
    n_iterations: 100
    objectives:
      - name: f1_score
        weight: 0.7
      - name: inference_speed
        weight: 0.3
        max_threshold: 50ms

  monitoring:
    enabled: true
    dashboard: true
    alert_thresholds:
      error_count: 5
      memory_usage: 80%
      duration: 2h
```

### **4.3 Backward Compatibility**

All existing code continues to work. New features are opt-in:

```python
# Existing code works unchanged
model, features, metrics = develop_production_model(...)

# New features available when you're ready
model, features, metrics, logs, search_results = (
    develop_production_model_with_enhanced_logging(...)
)
```

---

## **5. Advanced Features**

### **5.1 Transfer Learning Across Symbols**

```python
# Learn from previous optimizations
from model_development import TransferLearningSearch

# Initialize with knowledge base
knowledge_base = TransferLearningSearch.load_knowledge_base(
    "knowledge/parameter_knowledge.db"
)

# Warm start new search
search = TransferLearningSearch(
    knowledge_base=knowledge_base,
    similarity_threshold=0.7  # Use knowledge from similar symbols
)

# Search starts with intelligent priors
results = search.optimize(
    symbol="GBPUSD",
    timeframe="1min",
    initial_parameters="from_similar"  # Start from EURUSD 1min params
)
```

### **5.2 Automated Experiment Tracking**

```python
# Track every experiment automatically
from model_development import ExperimentTracker

tracker = ExperimentTracker(
    project_name="forex_trading_models",
    tracking_uri="http://localhost:5000",  # MLflow or similar
    autolog=True  # Automatically log parameters, metrics, artifacts
)

with tracker.start_run(experiment_name="EURUSD_1min_optimization"):
    # All metrics automatically tracked
    model, features, metrics = develop_production_model_with_enhanced_logging(...)

    # Manually log additional information
    tracker.log_artifact("feature_importance.png")
    tracker.log_param("market_regime", "high_volatility")
    tracker.log_metric("sharpe_ratio", 2.5)
```

### **5.3 Ensemble Optimization**

```python
# Optimize ensemble of models
from model_development import EnsembleOptimizer

ensemble_config = {
    'models': [
        {'type': 'RandomForest', 'params': {...}},
        {'type': 'XGBoost', 'params': {...}},
        {'type': 'LightGBM', 'params': {...}}
    ],
    'combination_methods': ['voting', 'stacking', 'averaging'],
    'optimization_objective': 'risk_adjusted_returns'
}

optimizer = EnsembleOptimizer(ensemble_config)

# Find best ensemble combination
best_ensemble = optimizer.optimize(
    X_train, y_train,
    validation_method='purged_cv',
    n_iterations=100
)

# Returns optimal weights and combination method
print(f"Best weights: {best_ensemble.weights}")
print(f"Combination method: {best_ensemble.method}")
```

### **5.4 Real-time Parameter Adaptation**

```python
# Adapt parameters in real-time based on performance
from model_development import AdaptiveModelManager

manager = AdaptiveModelManager(
    base_model=your_model,
    adaptation_strategy='performance_based',
    adaptation_frequency='daily',  # Can be 'hourly', 'daily', 'weekly'
    performance_window=30          # Look back 30 periods
)

# Deploy adaptive model
adaptive_model = manager.deploy()

# Model automatically adjusts parameters when performance degrades
predictions = adaptive_model.predict_live(data)
```

---

## **6. Troubleshooting**

### **Common Issues and Solutions**

| **Issue**                        | **Symptoms**                        | **Solution**                                        |
| -------------------------------- | ----------------------------------- | --------------------------------------------------- |
| **Logs not appearing**           | No log files created                | Check `log_level` setting (should be INFO or DEBUG) |
| **Search taking too long**       | Optimization running for hours      | Set `time_budget` or reduce `n_iterations`          |
| **Memory errors**                | Out of memory during search         | Enable `memory_efficient=True`, reduce search space |
| **No improvement in scores**     | Scores plateau early                | Increase `random_starts`, try different strategy    |
| **Errors in parallel execution** | Random failures in multi-processing | Set `n_jobs=1` to debug, then gradually increase    |

### **Debug Mode**

```python
# Enable debug mode for detailed troubleshooting
pipeline = LoggingModelDevelopmentPipeline(
    ...,
    debug_mode=True,  # Enables additional checks and verbose output
    fail_fast=False   # Continue after errors for debugging
)

# Check pipeline health
health_report = pipeline.get_health_report()
if not health_report['healthy']:
    print(f"Issues found: {health_report['issues']}")

# Generate debugging information
debug_info = pipeline.generate_debug_info(
    include=['memory_usage', 'performance_metrics', 'system_info']
)
```

### **Performance Profiling**

```python
# Profile pipeline performance
from model_development import PipelineProfiler

profiler = PipelineProfiler(pipeline)
profiler.profile(
    n_runs=3,            # Run multiple times for stability
    warmup_runs=1,       # Warmup run discarded
    profile_memory=True, # Track memory usage
    profile_cpu=True     # Track CPU usage
)

# Get performance report
report = profiler.get_report()
print(f"Slowest step: {report.slowest_step}")
print(f"Memory peak: {report.memory_peak_mb}MB")
print(f"Optimization suggestions: {report.suggestions}")
```

---

## **7. Best Practices**

### **7.1 Logging Best Practices**

```python
# DO: Use structured logging
logger.info("feature_generation_complete",
            feature_count=len(features),
            generation_time=45.2,
            memory_usage="150MB")

# DON'T: Use print statements
print(f"Generated {len(features)} features in 45.2 seconds")

# DO: Set appropriate log levels
logger.set_level("INFO")  # Production
logger.set_level("DEBUG") # Development

# DO: Separate logs by component
train_logger = ModelDevelopmentLogger("training")
search_logger = ModelDevelopmentLogger("search")
data_logger = ModelDevelopmentLogger("data_loading")
```

### **7.2 Search Optimization Tips**

```python
# 1. Start broad, then narrow
search_config = {
    'phase_1': {  # Broad exploration
        'strategy': 'random',
        'n_iterations': 20,
        'param_space': 'broad'
    },
    'phase_2': {  # Focused refinement
        'strategy': 'bayesian',
        'n_iterations': 30,
        'param_space': 'narrowed'  # Based on phase 1 results
    }
}

# 2. Use prior knowledge
search = IntelligentParameterSearch(
    warm_start=True,
    prior_knowledge=load_previous_best_params(),
    exploration_factor=0.3  # 30% exploration, 70% exploitation
)

# 3. Monitor search progress
search.set_progress_callback(
    lambda iteration, score, params:
        logger.info("search_progress",
                   iteration=iteration,
                   current_score=score,
                   best_score=search.best_score)
)
```

### **7.3 Production Deployment Checklist**

```python
# Before deploying to production:
checklist = {
    'logging': [
        '✅ Log level set to INFO or higher',
        '✅ Log files properly rotated',
        '✅ Error alerts configured',
        '✅ Log storage has sufficient space'
    ],
    'search': [
        '✅ Time budget enforced',
        '✅ Memory limits configured',
        '✅ Early stopping enabled',
        '✅ Results cached for reproducibility'
    ],
    'monitoring': [
        '✅ Dashboard accessible',
        '✅ Performance metrics tracked',
        '✅ Alert thresholds set',
        '✅ Backup procedures in place'
    ]
}

# Run production readiness check
from model_development import ProductionReadinessChecker

checker = ProductionReadinessChecker(pipeline)
readiness_report = checker.run_checks()

if readiness_report['ready_for_production']:
    print("✅ Pipeline ready for production deployment")
else:
    print(f"⚠️ Issues found: {readiness_report['issues']}")
```

### **7.4 Performance Tuning**

```python
# Tune for different scenarios
tuning_configs = {
    'development': {
        'logging_level': 'DEBUG',
        'search_iterations': 10,
        'parallel_jobs': 2,
        'cache_size': '1GB'
    },
    'testing': {
        'logging_level': 'INFO',
        'search_iterations': 30,
        'parallel_jobs': 4,
        'cache_size': '2GB'
    },
    'production': {
        'logging_level': 'WARNING',
        'search_iterations': 50,
        'parallel_jobs': -1,  # Use all cores
        'cache_size': '5GB',
        'memory_limit': '80%'
    }
}

# Apply configuration
config = tuning_configs[environment]
pipeline.apply_configuration(config)
```

---

## **Quick Reference Card**

### **Most Used Commands**

```python
# 1. Run with enhanced features
develop_production_model_with_enhanced_logging(...)

# 2. Create logged pipeline
pipeline = LoggingModelDevelopmentPipeline(...)

# 3. Check pipeline status
pipeline.get_status()

# 4. View logs
pipeline.view_logs(tail=100)  # Last 100 lines

# 5. Export results
pipeline.export_results("output/")

# 6. Generate report
pipeline.generate_report()

# 7. Compare multiple runs
compare_runs(["run_1", "run_2", "run_3"])
```

### **Configuration Reference**

| **Parameter**     | **Default**  | **Description**         |
| ----------------- | ------------ | ----------------------- |
| `log_level`       | `"INFO"`     | Logging verbosity       |
| `search_strategy` | `"adaptive"` | Parameter search method |
| `n_iterations`    | `50`         | Search iterations       |
| `time_budget`     | `None`       | Maximum search time     |
| `parallel_jobs`   | `-1`         | Number of parallel jobs |
| `cache_size`      | `"2GB"`      | Cache memory limit      |
| `early_stopping`  | `True`       | Stop if no improvement  |
| `save_artifacts`  | `True`       | Save logs and results   |

---

## **Getting Help**

### **Useful Debugging Commands**

```python
# Get detailed pipeline information
pipeline.info()  # Shows configuration, status, metrics

# Test individual components
pipeline.test_component("feature_engineering")
pipeline.test_component("model_training")

# Generate diagnostic report
diagnostics = pipeline.run_diagnostics()
if diagnostics['has_issues']:
    pipeline.fix_issues(diagnostics['issues'])

# Compare with previous runs
comparison = pipeline.compare_with_previous()
```

### **Common Error Codes**

| **Code**     | **Meaning**                | **Action**                      |
| ------------ | -------------------------- | ------------------------------- |
| `LOG-001`    | Log directory not writable | Check permissions               |
| `SEARCH-002` | Parameter space too large  | Reduce search space             |
| `MEM-003`    | Memory limit exceeded      | Increase cache or reduce data   |
| `TIME-004`   | Time budget exceeded       | Reduce iterations or complexity |
| `DATA-005`   | Data validation failed     | Check input data quality        |

---

**Remember**: The enhanced features are designed to be **opt-in**. You can start with basic logging and gradually enable more advanced features as you become comfortable with the system.
