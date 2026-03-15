# MetaTrader 5 Machine Learning Blueprint

> **A production-grade financial machine learning library for MetaTrader 5.**
> Implements and extends the techniques from *Advances in Financial Machine Learning*
> by Marcos López de Prado — with MT5-specific engineering, Numba-accelerated
> algorithms, and an intelligent caching architecture built for real-world trading
> research.

[![Python](https://img.shields.io/pypi/pyversions/mlfinlab.svg)](https://www.python.org/)
[![Build Status](https://travis-ci.com/pnjoroge54/Machine-Learning-Blueprint.svg?branch=main)](https://travis-ci.com/pnjoroge54/Machine-Learning-Blueprint)
[![codecov](https://codecov.io/gh/pnjoroge54/Machine-Learning-Blueprint/branch/main/graph/badge.svg)](https://codecov.io/gh/pnjoroge54/Machine-Learning-Blueprint)
[![pylint Score](https://mperlet.github.io/pybadge/badges/10.svg)](https://github.com/pnjoroge54/Machine-Learning-Blueprint)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.txt)

---

## About This Project

This repository is the companion codebase for the
**MetaTrader 5 Machine Learning Blueprint** article series, published on
[MQL5.com](https://www.mql5.com/en/users/patricknjoroge743/publications)
by **[Patrick Murimi Njoroge](https://www.mql5.com/en/users/patricknjoroge743)**.

The series constructs a complete, reproducible ML pipeline — from raw tick data
ingestion in MetaTrader 5 through to ONNX model deployment in a live EA —
addressing the real-world pitfalls that cause most trading ML projects to fail
out-of-sample.

---

## Article Series

| # | Title | Published | Core Topics |
|---|-------|-----------|-------------|
| [1](https://www.mql5.com/en/articles/17520) | Data Leakage & Timestamp Fixes | Jun 2025 | MT5 timestamp trap · activity-driven bars · look-ahead bias |
| [2](https://www.mql5.com/en/articles/18864) | Labeling Financial Data for ML | Aug 2025 | Triple-Barrier Method · meta-labeling · CUSUM filter |
| [3](https://www.mql5.com/en/articles/19253) | Trend-Scanning Labeling | Oct 2025 | Adaptive t-stat horizons · 350× Numba speedup |
| [4](https://www.mql5.com/en/articles/19850) | Label Concurrency | Oct 2025 | Sample weights · average uniqueness · IID correction |
| [5](https://www.mql5.com/en/articles/20059) | Sequential Bootstrapping | Nov 2025 | Debiased sampling · SB bagging · Monte Carlo validation |
| [6](https://www.mql5.com/en/articles/20302) | Production-Grade Caching System | Nov 2025 | AFML cache · finance-aware invalidation · session vs disk |
| [7](https://www.mql5.com/en/articles/20451) | Reproducible Research Pipeline | Feb 2026 | End-to-end pipeline · TickDataLoader · ONNX export |

---

## Repository Structure

```
Machine-Learning-Blueprint/
│
├── afml/                       # Core library
│   └── cache/                  # Production caching system (Part 6)
│       ├── backtest_cache.py   # Backtest-stage cache
│       ├── cache_monitoring.py # Hit/miss telemetry
│       ├── cv_cache.py         # Cross-validation cache
│       ├── data_access_tracker.py  # Look-ahead bias guard
│       ├── robust_cache_keys.py    # Pandas/NumPy-safe hashing
│       └── selective_cleaner.py    # Code-change invalidation
│
├── MQL5/                       # Expert Advisors & Python bridge EA
│
├── data/                       # Synthetic / sample market data only
│
├── notebooks/                  # Jupyter notebooks (one per article)
│
├── testing_tuning/             # Walk-forward & parameter search
│
├── tools/                      # Standalone utility scripts
│
├── util/                       # Shared helpers (multiprocessing, logging)
│
├── performance_attribution.py
├── requirements.txt
├── environment.yml
└── setup.py
```

---

## Key Features

### Data Structures — Part 1
- Tick, volume, and dollar bar construction from raw MT5 tick data
- MT5 timestamp correction (bar-close alignment) to eliminate look-ahead bias
- Activity-driven sampling for improved statistical properties

### Labeling — Parts 2 & 3
- **Triple-Barrier Method** with dynamic, volatility-scaled (`get_daily_vol`) barriers
- **Meta-Labeling** — secondary model to size position conviction
- **Trend-Scanning** — t-stat-driven adaptive horizons with a Numba-JIT core
  (~350× faster than the original López de Prado reference implementation)

### Sample Weighting — Parts 4 & 5
- Concurrency counting (`num_concurrent_events`) and average uniqueness
- Time-decay weights for recency-aware training
- **Sequential Bootstrapping** — actively avoids temporally overlapping samples
  during resampling, correcting the IID violation at its source

### Caching System — Part 6
- Finance-aware persistent cache (disk-backed, survives Python restarts)
- Automatic invalidation when source code or upstream data changes
- Look-ahead-bias guard on cached intermediate outputs
- Handles Pandas DataFrames and NumPy arrays that Python's `lru_cache` cannot hash

### Reproducible Pipeline — Part 7
- `TickDataLoader` — RAM cache with intelligent partial-range loading
- Dependency-graph-aware invalidation (changing hyperparameters skips re-running
  bar construction; changing symbol triggers a full recompute)
- Automatic research reports at every pipeline stage
- ONNX model export for direct deployment inside MetaTrader 5

---

## Getting Started

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal (for live data; sample data provided for offline use)

### Installation

**Conda (recommended):**
```bash
git clone https://github.com/pnjoroge54/Machine-Learning-Blueprint.git
cd Machine-Learning-Blueprint
conda env create -f environment.yml
conda activate ml-blueprint
pip install -e .
```

**pip:**
```bash
git clone https://github.com/pnjoroge54/Machine-Learning-Blueprint.git
cd Machine-Learning-Blueprint
pip install -r requirements.txt
pip install -e .
```

### Quick-Start Example

```python
from afml.data_structures import get_dollar_bars
from afml.labeling import get_events, get_bins, get_daily_vol
from afml.sample_weights import get_sample_weights

# 1. Build activity-driven bars from raw tick data
bars = get_dollar_bars(ticks, threshold=1_000_000)

# 2. Compute dynamic volatility targets
daily_vol = get_daily_vol(bars['close'], lookback=100)

# 3. Label with the Triple-Barrier Method
events = get_events(close=bars['close'], t_events=t_events,
                    pt_sl=[1, 1], target=daily_vol, min_ret=0.005)
labels = get_bins(events, bars['close'])

# 4. Correct for label concurrency before training
weights = get_sample_weights(events, bars['close'])

# 5. Train with corrected sample weights
model.fit(X_train, y_train, sample_weight=weights.loc[X_train.index])
```

See `notebooks/` for fully worked examples corresponding to each article.

---

## Running Tests

```bash
# Full test suite
pytest

# With HTML coverage report
pytest --cov=afml --cov-report=html
open htmlcov/index.html
```

---

## Code Quality

```bash
# Lint
pylint afml/

# Style
pycodestyle afml/
```

---

## Attribution & Acknowledgements

### Theoretical Foundation

All core algorithms implement techniques from:

- **Advances in Financial Machine Learning** — Marcos López de Prado (Wiley, 2018)
- **Machine Learning for Asset Managers** — Marcos López de Prado (Cambridge, 2020)

Every function that implements a named snippet from these books carries a
docstring citation with the book title, snippet number, and page number.

### Codebase Scaffold

The project structure, CI configuration, and initial module layout were
bootstrapped from **[MlFinLab](https://github.com/hudson-and-thames/mlfinlab)**
by Hudson & Thames Quantitative Research, used under the BSD-3-Clause licence.
The original licence is preserved in [LICENSE.txt](LICENSE.txt).

Original contributions in this repository — including the AFML caching system,
the Numba-accelerated trend-scanning implementation, the `TickDataLoader`, the
MetaTrader 5 Python bridge, and the end-to-end reproducible pipeline — are the
work of **Patrick Murimi Njoroge** and are documented in the MQL5 article series
above.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

BSD-3-Clause. See [LICENSE.txt](LICENSE.txt) for details.
