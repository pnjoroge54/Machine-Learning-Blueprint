import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .combinatorial import CombinatorialPurgedCV, CPCVAnalyzer, optimal_folds_number


# ── 1. Configure the CV generator ────────────────────────────────────────
# Use optimal_folds_number to find N and k that meet your targets.
# Here: aim for ~600 obs in each training set and 5 backtest paths.
n_folds, n_test_folds = optimal_folds_number(
    n_observations=1000,
    target_train_size=600,
    target_n_test_paths=5,
)
print(f"n_folds={n_folds}, n_test_folds={n_test_folds}")

cv = CombinatorialPurgedCV(
    n_folds=n_folds,
    n_test_folds=n_test_folds,
    t1=t1_outer,          # pd.Series: index=event start, values=event end
    pct_embargo=0.01,     # removes 1% of obs after each test block
)

print(cv.summary(X_outer))
# Number of Observations      1000
# Total Number of Folds          6
# Number of Test Folds           2
# Number of Test Paths           5
# Number of Training Combinations 15


# ── 2. Fit and predict across all combinatorial splits ───────────────────
# CPCVAnalyzer handles the parallel execution and path recombination.
analyzer = CPCVAnalyzer(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv_gen=cv,
    close_prices=close_prices,   # pd.Series of price for MtM Sharpe calculation
)

# fit_predict exhausts the full split() generator internally, which populates
# cv.index_train_test_ across ALL n_splits columns — not just split 0.
recombined_preds = analyzer.fit_predict(X_outer, y_outer)


# ── 3. Inspect the distribution of path Sharpe ratios ───────────────────
# Each path is an independent backtest covering the full timeline once.
# A tight, positive distribution signals robustness.
metrics = analyzer.get_distribution_metrics(primary_sides=sides)

path_sharpes = metrics.xs("binary", level="method")["mtm_sharpe"]
print("Path Sharpe ratios:")
print(path_sharpes.to_string())
print(f"Mean: {path_sharpes.mean():.3f}  Std: {path_sharpes.std():.3f}")


# ── 4. Visualise train/test/purge observation assignments ────────────────
#
# plot_train_test_index renders a colour-coded table:
#   Blue  (0)  → training observation for that split
#   Red   (1)  → test observation for that split
#   Green (-1) → excluded: purged (label overlap) or embargoed (positional buffer)
#
# Rows = individual observations; Columns = splits (0 … n_splits-1).
#
# The method now checks for index_train_test_ before doing any work:
#   - If fit_predict has already run, the matrix is fully populated and
#     the plot is rendered immediately with no additional computation.
#   - If the matrix does not yet exist, the full generator is exhausted
#     automatically before rendering — no manual list(cv.split()) needed.
fig_index = cv.plot_train_test_index(X_outer)   # ✓ always safe
fig_index.show()

# plot_train_test_folds shows the simpler fold-level view (no purge/embargo detail):
fig_folds = cv.plot_train_test_folds()
fig_folds.show()
