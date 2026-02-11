import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed
from math import comb
from numba import njit

# --- Numba Optimized Utilities ---

@njit(cache=True)
def fill_average_active_sides(num_close, t0_idx, t1_idx, side):
    """Calculates time-weighted average signal for overlapping events."""
    sum_side = np.zeros(num_close, dtype=np.float64)
    active_count = np.zeros(num_close, dtype=np.int32)
    
    for i in range(len(t0_idx)):
        start, end = t0_idx[i], t1_idx[i]
        if start != -1 and end != -1:
            sum_side[start : end + 1] += side[i]
            active_count[start : end + 1] += 1
            
    avg_side = np.zeros(num_close, dtype=np.float64)
    for t in range(num_close):
        if active_count[t] > 0:
            avg_side[t] = sum_side[t] / active_count[t]
    return avg_side

# --- Cross Validation ---

class CombinatorialPurgedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, n_test_splits=2, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        group_arrays = np.array_split(indices, self.n_splits)
        group_bounds = [(arr[0], arr[-1] + 1) for arr in group_arrays]
        
        for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = np.concatenate([indices[group_bounds[i][0]:group_bounds[i][1]] for i in test_group_ids])
            
            test_time_ranges = []
            for gid in test_group_ids:
                start_ix, end_ix = group_bounds[gid]
                test_time_ranges.append((self.t1.index[start_ix], self.t1.iloc[start_ix:end_ix].max()))

            train_mask = np.ones(len(indices), dtype=bool)
            train_mask[test_indices] = False
            train_indices = indices[train_mask]
            
            keep_mask = np.ones(len(train_indices), dtype=bool)
            embargo_offset = int(len(indices) * self.pct_embargo)
            
            train_starts = self.t1.index[train_indices]
            train_ends = self.t1.iloc[train_indices].values
            
            for test_start, test_end in test_time_ranges:
                idx_in_full = self.t1.index.searchsorted(test_end)
                embargo_cutoff = self.t1.index[min(idx_in_full + embargo_offset, len(indices)-1)] if idx_in_full + embargo_offset < len(indices) else pd.Timestamp.max
                
                is_overlapping = (train_starts <= embargo_cutoff) & (train_ends >= test_start)
                keep_mask &= ~is_overlapping
            
            yield train_indices[keep_mask], test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return comb(self.n_splits, self.n_test_splits)

# --- Performance Analysis ---

class CPCVAnalyzer:
    def __init__(self, estimator, cv_gen, close_prices, n_jobs=-1):
        self.estimator = estimator
        self.cv_gen = cv_gen
        self.close = close_prices
        self.n_jobs = n_jobs
        self._prediction_matrix = None
        self._X = None
        # Standard MtM Return: log(P_t1 / P_t0)
        self.log_returns = np.log(self.close).diff().shift(-1).fillna(0)

    def fit_predict(self, X, y, sample_weight=None):
        self._X = X
        n_splits = self.cv_gen.get_n_splits(X)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_fold)(self.estimator, X, y, train, test, i, sample_weight) 
            for i, (train, test) in enumerate(self.cv_gen.split(X, y))
        )

        self._prediction_matrix = pd.DataFrame(np.nan, index=X.index, columns=range(n_splits))
        for fold_idx, test_idx, preds in results:
            self._prediction_matrix.iloc[test_idx, fold_idx] = preds
            
        return self.recombined_predictions

    @property
    def backtest_paths(self):
        N, k = self.cv_gen.n_splits, self.cv_gen.n_test_splits
        J = comb(N - 1, k - 1)
        group_indices = np.array_split(np.arange(len(self._X)), N)
        
        paths = []
        for j in range(J):
            path_series = pd.Series(index=self._X.index, dtype=float)
            for g_idx in range(N):
                idx = group_indices[g_idx]
                # Find columns where this group was in the test set
                valid_cols = self._prediction_matrix.columns[self._prediction_matrix.iloc[idx[0]].notna()]
                path_series.iloc[idx] = self._prediction_matrix.iloc[idx, valid_cols[j]]
            paths.append(path_series)
        return paths

    def get_distribution_metrics(self, num_classes, step_size=0.0):
        from ..bet_sizing.ch10_snippets import discrete_signal, get_signal
        
        results_list = []
        for i, path_preds in enumerate(self.backtest_paths):
            # 1. Bet Sizing
            signals = get_signal(path_preds, num_classes)
            if step_size > 0:
                signals = discrete_signal(signals, step_size)
            
            # 2. MtM Timeline Mapping
            t0_idx = self.close.index.get_indexer(signals.index)
            t1_idx = self.close.index.get_indexer(self.cv_gen.t1.loc[signals.index])
            avg_pos = fill_average_active_sides(len(self.close), t0_idx, t1_idx, signals.values)
            
            # 3. Precision Returns
            log_rets = avg_pos * self.log_returns.values
            arithmetic_rets = np.expm1(log_rets)
            
            # 4. Metrics Calculation
            pos_sum = arithmetic_rets[arithmetic_rets > 0].sum()
            neg_sum = np.abs(arithmetic_rets[arithmetic_rets < 0].sum())
            
            results_list.append({
                'path_id': i,
                'mtm_profit_factor': pos_sum / neg_sum if neg_sum > 0 else 0,
                'mtm_sharpe': (log_rets.mean() / log_rets.std()) if log_rets.std() != 0 else 0,
                'max_drawdown': self._calculate_max_dd_precision(log_rets)
            })
        return pd.DataFrame(results_list)

    def _calculate_max_dd_precision(self, log_rets):
        cum_rets = np.exp(np.cumsum(log_rets))
        peak = np.maximum.accumulate(cum_rets)
        return np.min(cum_rets / peak - 1) if len(peak) > 0 else 0

    @property
    def recombined_predictions(self):
        return self._prediction_matrix.mean(axis=1)

def _fit_predict_fold(estimator, X, y, train_idx, test_idx, fold_idx, sample_weight=None):
    model = clone(estimator)
    if sample_weight is not None:
        model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sample_weight.iloc[train_idx])
    else:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    preds = model.predict_proba(X.iloc[test_idx])[:, 1] if hasattr(model, "predict_proba") else model.predict(X.iloc[test_idx])
    return fold_idx, test_idx, preds
