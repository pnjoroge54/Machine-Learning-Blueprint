import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed
from math import comb
from numba import njit
from scipy.stats import norm

# --- Numba Optimized Utilities ---

@njit(cache=True)
def fill_sides_numba(num_close, t0_idx, t1_idx, side):
    """
    Maps event-based signals to a continuous timeline by additive accumulation.

    Parameters
    ----------
    num_close : int
        The total number of bars in the reference price series.
    t0_idx : np.ndarray (int64)
        Integer indices for the start (entry) of each bet.
    t1_idx : np.ndarray (int64)
        Integer indices for the end (exit) of each bet.
    side : np.ndarray (float64)
        The signal values/sizes to be mapped.

    Returns
    -------
    np.ndarray (float64)
        A continuous timeline of accumulated bet sizes.
    """
    full_side = np.zeros(num_close, dtype=np.float64)
    for i in range(len(t0_idx)):
        start, end = t0_idx[i], t1_idx[i]
        if start != -1 and end != -1:
            full_side[start : end + 1] += side[i]
    return full_side

@njit(cache=True)
def fill_average_active_sides(num_close, t0_idx, t1_idx, side):
    """
    Maps event-based signals to a timeline by averaging concurrent signals.
    Implementation of AFML Snippet 10.3 logic.

    Parameters
    ----------
    num_close : int
        The total number of bars in the reference price series.
    t0_idx : np.ndarray (int64)
        Integer indices for signal entries.
    t1_idx : np.ndarray (int64)
        Integer indices for signal exits.
    side : np.ndarray (float64)
        The conviction/probability signals.

    Returns
    -------
    np.ndarray (float64)
        The time-weighted average signal at every timestamp.
    """
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
    """
    Combinatorial Purged Cross-Validation (CPCV) as defined by Marcos Lopez de Prado.
    
    Decomposes the dataset into N contiguous chunks and holds out k chunks as test sets.
    This creates multiple backtesting paths for a single model configuration.

    Parameters
    ----------
    n_splits : int, default 5
        The number of total groups (N) to split the data into.
    n_test_splits : int, default 2
        The number of groups (k) to be used in the test set per combination.
    t1 : pd.Series
        The information range (event end times). Index is start time, values are end times.
    pct_embargo : float, default 0.01
        Percentage of total samples to embargo after each test split to prevent leakage.
    """
    def __init__(self, n_splits=5, n_test_splits=2, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test sets.

        Yields
        ------
        train_indices : np.ndarray
            The purged and embargoed training indices.
        test_indices : np.ndarray
            The combinatorial test indices.
        """
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
                if idx_in_full + embargo_offset < len(indices):
                    embargo_cutoff = self.t1.index[idx_in_full + embargo_offset]
                else:
                    embargo_cutoff = pd.Timestamp.max
                
                is_overlapping = (train_starts <= embargo_cutoff) & (train_ends >= test_start)
                keep_mask &= ~is_overlapping
            
            yield train_indices[keep_mask], test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of combinatorial combinations (N over k)."""
        return comb(self.n_splits, self.n_test_splits)

# --- CPCV Analysis & Meta-Labeling Performance ---

class CPCVAnalyzer:
    """
    Manages the parallel execution of CPCV and performance metric distribution.
    
    This class handles the transition from event-based model predictions to 
    time-weighted portfolio performance across multiple backtest paths.

    Parameters
    ----------
    estimator : object
        A scikit-learn compatible estimator.
    cv_gen : CombinatorialPurgedKFold
        The CPCV cross-validation generator.
    close_prices : pd.Series
        Continuous price series used for Mark-to-Market calculation.
    n_jobs : int, default -1
        Number of cores to use for parallel model training.
    """
    def __init__(self, estimator, cv_gen, close_prices, n_jobs=-1):
        self.estimator = estimator
        self.cv_gen = cv_gen
        self.close = close_prices
        self.n_jobs = n_jobs
        self._prediction_matrix = None
        self._X = None
        # High-frequency log returns: r_t = ln(P_{t+1}/P_t)
        self.log_returns = np.log(self.close).diff().shift(-1).fillna(0)

    def fit_predict(self, X, y, sample_weight=None):
        """
        Trains the estimator across all CPCV splits in parallel.

        Returns
        -------
        pd.Series
            The 'recombined' (bagged) predictions for each timestamp.
        """
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
        """
        Assembles J backtest paths from the prediction matrix.
        
        Each path is constructed by taking exactly one OOS prediction per 
        contiguous time interval across the full dataset.

        Returns
        -------
        list of pd.Series
            A list containing J pandas Series, where each series represents 
            one unique out-of-sample backtest timeline.
        """
        N, k = self.cv_gen.n_splits, self.cv_gen.n_test_splits
        J = comb(N - 1, k - 1)
        group_indices = np.array_split(np.arange(len(self._X)), N)
        
        paths = []
        for j in range(J):
            path_series = pd.Series(index=self._X.index, dtype=float)
            for g_idx in range(N):
                idx = group_indices[g_idx]
                valid_cols = self._prediction_matrix.columns[self._prediction_matrix.iloc[idx[0]].notna()]
                path_series.iloc[idx] = self._prediction_matrix.iloc[idx, valid_cols[j]]
            paths.append(path_series)
        return paths

    def get_distribution_metrics(self, primary_sides, num_classes=2, step_size=0.0):
        """
        Calculates granular MtM performance metrics for all paths and bet-sizing methods.

        Parameters
        ----------
        primary_sides : pd.Series
            The direction (+1/-1) suggested by the primary model for meta-labeling.
        num_classes : int, default 2
            Number of classes used for confidence z-score calculation.
        step_size : float, default 0.0
            Discretization step size for bet sizes. 0.0 for continuous.

        Returns
        -------
        pd.DataFrame
            A MultiIndex DataFrame [method, path_id] containing PF, Sharpe, PSR, MaxDD, and Turnover.
        """
        results_list = []
        
        for i, path_probs in enumerate(self.backtest_paths):
            clean_probs = path_probs.dropna()
            clean_sides = primary_sides.loc[clean_probs.index]

            for method in ['sigmoid', 'power', 'binary']:
                signals = self._apply_bet_method(method, clean_probs, clean_sides, num_classes)
                
                if step_size > 0:
                    from ..bet_sizing.ch10_snippets import discrete_signal
                    signals = discrete_signal(signals, step_size)
                
                # Timeline Mapping
                t0_idx = self.close.index.get_indexer(signals.index)
                t1_idx = self.close.index.get_indexer(self.cv_gen.t1.loc[signals.index])
                
                avg_pos = fill_average_active_sides(len(self.close), t0_idx, t1_idx, signals.values)
                
                # Performance Attribution (OOS Only)
                full_log_rets = avg_pos * self.log_returns.values
                active_mask = (avg_pos != 0)
                oos_log_rets = full_log_rets[active_mask]
                
                if len(oos_log_rets) < 5: continue 

                # Precision calculations
                arith_rets = np.expm1(oos_log_rets)
                sr_raw = oos_log_rets.mean() / oos_log_rets.std()
                
                # Turnover estimation: Sum of absolute changes in position
                turnover = np.abs(np.diff(avg_pos)).sum()

                results_list.append({
                    'method': method,
                    'path_id': i,
                    'mtm_profit_factor': arith_rets[arith_rets > 0].sum() / abs(arith_rets[arith_rets < 0].sum()) if arith_rets[arith_rets < 0].sum() != 0 else 0,
                    'mtm_sharpe': sr_raw,
                    'psr': self._calculate_psr(sr_raw, len(oos_log_rets), pd.Series(oos_log_rets).skew(), pd.Series(oos_log_rets).kurt()),
                    'max_drawdown': self._calculate_max_dd_precision(full_log_rets),
                    'turnover': turnover
                })
                
        return pd.DataFrame(results_list).set_index(['method', 'path_id'])

    def _apply_bet_method(self, method, probs, sides, num_classes):
        """Utility factory to apply different betting philosophies."""
        from ..bet_sizing.ch10_snippets import get_signal
        if method == 'sigmoid':
            return get_signal(probs, num_classes, pred=sides)
        elif method == 'power':
            conviction = get_signal(probs, num_classes)
            return sides * (conviction**2)
        elif method == 'binary':
            return sides * (probs > 0.5).astype(float)
        return probs

    def _calculate_psr(self, sr, n, skew, kurt, sr_benchmark=0):
        """Calculates Probabilistic Sharpe Ratio using raw high-frequency returns."""
        std_sr = np.sqrt((1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (n - 1))
        return norm.cdf((sr - sr_benchmark) / std_sr)

    def _calculate_max_dd_precision(self, log_rets):
        """Peak-to-trough ratio using log-return compounding for numerical stability."""
        cum_rets = np.exp(np.cumsum(log_rets))
        peak = np.maximum.accumulate(cum_rets)
        return np.min(cum_rets / peak - 1) if len(peak) > 0 else 0

    @property
    def recombined_predictions(self):
        """Mean prediction across all folds where a sample was OOS."""
        return self._prediction_matrix.mean(axis=1)
        

def _fit_predict_fold(estimator, X, y, train_idx, test_idx, fold_idx, sample_weight=None):
    """Worker function for parallelized CV training."""
    model = clone(estimator)
    if sample_weight is not None:
        model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sample_weight.iloc[train_idx])
    else:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    preds = model.predict_proba(X.iloc[test_idx])[:, 1] if hasattr(model, "predict_proba") else model.predict(X.iloc[test_idx])
    return fold_idx, test_idx, preds
