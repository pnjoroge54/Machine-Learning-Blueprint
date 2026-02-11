import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed
from math import comb
from numba import njit


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    This splitter decomposes the dataset into N contiguous chunks.
    For every split, it holds out 'k' of these chunks as the Test Set, 
    and uses the remaining N-k chunks as the Training Set (with purging/embargo).
    
    :param n_splits: (int) N, the number of total groups to split the data into.
    :param n_test_splits: (int) k, the number of groups to be in the test set.
    :param t1: (pd.Series) The information range (event end times).
    :param pct_embargo: (float) Percent of data to embargo after a test split.
    """
    def __init__(self, n_splits=5, n_test_splits=2, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        # Validation
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits (k) must be less than n_splits (N)")

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        n_samples = len(indices)
        
        # 1. Define the N contiguous groups using array_split (matches AFML style)
        # This creates a list of arrays, e.g., [arr([0,1,2]), arr([3,4,5]), ...]
        group_arrays = np.array_split(indices, self.n_splits)
        
        # Store bounds for easy access: (start_index, end_index)
        # end_index is exclusive for slice usage
        group_bounds = [(arr[0], arr[-1] + 1) for arr in group_arrays]
        
        # 2. Iterate over all combinations of k groups
        # combinations(range(5), 2) -> (0,1), (0,2), ... (3,4)
        for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):
            
            # --- Construct Test Set ---
            test_indices_list = []
            test_time_ranges = [] # To store (start_time, end_time) of test events
            
            for gid in test_group_ids:
                start_ix, end_ix = group_bounds[gid]
                test_indices_list.append(indices[start_ix:end_ix])
                
                # Get time boundaries for purging
                # t1.index is event start, t1.values is event end
                s_time = self.t1.index[start_ix]
                e_time = self.t1.iloc[start_ix:end_ix].max()
                test_time_ranges.append((s_time, e_time))
            
            test_indices = np.concatenate(test_indices_list)
            
            # --- Construct Train Set ---
            # Start with all indices that are NOT in the test set
            mask = np.ones(n_samples, dtype=bool)
            mask[test_indices] = False
            train_indices = indices[mask]
            
            # --- Purging & Embargo ---
            # We must purge training samples that overlap with ANY of the test groups
            train_starts = self.t1.index[train_indices]
            train_ends = self.t1.iloc[train_indices].values
            
            keep_mask = np.ones(len(train_indices), dtype=bool)
            embargo_offset = int(n_samples * self.pct_embargo)
            
            for test_start, test_end in test_time_ranges:
                # 1. Determine Embargo cutoff
                # Find where test_end falls in the index, shift by embargo size
                # usage of searchsorted ensures we handle non-contiguous time indices
                idx_in_full = self.t1.index.searchsorted(test_end)
                if idx_in_full + embargo_offset < n_samples:
                    embargo_cutoff = self.t1.index[idx_in_full + embargo_offset]
                else:
                    embargo_cutoff = pd.Timestamp.max
                
                # 2. Identify Overlaps
                # Drop if: Train_Start <= Embargo_End AND Train_End >= Test_Start
                is_overlapping = (train_starts <= embargo_cutoff) & \
                                 (train_ends >= test_start)
                                 
                keep_mask = keep_mask & ~is_overlapping
            
            yield train_indices[keep_mask], test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class CPCVAnalyzer:
    """
    Manages Combinatorial Purged CV execution and result recombination.
    Encapsulates the 'variable tracking' so the user doesn't have to.
    """
    def __init__(self, estimator, cv):
        self.estimator = estimator
        self.cv = cv
        self._prediction_matrix = None
        self._X = None
        self._y = None

    def fit_predict(self, X, y, sample_weight=None):
        """Runs the CV and stores results internally."""
        self._X = X
        self._y = y
        n_splits = self.cv.get_n_splits(X)
        
        # Initialize matrix: Rows = Samples, Cols = Split Index
        self._prediction_matrix = pd.DataFrame(
            np.nan, 
            index=X.index, 
            columns=[f"split_{i}" for i in range(n_splits)]
        )

        for i, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            model = clone(self.estimator)
            
            # Fit & Predict
            if sample_weight is not None:
                model.fit(X.iloc[train_idx], y.iloc[train_idx], 
                          sample_weight=sample_weight.iloc[train_idx])
            else:
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            # Use predict_proba for financial ranking/scoring if available
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X.iloc[test_idx])[:, 1]
            else:
                preds = model.predict(X.iloc[test_idx])
                
            self._prediction_matrix.iloc[test_idx, i] = preds
            
        return self.recombined_predictions

    @property
    def prediction_matrix(self):
        """The raw matrix of predictions from all combinatorial folds."""
        if self._prediction_matrix is None:
            raise ValueError("Run .fit_predict() first.")
        return self._prediction_matrix

    @property
    def recombined_predictions(self):
        """
        The 'Bagged' prediction for each timestamp.
        Calculates the mean across all folds where the sample was in the test set.
        """
        return self.prediction_matrix.mean(axis=1)

    @property
    def num_predictions_per_sample(self):
        """Returns how many times each sample was 'tested'."""
        return self.prediction_matrix.count(axis=1)

    @property
    def backtest_paths(self):
        """
        Returns the individual backtest paths as described in AFML.
        Each path is a series covering the full history.
        """
        # Logic to partition the combinations into J = (N-1 choose k-1) paths
        # This is a complex combinatorial task; for simplicity, many users 
        # use the prediction_matrix directly for distribution analysis.
        pass
            
    
@njit(cache=True)
def fill_sides_numba(num_close, t0_idx, t1_idx, side):
    full_side = np.zeros(num_close, dtype=np.float64)
    for i in range(len(t0_idx)):
        start, end = t0_idx[i], t1_idx[i]
        if start != -1 and end != -1:
            full_side[start : end + 1] += side[i]
    return full_side
    

class CPCVAnalyzer:
    def __init__(self, estimator, cv_gen, close_prices, n_jobs=-1):
        self.estimator = estimator
        self.cv_gen = cv_gen # Renamed as requested
        self.close = close_prices
        self.n_jobs = n_jobs
        self._prediction_matrix = None
        self._X = None
        # Align log returns for MtM: r_t = log(P_{t+1}/P_t)
        self.daily_returns = np.log(self.close).diff().shift(-1).fillna(0)

    def fit_predict(self, X, y, sample_weight=None):
        """Parallelized training using joblib."""
        self._X = X
        n_splits = self.cv_gen.get_n_splits(X)
        
        # Dispatching folds to parallel workers
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_fold)(
                self.estimator, X, y, train, test, i, sample_weight
            ) for i, (train, test) in enumerate(self.cv_gen.split(X, y))
        )

        self._prediction_matrix = pd.DataFrame(
            np.nan, index=X.index, columns=range(n_splits)
        )
        
        for fold_idx, test_idx, preds in results:
            self._prediction_matrix.iloc[test_idx, fold_idx] = preds
            
        return self.recombined_predictions

    @property
    def backtest_paths(self):
        """Assembles the J = (N-1 choose k-1) unique paths."""
        N, k = self.cv_gen.n_splits, self.cv_gen.n_test_splits
        J = comb(N - 1, k - 1)
        group_indices = np.array_split(np.arange(len(self._X)), N)
        
        paths = []
        for j in range(J):
            path_series = pd.Series(index=self._X.index, dtype=float)
            for g_idx in range(N):
                valid_cols = self._prediction_matrix.columns[
                    self._prediction_matrix.iloc[group_indices[g_idx][0]].notna()
                ]
                target_col = valid_cols[j]
                idx = group_indices[g_idx]
                path_series.iloc[idx] = self._prediction_matrix.iloc[idx, target_col]
            paths.append(path_series)
        return paths

    def get_distribution_metrics(self):
        """
        Efficiently collects metrics into a list of dicts before
        converting to a DataFrame.
        """
        paths = self.backtest_paths
        results_list = [] # The efficient way
        
        for i, path_side in enumerate(paths):
            # Vectorized indexing
            t0_idx = self.close.index.get_indexer(path_side.index)
            t1_idx = self.close.index.get_indexer(self.cv_gen.t1.loc[path_side.index])
            
            # Numba MtM
            pos = fill_sides_numba(len(self.close), t0_idx, t1_idx, path_side.values)
            rets = pd.Series(pos * self.daily_returns.values, index=self.close.index)
            
            # Append dict to list (O(1) per iteration)
            results_list.append({
                'path_id': i,
                'sharpe': (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() != 0 else 0,
                'max_dd': ((rets.cumsum().apply(np.exp) / rets.cumsum().apply(np.exp).expanding().max()) - 1).min()
            })
            
        return pd.DataFrame(results_list) # Single O(N) conversion

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
    return fold_idx, test_idx, preds                self.all_test_indices.append(test_indices)
                yield train_indices, test_indices
    
    def get_n_splits(
        self, 
        X: Optional[pd.DataFrame] = None, 
        y: Optional[pd.Series] = None, 
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return the number of splits.
        
        If n_paths is specified, returns n_paths.
        Otherwise returns total number of combinations.
        """
        if self.n_paths is not None:
            return min(self.n_paths, self.total_combinations)
        return self.total_combinations
    
    def recombine_test_predictions(
        self, 
        all_test_indices: List[np.ndarray], 
        all_predictions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Recombine predictions from different splits to form complete backtest paths.
        
        Parameters
        ----------
        all_test_indices : List[np.ndarray]
            List of test indices from each split.
            
        all_predictions : List[np.ndarray]
            List of predictions corresponding to each test split.
            
        Returns
        -------
        List[np.ndarray]
            List of recombined prediction paths.
        """
        # Create a mapping from index to predictions across all splits
        index_to_preds = {}
        for indices, preds in zip(all_test_indices, all_predictions):
            for idx, pred in zip(indices, preds):
                index_to_preds.setdefault(idx, []).append(pred)
        
        # For each index, we now have predictions from different splits
        # We need to form coherent paths. This is complex and depends on
        # how you want to handle multiple predictions per index.
        # One approach: average predictions for each index
        averaged_predictions = {
            idx: np.mean(pred_list) for idx, pred_list in index_to_preds.items()
        }
        
        # Sort by index and return as array
        sorted_indices = sorted(averaged_predictions.keys())
        return [averaged_predictions[idx] for idx in sorted_indices]


"""
# Assuming X is your feature DataFrame and t1 defines event windows
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create sample data
n_samples = 1000
dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
X = pd.DataFrame(np.random.randn(n_samples, 5), index=dates)
returns = np.random.randn(n_samples) * 0.01

# Define event windows: e.g., 5-day forward returns
t1_values = dates + pd.Timedelta(days=5)
t1 = pd.Series(t1_values, index=dates)

# Create CV object
cpcv = CombinatorialPurgedCV(
    n_folds=10,
    n_test_folds=3,
    t1=t1,
    pct_embargo=0.01,
    n_paths=20  # Generate 20 different backtest paths
)

# Use in cross-validation
all_test_indices = []
all_predictions = []
all_true_values = []

for train_idx, test_idx in cpcv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = (returns[train_idx] > 0).astype(int)
    y_test = (returns[test_idx] > 0).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Get predictions
    preds = model.predict_proba(X_test)[:, 1]
    
    # Store for later recombination
    all_test_indices.append(test_idx)
    all_predictions.append(preds)
    all_true_values.append(y_test)

# Recombine predictions to analyze performance distribution
recombined_preds = cpcv.recombine_test_predictions(all_test_indices, all_predictions)
"""


