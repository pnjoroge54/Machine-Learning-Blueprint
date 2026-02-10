from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from itertools import combinations


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation for financial time series.
    
    This class extends the PurgedKFold concept to generate multiple 
    backtest paths using combinatorial splits while preventing data 
    leakage through event-based purging and embargo.
    
    Parameters
    ----------
    n_folds : int, default=6
        Total number of folds to split the data into.
        
    n_test_folds : int, default=2
        Number of folds used for testing in each split.
        
    t1 : pd.Series
        The information range on which each record is constructed.
        - t1.index: Time when information extraction started
        - t1.value: Time when information extraction ended
        
    pct_embargo : float, default=0.01
        Percent that determines the embargo size.
        
    n_paths : Optional[int], default=None
        Maximum number of combinatorial paths to generate.
        If None, generates all possible combinations.
    """
    
    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
        n_paths: Optional[int] = None
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series")
            
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.n_paths = n_paths
        
        # Store the original index for alignment
        self.t1_index = t1.index
        
        # Calculate total number of possible combinations
        self.total_combinations = self._calculate_total_combinations()
        
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of possible combinations."""
        from math import comb
        return comb(self.n_folds, self.n_test_folds)
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorial train/test splits with purging.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix aligned with t1 index.
            
        y : pd.Series, optional
            Target labels (not used for splitting but kept for compatibility).
            
        groups : array-like, optional
            Group labels (not used but kept for compatibility).
            
        Yields
        ------
        train_indices : np.ndarray
            Indices for training set (purged of overlapping events).
            
        test_indices : np.ndarray
            Indices for test set.
        """
        # Validate alignment
        if (X.index != self.t1_index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")
            
        indices = np.arange(X.shape[0])
        n_samples = X.shape[0]
        embargo_size = int(n_samples * self.pct_embargo)
        
        # 1. Create sequential folds
        fold_indices = np.array_split(indices, self.n_folds)
        
        # 2. Generate all possible test fold combinations
        fold_numbers = range(self.n_folds)
        test_combinations = list(combinations(fold_numbers, self.n_test_folds))
        
        # Limit number of paths if specified
        if self.n_paths is not None and self.n_paths < len(test_combinations):
            # Randomly sample combinations (can be made deterministic with random_state)
            rng = np.random.default_rng(42)
            selected_idx = rng.choice(
                len(test_combinations), 
                size=self.n_paths, 
                replace=False
            )
            test_combinations = [test_combinations[i] for i in selected_idx]

        # List of test indices to be used in filling backtest paths
        self.all_test_indices = []
        
        # 3. Generate each combinatorial split
        for test_fold_nums in test_combinations:
            # Get test indices from selected folds
            test_indices = np.concatenate([fold_indices[i] for i in test_fold_nums])
            
            # Get train indices from remaining folds
            train_fold_nums = [i for i in fold_numbers if i not in test_fold_nums]
            initial_train_indices = np.concatenate([fold_indices[i] for i in train_fold_nums])
            
            # 4. Apply event-based purging
            if len(initial_train_indices) > 0:
                # Get test event times for purging
                test_times = pd.Series(
                    index=[self.t1.index[test_indices[0]]],
                    data=[self.t1.iloc[test_indices[-1]]]
                )
                
                # Get train event times
                initial_train_times = pd.Series(
                    index=self.t1.index[initial_train_indices],
                    data=self.t1.iloc[initial_train_indices].values
                )
                
                # Apply purging using ml_get_train_times
                purged_train_times = ml_get_train_times(initial_train_times, test_times)
                
                # Convert purged times back to indices
                train_indices = []
                for train_time in purged_train_times.index:
                    loc = self.t1.index.get_loc(train_time)
                    if isinstance(loc, int):
                        train_indices.append(loc)
                    else:
                        train_indices.extend(range(loc.start, loc.stop))
                
                train_indices = np.array(train_indices, dtype=int)
            else:
                train_indices = np.array([], dtype=int)
            
            # 5. Apply embargo
            if len(train_indices) > 0 and embargo_size > 0:
                # Find indices in train that come immediately after test
                test_end = test_indices[-1] + 1
                embargo_end = min(test_end + embargo_size, n_samples)
                
                # Remove training indices within embargo period
                mask = ~((train_indices >= test_end) & (train_indices < embargo_end))
                train_indices = train_indices[mask]
            
            # Yield only if we have valid training data
            if len(train_indices) > 0:
                self.all_test_indices.append(test_indices)
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


