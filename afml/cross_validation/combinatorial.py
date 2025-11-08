"""
Implements the Combinatorial Purged Cross-Validation class from Chapter 12
"""

from itertools import combinations

import numpy as np
import pandas as pd


class CombinatorialPurgedKFold:
    """
    Robust implementation of Combinatorial Purged Cross-Validation (CPCV) with proper embargo handling.

    This class implements the CPCV method from Chapter 12 of "Advances in Financial Machine Learning"
    by Marcos López de Prado, with corrections to ensure proper temporal separation between
    training and testing periods through purging and embargo mechanisms.

    CPCV addresses the limited number of backtests in traditional time series cross-validation
    by generating multiple combinations of test splits, allowing for more robust model validation
    while maintaining temporal dependencies.

    Key Features:
    - Combinatorial generation of train/test splits
    - Proper purging of overlapping samples
    - Configurable embargo period to prevent data leakage
    - Robust index handling for financial time series

    Args:
        n_splits (int): Total number of splits to divide the dataset into.
            Each split represents a contiguous time block. Default: 3.
        n_test_splits (int): Number of test splits to use in each combination.
            Must be less than n_splits. Default: 2.
        samples_info_sets (pd.Series): Series containing the information ranges for each sample.
            - Index: Time when information extraction started
            - Values: Time when information extraction ended
            Used for purging overlapping samples between train and test sets.
        pct_embargo (float): Percentage of total samples to use as embargo period (0-1).
            Embargo prevents using samples immediately after test period to avoid
            forward-looking bias. Default: 0.01 (1% embargo).
        random_state (int, optional): Random seed for reproducibility. Not currently
            used but included for future compatibility. Default: None.

    Attributes:
        n_splits (int): Stored parameter for total splits
        n_test_splits (int): Stored parameter for test splits per combination
        samples_info_sets (pd.Series): Stored information ranges
        pct_embargo (float): Stored embargo percentage
        random_state (int): Stored random state

    Raises:
        ValueError: If n_test_splits >= n_splits
        TypeError: If samples_info_sets is not a pandas Series

    Example:
        >>> from combinatorial import CombinatorialPurgedKFold
        >>> import pandas as pd
        >>> import numpy as np

        # Create sample financial time series data
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> samples_info = pd.Series(
        ...     index=dates,
        ...     data=dates + pd.Timedelta(days=5)  # 5-day information periods
        ... )
        >>> X = pd.DataFrame(np.random.randn(100, 5), index=dates)
        >>> y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        # Initialize CPCV with embargo
        >>> cv = CombinatorialPurgedKFold(
        ...     n_splits=5,
        ...     n_test_splits=2,
        ...     samples_info_sets=samples_info,
        ...     pct_embargo=0.01  # 1% embargo
        ... )

        # Generate splits
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        ...     # Model training and validation would go here

    Note:
        - The number of total combinations generated is C(n_splits, n_test_splits)
        - Each combination uses n_test_splits contiguous blocks as test set
        - Remaining (n_splits - n_test_splits) blocks form the training set
        - Purging removes any training samples whose information period overlaps with test period
        - Embargo removes additional samples after test period to prevent leakage

    Reference:
        López de Prado, M. (2018). Advances in Financial Machine Learning, Chapter 12.
        Wiley. ISBN: 978-1-119-48208-6
    """

    def __init__(
        self,
        n_splits: int = 3,
        n_test_splits: int = 2,
        samples_info_sets: pd.Series = None,
        pct_embargo: float = 0.01,
        random_state: int = None,
    ):
        # Validate input parameters
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits ({n_test_splits}) must be less than n_splits ({n_splits})"
            )

        if not isinstance(samples_info_sets, pd.Series):
            raise TypeError("samples_info_sets must be a pandas Series")

        if not 0 <= pct_embargo <= 1:
            raise ValueError(f"pct_embargo ({pct_embargo}) must be between 0 and 1")

        # Initialize instance variables
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.random_state = random_state

        # Calculate total number of combinations for information
        from scipy.special import comb

        self.n_combinations = int(comb(n_splits, n_test_splits))

        # Log initialization parameters (optional - for debugging)
        # print(f"Initialized CPCV with {n_splits} splits, {n_test_splits} test splits, "
        #       f"{pct_embargo:.1%} embargo, generating {self.n_combinations} combinations")

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and testing sets.

        This method implements the core combinatorial purged cross-validation logic:
        1. Divides the dataset into n_splits contiguous blocks
        2. Generates all combinations of n_test_splits blocks for testing
        3. For each combination:
           - Uses the selected blocks as test set
           - Uses remaining blocks as training set (after purging and embargo)

        Args:
            X (array-like, shape (n_samples, n_features)): Training data matrix
            y (array-like, shape (n_samples,), optional): Target variable vector
            groups (array-like, optional): Group labels for the samples used while
                splitting the dataset. Not used in this implementation but included
                for sklearn compatibility.

        Yields:
            tuple: (train_indices, test_indices) for each combination

        Raises:
            ValueError: If X has different number of samples than samples_info_sets

        Note:
            - The method yields C(n_splits, n_test_splits) combinations
            - Each yield provides indices for one train/test split
            - Training indices are purged of samples overlapping with test information periods
            - Embargo period is applied after each test block
        """
        # Validate input dimensions
        n_samples = len(X)
        if n_samples != len(self.samples_info_sets):
            raise ValueError(
                f"X has {n_samples} samples but samples_info_sets has "
                f"{len(self.samples_info_sets)} entries. They must match."
            )

        # Create array of sample indices [0, 1, 2, ..., n_samples-1]
        indices = np.arange(n_samples)

        # Calculate absolute number of samples for embargo
        # Using percentage of total samples provides consistent embargo size
        embargo_samples = int(n_samples * self.pct_embargo)

        # Split indices into n_splits contiguous blocks
        # Example: 100 samples, 5 splits → blocks of [0-19], [20-39], [40-59], [60-79], [80-99]
        base_splits = np.array_split(indices, self.n_splits)

        # Store split boundaries for easy reference
        # Each tuple contains (split_index, start_index, end_index)
        split_boundaries = []
        for i, split in enumerate(base_splits):
            start = split[0]  # First index in this split
            end = split[-1] + 1  # Last index + 1 (exclusive boundary)
            split_boundaries.append((i, start, end))

        # Generate all possible combinations of test splits
        # Example: 5 splits, choose 2 → C(5,2) = 10 combinations
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        # Log combination count (optional - for debugging)
        # print(f"Generating {len(test_combinations)} train/test combinations")

        # Iterate through each combination of test splits
        for test_split_indices in test_combinations:
            # test_split_indices contains the indices of splits used for testing
            # Example: (0, 2) means use first and third splits as test set

            test_indices = []  # Will contain all test sample indices
            test_ranges = []  # Will store start/end boundaries for each test split

            # Collect all indices from the selected test splits
            for split_idx in test_split_indices:
                # Get boundaries for this test split
                start, end = split_boundaries[split_idx][1], split_boundaries[split_idx][2]

                # Add all indices in this range to test set
                test_indices.extend(range(start, end))

                # Store the range for purging and embargo calculations
                test_ranges.append((start, end))

            # Get training indices with proper purging and embargo
            # This ensures no temporal leakage between train and test sets
            train_indices = self._get_purged_train_indices(test_ranges, embargo_samples, n_samples)

            # Convert to numpy arrays for consistency with sklearn API
            train_array = np.array(train_indices)
            test_array = np.array(test_indices)

            # Validate that train and test sets are disjoint
            if len(np.intersect1d(train_array, test_array)) > 0:
                raise RuntimeError(
                    f"Train and test sets overlap! This indicates a bug in purging logic. "
                    f"Train size: {len(train_array)}, Test size: {len(test_array)}, "
                    f"Intersection: {len(np.intersect1d(train_array, test_array))}"
                )

            # Yield the indices for this combination
            yield train_array, test_array

    def _get_purged_train_indices(self, test_ranges, embargo_samples, n_samples):
        """
        Calculate training indices after applying purging and embargo.

        This helper method implements the core data leakage prevention:
        - Purging: Remove any training samples whose information period overlaps with test period
        - Embargo: Remove additional samples immediately after test period to prevent
          leakage from serially correlated data

        Args:
            test_ranges (list of tuples): List of (start, end) index ranges for test splits
            embargo_samples (int): Number of samples to embargo after each test range
            n_samples (int): Total number of samples in dataset

        Returns:
            list: Sorted list of training indices after purging and embargo

        Note:
            - Purging uses samples_info_sets to determine temporal overlaps
            - Embargo uses simple index-based exclusion for efficiency
            - The method is conservative - it may remove more samples than strictly necessary
              to ensure no data leakage
        """
        # Start with all possible indices as candidates for training
        train_candidates = set(range(n_samples))

        # Step 1: Remove all test indices from training candidates
        for start, end in test_ranges:
            # Remove the entire test range from training candidates
            test_range_indices = set(range(start, end))
            train_candidates -= test_range_indices

        # Step 2: Apply embargo - remove samples immediately after each test range
        # This prevents using data that might be correlated with test period
        for start, end in test_ranges:
            # Calculate embargo range: from end of test to end + embargo_samples
            embargo_start = end
            embargo_end = min(end + embargo_samples, n_samples)  # Don't exceed dataset bounds

            # Remove embargoed samples from training candidates
            if embargo_start < embargo_end:  # Only if embargo period is valid
                embargo_indices = set(range(embargo_start, embargo_end))
                train_candidates -= embargo_indices

        # Step 3: Apply purging based on temporal information overlap
        # This is the most crucial step for financial data
        final_train_indices = self._apply_temporal_purging(train_candidates, test_ranges)

        # Return sorted list for consistent ordering
        return sorted(final_train_indices)

    def _apply_temporal_purging(self, candidate_indices, test_ranges):
        """
        Apply temporal purging to remove samples with overlapping information periods.

        This method uses the samples_info_sets to identify and remove training samples
        whose information collection period overlaps with any test sample's period.

        Args:
            candidate_indices (set): Candidate training indices after basic exclusion
            test_ranges (list): Test range boundaries for overlap checking

        Returns:
            set: Purged training indices with no temporal overlap with test set
        """
        if self.samples_info_sets is None:
            # If no temporal information provided, return candidates as-is
            return candidate_indices

        # Get all test sample indices for overlap checking
        all_test_indices = set()
        for start, end in test_ranges:
            all_test_indices.update(range(start, end))

        # Get information periods for test samples
        test_info_starts = self.samples_info_sets.index[list(all_test_indices)]
        test_info_ends = self.samples_info_sets.iloc[list(all_test_indices)].values

        # Convert candidate indices to list for processing
        candidate_list = list(candidate_indices)

        # Check each candidate training sample for temporal overlap
        safe_train_indices = set()

        for train_idx in candidate_list:
            train_info_start = self.samples_info_sets.index[train_idx]
            train_info_end = self.samples_info_sets.iloc[train_idx]

            # Check if this training sample overlaps with any test sample
            has_overlap = False

            for test_idx in all_test_indices:
                test_info_start = self.samples_info_sets.index[test_idx]
                test_info_end = self.samples_info_sets.iloc[test_idx]

                # Check for temporal overlap using the standard conditions
                # 1. Train starts during test period
                condition1 = test_info_start <= train_info_start <= test_info_end
                # 2. Train ends during test period
                condition2 = test_info_start <= train_info_end <= test_info_end
                # 3. Train completely contains test period
                condition3 = train_info_start <= test_info_start and test_info_end <= train_info_end

                if condition1 or condition2 or condition3:
                    has_overlap = True
                    break  # No need to check other test samples

            # Only keep training sample if it has no temporal overlap
            if not has_overlap:
                safe_train_indices.add(train_idx)

        return safe_train_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        This method provides compatibility with sklearn's cross-validation interface.

        Args:
            X (object): Always ignored, exists for compatibility
            y (object): Always ignored, exists for compatibility
            groups (object): Always ignored, exists for compatibility

        Returns:
            int: Number of train/test splits that will be generated

        Note:
            The number of splits is determined by the combinatorial formula:
            C(n_splits, n_test_splits) - the number of ways to choose n_test_splits
            test blocks from n_splits total blocks.
        """
        return self.n_combinations

    def __repr__(self):
        """
        Return string representation of the cross-validator.

        Returns:
            str: Informative string representation
        """
        return (
            f"CombinatorialPurgedKFold(n_splits={self.n_splits}, "
            f"n_test_splits={self.n_test_splits}, "
            f"pct_embargo={self.pct_embargo}, "
            f"n_combinations={self.n_combinations})"
        )


def demonstrate_combinatorial_cv():
    """
    Comprehensive demonstration of CombinatorialPurgedKFold usage.

    This example shows how to properly use the combinatorial CV with financial data,
    including validation of the embargo and purging functionality.
    """
    from datetime import timedelta

    import numpy as np
    import pandas as pd

    # Create realistic financial time series data
    print("Creating sample financial dataset...")
    n_samples = 252 * 2  # 2 years of daily data
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # Simulate information collection periods (typical in financial ML)
    # Each sample uses information from the past 10 days
    samples_info_sets = pd.Series(index=dates, data=[date + timedelta(days=10) for date in dates])

    # Create features and target
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        },
        index=dates,
    )

    y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)

    print(f"Dataset created: {n_samples} samples from {dates[0]} to {dates[-1]}")

    # Initialize combinatorial CV with meaningful parameters
    print("\nInitializing Combinatorial Purged Cross-Validation...")
    cv = CombinatorialPurgedKFold(
        n_splits=6,  # Divide data into 6 contiguous blocks
        n_test_splits=2,  # Use 2 blocks for testing in each combination
        samples_info_sets=samples_info_sets,
        pct_embargo=0.02,  # 2% embargo to prevent leakage
        random_state=42,
    )

    print(f"CV parameters: {cv}")
    print(f"Will generate {cv.get_n_splits()} train/test combinations")

    # Validate embargo functionality
    print("\nValidating embargo functionality...")
    embargo_violations = 0
    total_combinations = 0

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        total_combinations += 1

        # Basic validation
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected!"

        # Check that test indices are contiguous blocks
        test_indices_sorted = sorted(test_idx)
        is_contiguous = test_indices_sorted == list(
            range(test_indices_sorted[0], test_indices_sorted[-1] + 1)
        )

        if not is_contiguous:
            print(f"Warning: Test indices in combination {i} are not contiguous")

        # Check embargo: no train samples immediately after test period
        if len(test_idx) > 0:
            max_test_idx = max(test_idx)
            # Look for train samples in the embargo period (next 5 samples after test)
            embargo_violations_in_split = sum(
                1 for idx in train_idx if max_test_idx < idx <= max_test_idx + int(n_samples * 0.02)
            )

            if embargo_violations_in_split > 0:
                embargo_violations += 1
                print(f"Combination {i}: {embargo_violations_in_split} embargo violations")

    print(
        f"Embargo validation: {embargo_violations} violations in {total_combinations} combinations"
    )

    # Demonstrate usage with a classifier
    print("\nDemonstrating usage with Random Forest classifier...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        clf.fit(X_train, y_train)

        # Predict and score
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

        print(
            f"Fold {fold + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}, "
            f"Accuracy={accuracy:.4f}"
        )

    print(f"\nCross-validation results:")
    print(f"Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Min accuracy: {np.min(scores):.4f}")
    print(f"Max accuracy: {np.max(scores):.4f}")

    return cv, scores


# Run the demonstration
if __name__ == "__main__":
    cv, scores = demonstrate_combinatorial_cv()
