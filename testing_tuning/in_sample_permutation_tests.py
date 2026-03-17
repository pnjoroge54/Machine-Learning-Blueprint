import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Any, Optional, Union
from itertools import product
import warnings
from tqdm import tqdm

from .bar_permute import get_permutation

def insample_optimization_with_mcpt(
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    objective_function: Callable,
    n_permutations: int = 1000,
    direction: str = 'maximize',
    train_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform in‑sample grid search optimization and assess significance via MCPT.

    Parameters
    ----------
    data : pd.DataFrame
        Full OHLC dataset (must contain columns 'open','high','low','close').
    param_grid : dict
        Dictionary with parameter names as keys and lists of values to try.
    objective_function : callable
        Function that accepts (data, **params, **kwargs) and returns a scalar metric.
    n_permutations : int, default 1000
        Number of permuted datasets to generate for the Monte Carlo test.
    direction : str, default 'maximize'
        Either 'maximize' or 'minimize'.
    train_filter : callable, optional
        Function to subset the data for training (e.g., lambda df: df[df.index.year < 2020]).
        If None, the full dataset is used.
    seed : int, optional
        Random seed for reproducibility (affects both permutation and any randomness in objective).
    verbose : bool, default True
        If True, show progress bars.
    **kwargs : additional arguments
        Passed directly to the objective_function.

    Returns
    -------
    dict
        {
            'best_params': dict,           # best parameters on real data
            'real_score': float,            # objective value with best_params on real data
            'p_value': float,                # MCPT p‑value
            'permuted_scores': list,         # best scores from all permutations
            'n_permutations': int,
            'param_grid': dict,
            'direction': str
        }
    """
    # --- 1. Prepare training data ---
    if train_filter is not None:
        train_data = train_filter(data)
    else:
        train_data = data

    if verbose:
        print(f"Training data shape: {train_data.shape}")

    # --- 2. Optimize on real data ---
    real_best_params, real_best_score = _grid_search(
        data=train_data,
        param_grid=param_grid,
        objective_function=objective_function,
        direction=direction,
        verbose=verbose,
        **kwargs
    )

    if verbose:
        print(f"Real optimization complete. Best score: {real_best_score:.4f}")

    # --- 3. Monte Carlo Permutation Test ---
    permuted_scores = []
    better_count = 0

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Create a generator for permutations (to optionally use seeds per iteration)
    rng = np.random.default_rng(seed)

    perm_iterator = tqdm(range(n_permutations), desc="MCPT") if verbose else range(n_permutations)
    for i in perm_iterator:
        # Generate permuted data (preserves OHLC structure but shuffles bars)
        # Use a different seed for each permutation to ensure independence
        perm_data = get_permutation(train_data, seed=rng.integers(0, 2**31))

        # Optimize on permuted data
        _, perm_best_score = _grid_search(
            data=perm_data,
            param_grid=param_grid,
            objective_function=objective_function,
            direction=direction,
            verbose=False,          # keep quiet inside loop
            **kwargs
        )

        permuted_scores.append(perm_best_score)

        # Count how many permutations beat (or match) the real score
        if direction == 'maximize':
            if perm_best_score >= real_best_score:
                better_count += 1
        else:  # minimize
            if perm_best_score <= real_best_score:
                better_count += 1

    p_value = better_count / n_permutations

    # --- 4. Return results ---
    return {
        'best_params': real_best_params,
        'real_score': real_best_score,
        'p_value': p_value,
        'permuted_scores': permuted_scores,
        'n_permutations': n_permutations,
        'param_grid': param_grid,
        'direction': direction
    }


def _grid_search(
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    objective_function: Callable,
    direction: str = 'maximize',
    verbose: bool = False,
    **kwargs
) -> tuple:
    """
    Internal grid search helper (same as earlier, but returns best params and score).
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    best_score = -np.inf if direction == 'maximize' else np.inf
    best_params = None

    for values in combinations:
        params = dict(zip(param_names, values))
        try:
            score = objective_function(data=data, **params, **kwargs)
            if direction == 'maximize':
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception as e:
            warnings.warn(f"Error with params {params}: {str(e)}")
            continue

    return best_params, best_score