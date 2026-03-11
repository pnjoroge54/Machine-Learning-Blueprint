"""Combinatorial Purged Cross-Validation and CPCV-based performance analysis."""

import itertools
import math
import numbers
from collections.abc import Iterator
from math import comb
from itertools import combinations
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
import sklearn.utils as sku
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import norm, skew as scipy_skew, kurtosis as scipy_kurt

from .cross_validation import ml_get_train_times


_MAX_COMBINATIONS = 100_000


# ---------------------------------------------------------------------------
# Numba Optimized Utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper statistics functions
# ---------------------------------------------------------------------------

def _n_splits(n_folds: int, n_test_folds: int) -> int:
    """Number of splits = C(n_folds, n_test_folds)."""
    return math.comb(n_folds, n_test_folds)


def _n_test_paths(n_folds: int, n_test_folds: int) -> int:
    """Number of distinct backtest paths that can be reconstructed."""
    return _n_splits(n_folds=n_folds, n_test_folds=n_test_folds) * n_test_folds // n_folds


def _avg_train_size(n_observations: int, n_folds: int, n_test_folds: int) -> float:
    """Average number of observations in each training set."""
    return n_observations / n_folds * (n_folds - n_test_folds)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Provides train/test indices to split time series data samples based on
    Combinatorial Purged Cross-Validation [1]_.

    Compared to ``KFold``, which uses a single test fold, this class uses
    ``n_test_folds > 1`` test folds per split so that multiple independent
    backtest paths can be recombined from the train/test combinations.

    To prevent data leakage, **purging** and **embargoing** are applied:

    * **Purging** (event-based, via ``t1``) – removes every training
      observation whose label formation period ``[t1.index[i], t1.iloc[i]]``
      overlaps in time with any test-set event window.  This is the exact
      overlap test from López de Prado (2018) Ch. 7, implemented in
      :func:`ml_get_train_times`.

    * **Embargoing** (positional, via ``pct_embargo``) – after the last
      observation of each contiguous test block, the next ``embargo_size``
      training observations are excluded to guard against serial-correlation
      leakage (e.g. from ARMA-type features).  ``embargo_size`` is computed
      as ``int(n_samples * pct_embargo)`` at ``split`` time.

    Parameters
    ----------
    n_folds : int, default=10
        Total number of folds. Must be at least 3.

    n_test_folds : int, default=2
        Number of test folds per split. Must be at least 2 and strictly less
        than ``n_folds``. For one test fold use ``sklearn.model_selection.KFold``.

    t1 : pd.Series
        The information range for each observation.

        * ``t1.index``  – timestamp when information extraction *started*
          (bar open / observation start time).
        * ``t1.values`` – timestamp when the label was *finalised*
          (e.g. when a triple-barrier was hit).

        Must be aligned with ``X`` (same index). Used exclusively for
        event-based purging via :func:`ml_get_train_times`.

    pct_embargo : float, default=0.01
        Fraction of total observations to exclude from the start of each
        training segment immediately following a test block (positional
        embargo). Set to ``0.0`` to disable.

    Attributes
    ----------
    index_train_test_ : ndarray of shape (n_observations, n_splits)
        Populated after the generator from ``split()`` is fully exhausted.
        Encodes each observation's role per split:

        *  ``0`` – training
        *  ``1`` – test
        * ``-1`` – excluded (purged or embargoed)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from combinatorial import CombinatorialPurgedCV
    >>> dates = pd.date_range("2020-01-01", periods=12, freq="D")
    >>> X = pd.DataFrame(np.random.randn(12, 2), index=dates)
    >>> t1 = pd.Series(dates + pd.Timedelta(days=2), index=dates)
    >>> cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, t1=t1, pct_embargo=0.0)
    >>> for i, (train_idx, test_list) in enumerate(cv.split(X)):
    ...     print(f"Split {i}: train={train_idx.tolist()}")
    ...     for j, t in enumerate(test_list):
    ...         print(f"  test[{j}]={t.tolist()}")

    References
    ----------
    .. [1] "Advances in Financial Machine Learning",
       Marcos López de Prado (2018).
    """

    def __init__(
        self,
        n_folds: int = 10,
        n_test_folds: int = 2,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ) -> None:
        # ---- validate t1 ----
        if not isinstance(t1, pd.Series):
            raise TypeError(
                "`t1` must be a pd.Series whose index holds event-start times "
                "and whose values hold event-end times."
            )

        # ---- validate n_folds ----
        if not isinstance(n_folds, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                f"{n_folds} of type {type(n_folds)} was passed."
            )
        n_folds = int(n_folds)
        if n_folds <= 2:
            raise ValueError(f"`n_folds` must be at least 3, got `n_folds={n_folds}`.")

        # ---- validate n_test_folds ----
        if n_test_folds <= 1:
            raise ValueError(
                f"`n_test_folds` must be at least 2, got `n_test_folds={n_test_folds}`."
            )
        if n_test_folds >= n_folds:
            raise ValueError(
                "`n_folds` must be strictly greater than `n_test_folds`, "
                f"got n_folds={n_folds}, n_test_folds={n_test_folds}."
            )

        # ---- validate pct_embargo ----
        if not (0.0 <= pct_embargo < 1.0):
            raise ValueError("`pct_embargo` must be in [0, 1).")

        # ---- guard against combinatorial explosion ----
        n_combinations = math.comb(n_folds, n_test_folds)
        if n_combinations > _MAX_COMBINATIONS:
            raise ValueError(
                f"n_folds={n_folds} and n_test_folds={n_test_folds} produce "
                f"{n_combinations:,} splits, exceeding the maximum of "
                f"{_MAX_COMBINATIONS:,}. Reduce `n_folds` or move `n_test_folds` "
                "further from n_folds / 2."
            )

        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.t1 = t1
        self.pct_embargo = pct_embargo

    # ------------------------------------------------------------------
    # sklearn BaseCrossValidator interface
    # ------------------------------------------------------------------

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the total number of splits = C(n_folds, n_test_folds)."""
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Required stub; not useful here – use ``split()`` directly."""
        raise NotImplementedError(
            "Use `split()` directly – it yields (train_index, [test_0, test_1, ...])."
        )

    # --------------------------------------------------------------------------
    # Read-only combinatorial properties (depend only on n_folds / n_test_folds)
    # --------------------------------------------------------------------------

    @property
    def n_splits(self) -> int:
        """Total number of train/test splits."""
        return _n_splits(n_folds=self.n_folds, n_test_folds=self.n_test_folds)

    @property
    def n_test_paths(self) -> int:
        """Number of distinct backtest paths that can be reconstructed."""
        return _n_test_paths(n_folds=self.n_folds, n_test_folds=self.n_test_folds)

    @property
    def test_set_index(self) -> np.ndarray:
        """Shape (n_splits, n_test_folds) – fold numbers used as test per split."""
        return np.array(
            list(itertools.combinations(np.arange(self.n_folds), self.n_test_folds))
        ).reshape(-1, self.n_test_folds)

    @property
    def binary_train_test_sets(self) -> np.ndarray:
        """Shape (n_folds, n_splits) – 0 = train fold, 1 = test fold."""
        mat = np.zeros((self.n_folds, self.n_splits))
        mat[self.test_set_index, np.arange(self.n_splits)[:, np.newaxis]] = 1
        return mat

    @property
    def recombined_paths(self) -> np.ndarray:
        """
        Shape (n_folds, n_test_paths) – split index in which each fold appears
        as a test set, ordered by path.  Used to reconstruct backtest paths."""
        return np.argwhere(self.binary_train_test_sets == 1)[:, 1].reshape(
            self.n_folds, -1
        )

    # ------------------------------------------------------------------
    # Path IDs
    # ------------------------------------------------------------------

    def get_path_ids(self) -> np.ndarray:
        """Return the path id of each test set in each split.

        Returns
        -------
        path_ids : ndarray of shape (n_splits, n_test_folds)
            ``path_ids[i, j]`` is the path index that the j-th test fold of
            split *i* contributes to.
        """
        recombined = self.recombined_paths
        path_ids = np.zeros((self.n_splits, self.n_test_folds), dtype=int)
        for i in range(self.n_splits):
            for j in range(self.n_test_folds):
                path_ids[i, j] = int(np.argwhere(recombined == i)[j][1])
        return path_ids

    # ------------------------------------------------------------------
    # Core split
    # ------------------------------------------------------------------

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Generate train/test index arrays for each combinatorial split.

        Each yielded training set has already been:

        1. **Purged** (event-based) – observations whose ``t1`` window overlaps
           any test-set event window are removed via :func:`ml_get_train_times`.
        2. **Embargoed** (positional) – the first ``embargo_size`` training
           observations immediately following the end of each contiguous test
           block are removed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must share its index with ``self.t1``.
        y : array-like of shape (n_samples,), optional
        groups : ignored

        Yields
        ------
        train_index : ndarray
            Row indices for the purged + embargoed training set.
        test_index_list : list of ndarray
            One array of row indices per test fold, in ascending fold order.
        """
        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]

        # ---- validate X / t1 alignment ----
        x_index = X.index if hasattr(X, "index") else pd.RangeIndex(n_samples)
        if len(x_index) != len(self.t1) or not x_index.equals(self.t1.index):
            raise ValueError(
                "X and t1 must share the same index "
                "(same length and identical index labels)."
            )

        embargo_size = int(n_samples * self.pct_embargo)

        # Map every observation to a fold id  (0 … n_folds-1)
        fold_index_num = np.arange(n_samples) // (n_samples // self.n_folds)
        fold_index_num[fold_index_num == self.n_folds] = self.n_folds - 1

        # Pre-compute positional arrays for each fold
        fold_pos: dict = {
            fid: np.argwhere(fold_index_num == fid).reshape(-1)
            for fid in range(self.n_folds)
        }

        # Observation-role matrix:  0=train, 1=test, -1=excluded
        index_train_test = np.zeros((n_samples, self.n_splits), dtype=np.int8)

        test_set_index = self.test_set_index  # (n_splits, n_test_folds)
        recombined = self.recombined_paths    # (n_folds,  n_test_paths)

        for i in range(self.n_splits):
            # 1. Mark test observations
            for fid in test_set_index[i]:
                index_train_test[fold_pos[fid], i] = 1

            # 2. Event-based purging via t1
            test_pos = np.argwhere(index_train_test[:, i] == 1).reshape(-1)
            train_pos = np.argwhere(index_train_test[:, i] == 0).reshape(-1)

            t1_test = pd.Series(
                self.t1.iloc[test_pos].values,
                index=self.t1.index[test_pos],
            )
            t1_train_candidate = pd.Series(
                self.t1.iloc[train_pos].values,
                index=self.t1.index[train_pos],
            )

            t1_purged = ml_get_train_times(t1_train_candidate, t1_test)
            surviving = set(t1_purged.index)

            purged_pos = np.array(
                [p for p in train_pos if self.t1.index[p] not in surviving],
                dtype=int,
            )
            if purged_pos.size:
                index_train_test[purged_pos, i] = -1

            # 3. Positional embargo after each contiguous test block
            if embargo_size > 0:
                test_mask = index_train_test[:, i] == 1
                # Block-end: test obs followed by non-test obs (or end of series)
                block_ends = np.where(
                    test_mask & ~np.append(test_mask[1:], False)
                )[0]
                for end_pos in block_ends:
                    for ep in range(end_pos + 1, min(end_pos + 1 + embargo_size, n_samples)):
                        if index_train_test[ep, i] == 0:
                            index_train_test[ep, i] = -1

            # 4. Yield
            train_index = np.argwhere(index_train_test[:, i] == 0).reshape(-1)
            test_index_list = [
                fold_pos[fid] for fid, _ in np.argwhere(recombined == i)
            ]
            yield train_index, test_index_list

        # Store after all splits are generated
        self.index_train_test_ = index_train_test
        self._fold_index_num = fold_index_num

    # ------------------------------------------------------------------
    # Prediction recombination
    # ------------------------------------------------------------------

    def recombine_test_predictions(
        self,
        all_predictions: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Recombine per-split predictions into complete backtest paths.

        Each path covers every test observation exactly once, assigned to
        paths via :meth:`get_path_ids`.

        Parameters
        ----------
        all_predictions : list of ndarray, length ``n_splits``
            ``all_predictions[i]`` contains model predictions for split *i*,
            ordered to match ``np.concatenate(test_index_list)`` from
            ``split()``.

        Returns
        -------
        paths : list of ndarray, length ``n_test_paths``
            Each array holds predictions for all test observations on that
            path, ordered by their original row position in ``X``.
        """
        if not hasattr(self, "index_train_test_") or not hasattr(self, "_fold_index_num"):
            raise RuntimeError(
                "Exhaust the `split(X)` generator fully before calling "
                "`recombine_test_predictions`."
            )

        path_ids = self.get_path_ids()        # (n_splits, n_test_folds)
        test_set_index = self.test_set_index  # (n_splits, n_test_folds)

        # path_store[p] : {row_position -> [prediction, ...]}
        path_store: List[dict] = [{} for _ in range(self.n_test_paths)]

        for i, (preds, fold_ids, pid_row) in enumerate(
            zip(all_predictions, test_set_index, path_ids)
        ):
            split_preds = np.asarray(preds)
            # Test observations for this split in positional order
            test_pos_all = np.argwhere(self.index_train_test_[:, i] == 1).reshape(-1)
            offset = 0
            for j, fold_id in enumerate(fold_ids):
                path_id = int(pid_row[j])
                fold_size = int((self._fold_index_num == fold_id).sum())
                chunk_pos = test_pos_all[offset: offset + fold_size]
                chunk_preds = split_preds[offset: offset + fold_size]
                for pos, pred_val in zip(chunk_pos, chunk_preds):
                    path_store[path_id].setdefault(int(pos), []).append(float(pred_val))
                offset += fold_size

        paths = []
        for store in path_store:
            sorted_keys = sorted(store.keys())
            paths.append(np.array([np.mean(store[k]) for k in sorted_keys]))
        return paths

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, X) -> pd.Series:
        """
        Return a human-readable summary of the CV configuration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pd.Series
        """
        n_obs = np.asarray(X).shape[0]
        return pd.Series(
            {
                "Number of Observations": n_obs,
                "Total Number of Folds": self.n_folds,
                "Number of Test Folds": self.n_test_folds,
                "Embargo Size (obs)": int(n_obs * self.pct_embargo),
                "Average Training Size": int(
                    _avg_train_size(n_obs, self.n_folds, self.n_test_folds)
                ),
                "Number of Test Paths": self.n_test_paths,
                "Number of Training Combinations": self.n_splits,
            }
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_train_test_folds(self) -> go.Figure:
        """Plot the train/test fold locations."""
        values = self.binary_train_test_sets
        fill_color = np.where(values == 0, "blue", "red")
        fill_color = fill_color.astype(object)
        fill_color = np.insert(
            fill_color, 0, np.array(["darkblue" for _ in range(self.n_splits)]), axis=0
        )
        values = np.insert(values, 0, np.arange(self.n_splits), axis=0)
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Train Combinations"]
                        + [f"Fold {i}" for i in range(self.n_folds)],
                        fill_color="darkblue",
                        font=dict(color="white"),
                        align="left",
                    ),
                    cells=dict(
                        values=values,
                        font=dict(color="white"),
                        fill_color=fill_color,
                        line_color="grey",
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(title="Split Train (0) /Test (1) Folds per Combination")
        return fig

    def plot_train_test_index(self, X) -> go.Figure:
        """
        Plot the training and test indices for each split by assigning ``0`` to
        training, ``1`` to test, and ``-1`` to purged or embargoed observations.

        Each column of the resulting table corresponds to one of the
        ``n_splits`` combinatorial train/test splits; each row corresponds to
        one observation in ``X``.  The colour encoding is:

        * **Blue**  (0)  – observation is in the training set for that split.
        * **Red**   (1)  – observation is in the test set for that split.
        * **Green** (−1) – observation has been excluded from the training set
          due to purging (event-label overlap with the test window) or
          embargoing (positional buffer immediately after the test block).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must share its index with ``self.t1``.  Only the shape and index
            are used; the feature values are not accessed.

        Returns
        -------
        go.Figure
            An interactive Plotly table figure.

        Notes
        -----
        This method requires ``index_train_test_`` to be fully populated
        across **all** ``n_splits`` columns before it can render a correct
        visualisation.  ``index_train_test_`` is built incrementally inside
        the ``split()`` generator and is only finalised once the generator is
        fully exhausted.

        To guarantee this, the method checks whether ``index_train_test_``
        already exists (e.g. because ``CPCVAnalyzer.fit_predict`` or a manual
        ``list(self.split(X))`` call has already run).  If it does not exist,
        the full generator is exhausted here automatically.  The previous
        behaviour of calling ``next(self.split(X))`` — which populated only
        the first column — has been removed, as it produced an incorrect
        visualisation for all splits after the first.
        """
        if not hasattr(self, "index_train_test_"):
            # Exhaust the full generator so every column of index_train_test_
            # is populated before we attempt to render the table.
            for _ in self.split(X):
                pass

        n_samples = X.shape[0]
        cond = [
            self.index_train_test_ == -1,
            self.index_train_test_ == 0,
            self.index_train_test_ == 1,
        ]
        values = self.index_train_test_.T
        values = np.insert(values, 0, np.arange(n_samples), axis=0)
        fill_color = np.select(cond, ["green", "blue", "red"], default="green").T
        fill_color = fill_color.astype(object)
        fill_color = np.insert(
            fill_color, 0, np.array(["darkblue" for _ in range(n_samples)]), axis=0
        )
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["observations"]
                        + [f"Split {i}" for i in range(self.n_splits)],
                        fill_color="darkblue",
                        font=dict(color="white"),
                        align="left",
                    ),
                    cells=dict(
                        values=values,
                        font=dict(color="white"),
                        fill_color=fill_color,
                        line_color="grey",
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(
            title="Train (0), Test (1) and Purge/Embargo (-1) observations per splits"
        )

        return fig


# ---------------------------------------------------------------------------
# Optimal-folds search
# ---------------------------------------------------------------------------

def optimal_folds_number(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
    weight_train_size: float = 1.0,
    weight_n_test_paths: float = 1.0,
) -> Tuple[int, int]:
    r"""Find the ``(n_folds, n_test_folds)`` pair that best matches the targets.

    Minimises the weighted relative distance:

    .. math::

        \text{cost}(x,y) =
            w_f \left|\frac{f(x,y) - f_{\text{target}}}{f_{\text{target}}}\right|
          + w_g \left|\frac{g(x,y) - g_{\text{target}}}{g_{\text{target}}}\right|

    where :math:`f` is the average training size and :math:`g` is the number
    of test paths.

    Parameters
    ----------
    n_observations : int
    target_train_size : int
    target_n_test_paths : int
    weight_train_size : float, default=1.0
    weight_n_test_paths : float, default=1.0

    Returns
    -------
    n_folds : int
    n_test_folds : int
    """

    def _cost(x: int, y: int) -> float:
        n_paths = _n_test_paths(n_folds=x, n_test_folds=y)
        avg_tr = _avg_train_size(n_observations, x, y)
        return (
            weight_n_test_paths * abs(n_paths - target_n_test_paths) / target_n_test_paths
            + weight_train_size * abs(avg_tr - target_train_size) / target_train_size
        )

    costs, res = [], []
    for n_folds in range(3, n_observations + 1):
        cutoff = None
        for n_test_folds in range(2, n_folds):
            if cutoff is None or n_folds - n_test_folds <= cutoff:
                c = _cost(n_folds, n_test_folds)
                costs.append(c)
                res.append((n_folds, n_test_folds))
                if cutoff is None and c > 1e5:
                    cutoff = n_test_folds
    return res[int(np.argmin(costs))]



# ---------------------------------------------------------------------------
# CPCVAnalyzer — module-level worker functions
# (must be at module scope so joblib can pickle them across processes)
# ---------------------------------------------------------------------------

def _fit_predict_fold(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,       # flat array — caller concatenates test_index_list
    fold_idx: int,
    sample_weight: Optional[pd.Series] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Train one fold and return (fold_idx, test_idx, predictions).

    Returns the fold index so that results can be reordered after parallel
    execution regardless of completion order.
    """
    model = clone(estimator)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.iloc[train_idx]
    model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)

    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X.iloc[test_idx])[:, 1]
    else:
        preds = model.predict(X.iloc[test_idx])

    return fold_idx, test_idx, preds


def _apply_bet_method_static(
    method: str,
    probs: pd.Series,
    sides: pd.Series,
    num_classes: int,
):
    """
    Apply a bet-sizing method and return a signal Series.

    Extracted to module scope so it is picklable by joblib workers.
    """
    from ..bet_sizing.ch10_snippets import get_signal  # project-local import
    if method == "sigmoid":
        return get_signal(probs, num_classes, pred=sides)
    elif method == "power":
        conviction = get_signal(probs, num_classes)
        return sides * (conviction ** 2)
    elif method == "binary":
        return sides * (probs > 0.5).astype(float)
    raise ValueError(f"Unknown bet method: {method!r}")


def _compute_path_metrics(
    path_id: int,
    path_probs: pd.Series,
    method: str,
    primary_sides: pd.Series,
    close_index: pd.Index,
    t1: pd.Series,
    log_returns_values: np.ndarray,
    num_classes: int,
    step_size: float,
) -> Optional[dict]:
    """
    Compute all performance metrics for one (path, method) combination.

    Module-level so joblib can serialise it. Returns None when there are
    insufficient OOS observations to produce meaningful statistics.
    """
    clean_probs = path_probs.dropna()
    if clean_probs.empty:
        return None

    clean_sides = primary_sides.loc[clean_probs.index]
    signals = _apply_bet_method_static(method, clean_probs, clean_sides, num_classes)

    if step_size > 0:
        from ..bet_sizing.ch10_snippets import discrete_signal
        signals = discrete_signal(signals, step_size)

    # Positional index arrays for the MtM averaging kernel
    t0_idx = close_index.get_indexer(signals.index)
    t1_idx = close_index.get_indexer(t1.loc[signals.index])

    avg_pos = fill_average_active_sides(
        len(close_index), t0_idx, t1_idx, signals.values
    )

    full_log_rets = avg_pos * log_returns_values
    active_mask = avg_pos != 0
    oos_log_rets = full_log_rets[active_mask]

    if len(oos_log_rets) < 5:
        return None

    # ---- summary statistics (scipy avoids pd.Series allocation overhead) ----
    arith_rets = np.expm1(oos_log_rets)
    sr_raw = oos_log_rets.mean() / oos_log_rets.std(ddof=1)
    skewness = float(scipy_skew(oos_log_rets))
    excess_kurt = float(scipy_kurt(oos_log_rets, fisher=True))  # excess kurtosis
    n = len(oos_log_rets)

    # ---- profit factor ----
    neg_sum = arith_rets[arith_rets < 0].sum()
    pf = (arith_rets[arith_rets > 0].sum() / abs(neg_sum)) if neg_sum != 0 else 0.0

    # ---- PSR (López de Prado AFML Ch. 14) ----
    var_sr = (1 - skewness * sr_raw + ((excess_kurt - 1) / 4) * sr_raw ** 2) / (n - 1)
    psr = float(norm.cdf(sr_raw / np.sqrt(var_sr))) if var_sr > 0 else 0.0

    # ---- max drawdown (log-space for numerical stability) ----
    cum = np.exp(np.cumsum(full_log_rets))
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / peak - 1))

    # ---- turnover ----
    turnover = float(np.abs(np.diff(avg_pos)).sum())

    return {
        "method": method,
        "path_id": path_id,
        "mtm_profit_factor": pf,
        "mtm_sharpe": sr_raw,
        "psr": psr,
        "max_drawdown": max_dd,
        "turnover": turnover,
    }


# ---------------------------------------------------------------------------
# CPCVAnalyzer
# ---------------------------------------------------------------------------

class CPCVAnalyzer:
    """
    Parallel CPCV execution and MtM performance metric distribution.

    Handles the full pipeline from raw model predictions to a distribution of
    time-weighted portfolio performance across all combinatorial backtest paths.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    cv_gen : CombinatorialPurgedCV
        The configured CPCV generator. Must have ``t1`` set.
    close_prices : pd.Series
        Continuous price series for Mark-to-Market calculation.
    n_jobs : int, default=-1
        Joblib parallelism for both ``fit_predict`` and ``get_distribution_metrics``.
    """

    _BET_METHODS = ("sigmoid", "power", "binary")

    def __init__(
        self,
        estimator,
        cv_gen: CombinatorialPurgedCV,
        close_prices: pd.Series,
        n_jobs: int = -1,
    ) -> None:
        self.estimator = estimator
        self.cv_gen = cv_gen
        self.close = close_prices
        self.n_jobs = n_jobs if n_jobs != -1 else joblib.cpu_count(only_physical_cores=True)

        # Pre-compute log returns once; aligned to close_prices index
        self.log_returns: pd.Series = np.log(close_prices).diff().shift(-1).fillna(0)

        # Set by fit_predict
        self._X: Optional[pd.DataFrame] = None
        self._split_predictions: Optional[List[np.ndarray]] = None
        self._backtest_paths_cache: Optional[List[pd.Series]] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Train the estimator across all CPCV splits in parallel.

        The split generator is exhausted eagerly so that ``cv_gen`` stores
        ``index_train_test_`` and ``_fold_index_num`` before any downstream
        method reads them.

        Returns
        -------
        pd.Series
            Recombined (bagged) predictions — mean across all paths per timestamp.
        """
        self._X = X
        self._backtest_paths_cache = None  # invalidate on refit

        # Exhaust the generator eagerly: this populates cv_gen.index_train_test_
        # and cv_gen._fold_index_num, which recombine_test_predictions requires.
        splits = [
            (train, np.concatenate(test_list))
            for train, test_list in self.cv_gen.split(X, y)
        ]  # length == cv_gen.n_splits

        results: List[Tuple[int, np.ndarray, np.ndarray]] = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_fold)(
                self.estimator, X, y, train, test, i, sample_weight
            )
            for i, (train, test) in enumerate(splits)
        )

        # Sort by fold index — Parallel does not guarantee return order
        results.sort(key=lambda r: r[0])
        self._split_predictions = [preds for _, _, preds in results]

        return self.recombined_predictions

    # ------------------------------------------------------------------
    # Derived views (cached)
    # ------------------------------------------------------------------

    @property
    def backtest_paths(self) -> List[pd.Series]:
        """Assemble the ``n_test_paths`` backtest paths from split predictions.

        Delegates entirely to ``cv_gen.recombine_test_predictions`` — no
        manual fold-index arithmetic here.  The result is cached after the
        first call and invalidated by ``fit_predict``.

        Returns
        -------
        list of pd.Series, length ``cv_gen.n_test_paths``
            Each Series covers all observations in the dataset, indexed by
            the original timestamps from ``X``.
        """
        if self._split_predictions is None:
            raise RuntimeError("Call fit_predict before accessing backtest_paths.")

        if self._backtest_paths_cache is None:
            raw: List[np.ndarray] = self.cv_gen.recombine_test_predictions(
                self._split_predictions
            )
            self._backtest_paths_cache = [
                pd.Series(path, index=self._X.index, name=f"path_{j}")
                for j, path in enumerate(raw)
            ]

        return self._backtest_paths_cache

    @property
    def recombined_predictions(self) -> pd.Series:
        """Mean prediction across all paths at each timestamp.

        Each observation appears in every path exactly once, so this is
        equivalent to averaging across splits where the observation was OOS.
        """
        return (
            pd.concat(self.backtest_paths, axis=1)
            .mean(axis=1)
            .rename("recombined_prediction")
        )

    # ------------------------------------------------------------------
    # Performance distribution
    # ------------------------------------------------------------------

    def get_distribution_metrics(
        self,
        primary_sides: pd.Series,
        num_classes: int = 2,
        step_size: float = 0.0,
    ) -> pd.DataFrame:
        """Calculate MtM performance metrics for all paths and bet-sizing methods.

        All ``n_test_paths × 3`` (path, method) pairs are evaluated in parallel
        so the wall time is roughly that of the single slowest combination.

        Parameters
        ----------
        primary_sides : pd.Series
            Direction (+1 / -1) from the primary model for meta-labeling.
        num_classes : int, default=2
            Number of classes for confidence z-score in ``get_signal``.
        step_size : float, default=0.0
            Discretisation step for bet sizes; 0.0 = continuous.

        Returns
        -------
        pd.DataFrame
            MultiIndex ``[method, path_id]`` with columns:
            ``mtm_profit_factor``, ``mtm_sharpe``, ``psr``,
            ``max_drawdown``, ``turnover``.
        """
        # Snapshot heavy attributes once — avoids repeated self-lookups inside workers
        close_index = self.close.index
        t1 = self.cv_gen.t1
        log_ret_vals = self.log_returns.values

        raw_results: List[Optional[dict]] = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_path_metrics)(
                path_id=i,
                path_probs=path,
                method=method,
                primary_sides=primary_sides,
                close_index=close_index,
                t1=t1,
                log_returns_values=log_ret_vals,
                num_classes=num_classes,
                step_size=step_size,
            )
            for i, path in enumerate(self.backtest_paths)
            for method in self._BET_METHODS
        )

        records = [r for r in raw_results if r is not None]
        if not records:
            return pd.DataFrame(
                columns=["mtm_profit_factor", "mtm_sharpe", "psr", "max_drawdown", "turnover"]
            )

        return (
            pd.DataFrame(records)
            .set_index(["method", "path_id"])
            .sort_index()
        )