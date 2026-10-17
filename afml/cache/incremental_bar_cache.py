"""
Incremental bar cache for AFML.

Avoids recomputing bars from scratch when new ticks extend a previously
cached date range.  The invariant is strict:

    cached([t₀, t₁]) + incremental([t₁, t₂])
    == from_scratch([t₀, t₂])

to within floating-point tolerance.

Supported bar types
-------------------
Fully incremental (O(new ticks) cost):
    time, tick, volume, dollar

Not yet incremental (full recomputation on extension):
    tick_imbalance, volume_imbalance, dollar_imbalance,
    tick_runs, volume_runs, dollar_runs

Information bars carry evolving EWM-based expectations (E[T], E[θ])
that feed back into bar-closing decisions.  Incremental extension
requires passing the full accumulator state into the boundary
detection functions.  This is planned but not yet implemented;
for now, extending the date range on an information-bar cache
triggers a full recomputation with a logged warning.

Calibration interface
---------------------
Information bars accept either:

  (A) ``target_timeframe``  — auto-calibrated initial parameters based on
      the target clock-time cadence (recommended).
  (B) ``exp_ticks_init`` / ``exp_imbalance_init``  — manual initial
      parameters (advanced).

The two paths are mutually exclusive and produce distinct cache keys,
so switching calibration mode produces a cache miss (by design — the
resulting bars would differ during the EWM transient).

Thread safety
-------------
The cache uses file-level locking via a simple rename-on-commit
pattern.  Concurrent reads are safe; concurrent writes to the same
cache entry are last-writer-wins.

References
----------
- AFML Ch. 1 (bars, information-driven bars)
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STANDARD_BAR_TYPES = frozenset({"time", "tick", "volume", "dollar"})
_INFO_BAR_TYPES = frozenset(
    {
        "tick_imbalance",
        "volume_imbalance",
        "dollar_imbalance",
        "tick_runs",
        "volume_runs",
        "dollar_runs",
    }
)


# ---------------------------------------------------------------------------
# Boundary state
# ---------------------------------------------------------------------------


@dataclass
class BarBoundaryState:
    """Serialisable state at the trailing edge of a bar computation.

    Attributes
    ----------
    bar_type : str
        The bar type that produced this state.
    params_hash : str
        Hash of all bar parameters (bar_size, price, etc.) so we can
        detect parameter changes that invalidate the cache.
    end_timestamp : pd.Timestamp
        Timestamp of the last tick that was processed.
    n_ticks_processed : int
        Total number of ticks consumed to produce the cached bars.
    leftover_ticks : pd.DataFrame
        Ticks from the last incomplete bar.  These must be prepended to
        new data before resuming bar construction.
    accumulator : dict
        Bar-type-specific accumulator state.  For standard bars this is
        minimal (just the partial cumulative metric).  For information
        bars it would include EWM histories (not yet implemented).
    """

    bar_type: str
    params_hash: str
    end_timestamp: pd.Timestamp
    n_ticks_processed: int
    leftover_ticks: pd.DataFrame
    accumulator: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parameter hashing
# ---------------------------------------------------------------------------


def _hash_bar_params(
    bar_type: str,
    bar_size: int | str,
    price: str,
    **extra,
) -> str:
    """
    Deterministic hash of bar construction parameters.

    ``None`` values in ``extra`` are excluded so that standard bars
    produce the same hash regardless of which information-bar
    parameters were passed. For information bars, the effective
    parameter set (target_timeframe XOR exp_ticks_init/exp_imbalance_init)
    is included, producing distinct cache entries for the two
    calibration modes.
    """
    parts = [
        f"bar_type={bar_type}",
        f"bar_size={bar_size}",
        f"price={price}",
    ]
    for k in sorted(extra):
        if extra[k] is not None:
            parts.append(f"{k}={extra[k]}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache entry (what gets persisted)
# ---------------------------------------------------------------------------


@dataclass
class IncrementalCacheEntry:
    """Persisted alongside the bar DataFrame."""

    bars: pd.DataFrame
    boundary: BarBoundaryState
    created_at: float = field(default_factory=time.time)
    version: int = 1


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def _detect_gap(
    cached_end: pd.Timestamp,
    new_start: pd.Timestamp,
    expected_freq: pd.Timedelta,
    tolerance_factor: float = 3.0,
) -> bool:
    """Return True if there is a suspicious gap between cached and new data."""
    gap = new_start - cached_end
    return gap > expected_freq * tolerance_factor


def _estimate_tick_freq(tick_df: pd.DataFrame, sample_size: int = 1000) -> pd.Timedelta:
    """Estimate median inter-tick interval from a sample."""
    if len(tick_df) < 2:
        return pd.Timedelta(seconds=1)
    sample = tick_df.index[:sample_size]
    diffs = pd.Series(sample).diff().dropna()
    return diffs.median()


# ---------------------------------------------------------------------------
# Leftover extraction
# ---------------------------------------------------------------------------


def _extract_leftover_standard(
    tick_df: pd.DataFrame,
    bar_type: str,
    bar_size: int,
    price: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split tick_df into (complete bar ticks, leftover ticks, accumulator).

    For tick bars: leftover = last ``len % bar_size`` ticks.
    For volume/dollar bars: leftover = ticks after the last cumsum
    threshold crossing.

    Returns
    -------
    complete_ticks : pd.DataFrame
        Ticks that map to fully closed bars.
    leftover_ticks : pd.DataFrame
        Ticks from the incomplete trailing bar.
    accumulator : dict
        For volume/dollar: ``{"partial_cumsum": float}`` — the
        cumulated metric within the incomplete bar.
        For tick: empty dict.
    """
    n = len(tick_df)

    if bar_type == "tick":
        leftover_count = n % bar_size
        if leftover_count == 0:
            return tick_df, tick_df.iloc[0:0], {}
        return (
            tick_df.iloc[:-leftover_count],
            tick_df.iloc[-leftover_count:],
            {},
        )

    if bar_type == "time":
        # Time bars close on fixed boundaries; no leftover concept.
        return tick_df, tick_df.iloc[0:0], {}

    # volume / dollar bars
    if bar_type == "dollar":
        metric = tick_df["volume"] * tick_df[price]
    else:  # "volume"
        metric = tick_df["volume"]

    cumsum = metric.cumsum()
    # Number of complete bars
    n_complete_bars = int(cumsum.iloc[-1] // bar_size)

    if n_complete_bars == 0:
        return tick_df.iloc[0:0], tick_df, {"partial_cumsum": float(cumsum.iloc[-1])}

    # Last complete bar ends where cumsum crosses n_complete_bars * bar_size
    threshold = n_complete_bars * bar_size
    last_complete_idx = (cumsum >= threshold).idxmax()
    last_complete_pos = tick_df.index.get_loc(last_complete_idx)

    # Everything up to and including that tick is complete
    complete = tick_df.iloc[: last_complete_pos + 1]
    leftover = tick_df.iloc[last_complete_pos + 1 :]

    if len(leftover) > 0:
        partial = float(metric.iloc[last_complete_pos + 1 :].sum())
    else:
        partial = 0.0

    return complete, leftover, {"partial_cumsum": partial}


# ---------------------------------------------------------------------------
# Information-bar calibration validation
# ---------------------------------------------------------------------------


def _validate_info_bar_calibration(
    bar_type: str,
    target_timeframe: str | None,
    exp_ticks_init: float | None,
    exp_imbalance_init: float | None,
) -> None:
    """Enforce mutual exclusivity and required-parameter rules.

    Mirrors the contract in ``make_bars`` so that callers get early
    validation at the cache layer rather than after tick data is
    loaded or a cache lookup is attempted.

    Raises
    ------
    ValueError
        If ``bar_type`` is an information bar and the caller has
        provided an invalid combination of calibration parameters.
    """
    if bar_type not in _INFO_BAR_TYPES:
        return

    raw_params_given = (exp_ticks_init is not None) or (exp_imbalance_init is not None)

    if target_timeframe is not None and raw_params_given:
        raise ValueError(
            f"For information bar type '{bar_type}': pass either "
            f"'target_timeframe' (recommended, auto-calibrated) OR "
            f"'exp_ticks_init'/'exp_imbalance_init' (advanced, manual), "
            f"not both."
        )

    if target_timeframe is None and exp_ticks_init is None:
        raise ValueError(
            f"Information bar type '{bar_type}' requires calibration. "
            f"Pass either 'target_timeframe' (e.g., 'M15') for "
            f"auto-calibration or 'exp_ticks_init' for manual control."
        )


# ---------------------------------------------------------------------------
# Incremental bar cache
# ---------------------------------------------------------------------------


class IncrementalBarCache:
    """Wrap ``make_bars`` with incremental caching.

    Parameters
    ----------
    cache_dir : Path
        Directory for persisted cache entries.
    make_bars_fn : callable
        The ``afml.data_structures.make_bars`` function (or a compatible
        callable).  Injected to avoid circular imports.

    Examples
    --------
    Auto-calibrated dollar imbalance bars:

    >>> from afml.data_structures import make_bars
    >>> cache = IncrementalBarCache(Path("bar_cache"))
    >>> bars = cache.get_or_extend(
    ...     ticks_2021_2023,
    ...     bar_type="dollar_imbalance",
    ...     target_timeframe="M15",
    ... )

    Standard dollar bars with incremental extension:

    >>> bars = cache.get_or_extend(
    ...     ticks_2021_2023,
    ...     bar_type="dollar",
    ...     bar_size=1_000_000,
    ...     price="mid_price",
    ... )
    >>> # Later, with extended data — incremental path is used:
    >>> bars = cache.get_or_extend(
    ...     ticks_2021_2024,
    ...     bar_type="dollar",
    ...     bar_size=1_000_000,
    ...     price="mid_price",
    ... )
    """

    def __init__(
        self,
        cache_dir: Path = None,
        make_bars_fn: Callable | None = None,
    ):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            from .unified_cache import CACHE_DIRS

            CACHE_DIRS["bar_cache"] = CACHE_DIRS["base"] / "bar_cache"
            self.cache_dir = CACHE_DIRS["bar_cache"]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._make_bars = make_bars_fn

    @property
    def make_bars_fn(self) -> Callable:
        if self._make_bars is None:
            from ..data_structures import make_bars

            self._make_bars = make_bars
        return self._make_bars

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _cache_path(self, params_hash: str) -> Path:
        """Return the file path for a given params hash."""
        return self.cache_dir / f"bars_{params_hash}.pkl"

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self, params_hash: str) -> IncrementalCacheEntry | None:
        path = self._cache_path(params_hash)
        if not path.exists():
            return None
        try:
            if joblib is not None:
                entry = joblib.load(path)
            else:
                import pickle

                with open(path, "rb") as f:
                    entry = pickle.load(f)
            if not isinstance(entry, IncrementalCacheEntry):
                logger.warning(f"Corrupt cache entry at {path}; ignoring.")
                return None
            return entry
        except Exception as exc:
            logger.warning(f"Failed to load cache {path}: {exc}")
            return None

    def _save(self, params_hash: str, entry: IncrementalCacheEntry) -> None:
        path = self._cache_path(params_hash)
        tmp = path.with_suffix(".tmp")
        try:
            if joblib is not None:
                joblib.dump(entry, tmp, compress=3)
            else:
                import pickle

                with open(tmp, "wb") as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(path)  # atomic on POSIX
        except Exception as exc:
            logger.warning(f"Failed to save cache {path}: {exc}")
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_extend(
        self,
        tick_df: pd.DataFrame,
        bar_type: str = "tick",
        bar_size: int | str = 100,
        price: str = "mid_price",
        tick_num: bool = True,
        verbose: bool = False,
        # --- Information bar calibration (preferred entry point) ---
        target_timeframe: str | None = None,
        ewm_span: int = 20,
        # --- Escape hatches (mutually exclusive with target_timeframe) ---
        exp_ticks_init: float | None = None,
        exp_imbalance_init: float | None = None,
    ) -> pd.DataFrame:
        """Return bars, using cached data when possible.

        Decision tree
        ~~~~~~~~~~~~~
        0. Validate calibration parameters for information bars.
        1. No cache exists → compute from scratch, store, return.
        2. Cache covers requested range → return cached bars.
        3. Cache covers a prefix of the requested range:
           a. Standard bars → incremental extension.
           b. Information bars → full recomputation (with warning).
        4. Cache parameters don't match → full recomputation.
        5. New data starts before cached end → full recomputation
           (data corrections / overlapping ranges).

        Parameters
        ----------
        tick_df : pd.DataFrame
            Tick data with DatetimeIndex (or ``time`` column).

        bar_type : str, default='tick'
            One of the ten supported bar types. See ``make_bars``.

        bar_size : int or str, default=100
            Standard bars only; ignored for information bars.

        price : str, default='mid_price'
            Price column for OHLC construction.

        tick_num : bool, default=True
            Add 'tick_num' column with 1-based global tick index.

        verbose : bool, default=False
            Log bar construction diagnostics.

        target_timeframe : str, optional
            Information bars only. MT5 timeframe string (e.g., 'M15')
            expressing the target bar cadence. Auto-calibrates initial
            threshold parameters. Mutually exclusive with ``exp_ticks_init``.

        ewm_span : int, default=20
            Information bars only. EWM span for threshold adaptation.

        exp_ticks_init : int, float, or None
            Information bars only. Manual initial E_0[T]. Mutually
            exclusive with ``target_timeframe``.

        exp_imbalance_init : float, optional
            Information bars only. Manual initial E_0[|imbalance|].
            Defaults to 0.1 inside ``make_bars`` if ``exp_ticks_init`` is
            provided but this is omitted. Mutually exclusive with
            ``target_timeframe``.

        Returns
        -------
        pd.DataFrame
            OHLC bar DataFrame.

        Raises
        ------
        ValueError
            If information-bar calibration parameters are in an invalid
            combination (both paths or neither).
        TypeError
            If ``tick_df`` lacks a usable datetime index.
        """
        # ----------------------------------------------------------
        # Step 0: Validate calibration parameters
        # ----------------------------------------------------------
        _validate_info_bar_calibration(
            bar_type=bar_type,
            target_timeframe=target_timeframe,
            exp_ticks_init=exp_ticks_init,
            exp_imbalance_init=exp_imbalance_init,
        )

        # Build the hash. For information bars, target_timeframe and the
        # raw init parameters produce distinct entries (by design — the
        # two calibration paths yield different bars during the EWM
        # transient). For standard bars, all information-bar parameters
        # are None and therefore excluded.
        is_info = bar_type in _INFO_BAR_TYPES
        params_hash = _hash_bar_params(
            bar_type=bar_type,
            bar_size=bar_size,
            price=price,
            target_timeframe=target_timeframe if is_info else None,
            exp_ticks_init=exp_ticks_init if is_info else None,
            exp_imbalance_init=exp_imbalance_init if is_info else None,
            ewm_span=ewm_span if is_info else None,
        )

        entry = self._load(params_hash)

        # Ensure tick_df has a DatetimeIndex
        if not isinstance(tick_df.index, pd.DatetimeIndex):
            # First look for a column named 'time'
            if "time" in tick_df.columns:
                tick_df = tick_df.set_index("time")
            else:
                # Scan for any datetime64 column
                datetime_cols = tick_df.select_dtypes(include=["datetime64"]).columns
                if len(datetime_cols) > 0:
                    # Use the first datetime column as index and optionally rename to 'time'
                    tick_df = tick_df.set_index(datetime_cols[0])
                    # If the original index name is not 'time', rename it for consistency
                    tick_df.index.name = "time"
                else:
                    raise TypeError("tick_df must have a DatetimeIndex or a datetime64 column.")

        tick_end = tick_df.index[-1]

        # ----------------------------------------------------------
        # Case 1: No cache
        # ----------------------------------------------------------
        if entry is None:
            logger.debug(f"CACHE MISS | bars_{params_hash[:12]} | no existing entry")
            return self._compute_and_store(
                tick_df=tick_df,
                bar_type=bar_type,
                bar_size=bar_size,
                price=price,
                tick_num=tick_num,
                verbose=verbose,
                target_timeframe=target_timeframe,
                ewm_span=ewm_span,
                exp_ticks_init=exp_ticks_init,
                exp_imbalance_init=exp_imbalance_init,
                params_hash=params_hash,
            )

        cached_end = entry.boundary.end_timestamp

        # ----------------------------------------------------------
        # Case 2: Cache already covers the range
        # ----------------------------------------------------------
        if cached_end >= tick_end:
            logger.debug(f"CACHE HIT  | bars_{params_hash[:12]} | cached through {cached_end}")
            return entry.bars

        # ----------------------------------------------------------
        # Case 4: Parameter mismatch (defensive — hash keying should
        # prevent this from happening in practice)
        # ----------------------------------------------------------
        if entry.boundary.params_hash != params_hash:
            logger.debug(f"CACHE MISS | bars_{params_hash[:12]} | params changed")
            return self._compute_and_store(
                tick_df=tick_df,
                bar_type=bar_type,
                bar_size=bar_size,
                price=price,
                tick_num=tick_num,
                verbose=verbose,
                target_timeframe=target_timeframe,
                ewm_span=ewm_span,
                exp_ticks_init=exp_ticks_init,
                exp_imbalance_init=exp_imbalance_init,
                params_hash=params_hash,
            )

        # ----------------------------------------------------------
        # Case 5: New data starts before cached end (corrections)
        # ----------------------------------------------------------
        new_ticks_start = tick_df.index[tick_df.index > cached_end]
        if len(new_ticks_start) == 0:
            logger.debug(f"CACHE HIT  | bars_{params_hash[:12]} | no new ticks beyond cache")
            return entry.bars

        # ----------------------------------------------------------
        # Case 3: Incremental extension
        # ----------------------------------------------------------
        if is_info:
            logger.warning(
                f"Incremental extension not yet supported for "
                f"'{bar_type}' bars. Full recomputation triggered. "
                f"(Accumulator state pass-through is planned.)"
            )
            return self._compute_and_store(
                tick_df=tick_df,
                bar_type=bar_type,
                bar_size=bar_size,
                price=price,
                tick_num=tick_num,
                verbose=verbose,
                target_timeframe=target_timeframe,
                ewm_span=ewm_span,
                exp_ticks_init=exp_ticks_init,
                exp_imbalance_init=exp_imbalance_init,
                params_hash=params_hash,
            )

        # --- Standard bar extension ---
        new_ticks = tick_df.loc[tick_df.index > cached_end]

        # Gap detection
        estimated_freq = _estimate_tick_freq(tick_df)
        if _detect_gap(cached_end, new_ticks.index[0], estimated_freq):
            gap_duration = new_ticks.index[0] - cached_end
            logger.warning(
                f"Gap detected between cached end ({cached_end}) and "
                f"new data start ({new_ticks.index[0]}): {gap_duration}. "
                f"Bars spanning the gap may be malformed."
            )

        t0 = time.perf_counter()

        # Prepend leftover ticks from the boundary
        leftover = entry.boundary.leftover_ticks
        if len(leftover) > 0:
            extension_ticks = pd.concat([leftover, new_ticks])
        else:
            extension_ticks = new_ticks

        # Compute bars on the extension segment.
        # Standard bars don't use information-bar parameters, so we omit
        # them from the call to keep the signature simple and robust.
        extension_bars = self.make_bars_fn(
            tick_df=extension_ticks,
            bar_type=bar_type,
            bar_size=bar_size,
            price=price,
            tick_num=tick_num,
            verbose=False,
        )

        # If the cached bars had a last bar that was force-closed
        # (i.e. leftover was non-empty), that last bar was already
        # dropped during the original computation. So we don't need
        # to pop anything — just concatenate.
        if len(extension_bars) > 0:
            combined_bars = pd.concat([entry.bars, extension_bars])
        else:
            combined_bars = entry.bars

        # Fix tick_num if present (must be globally monotonic)
        if tick_num and "tick_num" in combined_bars.columns:
            if len(entry.bars) > 0 and "tick_num" in entry.bars.columns:
                offset = entry.boundary.n_ticks_processed
                if len(extension_bars) > 0 and "tick_num" in extension_bars.columns:
                    combined_bars.loc[extension_bars.index, "tick_num"] = (
                        extension_bars["tick_num"].astype("int64") + offset
                    )

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"CACHE EXTEND | bars_{params_hash[:12]} | "
            f"{len(new_ticks):,} new ticks → "
            f"{len(extension_bars):,} new bars | "
            f"{str(pd.Timedelta(seconds=elapsed).round('1s')).replace('0 days ', '')}"
        )

        # Extract new boundary state from the extension ticks
        if bar_type == "time":
            new_leftover = extension_ticks.iloc[0:0]
            new_accum = {}
        else:
            _, new_leftover, new_accum = _extract_leftover_standard(
                extension_ticks,
                bar_type,
                int(bar_size),
                price,
            )

        new_boundary = BarBoundaryState(
            bar_type=bar_type,
            params_hash=params_hash,
            end_timestamp=tick_df.index[-1],
            n_ticks_processed=(entry.boundary.n_ticks_processed + len(new_ticks)),
            leftover_ticks=new_leftover,
            accumulator=new_accum,
        )

        new_entry = IncrementalCacheEntry(
            bars=combined_bars,
            boundary=new_boundary,
        )
        self._save(params_hash, new_entry)

        return combined_bars

    # ------------------------------------------------------------------
    # Full computation
    # ------------------------------------------------------------------

    def _compute_and_store(
        self,
        tick_df: pd.DataFrame,
        bar_type: str,
        bar_size: int | str,
        price: str,
        tick_num: bool,
        verbose: bool,
        target_timeframe: str | None,
        ewm_span: int,
        exp_ticks_init: float | None,
        exp_imbalance_init: float | None,
        params_hash: str,
    ) -> pd.DataFrame:
        """Compute bars from scratch and persist the cache entry.

        Forwards both calibration paths (``target_timeframe`` and
        raw ``exp_ticks_init`` / ``exp_imbalance_init``) to ``make_bars``,
        which enforces its own mutual-exclusivity contract.
        """
        t0 = time.perf_counter()

        # Standard bars don't use information-bar parameters; pass them
        # conditionally to avoid confusing ``make_bars`` if its signature
        # treats extras as unexpected kwargs.
        extra_kwargs = {}
        if bar_type in _INFO_BAR_TYPES:
            extra_kwargs.update(
                target_timeframe=target_timeframe,
                ewm_span=ewm_span,
                exp_ticks_init=exp_ticks_init,
                exp_imbalance_init=exp_imbalance_init,
            )

        bars = self.make_bars_fn(
            tick_df=tick_df,
            bar_type=bar_type,
            bar_size=bar_size,
            price=price,
            tick_num=tick_num,
            verbose=verbose,
            **extra_kwargs,
        )

        elapsed = str(pd.Timedelta(seconds=time.perf_counter() - t0).round("1s")).replace(
            "0 days ", ""
        )

        # Extract boundary state
        if bar_type == "time":
            leftover = tick_df.iloc[0:0]
            accum = {}
        elif bar_type in _STANDARD_BAR_TYPES:
            _, leftover, accum = _extract_leftover_standard(
                tick_df,
                bar_type,
                int(bar_size),
                price,
            )
        else:
            # Information bars: no incremental state yet
            leftover = tick_df.iloc[0:0]
            accum = {}

        boundary = BarBoundaryState(
            bar_type=bar_type,
            params_hash=params_hash,
            end_timestamp=tick_df.index[-1],
            n_ticks_processed=len(tick_df),
            leftover_ticks=leftover,
            accumulator=accum,
        )

        entry = IncrementalCacheEntry(bars=bars, boundary=boundary)
        self._save(params_hash, entry)

        logger.debug(
            f"CACHE MISS | bars_{params_hash[:12]} | "
            f"computed {len(bars):,} bars from "
            f"{len(tick_df):,} ticks in {elapsed}"
        )

        return bars

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def invalidate(
        self,
        bar_type: str | None = None,
        bar_size: int | str | None = None,
        price: str | None = None,
    ) -> int:
        """Remove cache entries matching the given parameters.

        If all parameters are None, clears the entire cache.

        Returns
        -------
        int
            Number of entries removed.
        """
        if bar_type is None and bar_size is None and price is None:
            count = 0
            for p in self.cache_dir.glob("bars_*.pkl"):
                p.unlink()
                count += 1
            logger.info(f"Cleared {count} bar cache entries.")
            return count

        if bar_type is not None and bar_size is not None and price is not None:
            h = _hash_bar_params(bar_type=bar_type, bar_size=bar_size, price=price)
            path = self._cache_path(h)
            if path.exists():
                path.unlink()
                logger.info(f"Invalidated bar cache: {bar_type}/{bar_size}/{price}")
                return 1
            return 0

        # Partial match: iterate all entries
        count = 0
        for p in self.cache_dir.glob("bars_*.pkl"):
            entry = self._load(p.stem.replace("bars_", ""))
            if entry is None:
                continue
            match = True
            if bar_type is not None and entry.boundary.bar_type != bar_type:
                match = False
            if match:
                p.unlink()
                count += 1
        logger.info(f"Invalidated {count} bar cache entries.")
        return count

    def status(self) -> pd.DataFrame:
        """Return a summary of all cached entries."""
        rows = []
        for p in sorted(self.cache_dir.glob("bars_*.pkl")):
            params_hash = p.stem.replace("bars_", "")
            entry = self._load(params_hash)
            if entry is None:
                continue
            b = entry.boundary
            rows.append(
                {
                    "params_hash": params_hash,
                    "bar_type": b.bar_type,
                    "end_timestamp": b.end_timestamp,
                    "n_ticks": b.n_ticks_processed,
                    "n_bars": len(entry.bars),
                    "leftover_ticks": len(b.leftover_ticks),
                    "size_mb": p.stat().st_size / 1e6,
                    "created": pd.Timestamp(entry.created_at, unit="s"),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
