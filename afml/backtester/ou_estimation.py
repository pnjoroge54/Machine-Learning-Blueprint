"""
afml.backtester.ou_estimation
=============================
Correct Ornstein-Uhlenbeck parameter estimation for the Optimal Trading
Rule framework (AFML Chapter 13), implementing equation (13.6).

The O-U process models the mark-to-market P&L of an OPEN POSITION as it
evolves bar-by-bar over the holding period:

    P_{i,t} = E_0[P_{i,T_i}] + phi (P_{i,t-1} - E_0[P_{i,T_i}]) + xi_t

Equation (13.6) forms the regression by *sequencing opportunities*: for each
trade i it stacks the within-trade path {P_{i,0}, ..., P_{i,T-1}} (lagged)
against {P_{i,1}, ..., P_{i,T}} (current), and regresses across the pooled
within-trade pairs.  It NEVER regresses one trade's outcome on another
trade's outcome.

Performance notes
-----------------
The path builder is *pandas-indexing bound*, not compute bound: profiling the
original implementation attributes roughly 88% of wall time to
``Series.loc`` / ``Series.iloc`` machinery and only ~12% to the arithmetic.
Distributing that loop across processes parallelizes the overhead instead of
removing it.  This module therefore hoists the pandas layer out of the loop
(one ``to_numpy`` conversion up front, positional integer indexing inside)
and JIT-compiles the remaining scan with Numba.  A pure-NumPy fallback is
used when Numba is unavailable.

Paths are materialized in a CSR-style ``(flat, offsets)`` buffer.  The pooled
regression never needs the ragged Python list; building it is pure allocation
overhead.  ``build_within_trade_paths`` still returns the list for backward
compatibility, but new call sites should prefer the ``_flat`` variants.

References
----------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Wiley. Chapter 13, section 13.5.1, eq. (13.5)-(13.7).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

try:  # Numba is optional; the NumPy fallback is exact, just slower.
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap(args[0]) if args and callable(args[0]) else _wrap

    prange = range  # type: ignore[assignment]


__all__ = [
    "OUParams",
    "build_within_trade_paths",
    "build_within_trade_paths_flat",
    "estimate_ou_from_signals",
    "estimate_ou_params",
    "estimate_ou_params_flat",
]


class OUParams(NamedTuple):
    """Estimated O-U parameters."""

    phi: float  # bar-to-bar mean-reversion coefficient phi in (-1, 1)
    sigma: float  # per-bar innovation standard deviation
    hl: float  # implied half-life in bars
    forecast: float  # long-run equilibrium E_0[P_T]


# ---------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------
def _scan_body(close, sig_pos, sides, pt_arr, sl_arr, max_hold, pip, seed_zero):
    """Fixed-stride scan; writes are independent per trade, so prange-safe."""
    n = sig_pos.shape[0]
    n_bars = close.shape[0]
    stride = max_hold + 1

    slots = np.empty(n * stride, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    terminal = np.empty(n, dtype=np.float64)
    keep = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        p = sig_pos[i]
        side = sides[i]
        entry = close[p]

        end = p + max_hold
        end = min(end, n_bars - 1)
        if end <= p:  # no future bars -> trade dropped, matches reference
            continue

        base = i * stride
        k = 0
        if seed_zero:
            slots[base] = 0.0
            k = 1

        pt = pt_arr[i]
        sl = sl_arr[i]
        term = 0.0
        touched = False

        for j in range(p + 1, end + 1):
            mtm = side * (close[j] - entry)
            mtm_pips = mtm / pip
            slots[base + k] = mtm_pips
            k += 1
            if mtm >= pt or mtm <= -sl:
                term = mtm_pips
                touched = True
                break

        if not touched:
            term = slots[base + k - 1]

        lengths[i] = k
        terminal[i] = term
        keep[i] = True

    return slots, lengths, terminal, keep


_scan_serial = njit(cache=True)(_scan_body)
_scan_parallel = njit(cache=True, parallel=True)(_scan_body) if _HAS_NUMBA else _scan_body


@njit(cache=True)
def _compact(slots, lengths, terminal, keep, stride):
    """Compact the fixed-stride slot buffer into a contiguous CSR buffer."""
    n = lengths.shape[0]
    n_keep = 0
    total = 0
    for i in range(n):
        if keep[i]:
            n_keep += 1
            total += lengths[i]

    flat = np.empty(total, dtype=np.float64)
    offsets = np.empty(n_keep + 1, dtype=np.int64)
    term_out = np.empty(n_keep, dtype=np.float64)

    offsets[0] = 0
    c = 0
    pos = 0
    for i in range(n):
        if not keep[i]:
            continue
        base = i * stride
        length = lengths[i]
        for k in range(length):
            flat[pos + k] = slots[base + k]
        pos += length
        c += 1
        offsets[c] = pos
        term_out[c - 1] = terminal[i]

    return flat, offsets, term_out


# ---------------------------------------------------------------------------
# Within-trade path construction
# ---------------------------------------------------------------------------
def build_within_trade_paths_flat(
    close: pd.Series,
    signals: pd.Series,
    atr: pd.Series | None = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 48,
    pip: float = 0.0001,
    seed_zero: bool = True,
    parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build within-trade mark-to-market P&L paths in CSR layout.

    Trade ``i`` occupies ``flat[offsets[i]:offsets[i + 1]]``.  This is the
    preferred entry point: it avoids allocating one small ndarray per trade.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed like ``signals``.
    signals : pd.Series
        +1 / -1 / 0 entry signals, aligned to ``close``.
    atr : pd.Series or None
        ATR series for barrier placement.  If None, paths run the full
        ``max_hold`` with no early barrier truncation.
    pt_mult, sl_mult : float
        Barrier multiples of ATR (only used when ``atr`` is provided).
    max_hold : int
        Vertical barrier in bars.  MUST equal the simulation ``max_hp`` and
        the labelling ``max_hold``.
    pip : float
        Pip size in price units.
    seed_zero : bool
        If True, prepend the seed value 0.0 to each path (P_{i,0} = 0).
    parallel : bool
        Use the ``prange`` kernel.  Only worth enabling above roughly 10^5
        signals; below that, thread launch dominates the scan.

    Returns
    -------
    (flat, offsets, terminal) : tuple of np.ndarray
        flat     : concatenated paths (pips), float64.
        offsets  : length n_trades + 1, int64.
        terminal : terminal P&L per trade (pips, NO spread/cost applied).
    """
    if not close.index.equals(signals.index):
        raise ValueError("close and signals must share an identical index.")
    if atr is not None and not atr.index.equals(close.index):
        raise ValueError("atr must share close's index.")

    close_v = np.ascontiguousarray(close.to_numpy(dtype=np.float64))
    sig_v = signals.to_numpy()
    mask = sig_v != 0
    sig_pos = np.flatnonzero(mask).astype(np.int64)
    sides = sig_v[mask].astype(np.float64)

    if sig_pos.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    if atr is not None:
        av = atr.to_numpy(dtype=np.float64)[mask]
        pt_arr = np.ascontiguousarray(pt_mult * av)
        sl_arr = np.ascontiguousarray(sl_mult * av)
    else:
        pt_arr = np.full(sig_pos.size, np.inf)
        sl_arr = np.full(sig_pos.size, np.inf)

    kernel = _scan_parallel if (parallel and _HAS_NUMBA) else _scan_serial
    slots, lengths, terminal, keep = kernel(
        close_v, sig_pos, sides, pt_arr, sl_arr, int(max_hold), float(pip), bool(seed_zero)
    )
    return _compact(slots, lengths, terminal, keep, int(max_hold) + 1)


def build_within_trade_paths(
    close: pd.Series,
    signals: pd.Series,
    atr: pd.Series | None = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 48,
    pip: float = 0.0001,
    seed_zero: bool = True,
    parallel: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Backward-compatible wrapper returning the ragged list of paths.

    Returns
    -------
    (within_paths, terminal_pnl) : (list of np.ndarray, np.ndarray)
    """
    flat, offsets, terminal = build_within_trade_paths_flat(
        close, signals, atr, pt_mult, sl_mult, max_hold, pip, seed_zero, parallel
    )
    paths = [flat[offsets[i] : offsets[i + 1]] for i in range(offsets.size - 1)]
    return paths, terminal


# ---------------------------------------------------------------------------
# Equation (13.6) estimator
# ---------------------------------------------------------------------------
def _fit(X: np.ndarray, Y: np.ndarray, forecast: float) -> OUParams:
    """Shared OLS core.  Both moments use ddof=0 (see module notes)."""
    if X.size == 0:
        warnings.warn(
            "No within-trade pairs available; returning degenerate O-U.",
            UserWarning,
            stacklevel=3,
        )
        return OUParams(phi=0.0, sigma=0.0, hl=float("inf"), forecast=forecast)

    var_x = np.var(X)
    if var_x < 1e-12:
        warnings.warn(
            "Degenerate regressor variance in O-U fit.",
            UserWarning,
            stacklevel=3,
        )
        return OUParams(phi=0.0, sigma=float(np.std(Y)), hl=float("inf"), forecast=forecast)

    phi = float(np.cov(Y, X, ddof=0)[0, 1] / var_x)
    phi = float(np.clip(phi, -0.9999, 0.9999))

    resid = Y - forecast - phi * X
    sigma = float(np.std(resid))
    hl = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")
    return OUParams(phi=phi, sigma=sigma, hl=hl, forecast=forecast)


def estimate_ou_params_flat(
    flat: np.ndarray,
    offsets: np.ndarray,
    forecast: float = 0.0,
) -> OUParams:
    """
    Estimate {phi, sigma} via eq. (13.6) directly on the CSR buffer.

    Pairs are formed INSIDE each trade only: the last bar of trade i is never
    paired with the first bar of trade i+1.  The boundary is enforced by
    masking out each trade's first element from the response and each trade's
    last element from the regressor, which is exact for singleton paths too
    (they contribute to neither).
    """
    n_trades = int(offsets.size) - 1
    if n_trades <= 0 or flat.size == 0:
        return _fit(np.empty(0), np.empty(0), forecast)

    is_start = np.zeros(flat.size, dtype=bool)
    is_end = np.zeros(flat.size, dtype=bool)
    is_start[offsets[:-1]] = True
    is_end[offsets[1:] - 1] = True

    Y = flat[~is_start]
    X = flat[~is_end] - forecast
    return _fit(X, Y, forecast)


def estimate_ou_params(
    within_paths,
    forecast: float = 0.0,
) -> OUParams:
    """
    Estimate {phi, sigma} via eq. (13.6) from a ragged list of paths.

    Accepts either a list of 1-D arrays (legacy) or a ``(flat, offsets)``
    tuple, in which case it delegates to ``estimate_ou_params_flat``.
    """
    if isinstance(within_paths, tuple) and len(within_paths) == 2:
        return estimate_ou_params_flat(*within_paths, forecast=forecast)

    xs, ys = [], []
    for path in within_paths:
        if len(path) < 2:
            continue
        ys.append(path[1:])
        xs.append(path[:-1] - forecast)

    if not xs:
        return _fit(np.empty(0), np.empty(0), forecast)

    return _fit(np.concatenate(xs), np.concatenate(ys), forecast)


# ---------------------------------------------------------------------------
# Convenience: full estimation from raw signals
# ---------------------------------------------------------------------------
def estimate_ou_from_signals(
    close: pd.Series,
    signals: pd.Series,
    atr: pd.Series | None = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 48,
    pip: float = 0.0001,
    forecast_case: str = "A",
    spread_pips: float = 1.2,
    parallel: bool = False,
) -> tuple[OUParams, dict]:
    """
    End-to-end O-U estimation from raw signals.

    Builds within-trade paths, computes the forecast for the requested case
    from the POST-cost terminal P&L, and fits {phi, sigma} via eq. (13.6) on
    the RAW mark-to-market paths (path dynamics are cost-independent; cost is
    a terminal offset, not a per-bar effect).

    Parameters
    ----------
    forecast_case : {"A", "B", "C"}
        A : forecast = 0                  (mean-reversion)
        B : forecast = E[post-cost P&L]   (directional realistic)
        C : forecast = E[post-cost win]   (directional optimistic)
    spread_pips : float
        Round-trip spread deducted from terminal P&L for Case B / C.

    Returns
    -------
    (ou, info) : (OUParams, dict)
    """
    flat, offsets, terminal_raw = build_within_trade_paths_flat(
        close,
        signals,
        atr,
        pt_mult,
        sl_mult,
        max_hold,
        pip,
        seed_zero=True,
        parallel=parallel,
    )

    terminal_cost = terminal_raw - spread_pips
    case = forecast_case.upper()

    if case == "A":
        forecast = 0.0
    elif case == "B":
        forecast = float(terminal_cost.mean()) if terminal_cost.size else 0.0
    elif case == "C":
        wins = terminal_cost[terminal_cost > 0]
        forecast = float(wins.mean()) if wins.size else 0.0
    else:
        raise ValueError(f"forecast_case must be A/B/C; got {forecast_case!r}")

    ou = estimate_ou_params_flat(flat, offsets, forecast=forecast)

    has_win = bool((terminal_cost > 0).any())
    info = {
        "n_trades": int(offsets.size) - 1,
        "mean_pnl": float(terminal_cost.mean()) if terminal_cost.size else float("nan"),
        "mean_win": float(terminal_cost[terminal_cost > 0].mean()) if has_win else float("nan"),
        "terminal_dispersion": float(terminal_cost.std()) if terminal_cost.size else float("nan"),
    }
    return ou, info
