"""
Test suite for afml.backtester.ou_estimation (optimized).

Run:  python test_ou_estimation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from afml.backtester.ou_estimation import (
    build_within_trade_paths,
    build_within_trade_paths_flat,
    estimate_ou_from_signals,
    estimate_ou_params,
    estimate_ou_params_flat,
)


# --------------------------------------------------------------- reference ---
def ref_build(
    close, signals, atr=None, pt_mult=2.0, sl_mult=1.0, max_hold=48, pip=0.0001, seed_zero=True
):
    """Original pandas implementation, used only as ground truth."""
    sig_idx = signals[signals != 0].index
    paths, terminal = [], []
    for idx in sig_idx:
        side = int(signals.loc[idx])
        entry = float(close.loc[idx])
        future = close.loc[idx:].iloc[1 : max_hold + 1]
        if len(future) == 0:
            continue
        if atr is not None:
            av = float(atr.loc[idx])
            pt, sl = pt_mult * av, sl_mult * av
        else:
            pt = sl = np.inf
        path_pips, term = [], None
        for fp in future.to_numpy(dtype=float):
            mtm = side * (fp - entry)
            path_pips.append(mtm / pip)
            if mtm >= pt or mtm <= -sl:
                term = mtm / pip
                break
        if term is None:
            term = path_pips[-1]
        arr = np.asarray(path_pips, dtype=float)
        if seed_zero:
            arr = np.concatenate([[0.0], arr])
        paths.append(arr)
        terminal.append(term)
    return paths, np.asarray(terminal, dtype=float)


def ref_fit(paths, forecast=0.0):
    xs, ys = [], []
    for p in paths:
        if len(p) < 2:
            continue
        ys.append(p[1:])
        xs.append(p[:-1] - forecast)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    var_x = np.var(X)
    phi = float(np.clip(np.cov(Y, X, ddof=0)[0, 1] / var_x, -0.9999, 0.9999))
    sigma = float(np.std(Y - forecast - phi * X))
    return phi, sigma


def make_data(n_bars=8000, n_signals=900, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="h")
    close = pd.Series(1.15 * np.exp(np.cumsum(rng.normal(0, 1.1e-4, n_bars))), index=idx)
    atr = close.diff().abs().rolling(14).mean().bfill().clip(lower=1e-5)
    signals = pd.Series(0, index=idx, dtype=np.int64)
    pos = np.sort(rng.choice(n_bars, size=n_signals, replace=False))
    signals.iloc[pos] = rng.choice([-1, 1], size=n_signals)
    return close, signals, atr


# ------------------------------------------------------------------ tests ---
def test_paths_exact():
    close, signals, atr = make_data()
    kw = dict(atr=atr, pt_mult=2.0, sl_mult=1.0, max_hold=48, pip=0.0001)
    pr, tr = ref_build(close, signals, **kw)
    po, to = build_within_trade_paths(close, signals, **kw)
    assert len(pr) == len(po), (len(pr), len(po))
    assert all(np.array_equal(a, b) for a, b in zip(pr, po))
    assert np.array_equal(tr, to)


def test_no_atr_and_no_seed():
    close, signals, _ = make_data()
    for seed_zero in (True, False):
        kw = dict(atr=None, max_hold=12, pip=0.0001, seed_zero=seed_zero)
        pr, tr = ref_build(close, signals, **kw)
        po, to = build_within_trade_paths(close, signals, **kw)
        assert all(np.array_equal(a, b) for a, b in zip(pr, po))
        assert np.array_equal(tr, to)


def test_tail_signals_dropped_identically():
    """Signals within max_hold of the series end, and on the final bar."""
    close, signals, atr = make_data(n_bars=300, n_signals=0)
    signals.iloc[[290, 295, 298, 299]] = [1, -1, 1, -1]
    kw = dict(atr=atr, max_hold=48, pip=0.0001)
    pr, tr = ref_build(close, signals, **kw)
    po, to = build_within_trade_paths(close, signals, **kw)
    assert len(pr) == len(po) == 3  # bar 299 has no future bar
    assert all(np.array_equal(a, b) for a, b in zip(pr, po))
    assert np.array_equal(tr, to)


def test_csr_layout_matches_list():
    close, signals, atr = make_data()
    kw = dict(atr=atr, max_hold=48, pip=0.0001)
    paths, term = build_within_trade_paths(close, signals, **kw)
    flat, offsets, term_f = build_within_trade_paths_flat(close, signals, **kw)
    assert offsets.size - 1 == len(paths)
    assert offsets[-1] == flat.size
    assert np.array_equal(term, term_f)
    for i, p in enumerate(paths):
        assert np.array_equal(p, flat[offsets[i] : offsets[i + 1]])


def test_estimators_agree():
    close, signals, atr = make_data()
    kw = dict(atr=atr, max_hold=48, pip=0.0001)
    paths, _ = build_within_trade_paths(close, signals, **kw)
    flat, offsets, _ = build_within_trade_paths_flat(close, signals, **kw)
    for fc in (0.0, 2.5, -1.75):
        phi_r, sig_r = ref_fit(paths, fc)
        o_list = estimate_ou_params(paths, forecast=fc)
        o_flat = estimate_ou_params_flat(flat, offsets, forecast=fc)
        o_tup = estimate_ou_params((flat, offsets), forecast=fc)
        assert abs(o_list.phi - phi_r) < 1e-12, (o_list.phi, phi_r)
        assert abs(o_list.sigma - sig_r) < 1e-12
        assert abs(o_flat.phi - o_list.phi) < 1e-12
        assert abs(o_flat.sigma - o_list.sigma) < 1e-12
        assert o_tup == o_flat


def test_parallel_kernel_identical():
    close, signals, atr = make_data(n_bars=20000, n_signals=3000, seed=3)
    kw = dict(atr=atr, max_hold=48, pip=0.0001)
    f1, o1, t1 = build_within_trade_paths_flat(close, signals, parallel=False, **kw)
    f2, o2, t2 = build_within_trade_paths_flat(close, signals, parallel=True, **kw)
    assert np.array_equal(f1, f2)
    assert np.array_equal(o1, o2)
    assert np.array_equal(t1, t2)


def test_within_trade_boundary_not_crossed():
    """Two synthetic trades with disjoint levels: a crossed pair would show up."""
    flat = np.array([0.0, 1.0, 2.0, 0.0, 100.0, 200.0])
    offsets = np.array([0, 3, 6], dtype=np.int64)
    o = estimate_ou_params_flat(flat, offsets, forecast=0.0)
    paths = [flat[0:3], flat[3:6]]
    phi_r, sig_r = ref_fit(paths, 0.0)
    assert abs(o.phi - phi_r) < 1e-12
    # A boundary-crossing fit would pool the pair (2.0 -> 0.0); confirm absent.
    is_s = np.zeros(6, bool)
    is_e = np.zeros(6, bool)
    is_s[offsets[:-1]] = True
    is_e[offsets[1:] - 1] = True
    assert flat[~is_e].size == 4 and flat[~is_s].size == 4


def test_singleton_and_empty():
    # Singleton path contributes to neither X nor Y.
    flat = np.array([0.0, 1.0, 2.0, 5.0])
    offsets = np.array([0, 3, 4], dtype=np.int64)
    o_mix = estimate_ou_params_flat(flat, offsets, 0.0)
    o_one = estimate_ou_params_flat(flat[:3], np.array([0, 3], dtype=np.int64), 0.0)
    assert abs(o_mix.phi - o_one.phi) < 1e-12

    # No signals at all.
    close, signals, atr = make_data(n_bars=500, n_signals=0)
    flat, offsets, term = build_within_trade_paths_flat(close, signals, atr=atr)
    assert flat.size == 0 and term.size == 0 and offsets.tolist() == [0]


def test_known_phi_recovery():
    """Simulate a true O-U with known phi; check the estimator recovers it."""
    rng = np.random.default_rng(0)
    phi_true, sigma_true, n_tr, n_bar = 0.85, 1.0, 4000, 40
    flat = np.empty(n_tr * n_bar)
    for i in range(n_tr):
        p = 0.0
        for k in range(n_bar):
            flat[i * n_bar + k] = p
            p = phi_true * p + rng.normal(0, sigma_true)
    offsets = (np.arange(n_tr + 1) * n_bar).astype(np.int64)
    o = estimate_ou_params_flat(flat, offsets, forecast=0.0)
    assert abs(o.phi - phi_true) < 0.01, o.phi
    assert abs(o.sigma - sigma_true) < 0.02, o.sigma
    assert abs(o.hl - (-np.log(2) / np.log(phi_true))) < 0.1


def test_end_to_end_cases():
    close, signals, atr = make_data()
    for case in ("A", "B", "C"):
        ou, info = estimate_ou_from_signals(
            close, signals, atr, max_hold=48, forecast_case=case, spread_pips=1.2
        )
        assert -1 < ou.phi < 1
        assert ou.sigma >= 0
        assert info["n_trades"] > 0
    try:
        estimate_ou_from_signals(close, signals, atr, forecast_case="Z")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_index_mismatch_raises():
    close, signals, atr = make_data(n_bars=500, n_signals=50)
    try:
        build_within_trade_paths_flat(close, signals.iloc[:-1])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
