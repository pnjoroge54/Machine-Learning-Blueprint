"""
MetaTrader 5 Machine Learning Blueprint (Part 26): Process Permutation.
Reproduces every number, table and figure in the article.

Data: EURUSD H1 OHLC, 2018-01-02 to 2024-12-31 (43,103 bars), attached next to this
script as EURUSD_H1_time_2018-01-01-2024-12-31.parq. Pass --data to use another file.

Usage (from MQL5/Files, the folder containing afml/):
    python afml/process_permutation_study.py                 # all stages, ~25 min
    python afml/process_permutation_study.py --stage curve   # one stage
    python afml/process_permutation_figures.py
    python -m pytest afml/test_process_permutation.py -q

Development process under test
------------------------------
Random search over an MA crossover's (fast, slow) windows, fast in [2, 40),
slow in [20, 250), drawn once from GRID_SEED and held fixed for the real run and
every permuted replicate. Fixing the grid means replicates differ only in the
data they see, never in which parameters were tried. Pairs with fast >= slow are
skipped (scored -10, never selected).

Backtest: position p_t = sign(MA_fast - MA_slow) in {-1, 0, +1}, known at the
close of bar t and held over bar t+1. Return: close-to-close log return,
including weekend and holiday gaps. Cost: 1 bp times |p_t - p_{t-1}|, so a
long-to-short reversal costs 2 bp. No spread, swap, slippage or execution delay
is modeled. Sharpe: per-bar mean / std of net P&L, times sqrt(6048)
(252 days x 24 h). The data average 6,158 bars per year; the constant scales
real and permuted Sharpe identically, so it affects levels, not comparisons.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afml.cross_validation.process_permutation import (
    assert_cache_isolation, permute_development_process, spurious_performance_curve,
)
from afml.cross_validation.selection_overfit import effective_trial_count, expected_max_sharpe

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "EURUSD_H1_time_2018-01-01-2024-12-31.parq"
ANN = np.sqrt(252 * 24)
COST_BPS = 1.0
GRID_SEED = 1
PERM_SEED = 3          # shared by the point comparison and the curve
N_PERM = 500
BUDGETS = (5, 25, 100, 300)
SENS_SEEDS = range(1, 21)
SENS_N_PERM = 200


def load_bars(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)[["open", "high", "low", "close"]].astype(float)


def net_pnl(close: np.ndarray, fast: int, slow: int, cost_bps: float = COST_BPS) -> np.ndarray | None:
    """Per-bar net log P&L of the crossover, or None for an invalid pair."""
    if fast >= slow or slow >= close.size // 4:
        return None
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    ma_f = pd.Series(close).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(close).rolling(slow).mean().to_numpy()
    pos = np.nan_to_num(np.sign(ma_f - ma_s))
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos[:-1] * ret[1:] - turn[:-1] * cost_bps * 1e-4


def sharpe(pnl: np.ndarray | None) -> float:
    if pnl is None or pnl.std() == 0:
        return -10.0
    return float(pnl.mean() / pnl.std() * ANN)


def make_grid(n_trials: int, seed: int = GRID_SEED) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    return [(int(f), int(s)) for f, s in zip(rng.integers(2, 40, n_trials), rng.integers(20, 250, n_trials))]


def searching_process(n_trials: int, seed: int = GRID_SEED):
    grid = make_grid(n_trials, seed)          # drawn once, reused on real and permuted bars

    def develop(bars: pd.DataFrame) -> float:
        close = bars["close"].to_numpy(dtype=float)
        return max(sharpe(net_pnl(close, f, s)) for f, s in grid)

    return develop


def fixed_process(fast: int = 10, slow: int = 50):
    def develop(bars: pd.DataFrame) -> float:
        return sharpe(net_pnl(bars["close"].to_numpy(dtype=float), fast, slow))

    return develop


def deflated_sharpe(bars: pd.DataFrame, n_trials: int = 100, seed: int = GRID_SEED) -> dict:
    """DSR (Bailey and Lopez de Prado, 2014) for the same search, per-bar units."""
    close = bars["close"].to_numpy(dtype=float)
    pnls = [p for p in (net_pnl(close, f, s) for f, s in make_grid(n_trials, seed)) if p is not None]
    P = np.vstack(pnls)
    sr = P.mean(axis=1) / P.std(axis=1)                  # per-bar Sharpe of each valid trial
    best = int(np.argmax(sr))
    x = P[best]
    T, g3, g4 = x.size, skew(x), kurtosis(x, fisher=False)
    etc = effective_trial_count(P)
    out = {"n_valid": int(P.shape[0]), "n_eff_eigen": etc.n_eff_eigen, "n_eff_kish": etc.n_eff_kish,
           "mean_corr": etc.mean_correlation, "sr_best_ann": float(sr[best] * ANN),
           "skew": float(g3), "kurt": float(g4), "T": int(T)}
    for label, n in (("raw", P.shape[0]), ("eigen", etc.n_eff_eigen), ("kish", etc.n_eff_kish)):
        sr0 = expected_max_sharpe(n, float(np.var(sr, ddof=1)))
        z = (sr[best] - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr[best] + (g4 - 1) / 4 * sr[best] ** 2)
        out[f"sr0_ann_{label}"] = float(sr0 * ANN)
        out[f"dsr_{label}"] = float(norm.cdf(z))
    return out


def run_point(bars: pd.DataFrame) -> dict:
    """Budget-100 search and search-free control, N_PERM replicates, plus the
    published replicate counts rerun with their original seeds."""
    assert_cache_isolation(searching_process(100), bars, seed=1)
    out: dict = {"cache_check": "passed"}
    for name, fn, budget in (("search", searching_process(100), 100), ("fixed", fixed_process(), 1)):
        r = permute_development_process(fn, bars, n_perm=N_PERM, seed=PERM_SEED,
                                        trial_budget=budget, check_cache_isolation=False)
        d = r.to_dict()
        d["median_ci_coverage"] = r.median_ci()[2]
        d["q90"] = r.quantile(0.9)
        d["q90_ci"] = list(r.quantile_ci(0.9)[:2])
        d["null_values"] = r.null_values.tolist()
        out[name] = d
        print(name, r, flush=True)
    for n, sd in ((30, 3), (25, 2)):
        r = permute_development_process(searching_process(100), bars, n_perm=n, seed=sd,
                                        check_cache_isolation=False)
        out[f"small_n{n}_seed{sd}"] = {"median": r.spurious_center, "ci": list(r.median_ci()[:2]),
                                       "percentile": r.null_percentile, "p": r.p_value,
                                       "q90_ci": list(r.quantile_ci(0.9)[:2])}
    return out


def run_curve(bars: pd.DataFrame, budgets=BUDGETS) -> dict:
    """Null quantiles with CIs, and the observed result, at each budget."""
    curve = spurious_performance_curve(searching_process, bars, budgets=budgets, n_perm=N_PERM,
                                       seed=PERM_SEED, include_observed=True)
    print(curve.round(3).to_string(), flush=True)
    return curve.reset_index().to_dict(orient="list")


def run_sensitivity(bars: pd.DataFrame, seeds=SENS_SEEDS) -> dict:
    """Budget-100 search repeated over different grid draws."""
    rows = []
    for gs in seeds:
        r = permute_development_process(searching_process(100, seed=gs), bars, n_perm=SENS_N_PERM,
                                        seed=PERM_SEED, check_cache_isolation=False)
        rows.append({"grid_seed": gs, "observed": r.observed, "median": r.spurious_center,
                     "percentile": r.null_percentile, "p": r.p_value})
        print(rows[-1], flush=True)
    return {"rows": rows}


STAGES = {"point": run_point, "curve": run_curve, "sensitivity": run_sensitivity,
          "dsr": deflated_sharpe}


def main(out_dir: Path = HERE, stages=tuple(STAGES), data: Path = DATA_PATH) -> dict:
    """Run the requested stages and merge them into study_results.json.

    The full study takes about 25 minutes on one core. Stages can be run
    separately (python process_permutation_study.py --stage curve) and are
    merged into the same results file.
    """
    bars = load_bars(data)
    path = out_dir / "study_results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.update(n_bars=len(bars), start=str(bars.index.min()), end=str(bars.index.max()),
               bars_per_year=len(bars) / 7.0)
    for name in stages:
        t0 = time.time()
        res[name] = STAGES[name](bars)
        print(f"stage {name}: {time.time() - t0:.0f}s", flush=True)
        path.write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", choices=list(STAGES),
                    help="run only these stages (repeatable); default runs all")
    ap.add_argument("--data", type=Path, default=DATA_PATH,
                    help="EURUSD H1 OHLC parquet (default: next to this script)")
    args = ap.parse_args()
    main(stages=tuple(args.stage) if args.stage else tuple(STAGES), data=args.data)
