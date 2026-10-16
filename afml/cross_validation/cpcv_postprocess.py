"""
cpcv_postprocess.py
───────────────────
Reads per-path equity CSVs written by CPCVBacktest.mq5's OnDeinit(),
constructs the returns matrix, computes the path Sharpe distribution,
and runs the PBO audit.

Usage
-----
    python cpcv_postprocess.py \\
        --results-dir "C:\\...\\MQL5\\Files\\ml_artifacts\\results" \\
        --n-paths 5

Output (stdout and summary.json):
  Median path Sharpe, path Sharpe std, PBO
  → three-number deployment decision

Notes
-----
For N=6, k=2: phi = 5 paths (C(6,2)*2//6).  Pass --n-paths 5.

The PBO computation requires a t1 series aligned with the returns
matrix index.  Since CSCV is symmetric (no temporal ordering), we
pass a neutral t1 = pd.Series(index, index=index).

Part of: MetaTrader 5 Machine Learning Blueprint (Part 17)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pbo import compute_pbo


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess CPCV path results")
    p.add_argument(
        "--results-dir", required=True, help="Directory containing path_N.csv files from the tester"
    )
    p.add_argument("--n-paths", type=int, required=True, help="Number of paths (phi = C(N,k)*k//N)")
    p.add_argument(
        "--pbo-folds",
        type=int,
        default=8,
        help="Number of CSCV subsets for PBO (default: 8, must be even)",
    )
    p.add_argument(
        "--annualize",
        type=int,
        default=252,
        help="Trading days per year for Sharpe annualisation (default: 252)",
    )
    p.add_argument(
        "--out", default="summary.json", help="Output summary JSON filename (default: summary.json)"
    )
    return p.parse_args()


def _sharpe(returns: pd.Series, annualize: int) -> float:
    """Annualised Sharpe ratio from a bar-level return series."""
    std = returns.std()
    if std < 1e-9:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualize))


def main() -> None:
    args = _parse_args()
    results = Path(args.results_dir)

    # ── Collect per-path equity series ────────────────────────────────
    path_series: list[pd.Series] = []
    missing = []

    for i in range(args.n_paths):
        csv_path = results / f"path_{i}.csv"
        if not csv_path.exists():
            print(f"  WARNING: path_{i}.csv not found — skipping")
            missing.append(i)
            continue
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if "equity" not in df.columns or df.empty:
            print(f"  WARNING: path_{i}.csv is empty or malformed — skipping")
            missing.append(i)
            continue
        equity = df.set_index("timestamp")["equity"]
        returns = equity.pct_change().fillna(0.0)
        path_series.append(returns.rename(i))

    if not path_series:
        print("ERROR: no path CSV files could be loaded.")
        return

    # ── Returns matrix (time × paths) ─────────────────────────────────
    returns_matrix = pd.concat(path_series, axis=1).fillna(0.0)

    # ── Path Sharpe distribution ───────────────────────────────────────
    path_sharpes = returns_matrix.apply(lambda s: _sharpe(s, args.annualize))

    median_sr = float(path_sharpes.median())
    std_sr = float(path_sharpes.std(ddof=1))
    min_sr = float(path_sharpes.min())
    max_sr = float(path_sharpes.max())

    # ── PBO audit ─────────────────────────────────────────────────────
    # Neutral t1: timestamps serve as both index and event end-times.
    t1_neutral = pd.Series(returns_matrix.index, index=returns_matrix.index)
    pbo_result = compute_pbo(
        returns_matrix,
        t1=t1_neutral,
        n_folds=args.pbo_folds,
    )
    pbo_value = float(pbo_result["pbo"])

    # ── Print deployment decision ──────────────────────────────────────
    print()
    print("═" * 56)
    print("  CPCV DEPLOYMENT DECISION SUMMARY")
    print(f"  Paths evaluated : {len(path_series)}/{args.n_paths}")
    print(f"  Missing paths   : {missing if missing else 'none'}")
    print("─" * 56)
    print(f"  Median path Sharpe : {median_sr:+.3f}")
    print(f"  Sharpe std         : {std_sr:.3f}")
    print(f"  Sharpe range       : [{min_sr:+.3f}, {max_sr:+.3f}]")
    print("─" * 56)
    print(f"  PBO                : {pbo_value:.4f}")
    print(f"  PBO splits         : {pbo_result['n_splits']}")
    print(f"  Below-median OOS   : {pbo_result['below_median']}")
    print("─" * 56)

    deploy = (median_sr > 0.0) and (std_sr < 0.5) and (pbo_value < 0.5)
    print(
        f"  Decision           : {
            'PASS — proceed to demo forward test' if deploy else 'FAIL — do not deploy'
        }"
    )
    print("═" * 56)
    print()

    # ── Write summary.json ────────────────────────────────────────────
    summary = {
        "n_paths_evaluated": len(path_series),
        "n_paths_expected": args.n_paths,
        "missing_paths": missing,
        "median_sharpe": round(median_sr, 4),
        "sharpe_std": round(std_sr, 4),
        "sharpe_min": round(min_sr, 4),
        "sharpe_max": round(max_sr, 4),
        "pbo": round(pbo_value, 4),
        "pbo_n_splits": pbo_result["n_splits"],
        "deploy_decision": deploy,
        "path_sharpes": {str(k): round(v, 4) for k, v in path_sharpes.items()},
    }
    out_path = results / args.out
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()
