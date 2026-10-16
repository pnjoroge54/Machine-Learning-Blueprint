"""
export_pipeline_artifacts.py
────────────────────────────
Translates afml pipeline artifacts from Python-native formats to the flat
files the CPCVBacktest.mq5 EA expects in Common\\Files\\ml_artifacts\\.

Usage
-----
    python export_pipeline_artifacts.py \\
        --model-dir ./Models/my_strategy/EURUSD/.../a1b2c3d4 \\
        --mql5-dir  "C:\\Users\\...\\AppData\\Roaming\\MetaQuotes\\Terminal\\...\\MQL5\\Files" \\
        --n-folds   6 \\
        --k-test    2

The script writes:
  ml_artifacts/model.onnx
  ml_artifacts/calibrator.csv
  ml_artifacts/calibrator_meta.json
  ml_artifacts/feature_spec.json
  ml_artifacts/path_0.csv … path_{phi-1}.csv
  ml_artifacts/cpcv_meta.json

Notes
-----
* The ONNX file contains the full sklearn pipeline (StandardScaler +
  classifier).  MQL5 passes raw feature values directly to OnnxRun();
  it must NOT apply z-score normalization.  The feature_spec.json
  exports mean and std for diagnostic validation only.

* For N=6, k=2: C(6,2)=15 splits, phi = 15*2//6 = 5 paths.
  The Strategy Tester optimization must be configured with
  InpPathIndex from=0, to=4, step=1.

Part of: MetaTrader 5 Machine Learning Blueprint (Part 17)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..cross_validation.combinatorial import CombinatorialPurgedCV
from .file_manager import ModelFileManager

# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export afml artifacts to MQL5 format")
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to the versioned model directory (config-hash leaf)",
    )
    p.add_argument(
        "--mql5-dir", required=True, help="Path to the MQL5\\Files\\ root of the target terminal"
    )
    p.add_argument("--n-folds", type=int, default=6, help="CPCV N (default: 6)")
    p.add_argument("--k-test", type=int, default=2, help="CPCV k (default: 2)")
    p.add_argument(
        "--embargo", type=float, default=0.01, help="Purge embargo fraction (default: 0.01)"
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _n_paths(n_folds: int, k_test: int) -> int:
    """Number of distinct backtest paths: C(N,k)*k // N."""
    from math import comb

    return comb(n_folds, k_test) * k_test // n_folds


# ── Export stages ──────────────────────────────────────────────────────────────


def export_onnx(model_dir: Path, out_dir: Path) -> None:
    """Copy the ONNX file to the output directory."""
    onnx_candidates = list(model_dir.glob("model_*.onnx"))
    if not onnx_candidates:
        raise FileNotFoundError(f"No model_*.onnx found in {model_dir}")
    src = onnx_candidates[0]
    dst = out_dir / "model.onnx"
    shutil.copy2(src, dst)
    print(f"  [ONNX]  {src.name} → {dst}")


def export_calibrator(calibrator, out_dir: Path) -> None:
    """
    Decompose the calibrator into flat files.

    Isotonic regression: writes calibrator.csv (two columns: x, y) and
    calibrator_meta.json {"method":"isotonic","n_breakpoints":N}.

    Platt scaling: writes calibrator_meta.json {"method":"platt","A":A,"B":B}.
    The two-parameter sigmoid is fully described by A and B; no separate CSV.
    """
    if isinstance(calibrator, IsotonicRegression):
        try:
            x_pts = calibrator.X_thresholds_
            y_pts = calibrator.y_thresholds_
        except AttributeError:
            # Older sklearn versions use f_.x / f_.y
            x_pts = calibrator.f_.x
            y_pts = calibrator.f_.y

        cal_df = pd.DataFrame({"x": x_pts, "y": y_pts})
        cal_df.to_csv(out_dir / "calibrator.csv", index=False)

        meta = {"method": "isotonic", "n_breakpoints": len(x_pts)}
        print(f"  [CAL]   isotonic — {len(x_pts)} breakpoints")

    elif isinstance(calibrator, LogisticRegression):
        A = float(calibrator.coef_[0, 0])
        B = float(calibrator.intercept_[0])
        meta = {"method": "platt", "A": A, "B": B}
        print(f"  [CAL]   Platt — A={A:.6f}, B={B:.6f}")

    else:
        raise TypeError(f"Unsupported calibrator type: {type(calibrator)}")

    with open(out_dir / "calibrator_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def export_feature_spec(model, feature_names: list[str], out_dir: Path) -> None:
    """
    Write feature_spec.json.

    The JSON exports mean and std from the fitted preprocessor for
    diagnostic validation only.  MQL5 must NOT apply z-score
    normalization before calling OnnxRun(); the StandardScaler is
    baked into the ONNX graph.
    """
    preprocessor = model.steps[0][1]  # (name, transformer) → transformer
    has_mean = hasattr(preprocessor, "mean_")
    has_scale = hasattr(preprocessor, "scale_")

    specs = []
    for i, name in enumerate(feature_names):
        spec: dict = {
            "name": name,
            "index": i,
            "lookback": 14,  # placeholder — adjust per feature
            "type": "RSI",  # placeholder — adjust per feature
            "mean": float(preprocessor.mean_[i]) if has_mean else 0.0,
            "std": float(preprocessor.scale_[i]) if has_scale else 1.0,
        }
        specs.append(spec)

    with open(out_dir / "feature_spec.json", "w") as f:
        json.dump(specs, f, indent=2)
    print(f"  [SPEC]  {len(specs)} features written to feature_spec.json")
    print("          NOTE: 'type' and 'lookback' are placeholders.")
    print("          Edit feature_spec.json to match your feature engineering.")


def export_cpcv_masks(
    events: pd.DataFrame,
    n_folds: int,
    k_test: int,
    embargo: float,
    config: dict,
    out_dir: Path,
) -> int:
    """
    Precompute the CPCV path-to-bar mapping and write one CSV per path.

    Each path_N.csv contains the sorted timestamps of all bars belonging
    to that path's test window.  The EA loads one CSV per optimization
    pass and skips bars not in the file.

    Returns the number of paths written.
    """
    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=k_test,
        t1=events["t1"],
        pct_embargo=embargo,
    )

    n_test_paths = cv.n_test_paths
    path_ids = cv.get_path_ids()  # shape (n_splits, k_test)
    X_dummy = pd.DataFrame(np.zeros((len(events), 1)), index=events.index)

    path_bars: dict[int, list] = {p: [] for p in range(n_test_paths)}

    for split_idx, (_, test_lists) in enumerate(cv.split(X_dummy)):
        for fold_j, test_idx in enumerate(test_lists):
            path_id = path_ids[split_idx, fold_j]
            timestamps = events.index[test_idx]
            path_bars[path_id].extend(timestamps.tolist())

    for path_id, timestamps in path_bars.items():
        ts_series = pd.Series(sorted(set(timestamps)), name="timestamp")
        csv_path = out_dir / f"path_{path_id}.csv"
        ts_series.to_csv(csv_path, index=False)
        print(f"  [MASK]  path_{path_id}.csv — {len(ts_series)} bars")

    meta = {
        "n_folds": n_folds,
        "k_test": k_test,
        "n_paths": n_test_paths,
        "n_features": 0,  # filled by export_feature_spec
        "symbol": config.get("symbol", ""),
        "embargo": embargo,
    }
    with open(out_dir / "cpcv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_formula = _n_paths(n_folds, k_test)
    print(
        f"  [META]  N={n_folds}, k={k_test} → φ={n_formula} paths "
        f"(C({n_folds},{k_test})={n_test_paths // k_test * n_folds} splits)"
    )
    return n_test_paths


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.mql5_dir) / "ml_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)

    print(f"\nExporting artifacts from:\n  {model_dir}")
    print(f"To:\n  {out_dir}\n")

    # ── Load all artifacts ────────────────────────────────────────────
    mgr = ModelFileManager()
    arts = mgr.load_from_path(model_dir)

    model = arts["model"]  # sklearn Pipeline
    calibrator = arts.get("calibrator")  # fitted calibrator
    feature_names = arts.get("feature_names", [])
    events = arts.get("events")  # DataFrame with t1
    config = arts.get("config", {})

    if events is None:
        raise ValueError("events artifact not found — re-run pipeline with save=True")
    if calibrator is None:
        raise ValueError("calibrator artifact not found — re-run with calibrate=True")

    # ── Stage 1: ONNX model ───────────────────────────────────────────
    export_onnx(model_dir, out_dir)

    # ── Stage 2: Calibrator ───────────────────────────────────────────
    export_calibrator(calibrator, out_dir)

    # ── Stage 3: Feature specification ───────────────────────────────
    export_feature_spec(model, feature_names, out_dir)

    # ── Stage 4: CPCV path masks ──────────────────────────────────────
    n_paths = export_cpcv_masks(
        events=events,
        n_folds=args.n_folds,
        k_test=args.k_test,
        embargo=args.embargo,
        config=config,
        out_dir=out_dir,
    )

    print(f"\n{'─' * 60}")
    print("Export complete.")
    print(f"  φ = {n_paths} paths written to {out_dir}")
    print(f"  Strategy Tester: InpPathIndex from=0, to={n_paths - 1}, step=1")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
