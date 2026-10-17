"""
Battery calibration (plan Section 10.2): run the validation battery on synthetic
families whose answer is known, before trusting it on real ones.

* Noise families (``signal_strength = 0``): the battery should pass no more often
  than the registered level. Each test's pass rate is reported too: PBO should sit
  near 0.5, the DSR mostly below 0.95, permutation p-values roughly uniform.
* Planted families (an AR(1) momentum term the features can learn): the battery
  should pass.

Calibration runs in a throwaway studio home, so its synthetic families never enter
the research ledger. Only the summary is written back, to
``<studio home>/calibration/``, where the Validation page reads it.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from . import paths

# A trend-following primary, so the planted momentum is an edge the pipeline can use;
# noise and planted families run the identical pipeline and differ only in the data.
DEFAULT_LABELS = {
    "strategy": {"name": "ma_crossover", "params": {"fast_window": 5, "slow_window": 20}},
    "target": {"name": "daily_vol", "params": {"lookback": 50}},
    "max_holding_period": {"hours": 12},
}
DEFAULT_FEATURES = {"steps": [{"name": "log_returns", "params": {"lags": [1, 2, 4]}},
                              {"name": "volatility", "params": {"window": 10}}]}
DEFAULT_MODEL = {
    "search": "optuna", "n_trials": 12, "cv_splits": 4, "pruner": "median",
    "estimator": {"name": "random_forest", "params": {"n_estimators": 30, "n_jobs": 1}},
    "search_space": {"max_depth": {"int": [2, 6]}, "min_weight_fraction_leaf": {"float": [0.025, 0.2]}},
}


def calibration_dir() -> Path:
    d = paths.studio_home() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest() -> Optional[Dict[str, Any]]:
    """The most recent calibration summary, if any."""
    f = calibration_dir() / "latest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def summarize(records: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
    """
    Pass rates per test and for the whole battery, by family kind.

    :param records: one per family: ``kind`` ("noise" or "signal"), ``passed``
        (whole battery) and ``tests`` mapping test number to passed.
    """
    out: Dict[str, Any] = {"alpha": alpha, "kinds": {}}
    for kind in ("noise", "signal"):
        rows = [r for r in records if r["kind"] == kind and r.get("passed") is not None]
        n = len(rows)
        k = sum(bool(r["passed"]) for r in rows)
        per_test = {}
        for t in sorted({int(t) for r in rows for t in r["tests"]}):
            vals = [bool(r["tests"][str(t)] if str(t) in r["tests"] else r["tests"][t]) for r in rows
                    if str(t) in r["tests"] or t in r["tests"]]
            per_test[str(t)] = {"pass_rate": float(np.mean(vals)) if vals else None,
                                "ci": wilson(sum(vals), len(vals)), "n": len(vals)}
        out["kinds"][kind] = {"n_families": n, "battery_pass_rate": k / n if n else None,
                              "ci": wilson(k, n), "tests": per_test,
                              "n_errors": sum(1 for r in records if r["kind"] == kind and r.get("passed") is None)}
    noise = out["kinds"]["noise"]
    signal = out["kinds"]["signal"]
    out["false_pass_rate"] = noise["battery_pass_rate"]
    out["power"] = signal["battery_pass_rate"]
    out["false_pass_ok"] = bool(noise["n_families"] and noise["battery_pass_rate"] <= alpha)
    out["power_ok"] = bool(signal["n_families"] and signal["battery_pass_rate"] >= 0.5)
    return out


@contextmanager
def _isolated_home() -> Iterator[Path]:
    old = os.environ.get("AFML_STUDIO_HOME")
    home = Path(tempfile.mkdtemp(prefix="afml_calibration_"))
    os.environ["AFML_STUDIO_HOME"] = str(home)
    try:
        yield home
    finally:
        if old is None:
            os.environ.pop("AFML_STUDIO_HOME", None)
        else:
            os.environ["AFML_STUDIO_HOME"] = old


def run_calibration(n_noise: int = 20, n_signal: int = 6, signal_strength: float = 0.3,
                    months: int = 6, model: Optional[Dict[str, Any]] = None,
                    prereg: Optional[Dict[str, Any]] = None, seed: int = 0,
                    log=print) -> Dict[str, Any]:
    """
    Run the battery (tests 1 to 6) on ``n_noise`` noise families and ``n_signal``
    planted ones, and save the summary to the studio home. Each family is its own
    symbol, so each has its own research_end and its own trial count.
    """
    from . import jobs, validation, vault
    from .ledger import get_ledger
    from .worker import run_worker

    out_dir = calibration_dir()  # the real home, resolved before switching
    prereg = prereg or {}
    model = model or DEFAULT_MODEL
    start = pd.Timestamp("2024-01-01")
    end = (start + pd.DateOffset(months=months) - pd.Timedelta(days=1)).date().isoformat()
    plan = [("noise", 0.0)] * n_noise + [("signal", signal_strength)] * n_signal
    records: List[Dict[str, Any]] = []
    alpha = None
    with _isolated_home() as home:
        led = get_ledger()
        model_runs = []
        for i, (kind, strength) in enumerate(plan):
            symbol = f"CAL{i:03d}"
            vault.set_research_end(symbol, "2030-12-31", ledger=led)
            cfg = {"stage": "model", "seed": seed + i,
                   "data": {"symbol": symbol, "source": "synthetic", "bar_type": "time", "bar_size": "M15",
                            "source_params": {"seed": seed + 1000 + i, "freq": "15min",
                                              "signal_strength": strength},
                            "start_date": start.date().isoformat(), "end_date": end},
                   "labels": DEFAULT_LABELS, "features": DEFAULT_FEATURES, "model": model}
            model_runs.append((kind, strength, jobs.submit(cfg, ledger=led)))
        log(f"calibration: {len(plan)} families in {home}")
        run_worker(poll_s=0.5, once=True, ledger=led)
        val_runs = []
        for kind, strength, run_id in model_runs:
            run = led.get_run(run_id)
            if run["status"] != "done":
                records.append({"kind": kind, "signal_strength": strength, "passed": None, "tests": {},
                                "error": run["error"]})
                continue
            reg = validation.Preregistration(source_run=run_id, **prereg)
            alpha = reg.thresholds.permutation_p_max
            validation.preregister(reg, led)
            val_runs.append((kind, strength, validation.submit_validation(run_id, led)))
        run_worker(poll_s=0.5, once=True, ledger=led)
        for kind, strength, run_id in val_runs:
            run = led.get_run(run_id)
            if run["status"] != "done":
                records.append({"kind": kind, "signal_strength": strength, "passed": None, "tests": {},
                                "error": run["error"]})
                continue
            verdict = pd.read_parquet(Path(run["run_dir"]) / "tables" / "verdict.parquet")
            v = run["summary"]["validation"]
            records.append({"kind": kind, "signal_strength": strength, "passed": bool(v["passed"]),
                            "tests": {str(t): bool(p) for t, p in verdict["passed"].items()},
                            "values": {str(t): (None if pd.isna(x) else float(x))
                                       for t, x in verdict["value"].items()}})
            log(f"  {kind} {run_id}: {'pass' if v['passed'] else 'fail'}")

    result = summarize(records, alpha if alpha is not None else 0.05)
    result.update(created_at=datetime.now().isoformat(timespec="seconds"), n_noise=n_noise,
                  n_signal=n_signal, signal_strength=signal_strength, months=months, model=model,
                  prereg=prereg, families=records)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in (f"calibration-{stamp}.json", "latest.json"):
        (out_dir / name).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
