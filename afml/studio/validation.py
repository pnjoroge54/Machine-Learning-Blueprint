"""
Phase 3: pre-registration, the validation entry gate, and validation submission.

Validation can only reject. A family reaches it once a model run's configuration is
frozen and its thresholds are written down; results are judged against that record
and nothing on the Validation page feeds back into tuning (AFML Section 11.4).

* ``preregister`` records the frozen config hash, the tests, every threshold, the
  significance level and the hold-out decision rule. Each call is a new version;
  earlier versions stay in the ledger beside it.
* ``entry_gate`` lists what still blocks validation for a model run.
* ``submit_validation`` and ``submit_process_permutation`` build the job from the
  frozen run's config, so the battery sees exactly the data, labels, features and
  model that were frozen. Every submission is a new run and a new trial; earlier
  reports are never overwritten.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import jobs, vault
from .ledger import Ledger, get_ledger, utcnow

BATTERY = {
    1: "Signal permutation",
    2: "CPCV path distribution",
    3: "Deflated Sharpe",
    4: "PBO",
    5: "Walk-forward degradation",
    6: "Sequence risk",
    7: "Process permutation",
}


class Thresholds(BaseModel):
    """Acceptance thresholds (plan Section 10.3). Only the DSR level comes from the
    literature; the rest are judgment calls set once per family, before looking."""

    model_config = ConfigDict(extra="forbid")

    dsr_min: float = Field(0.95, gt=0, lt=1)
    permutation_p_max: float = Field(0.05, gt=0, lt=1)  # tests 1 and 7
    pbo_max: float = Field(0.20, gt=0, lt=1)
    cpcv_median_sharpe_min: float = 0.0
    cpcv_positive_share_min: float = Field(0.80, ge=0, le=1)
    breach_probability_max: float = Field(0.05, gt=0, lt=1)


class HoldoutRule(BaseModel):
    """The hold-out decision rule, registered now and executed in Phase 4."""

    model_config = ConfigDict(extra="forbid")

    psr_min: float = Field(0.95, gt=0, lt=1)  # hold-out PSR against zero
    path_percentile_low: float = Field(5.0, ge=0, le=50)
    path_percentile_high: float = Field(95.0, ge=50, le=100)
    holdout_end: Optional[date] = None  # last hold-out day you plan to evaluate


class Account(BaseModel):
    """The account the CPCV and sequence-risk tests trade (afml BacktestConfig)."""

    model_config = ConfigDict(extra="forbid")

    initial_balance: float = Field(100_000.0, gt=0)
    lot_size: float = Field(1.0, gt=0)  # lots held at full signal strength
    pip_value: float = Field(10.0, gt=0)  # account currency per pip per lot
    pip_size: float = Field(0.0001, gt=0)
    commission_per_lot: float = Field(5.0, ge=0)
    slippage_pips: float = Field(0.5, ge=0)
    stop_loss_pct: Optional[float] = Field(None, gt=0, lt=1)  # needed when the rules cap risk per trade


class Preregistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run: str  # the frozen Optuna model run
    thresholds: Thresholds = Field(default_factory=Thresholds)
    holdout: HoldoutRule = Field(default_factory=HoldoutRule)
    alpha: float = Field(0.05, gt=0, lt=1)
    rule_set: str = "fundednext_stellar_2step"  # prop-firm limits for tests 2 and 6
    account: Account = Field(default_factory=Account)
    n_perm_signal: int = Field(1000, ge=100, le=100_000)
    n_perm_process: int = Field(30, ge=5, le=1000)
    n_boot: int = Field(5000, ge=500, le=100_000)
    tests: List[int] = Field(default_factory=lambda: list(BATTERY))
    notes: str = ""


class GateError(RuntimeError):
    pass


# =========================================================================== gate


def _model_run(run_id: str, ledger: Ledger) -> Dict[str, Any]:
    run = ledger.get_run(run_id)
    if run is None:
        raise GateError(f"unknown run {run_id}")
    return run


def entry_gate(source_run: str, ledger: Optional[Ledger] = None) -> List[str]:
    """
    Reasons validation cannot start for this model run (empty when it can).

    The plan's four conditions: the config is frozen and its Optuna study closed,
    the ledger holds per-fold scores for the family's trials, the thresholds are
    pre-registered, and every bar ends at or before ``research_end``.
    """
    ledger = ledger or get_ledger()
    problems: List[str] = []
    run = _model_run(source_run, ledger)
    if run["stage"] != "model":
        return [f"{source_run} is a {run['stage']} run; validation needs a model run"]
    model = (run.get("summary") or {}).get("model") or {}
    if run["status"] != "done":
        problems.append(f"the model run is {run['status']}; its Optuna study is not closed")
    elif model.get("search") != "optuna":
        problems.append("validation needs an Optuna model run (per-trial OOF probabilities)")
    fam = ledger.get_family(run["family_id"])
    if not fam or fam["frozen_config_hash"] != run["config_hash"]:
        problems.append("this run's config is not the family's frozen, pre-registered config")
    elif latest_preregistration(run["family_id"], ledger) is None:
        problems.append("no thresholds are pre-registered")
    stats = ledger.family_stats(run["family_id"])
    if (stats["n_metric"] or 0) < 2 or stats["n_eigen"] is None:
        problems.append("the ledger needs per-fold scores for at least two trials")
    cfg = jobs.clone_spec(source_run, ledger)
    d = cfg["data"]
    boundary = vault.get_research_end(d["symbol"], ledger)
    if boundary is None or vault._as_date(d["end_date"]) > boundary:
        problems.append(f"{d['symbol']} data must end at or before research_end ({boundary})")
    return problems


# =========================================================================== pre-registration


def preregister(prereg: Preregistration, ledger: Optional[Ledger] = None) -> int:
    """
    Freeze the source run's config and record the thresholds. Returns the
    pre-registration id. Registering again writes a new version; earlier ones stay.
    """
    from afml.prop_firms import PRESETS

    ledger = ledger or get_ledger()
    run = _model_run(prereg.source_run, ledger)
    if run["stage"] != "model" or run["status"] != "done":
        raise GateError("pre-register a completed model run")
    if prereg.rule_set not in PRESETS:
        raise GateError(f"unknown rule set {prereg.rule_set!r}; one of {sorted(PRESETS)}")
    payload = prereg.model_dump(mode="json")
    vault.preregister(run["family_id"], run["config_hash"], payload, ledger=ledger)
    row = ledger.one("SELECT MAX(id) AS id FROM preregistrations WHERE family_id = ?", (run["family_id"],))
    return int(row["id"])


def preregistrations(family_id: str, ledger: Optional[Ledger] = None) -> List[Dict[str, Any]]:
    """Every pre-registration of a family, oldest first, with its version number."""
    ledger = ledger or get_ledger()
    rows = ledger.query("SELECT * FROM preregistrations WHERE family_id = ? ORDER BY id", (family_id,))
    out = []
    for version, r in enumerate(rows, start=1):
        out.append({"id": r["id"], "version": version, "config_hash": r["config_hash"],
                    "created_at": r["created_at"], "payload": json.loads(r["payload_json"])})
    return out


def latest_preregistration(family_id: str, ledger: Optional[Ledger] = None) -> Optional[Dict[str, Any]]:
    regs = preregistrations(family_id, ledger)
    return regs[-1] if regs else None


def get_preregistration(prereg_id: int, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    ledger = ledger or get_ledger()
    row = ledger.one("SELECT * FROM preregistrations WHERE id = ?", (prereg_id,))
    if row is None:
        raise KeyError(f"unknown pre-registration {prereg_id}")
    reg = next(r for r in preregistrations(row["family_id"], ledger) if r["id"] == prereg_id)
    return {**reg, "family_id": row["family_id"]}


# =========================================================================== submission


def _frozen_job(source_run: str, stage: str, section: Dict[str, Any], ledger: Ledger) -> Dict[str, Any]:
    cfg = jobs.clone_spec(source_run, ledger)
    for k in ("importance", "screening", "validation", "process_permutation"):
        cfg.pop(k, None)
    cfg["stage"] = stage
    cfg["purpose"] = "research"
    cfg[stage] = section
    return cfg


def submit_validation(source_run: str, ledger: Optional[Ledger] = None) -> str:
    """Queue the battery (tests 1 to 6) against the latest pre-registration."""
    ledger = ledger or get_ledger()
    problems = entry_gate(source_run, ledger)
    if problems:
        raise GateError("; ".join(problems))
    reg = latest_preregistration(_model_run(source_run, ledger)["family_id"], ledger)
    section = {"source_run": source_run, "preregistration_id": reg["id"],
               "tests": [t for t in reg["payload"]["tests"] if t != 7]}
    return jobs.submit(_frozen_job(source_run, "validation", section, ledger), parent_runs=[source_run],
                       ledger=ledger)


def submit_process_permutation(source_run: str, ledger: Optional[Ledger] = None) -> str:
    """Queue test 7, the overnight process permutation, against the latest pre-registration."""
    ledger = ledger or get_ledger()
    problems = entry_gate(source_run, ledger)
    if problems:
        raise GateError("; ".join(problems))
    reg = latest_preregistration(_model_run(source_run, ledger)["family_id"], ledger)
    section = {"source_run": source_run, "preregistration_id": reg["id"]}
    return jobs.submit(_frozen_job(source_run, "process_permutation", section, ledger),
                       parent_runs=[source_run], ledger=ledger)


def close_family(family_id: str, reason: str, run_id: str, ledger: Optional[Ledger] = None) -> None:
    """
    Close a family after a failed battery. It stays in the ledger with ``closed``
    set; the reason and the report that closed it are logged beside its trials.
    """
    ledger = ledger or get_ledger()
    with ledger.transaction() as con:
        con.execute("UPDATE families SET closed = 1 WHERE family_id = ?", (family_id,))
    ledger.add_trial(run_id=run_id, family_id=family_id, kind="family_closed",
                     params={"reason": reason, "at": utcnow()})


# =========================================================================== hold-out registration


def battery_verdict(validation_run: str, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    """
    The verdict for a validation run, joined with the latest test 7 run against the
    same pre-registration. The family passes only when every test ran and passed.
    """
    import pandas as pd

    ledger = ledger or get_ledger()
    run = ledger.get_run(validation_run)
    if run is None or run["stage"] != "validation" or run["status"] != "done":
        raise GateError(f"{validation_run} is not a completed validation run")
    v = run["summary"]["validation"]
    verdict = pd.read_parquet(f"{run['run_dir']}/tables/verdict.parquet")
    perm = [r for r in ledger.list_runs(family_id=run["family_id"], stage="process_permutation", limit=200)
            if r["status"] == "done" and (r.get("summary") or {}).get("process_permutation", {})
            .get("preregistration_id") == v["preregistration_id"]]
    t7_run = perm[0]["run_id"] if perm else None
    if t7_run:
        verdict = pd.concat([verdict, pd.read_parquet(f"{perm[0]['run_dir']}/tables/verdict.parquet")])
    complete = set(verdict.index) >= set(BATTERY)
    return {"verdict": verdict, "summary": v, "process_permutation_run": t7_run, "complete": complete,
            "passed": bool(complete and verdict["passed"].astype(bool).all())}


def register_holdout(validation_run: str, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    """
    Register the family for its one hold-out evaluation (run in Phase 4).

    Needs every test of the battery, test 7 included, to have passed. Refuses a
    hold-out shorter than the minimum track record length at the observed Sharpe:
    such a test could not pass even with a real edge.
    """
    import numpy as np
    from afml.backtest_statistics.perfomance_statistics import minimum_track_record_length

    ledger = ledger or get_ledger()
    bv = battery_verdict(validation_run, ledger)
    if not bv["passed"]:
        raise GateError("hold-out registration needs every test, test 7 included, to have passed")
    run = ledger.get_run(validation_run)
    v = bv["summary"]
    reg = get_preregistration(v["preregistration_id"], ledger)
    rule = reg["payload"]["holdout"]
    if not rule.get("holdout_end"):
        raise GateError("the pre-registration has no holdout_end; register a new version with one")
    cfg = jobs.clone_spec(validation_run, ledger)
    boundary = vault.get_research_end(cfg["data"]["symbol"], ledger)
    t3 = v["tests"]["3"] if "3" in v["tests"] else v["tests"][3]
    ppy = float(t3["periods_per_year"])
    sr = float(t3["observed_sharpe"]) / np.sqrt(ppy)  # per event
    min_events = float(minimum_track_record_length(sr, 0.0, t3["skewness"], t3["excess_kurtosis"] + 3.0,
                                                   alpha=1.0 - rule["psr_min"])) if sr > 0 else float("inf")
    days = (vault._as_date(rule["holdout_end"]) - boundary).days
    events_per_day = v["n_events"] / max(1, (vault._as_date(cfg["data"]["end_date"])
                                             - vault._as_date(cfg["data"]["start_date"])).days + 1)
    expected = events_per_day * max(days, 0)
    record = {"validation_run": validation_run, "process_permutation_run": bv["process_permutation_run"],
              "preregistration_id": reg["id"], "holdout_start": str(boundary), "holdout_end": rule["holdout_end"],
              "expected_events": expected, "min_track_record_events": min_events,
              "observed_sharpe": t3["observed_sharpe"]}
    if not expected >= min_events:
        raise GateError(f"the hold-out ({expected:.0f} expected events to {rule['holdout_end']}) is shorter than "
                        f"the minimum track record length ({min_events:.0f} events at the observed Sharpe); "
                        "no result there could pass")
    ledger.add_trial(run_id=validation_run, family_id=run["family_id"], kind="holdout_registration",
                     params=record)
    return record
