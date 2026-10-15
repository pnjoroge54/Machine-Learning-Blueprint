"""
Job API used by the UI and the CLI: submit, cancel, resume, clone and status.

Nothing runs off the books: ``submit`` validates the spec, checks the vault, logs the
run to its family in the ledger and only then queues it.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from . import paths, vault
from .ledger import ACTIVE_STATUSES, Ledger, get_ledger
from .schemas import RunSpec
from .versioning import code_version, identity


def _new_run_id(config_hash: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{config_hash[:6]}-{secrets.token_hex(2)}"


def submit(spec: Union[RunSpec, Dict[str, Any]], *, parent_runs: Optional[List[str]] = None,
           ledger: Optional[Ledger] = None, enqueue: bool = True) -> str:
    """Validate, log and queue a run. Returns its run ID."""
    ledger = ledger or get_ledger()
    spec = spec if isinstance(spec, RunSpec) else RunSpec.model_validate(spec)
    resolved = spec.resolved()  # raises on unknown registry names or params
    config_hash = spec.config_hash()
    family_id = spec.family_id()

    # Fail closed before anything is queued
    d = resolved.data
    vault.check_request(d.symbol, d.start_date, d.end_date, spec.purpose,
                        family_id=family_id, config_hash=config_hash, ledger=ledger)

    version = code_version(spec)
    ident = identity(config_hash, version["hash"])
    run_id = _new_run_id(config_hash)
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    full = resolved.model_dump(mode="json")
    full["notes"], full["family_label"] = spec.notes, spec.family_label
    (run_dir / "config.json").write_text(json.dumps(full, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "code_version.json").write_text(json.dumps(version, indent=2, sort_keys=True),
                                               encoding="utf-8")

    ledger.upsert_family(family_id, spec.family_name(), spec.family_key())
    ledger.create_run(run_id=run_id, family_id=family_id, stage=spec.stage, purpose=spec.purpose,
                      config_hash=config_hash, code_version=version["hash"], identity=ident,
                      run_dir=str(run_dir), parent_runs=parent_runs, notes=spec.notes,
                      enqueue=enqueue)
    return run_id


def kill_tree(pid: int, timeout: float = 10.0) -> List[int]:
    """Kill a process and all its descendants. Returns the PIDs that were killed."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    procs = root.children(recursive=True) + [root]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:  # pragma: no cover - last resort
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    return [p.pid for p in procs]


def cancel(run_id: str, ledger: Optional[Ledger] = None) -> str:
    """Cancel a queued or running job. Returns the resulting status."""
    ledger = ledger or get_ledger()
    run = ledger.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    if run["status"] not in ACTIVE_STATUSES:
        return run["status"]
    ledger.request_cancel(run_id)
    job = ledger.get_job(run_id) or {}
    pid = job.get("pid")
    if pid:
        kill_tree(pid)
    ledger.set_status(run_id, "cancelled", error="cancelled by user")
    return "cancelled"


def clone_spec(run_id: str, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    """The run's config as a draft dict, for editing into a new run."""
    ledger = ledger or get_ledger()
    run = ledger.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    return json.loads((Path(run["run_dir"]) / "config.json").read_text(encoding="utf-8"))


def resume(run_id: str, ledger: Optional[Ledger] = None) -> str:
    """
    Resubmit a failed or cancelled run as a new run whose parent is the old one.

    With Optuna (newer afml) this maps to ``continue_study``; on this branch the
    model stage is a grid search, so resume reruns it and afml.cache supplies the
    steps that already finished.
    """
    ledger = ledger or get_ledger()
    return submit(clone_spec(run_id, ledger), parent_runs=[run_id], ledger=ledger)


def save_feature_set(run_id: str, columns: List[str], ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    """
    Save a column selection from an importance run as a feature set.

    Saving counts as a trial: choosing columns by out-of-fold MDA reuses the folds
    that will later score the model, so the choice is logged to the family. Returns
    the source run's config with ``features.selected_columns`` set, as a new draft.
    """
    ledger = ledger or get_ledger()
    run = ledger.get_run(run_id)
    if run is None or run["status"] != "done":
        raise ValueError(f"run {run_id} is not a completed run")
    if not columns:
        raise ValueError("select at least one column")
    cfg = clone_spec(run_id, ledger)
    cfg["features"]["selected_columns"] = list(columns)
    ledger.add_trial(run_id=run_id, family_id=run["family_id"], kind="feature_set",
                     params={"columns": list(columns), "source_run": run_id})
    return cfg


def read_log(run_dir: Union[str, Path], tail: int = 200) -> List[Dict[str, Any]]:
    """Parsed JSONL log records of a run (last ``tail``)."""
    path = Path(run_dir) / "log.jsonl"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-tail:]
    out = []
    for line in lines:
        try:
            rec = json.loads(line)["record"]
        except (ValueError, KeyError):
            continue
        out.append({
            "time": rec["time"]["repr"],
            "level": rec["level"]["name"],
            "module": rec.get("name"),
            "message": rec["message"],
            "stage": rec["extra"].get("stage"),
            "progress": rec["extra"].get("progress"),
        })
    return out


def progress(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Latest progress record of a run: stage, fraction and message."""
    for rec in reversed(read_log(run_dir, tail=400)):
        if rec["progress"] is not None:
            return {"stage": rec["stage"], "progress": rec["progress"], "message": rec["message"]}
    return {"stage": None, "progress": 0.0, "message": ""}


def is_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def status(run_id: str, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    ledger = ledger or get_ledger()
    run = ledger.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    run["progress"] = progress(run["run_dir"])
    run["stale"] = run["status"] == "running" and run.get("pid") and not is_alive(run["pid"])
    return run
