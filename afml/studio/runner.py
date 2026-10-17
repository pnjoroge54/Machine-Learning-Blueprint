"""
Run one studio job: resolve its spec, call the afml pipeline steps through the
adapter, and write the run folder.

Run folder layout::

    runs/<run_id>/
        config.json        resolved canonical spec (what the hash covers)
        code_version.json  git commit, function and package hashes
        env.json           Python, platform, packages, conda env export
        replay.py          generated headless equivalent of this run
        log.jsonl          every log record bound to this run (progress included)
        process.log        raw stdout/stderr of the subprocess
        summary.json       per-step summaries and cache hit/miss counts
        optuna.db          the Optuna study (model runs with search="optuna")
        oof/trial_*.parquet each complete Optuna trial's out-of-fold probabilities
        tables/*.parquet   every table a page plots from

Invoked as ``python -m afml.studio.runner <run_id>`` by the worker, one subprocess per
job, so Numba state, joblib memory and out-of-memory crashes stay isolated.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from afml.logging_config import configure_logging

from . import vault
from .adapters import pipelines as P
from .ledger import Ledger, get_ledger
from .progress import log_progress
from .schemas import RunSpec
from .versioning import environment_snapshot

STEP_ORDER = {
    "bars": ["bars"],
    "labels": ["bars", "labels"],
    "features": ["bars", "features"],
    "screening": ["bars", "labels", "features", "screening"],
    "importance": ["bars", "labels", "features", "importance"],
    "model": ["bars", "labels", "features", "model"],
    "validation": ["bars", "labels", "features", "validation"],
    "process_permutation": ["bars", "process_permutation"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunContext:
    """What a step may touch: the run folder, the ledger and progress reporting."""

    def __init__(self, run_id: str, run_dir: Path, spec: RunSpec, ledger: Ledger, family_id: str):
        self.run_id = run_id
        self.run_dir = run_dir
        self.spec = spec
        self.cfg = spec.resolved().model_dump(mode="json")
        self.ledger = ledger
        self.family_id = family_id
        self.tables_dir = run_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.summary: Dict[str, Any] = {}
        self.steps: List[str] = STEP_ORDER[spec.stage]
        self._step_index = 0

    def progress_for(self, step: str) -> Callable[[float, str], None]:
        i = self.steps.index(step)

        def report(p: float, msg: str = "") -> None:
            overall = (i + max(0.0, min(1.0, p))) / len(self.steps)
            log_progress(step, overall, msg or step, step_progress=p)

        return report

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.tables_dir / f"{name}.parquet"
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(path, index=True)
        self.ledger.add_artifact(self.run_id, f"tables/{name}.parquet", "table",
                                 sha256_file(path), path.stat().st_size)
        return path

    def write_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        for name, df in tables.items():
            self.write_table(name, df)

    def write_json(self, name: str, obj: Any, kind: str = "json") -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self.ledger.add_artifact(self.run_id, name, kind, sha256_file(path), path.stat().st_size)
        return path


# =========================================================================== steps


def step_bars(ctx: RunContext, state: Dict[str, Any]) -> None:
    data = ctx.cfg["data"]
    report = ctx.progress_for("bars")
    report(0.0, f"loading {data['symbol']} {data['start_date']}..{data['end_date']}")
    vault.check_request(data["symbol"], data["start_date"], data["end_date"], ctx.spec.purpose,
                        family_id=ctx.family_id, config_hash=ctx.spec.config_hash(), ledger=ctx.ledger)
    if ctx.spec.purpose == "holdout":
        first = vault.record_holdout_access(ctx.family_id, ctx.run_id, ctx.ledger)
        if not first:
            logger.warning("Second hold-out access: family {} is now contaminated", ctx.family_id)
    bars = P.load_bars(data)
    vault.check_frame(bars, data["symbol"], ctx.spec.purpose, ctx.ledger)
    if bars.empty:
        raise ValueError("no bars in the requested window")
    ctx.ledger.add_data_access(symbol=data["symbol"], start=bars.index[0], end=bars.index[-1],
                               purpose=ctx.spec.purpose, run_id=ctx.run_id,
                               family_id=ctx.family_id, n_rows=len(bars))
    try:  # mirror into afml's own tracker so its contamination report stays complete
        from afml.cache.data_access_tracker import log_data_access

        log_data_access(f"studio:{data['symbol']}:{ctx.spec.data.bar_spec}".lower(),
                        bars.index[0], bars.index[-1],
                        "test" if ctx.spec.purpose == "holdout" else "train", bars.shape)
    except Exception as exc:
        logger.debug("DataAccessTracker mirror failed: {}", exc)
    report(0.5, f"{len(bars):,} bars; computing ADF by d")
    tables, summary = P.bars_report(bars)
    if ctx.spec.stage != "bars":
        tables.pop("bars")  # downstream stages keep their own tables; bars stay cached
    ctx.write_tables(tables)
    ctx.summary["bars"] = summary
    state["bars"] = bars
    report(1.0, "bars done")


def step_labels(ctx: RunContext, state: Dict[str, Any]) -> None:
    report = ctx.progress_for("labels")
    report(0.0, "triple-barrier labels")
    events = P.compute_events(state["bars"], ctx.cfg["labels"])
    tables, summary = P.labels_report(events, state["bars"])
    ctx.write_tables(tables)
    ctx.summary["labels"] = summary
    state["events"] = events
    report(1.0, f"{len(events):,} events ({summary['role']} model)")


def step_features(ctx: RunContext, state: Dict[str, Any]) -> None:
    report = ctx.progress_for("features")
    X, catalog = P.build_feature_matrix(state["bars"], ctx.cfg["features"], progress=report)
    tables, summary = P.features_report(X, catalog)
    ctx.write_tables(tables)
    ctx.summary["features"] = summary
    state["X"] = X
    report(1.0, f"{X.shape[1]} features")


def step_screening(ctx: RunContext, state: Dict[str, Any]) -> None:
    report = ctx.progress_for("screening")
    tables, summary, trial = P.run_screening(state["X"], state["events"], ctx.cfg["screening"],
                                             seed=ctx.cfg["seed"], progress=report)
    ctx.write_tables(tables)
    ctx.summary["screening"] = summary
    # A screen chooses features, so it counts toward the family's N
    ctx.ledger.add_trial(run_id=ctx.run_id, family_id=ctx.family_id, kind="screening",
                         params=trial["params"])
    report(1.0, f"{summary['n_selected']}/{summary['n_features']} features pass all three gates")


def step_importance(ctx: RunContext, state: Dict[str, Any]) -> None:
    report = ctx.progress_for("importance")
    tables, summary, trial = P.run_importance(state["X"], state["events"], ctx.cfg["importance"],
                                              seed=ctx.cfg["seed"], progress=report)
    ctx.write_tables(tables)
    ctx.summary["importance"] = summary
    ctx.ledger.add_trial(run_id=ctx.run_id, family_id=ctx.family_id, kind="importance_cv",
                         metric=trial["metric"], score=trial["score"],
                         fold_scores=trial["fold_scores"],
                         params={"features": list(state["X"].columns)})
    report(1.0, f"CV {summary['scoring']} {summary['cv_score_mean']:.4f}")


def step_model(ctx: RunContext, state: Dict[str, Any]) -> None:
    report = ctx.progress_for("model")

    def log_trial(t: Dict[str, Any]) -> None:
        # Written as each trial finishes, so a cancelled study keeps every trial it ran
        ctx.ledger.add_trial(run_id=ctx.run_id, family_id=ctx.family_id, kind=t["kind"],
                             number=t["number"], metric=t["metric"], score=t["score"],
                             params=t["params"], fold_scores=t["fold_scores"])

    tables, summary, trials = P.run_model(state["X"], state["events"], ctx.cfg["model"],
                                          seed=ctx.cfg["seed"], progress=report,
                                          data_index=state["bars"].index, storage_dir=ctx.run_dir,
                                          on_trial=log_trial)
    ctx.write_tables(tables)
    ctx.summary["model"] = summary
    db = ctx.run_dir / "optuna.db"
    if db.exists():
        ctx.ledger.add_artifact(ctx.run_id, db.name, "optuna_study", sha256_file(db), db.stat().st_size)
    report(1.0, f"best {summary['scoring']} {summary['best_score']:.4f} over {len(trials)} trials")


def _frozen_source(ctx: RunContext, section: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The frozen model run and the pre-registration a Phase 3 job runs against."""
    from . import validation as studio_validation

    cfg = ctx.cfg[section]
    source = ctx.ledger.get_run(cfg["source_run"])
    if source is None or source["status"] != "done":
        raise RuntimeError(f"frozen run {cfg['source_run']} is not a completed run")
    reg = studio_validation.get_preregistration(cfg["preregistration_id"], ctx.ledger)
    if reg["family_id"] != ctx.family_id or reg["config_hash"] != source["config_hash"]:
        raise RuntimeError("the pre-registration does not freeze this run's config in this family")
    return source, reg


def step_validation(ctx: RunContext, state: Dict[str, Any]) -> None:
    from .adapters import validation as V

    report = ctx.progress_for("validation")
    source, reg = _frozen_source(ctx, "validation")
    model_summary = source["summary"]["model"]
    family = ctx.ledger.family_stats(ctx.family_id, model_summary["scoring"])
    tables, summary = V.run_battery(state["X"], state["events"], state["bars"], ctx.cfg["model"],
                                    model_summary, reg["payload"], family, ctx.cfg["validation"]["tests"],
                                    seed=ctx.cfg["seed"], progress=report)
    summary.update(source_run=source["run_id"], preregistration_id=reg["id"],
                   preregistration_version=reg["version"])
    ctx.write_tables(tables)
    ctx.summary["validation"] = summary
    # Rerunning the battery on the same frozen config is a new trial; reports never overwrite
    ctx.ledger.add_trial(run_id=ctx.run_id, family_id=ctx.family_id, kind="validation",
                         params={"source_run": source["run_id"], "preregistration_id": reg["id"],
                                 "passed": summary["passed"], "failed_tests": summary["n_failed"]})
    report(1.0, "battery " + ("passed" if summary["passed"] else f"failed {summary['n_failed']} test(s)"))


def step_process_permutation(ctx: RunContext, state: Dict[str, Any]) -> None:
    from .adapters import validation as V

    report = ctx.progress_for("process_permutation")
    source, reg = _frozen_source(ctx, "process_permutation")
    tables, summary = V.run_process_permutation(state["bars"], ctx.cfg, reg["payload"], seed=ctx.cfg["seed"],
                                                progress=report)
    summary.update(source_run=source["run_id"], preregistration_id=reg["id"],
                   preregistration_version=reg["version"])
    ctx.write_tables(tables)
    ctx.summary["process_permutation"] = summary
    ctx.ledger.add_trial(run_id=ctx.run_id, family_id=ctx.family_id, kind="process_permutation",
                         params={"source_run": source["run_id"], "preregistration_id": reg["id"],
                                 "passed": summary["passed"], "p_value": summary["p_value"]})
    report(1.0, "test 7 " + ("passed" if summary["passed"] else "failed"))


STEPS = {"bars": step_bars, "labels": step_labels, "features": step_features,
         "screening": step_screening, "importance": step_importance, "model": step_model,
         "validation": step_validation, "process_permutation": step_process_permutation}


# =========================================================================== replay script


def replay_script(run_id: str, spec: RunSpec) -> str:
    steps = STEP_ORDER[spec.stage]
    lines = [
        f'"""Headless replay of studio run {run_id} (stage: {spec.stage}).',
        "",
        "Generated by afml.studio. It calls the same adapter steps as the GUI run, without",
        "the ledger. To replay with the ledger and vault (counted as a new trial):",
        "",
        "    python -m afml.studio run config.json",
        '"""',
        "",
        "from pathlib import Path",
        "",
        "from afml.studio.adapters import pipelines as P",
        "from afml.studio.schemas import RunSpec",
        "",
        'spec = RunSpec.model_validate_json(Path(__file__).with_name("config.json").read_text())',
        'cfg = spec.resolved().model_dump(mode="json")',
        "",
        'bars = P.load_bars(cfg["data"])',
        "bar_tables, bar_summary = P.bars_report(bars)",
        "print(bar_summary)",
    ]
    if "labels" in steps:
        lines += ['events = P.compute_events(bars, cfg["labels"])',
                  "print(P.labels_report(events, bars)[1])"]
    if "features" in steps:
        lines += ['X, catalog = P.build_feature_matrix(bars, cfg["features"])',
                  "print(P.features_report(X, catalog)[1])"]
    if "screening" in steps:
        lines += ['tables, summary, trial = P.run_screening(X, events, cfg["screening"], seed=cfg["seed"])',
                  "print(summary)"]
    if "importance" in steps:
        lines += ['tables, summary, trial = P.run_importance(X, events, cfg["importance"], seed=cfg["seed"])',
                  "print(summary)"]
    if "model" in steps:
        lines += ['tables, summary, trials = P.run_model(X, events, cfg["model"], seed=cfg["seed"],',
                  '                                      data_index=bars.index, storage_dir=Path(__file__).parent / "replay")',
                  "print(summary)"]
    return "\n".join(lines) + "\n"


# =========================================================================== entry point


def execute_run(run_id: str, ledger: Optional[Ledger] = None) -> str:
    """Execute a submitted run in this process. Returns the final status."""
    ledger = ledger or get_ledger()
    row = ledger.get_run(run_id)
    if row is None:
        raise KeyError(f"unknown run {run_id}")
    run_dir = Path(row["run_dir"])
    spec = RunSpec.model_validate_json((run_dir / "config.json").read_text(encoding="utf-8"))

    ledger.set_status(run_id, "running")
    ledger.set_job_pid(run_id, os.getpid())
    # Prewarm imports afml modules; do it before logging is configured so any module
    # that still reconfigures loguru at import time cannot drop the run sink.
    P.prewarm()

    configure_logging()  # idempotent; afml configures loguru on import
    sink = logger.add(str(run_dir / "log.jsonl"), level="DEBUG", serialize=True,
                      filter=lambda r: r["extra"].get("run_id") == run_id)
    status = "failed"
    with logger.contextualize(run_id=run_id, stage="setup"):
        try:
            logger.info("run {} started (stage {}, purpose {})", run_id, spec.stage, spec.purpose)
            log_progress("setup", 0.0, "preparing environment")

            ctx = RunContext(run_id, run_dir, spec, ledger, row["family_id"])
            ctx.write_json("env.json", environment_snapshot(), kind="env")
            (run_dir / "replay.py").write_text(replay_script(run_id, spec), encoding="utf-8")

            P.cache_report(reset=True)
            state: Dict[str, Any] = {}
            for step in ctx.steps:
                with logger.contextualize(stage=step):
                    STEPS[step](ctx, state)
            ctx.summary["cache"] = P.cache_report()
            ctx.summary["capabilities"] = P.capabilities()
            ctx.write_json("summary.json", ctx.summary, kind="summary")
            ledger.update_run(run_id, summary=ctx.summary)
            status = "done"
            ledger.set_status(run_id, "done", exit_code=0)
            log_progress("done", 1.0, "run complete")
        except BaseException as exc:  # includes KeyboardInterrupt from a console cancel
            tb = traceback.format_exc()
            logger.error("run {} failed: {}\n{}", run_id, exc, tb)
            ledger.set_status(run_id, "failed", error=f"{type(exc).__name__}: {exc}", exit_code=1)
            status = "failed"
        finally:
            try:
                logger.remove(sink)
            except ValueError:  # a module reconfigured loguru mid-run (Section 5.6)
                pass
    return status


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m afml.studio.runner <run_id>", file=sys.stderr)
        return 2
    return 0 if execute_run(argv[0]) == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
