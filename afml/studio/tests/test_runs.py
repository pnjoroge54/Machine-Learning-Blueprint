"""
End-to-end checks from the plan's Phase 0:

* running a config writes a ledger row with its config hash;
* an identical rerun hits afml.cache and writes byte-identical output tables;
* cancel stops a running job and leaves no orphan processes.
"""

import time
from pathlib import Path

import psutil
import pytest

from afml.studio import jobs, vault
from afml.studio.adapters import pipelines as P
from afml.studio.schemas import RunSpec
from afml.studio.worker import _spawn, run_worker

from .conftest import base_config


def _run_queue(ledger):
    run_worker(concurrency=1, poll_s=0.2, once=True, ledger=ledger)


def test_identical_rerun_hits_cache_and_is_byte_identical(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    cfg = base_config(stage="importance")
    r1 = jobs.submit(cfg, ledger=home)
    r2 = jobs.submit(cfg, ledger=home)
    _run_queue(home)

    run1, run2 = home.get_run(r1), home.get_run(r2)
    assert run1["status"] == "done", run1["error"]
    assert run2["status"] == "done", run2["error"]
    assert run1["config_hash"] == run2["config_hash"] == RunSpec.model_validate(cfg).config_hash()
    assert run1["identity"] == run2["identity"]

    cache2 = run2["summary"]["cache"]
    assert cache2 and all(v["misses"] == 0 and v["hits"] > 0 for v in cache2.values())

    a1 = {a["path"]: a["sha256"] for a in home.artifacts(r1) if a["kind"] == "table"}
    a2 = {a["path"]: a["sha256"] for a in home.artifacts(r2) if a["kind"] == "table"}
    assert a1 and a1 == a2

    # Both runs count: two importance trials in the family
    stats = home.family_stats(run1["family_id"])
    assert stats["n_raw"] == 2 and stats["n_eigen"] is not None

    # Run folder holds a replay script and the environment
    d = Path(run1["run_dir"])
    assert (d / "replay.py").exists() and (d / "env.json").exists()
    assert any(r["progress"] is not None for r in jobs.read_log(d))


def test_cancel_kills_process_tree(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    cfg = base_config()
    cfg["features"]["steps"] = [{"name": "debug_sleep", "params": {"seconds": 120, "spawn_child": True}}]
    run_id = jobs.submit(cfg, ledger=home)
    assert home.claim_next_job(worker_pid=0) == run_id
    proc = _spawn(run_id, Path(home.get_run(run_id)["run_dir"]))
    home.set_job_pid(run_id, proc.pid)

    deadline = time.time() + 120
    children = []
    while time.time() < deadline:
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pytest.fail("runner exited early: " + str(home.get_run(run_id)["error"]))
        if children:
            break
        time.sleep(0.5)
    assert children, "debug_sleep never spawned its child"
    pids = [proc.pid] + [c.pid for c in children]

    assert jobs.cancel(run_id, ledger=home) == "cancelled"
    proc.wait(timeout=20)
    time.sleep(0.5)
    assert not [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]
    assert home.get_run(run_id)["status"] == "cancelled"


def test_model_stage_records_one_trial_per_candidate():
    spec = RunSpec.model_validate(base_config(
        stage="model",
        model={"estimator": {"name": "random_forest", "params": {"n_estimators": 10, "n_jobs": 1}},
               "param_grid": {"max_depth": [2, 4, 6]}, "cv_splits": 3},
    ))
    cfg = spec.resolved().model_dump(mode="json")
    bars = P.load_bars(cfg["data"])
    events = P.compute_events(bars, cfg["labels"])
    X, _ = P.build_feature_matrix(bars, cfg["features"])
    tables, summary, trials = P.run_model(X, events, cfg["model"], seed=1)
    assert len(trials) == 3 and all(len(t["fold_scores"]) == 3 for t in trials)
    assert summary["best_score"] == pytest.approx(max(t["score"] for t in trials))


def test_role_is_derived_from_events():
    spec = RunSpec.model_validate(base_config(stage="labels"))
    cfg = spec.resolved().model_dump(mode="json")
    bars = P.load_bars(cfg["data"])
    events = P.compute_events(bars, cfg["labels"])
    assert P.model_role(events) == "secondary"
    assert P.model_role(events.drop(columns="side")) == "primary"
    assert {"train_w", "w"} <= set(events.columns)


def test_optuna_study_logs_every_trial_in_its_family(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    n_trials = 12
    cfg = base_config(
        stage="model",
        model={"search": "optuna", "n_trials": n_trials, "cv_splits": 3,
               "estimator": {"name": "random_forest", "params": {"n_estimators": 10, "n_jobs": 1}},
               "search_space": {"max_depth": {"int": [2, 6]},
                                "min_weight_fraction_leaf": {"float": [0.025, 0.2]}}},
    )
    run_id = jobs.submit(cfg, ledger=home)
    _run_queue(home)
    run = home.get_run(run_id)
    assert run["status"] == "done", run["error"]

    trials = [t for t in home.trials(run["family_id"]) if t["run_id"] == run_id]
    assert len(trials) == n_trials and {t["kind"] for t in trials} == {"optuna_trial"}
    complete = [t for t in trials if t["score"] is not None]
    assert complete and run["summary"]["model"]["n_complete"] == len(complete)
    assert run["summary"]["model"]["best_score"] == pytest.approx(max(t["score"] for t in complete))

    # Every trial is a raw trial; the effective count and null threshold sit beside the best
    stats = home.family_stats(run["family_id"])
    assert stats["n_raw"] == n_trials
    assert stats["n_eigen"] is not None and stats["n_eigen"] <= len(complete)
    if len(complete) > 1:
        assert stats["null_threshold"] is not None

    paths = {a["path"] for a in home.artifacts(run_id)}
    assert {"optuna.db", "tables/cv_results.parquet", "tables/optuna_trials.parquet"} <= paths
