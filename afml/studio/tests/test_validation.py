"""
Phase 3: pre-registration, the entry gate, and the validation battery as a job.

Acceptance criteria covered here: a rerun on the same frozen config appears in the
ledger as a new trial and both reports remain; a hold-out load without a matching
pre-registration fails, and a second hold-out access flags the family.
"""

from pathlib import Path

import pandas as pd
import pytest

from afml.studio import jobs, vault
from afml.studio import validation as SV
from afml.studio.worker import run_worker

from .conftest import base_config

OPTUNA_MODEL = {
    "search": "optuna", "n_trials": 8, "cv_splits": 3, "pruner": "median",
    "estimator": {"name": "random_forest", "params": {"n_estimators": 15, "n_jobs": 1}},
    "search_space": {"max_depth": {"int": [2, 6]}, "min_weight_fraction_leaf": {"float": [0.025, 0.2]}},
}


def _run(ledger):
    run_worker(concurrency=1, poll_s=0.2, once=True, ledger=ledger)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """A completed Optuna model run in a fresh studio home."""
    import os

    from afml.studio.ledger import get_ledger

    home = tmp_path_factory.mktemp("phase3")
    old = os.environ.get("AFML_STUDIO_HOME")
    os.environ["AFML_STUDIO_HOME"] = str(home)
    led = get_ledger()
    vault.set_research_end("SYNTH", "2024-06-30", ledger=led)
    model_run = jobs.submit(base_config(stage="model", end="2024-03-31", model=OPTUNA_MODEL), ledger=led)
    _run(led)
    assert led.get_run(model_run)["status"] == "done", led.get_run(model_run)["error"]
    yield led, model_run
    if old is None:
        os.environ.pop("AFML_STUDIO_HOME", None)
    else:
        os.environ["AFML_STUDIO_HOME"] = old


def test_gate_refuses_until_the_config_is_frozen(frozen):
    led, model_run = frozen
    problems = SV.entry_gate(model_run, led)
    assert any("frozen" in p for p in problems)
    with pytest.raises(SV.GateError):
        SV.submit_validation(model_run, led)


def test_gate_refuses_a_grid_search_run(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    grid = jobs.submit(base_config(stage="model", model={
        "estimator": {"name": "random_forest", "params": {"n_estimators": 10, "n_jobs": 1}},
        "param_grid": {"max_depth": [2, 4]}, "cv_splits": 3}), ledger=home)
    _run(home)
    assert any("Optuna" in p for p in SV.entry_gate(grid, home))


def test_battery_reports_every_test_and_reruns_never_overwrite(frozen):
    led, model_run = frozen
    first_reg = SV.preregister(SV.Preregistration(source_run=model_run, n_perm_signal=100, n_boot=500), led)
    second_reg = SV.preregister(SV.Preregistration(source_run=model_run, n_perm_signal=100, n_boot=500,
                                                   notes="second version"), led)
    family = led.get_run(model_run)["family_id"]
    regs = SV.preregistrations(family, led)
    assert [r["version"] for r in regs] == [1, 2] and [r["id"] for r in regs] == [first_reg, second_reg]
    assert SV.entry_gate(model_run, led) == []

    runs = [SV.submit_validation(model_run, led) for _ in range(2)]
    _run(led)
    for run_id in runs:
        run = led.get_run(run_id)
        assert run["status"] == "done", run["error"]
        v = run["summary"]["validation"]
        assert v["preregistration_id"] == second_reg and v["preregistration_version"] == 2
        verdict = pd.read_parquet(Path(run["run_dir"]) / "tables" / "verdict.parquet")
        assert list(verdict.index) == [1, 2, 3, 4, 5, 6]  # every test reported
        assert verdict["passed"].dtype == bool and v["passed"] == bool(verdict["passed"].all())
        assert all("error" not in s for s in v["tests"].values()), v["tests"]

    trials = [t for t in led.trials(family) if t["kind"] == "validation"]
    assert len(trials) == 2  # the rerun is a new trial, and both reports remain
    assert all((Path(led.get_run(r)["run_dir"]) / "tables" / "verdict.parquet").exists() for r in runs)


def test_holdout_needs_a_matching_preregistration_and_contaminates_on_reuse(home):
    vault.set_research_end("SYNTH", "2024-01-15", ledger=home)
    cfg = base_config(stage="bars", purpose="holdout")
    with pytest.raises(vault.VaultError, match="not pre-registered"):
        jobs.submit(cfg, ledger=home)
    from afml.studio.schemas import RunSpec

    spec = RunSpec.model_validate(cfg)
    home.upsert_family(spec.family_id(), "f", spec.family_key())
    vault.preregister(spec.family_id(), spec.config_hash(), ledger=home)
    assert vault.record_holdout_access(spec.family_id(), "r1", home) is True
    assert vault.record_holdout_access(spec.family_id(), "r2", home) is False
    assert home.get_family(spec.family_id())["contaminated"]


def test_close_family_sets_the_flag_and_logs_the_reason(home):
    home.upsert_family("f-c", "c", {})
    home.create_run(run_id="v", family_id="f-c", stage="validation", purpose="research", config_hash="h",
                    code_version="c", identity="i", run_dir=".", enqueue=False)
    SV.close_family("f-c", "failed tests 3 and 4", "v", home)
    assert home.get_family("f-c")["closed"] == 1
    assert [t["kind"] for t in home.trials("f-c")] == ["family_closed"]


def test_process_permutation_refuses_to_start_when_cache_isolation_fails():
    from afml.studio.adapters import validation as V

    idx = pd.date_range("2024-01-01", periods=300, freq="15min")
    close = pd.Series(1.1 + 0.0001 * pd.Series(range(300)).sample(frac=1, random_state=0).cumsum().to_numpy(),
                      index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 1e-4, "low": close - 1e-4, "close": close,
                         "volume": 100.0}, index=idx)
    calls = []
    stale = lambda b: calls.append(1) or 1.25  # noqa: E731  (a cache serving one stale answer)
    prereg = SV.Preregistration(source_run="x", n_perm_process=5).model_dump(mode="json")
    with pytest.raises(RuntimeError, match="not responding to its input"):
        V.run_process_permutation(bars, {"model": {"n_trials": 3}}, prereg, develop_fn=stale)
    assert len(calls) == 3  # the isolation probes only; no replicate ran


def test_process_permutation_job_reports_test_7(frozen):
    led, model_run = frozen
    SV.preregister(SV.Preregistration(source_run=model_run, n_perm_process=5, n_boot=500), led)
    run_id = SV.submit_process_permutation(model_run, led)
    _run(led)
    run = led.get_run(run_id)
    assert run["status"] == "done", run["error"]
    s = run["summary"]["process_permutation"]
    assert s["cache_isolation"] == "passed" and s["n_perm"] == 5 and s["trial_budget"] == 8
    verdict = pd.read_parquet(Path(run["run_dir"]) / "tables" / "verdict.parquet")
    assert list(verdict.index) == [7]
    kinds = [t["kind"] for t in led.trials(run["family_id"])]
    assert "process_permutation" in kinds
