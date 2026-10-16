"""Headless tests for the studio core: schemas, registry, stats, ledger and vault."""

import numpy as np
import pytest

from afml.studio import jobs, registry, vault
from afml.studio.schemas import RunSpec
from afml.studio.stats import effective_trial_count, expected_max_sharpe, one_standard_error_selection

from .conftest import base_config


# --------------------------------------------------------------------------- schemas


def test_hash_ignores_explicit_defaults_unused_sections_and_notes():
    a = RunSpec.model_validate(base_config())
    cfg = base_config(notes="hello", family_label="mine")
    cfg["importance"]["n_splits"] = 7  # importance is unused at stage=features
    b = RunSpec.model_validate(cfg)
    assert a.config_hash() == b.config_hash()


def test_hash_fills_registry_defaults():
    cfg = base_config()
    cfg["features"]["steps"][1]["params"] = {}
    implicit = RunSpec.model_validate(cfg)
    cfg["features"]["steps"][1]["params"] = {"window": 20}
    explicit = RunSpec.model_validate(cfg)
    assert implicit.config_hash() == explicit.config_hash()


def test_family_is_the_research_question_not_the_feature_set():
    a = RunSpec.model_validate(base_config(stage="importance"))
    cfg = base_config(stage="importance")
    cfg["features"]["steps"].pop()
    b = RunSpec.model_validate(cfg)
    assert a.config_hash() != b.config_hash()
    assert a.family_id() == b.family_id()
    cfg["labels"]["profit_target"] = 2.0
    assert RunSpec.model_validate(cfg).family_id() != a.family_id()


def test_stage_requires_its_sections():
    cfg = base_config(stage="importance")
    del cfg["labels"]
    with pytest.raises(ValueError, match="needs sections"):
        RunSpec.model_validate(cfg)


def test_unknown_registry_param_is_rejected():
    cfg = base_config()
    cfg["features"]["steps"][0]["params"]["nope"] = 1
    with pytest.raises(ValueError, match="no params"):
        RunSpec.model_validate(cfg).config_hash()


def test_two_weight_fields_never_one():
    fields = RunSpec.model_fields["labels"].annotation.__args__[0].model_fields
    assert "train_weight_scheme" in fields and "score_weight_source" in fields
    assert "sample_weight" not in fields


# --------------------------------------------------------------------------- registry


def test_param_fields_widgets():
    fields = {f.name: f for f in registry.param_fields("estimator", "random_forest")}
    assert fields["criterion"].widget == "select"
    assert fields["criterion"].choices == ["entropy", "gini", "log_loss"]
    assert fields["n_estimators"].widget == "int" and fields["n_estimators"].min_value == 1
    assert fields["min_weight_fraction_leaf"].max_value == 0.5
    assert fields["max_features"].widget == "json"
    toggles = {f.name: f.widget for f in registry.param_fields("feature", "time_features")}
    assert toggles["forex"] == "toggle"
    # Pipeline-supplied inputs are not form fields
    assert "bars" not in {f.name for f in registry.param_fields("feature", "log_returns")}


def test_hidden_entries_not_listed():
    assert "debug_sleep" not in registry.names("feature")
    assert "debug_sleep" in registry.names("feature", include_hidden=True)


def test_register_conflict():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("feature", "log_returns")(lambda bars: bars)


# --------------------------------------------------------------------------- stats


def test_effective_trial_count_bounds():
    rng = np.random.default_rng(0)
    independent = rng.normal(size=(10, 200))
    tc = effective_trial_count(independent)
    assert 7 < tc.n_eigen <= 10 and 7 < tc.n_kish <= 10.01
    base = rng.normal(size=200)
    clones = base + rng.normal(scale=1e-3, size=(10, 200))
    tc = effective_trial_count(clones)
    # Li and Ji count the integer and fractional parts of each eigenvalue, so an
    # eigenvalue just under N contributes up to 2; Kish gives 1
    assert tc.n_eigen <= 2.0 + 1e-9 and tc.n_kish < 1.1
    assert not tc.rank_limited
    assert effective_trial_count(rng.normal(size=(12, 5))).rank_limited


def test_expected_max_sharpe_grows_with_n():
    vals = [expected_max_sharpe(n, 1.0) for n in (1, 2, 10, 100, 1000)]
    assert vals[0] == 0.0 and all(a < b for a, b in zip(vals, vals[1:]))
    # Known value: E[max] of 100 standard normals is about 2.5
    assert 2.4 < expected_max_sharpe(100, 1.0) < 2.6


def test_one_standard_error_selection_prefers_the_simpler_near_tie():
    folds = {2: [0.50, 0.52, 0.48, 0.50], 4: [0.57, 0.59, 0.58, 0.57], 8: [0.64, 0.54, 0.60, 0.60]}
    results = [{"params": {"max_depth": d}, "fold_scores": f} for d, f in folds.items()]
    chosen, diag = one_standard_error_selection(results, lambda p: p["max_depth"])
    assert chosen["params"]["max_depth"] == 4 and diag["n_within_1se"] == 2


def test_effective_trial_count_tolerates_tiny_matrices():
    assert effective_trial_count(np.array([[0.1, 0.2, 0.3]])).n_eigen == 1.0
    tc = effective_trial_count(np.array([[0.1, np.nan], [0.2, 0.3], [0.4, 0.5]]))
    assert tc.n_raw == 2 and tc.n_obs == 2


# --------------------------------------------------------------------------- vault


def test_missing_research_end_fails_closed(home):
    with pytest.raises(vault.VaultError, match="No research_end"):
        jobs.submit(base_config(), ledger=home)
    assert home.list_runs() == []


def test_research_end_is_never_moved(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)  # idempotent
    with pytest.raises(vault.VaultError, match="never moved"):
        vault.set_research_end("SYNTH", "2024-07-31", ledger=home)


def test_research_run_past_research_end_fails(home):
    vault.set_research_end("SYNTH", "2024-01-15", ledger=home)
    with pytest.raises(vault.VaultError, match="hold-out"):
        jobs.submit(base_config(end="2024-01-31"), ledger=home)


def test_research_end_must_follow_examined_bars(home):
    home.add_data_access(symbol="EURUSD", start="2024-01-01", end="2024-05-01", purpose="research")
    with pytest.raises(vault.VaultError, match="latest bar"):
        vault.set_research_end("EURUSD", "2024-03-01", ledger=home)


def test_holdout_requires_preregistration_and_second_access_contaminates(home):
    vault.set_research_end("SYNTH", "2024-01-15", ledger=home)
    spec = RunSpec.model_validate(base_config(purpose="holdout"))
    with pytest.raises(vault.VaultError, match="not pre-registered"):
        jobs.submit(spec, ledger=home)
    home.upsert_family(spec.family_id(), spec.family_name(), spec.family_key())
    vault.preregister(spec.family_id(), spec.config_hash(), ledger=home)
    run_id = jobs.submit(spec, ledger=home)
    assert vault.record_holdout_access(spec.family_id(), run_id, ledger=home) is True
    assert not home.get_family(spec.family_id())["contaminated"]
    assert vault.record_holdout_access(spec.family_id(), "again", ledger=home) is False
    assert home.get_family(spec.family_id())["contaminated"] == 1


# --------------------------------------------------------------------------- ledger


def test_submit_logs_run_with_hash_and_family(home):
    vault.set_research_end("SYNTH", "2024-06-30", ledger=home)
    spec = RunSpec.model_validate(base_config())
    run_id = jobs.submit(spec, ledger=home)
    run = home.get_run(run_id)
    assert run["status"] == "queued"
    assert run["config_hash"] == spec.config_hash()
    assert run["family_id"] == spec.family_id()
    assert home.get_job(run_id)["status"] == "queued"
    assert jobs.cancel(run_id, ledger=home) == "cancelled"
    assert home.claim_next_job(1) is None


def test_family_stats_count_every_trial(home):
    home.upsert_family("f-x", "x", {})
    rng = np.random.default_rng(1)
    for i in range(6):
        home.create_run(run_id=f"r{i}", family_id="f-x", stage="model", purpose="research",
                        config_hash="h", code_version="c", identity="i", run_dir=".", enqueue=False)
        home.add_trial(run_id=f"r{i}", family_id="f-x", kind="model_candidate", metric="acc",
                       score=0.5 + 0.01 * i, fold_scores=rng.normal(0.5, 0.05, size=5))
    home.add_trial(run_id="r0", family_id="f-x", kind="feature_set")  # saved feature set
    s = home.family_stats("f-x")
    assert s["n_raw"] == 7 and s["n_scored"] == 6
    assert s["best_score"] == pytest.approx(0.55)
    assert s["rank_limited"] is True  # 6 trials, 5 folds
    assert s["null_threshold"] is not None


def test_family_stats_never_rank_scores_across_metrics(home):
    home.upsert_family("f-m", "m", {})
    home.create_run(run_id="r", family_id="f-m", stage="model", purpose="research",
                    config_hash="h", code_version="c", identity="i", run_dir=".", enqueue=False)
    home.add_trial(run_id="r", family_id="f-m", kind="importance_cv", metric="neg_log_loss",
                   score=-0.69, fold_scores=[-0.7, -0.68, -0.69])
    for s in (0.40, 0.42):
        home.add_trial(run_id="r", family_id="f-m", kind="model_candidate", metric="f1",
                       score=s, fold_scores=[s - 0.01, s, s + 0.01])
    by_default = home.family_stats("f-m")
    assert by_default["metric"] == "f1" and by_default["n_raw"] == 3
    logloss = home.family_stats("f-m", "neg_log_loss")
    assert logloss["best_score"] == pytest.approx(-0.69)
    assert logloss["n_raw"] == 3 and logloss["n_metric"] == 1


@pytest.mark.parametrize("space, ok", [
    ({"max_depth": {"int": [2, 8]}, "C": {"float": [1e-3, 10], "log": True}, "f": ["sqrt", 1]}, True),
    ({}, False),                                  # Optuna needs a space
    ({"max_depth": {"int": [8, 2]}}, False),      # low >= high
    ({"C": {"float": [0, 1], "log": True}}, False),  # log range from zero
    ({"x": {"int": [1, 2], "float": [1, 2]}}, False),
    ({"x": []}, False),
])
def test_optuna_search_space_is_validated(space, ok):
    cfg = base_config(stage="model", model={"search": "optuna", "search_space": space})
    if ok:
        RunSpec.model_validate(cfg)
    else:
        with pytest.raises(ValueError):
            RunSpec.model_validate(cfg)
