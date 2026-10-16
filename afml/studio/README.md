# AFML Studio

A GUI for the path from feature importance to model development, built so that
every trial is counted. The core (`afml.studio`) owns every rule and never imports
Streamlit. The app (`app/`) only edits a draft config, launches jobs and reads
results.

## Quick start

```bash
# 1. Set the hold-out boundary for a symbol (once, never moved)
python -m afml.studio set-research-end EURUSD 2025-06-30

# 2. Start the worker (one heavy job at a time; later jobs queue)
python -m afml.studio worker

# 3. Start the app. Bind to your Tailscale IP to reach it from laptop or phone,
#    never to a public address.
python -m afml.studio app --address 127.0.0.1 --port 8501
```

Headless, with no browser:

```bash
python -m afml.studio example --stage importance > imp.json   # synthetic-data config
python -m afml.studio run imp.json                             # submit and execute now
python -m afml.studio families                                 # N, effective N, null threshold
```

`AFML_STUDIO_HOME` sets where the ledger and run folders live. The default is
`$AFML_DATA_DIR/studio`.

## Layout

| Module | Role |
| --- | --- |
| `schemas.py` | Pydantic specs. Config hash, family key, and `to_pipeline_kwargs()` |
| `registry.py` | Stable name → callable. Parameter forms are built from signatures |
| `ledger.py` | SQLite: families, runs, jobs, trials, fold scores, data access, artifacts |
| `vault.py` | `research_end` per symbol, pre-registration, contamination |
| `stats.py` | Family statistics through `afml.cross_validation.selection_overfit`: effective trial count, null threshold, noise floor, 1-SE rule |
| `jobs.py` / `worker.py` / `runner.py` | Submit, cancel (kills the process tree), resume; one subprocess per job |
| `versioning.py` | Code version: git, studio source, registered-function and package hashes |
| `adapters/pipelines.py` | **The only module that calls afml pipeline code** |
| `cli.py` | `python -m afml.studio ...` |

Each run folder holds `config.json`, `code_version.json`, `env.json`, `replay.py`
(a generated headless equivalent of the run), `log.jsonl` (progress included),
`summary.json` and `tables/*.parquet`. An Optuna model run also keeps its study in
`optuna.db`, which `optuna-dashboard sqlite:///<run folder>/optuna.db` opens.

## Rules enforced in code

1. Every launch is logged to a family before it is queued. Grid candidates and Optuna
   trials (pruned and failed ones included) are one trial each, written as each one
   finishes, and saving a feature set is a trial too.
2. Scores are shown beside the family's N, effective N and null threshold, and are
   only compared within the same metric.
3. A symbol with no `research_end` cannot start any run. Research runs cannot request
   or receive bars past it. A hold-out run needs a pre-registered config hash, and a
   second hold-out access contaminates the family permanently.
4. Two weight fields, never one: `train_weight_scheme` (fit) and `score_weight_source`
   (`events['w']`, scoring). The model role is derived from `side` in events.

## Model development (Phase 2)

The model stage runs either an Optuna study through
`afml.cross_validation.optuna_hyper_fit.optimize_trading_model` (`search: "optuna"`,
the default on page 5) or a purged-CV grid search through `clf_hyper_fit`
(`search: "grid"`). The Optuna study searches the fit weights (scheme and time decay)
along with the estimator's `search_space`, and scores every fold with `events['w']`.
It supports estimators that take `sample_weight` directly (`random_forest`,
`extra_trees`).

## Rewiring to the newer afml

afml also has `FeatureEngineeringPipeline`, `ModelDevelopmentPipeline`,
`screen_features`, `LearnedStrategy`, and the clustered, orthogonal and fingerprint
importance methods, which the studio does not call yet. Moving a stage onto them
means changing `adapters/pipelines.py` (the step functions and `capabilities()`)
and nothing else.

## Tests

```bash
python -m pytest afml/studio/tests
```

This covers the core, the Phase 0 checks (ledger row with config hash; an identical
rerun hits `afml.cache` and writes byte-identical tables; cancel leaves no orphan
processes), and a Streamlit `AppTest` render of every page.
