"""
Shared Streamlit helpers for AFML Studio pages.

The UI only edits a draft config, launches jobs through ``afml.studio.jobs`` and
reads results from run folders. ``st.session_state`` holds only the unsaved draft;
everything durable lives in the ledger and the run folders.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afml.studio import jobs, registry, vault  # noqa: E402
from afml.studio.ledger import get_ledger  # noqa: E402
from afml.studio.schemas import STAGE_SECTIONS, RunSpec  # noqa: E402

pio.templates.default = "plotly_white"

DEFAULT_DRAFT: Dict[str, Any] = {
    "purpose": "research",
    "seed": 42,
    "data": {
        "symbol": "SYNTH",
        "source": "synthetic",
        "source_params": {"seed": 7, "freq": "15min", "signal_strength": 0.05},
        "account": None,
        "bar_type": "time",
        "bar_size": "M15",
        "price": "mid_price",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
    },
    "labels": {
        "strategy": {"name": "bollinger", "params": {}},
        "target": {"name": "daily_vol", "params": {}},
        "profit_target": 1.0,
        "stop_loss": 1.0,
        "max_holding_period": {"hours": 12},
        "min_ret": 0.0,
        "vertical_barrier_zero": True,
        "filter_events": True,
        "train_weight_scheme": "uniqueness",
        "time_decay": 1.0,
        "score_weight_source": "return_attribution",
    },
    "features": {
        "steps": [
            {"name": "log_returns", "params": {}},
            {"name": "volatility", "params": {}},
            {"name": "frac_diff_close", "params": {}},
        ],
        "selected_columns": None,
    },
    "importance": {
        "methods": ["mdi", "mda"],
        "n_splits": 5,
        "pct_embargo": 0.01,
        "estimator": {"name": "random_forest", "params": {"n_estimators": 100}},
        "scoring": "neg_log_loss",
        "split_mode": False,
        "importance_fraction": 0.5,
    },
    "model": {
        "estimator": {"name": "random_forest", "params": {"n_estimators": 100}},
        "param_grid": {"max_depth": [3, 5, 8], "min_weight_fraction_leaf": [0.02, 0.05]},
        "cv_splits": 5,
        "pct_embargo": 0.01,
        "bagging_n_estimators": 0,
        "bagging_max_samples": 1.0,
        "bagging_max_features": 1.0,
        "rnd_search_iter": 0,
        "search": "optuna",
        "search_space": {},  # filled from the estimator's default on page 5
        "n_trials": 50,
        "timeout_s": 3600,
        "pruner": "hyperband",
        "metric": "neg_log_loss",
    },
}


# =========================================================================== state


def ledger():
    return get_ledger()


def draft() -> Dict[str, Any]:
    if "draft" not in st.session_state:
        st.session_state["draft"] = copy.deepcopy(DEFAULT_DRAFT)
    return st.session_state["draft"]


def load_into_draft(cfg: Dict[str, Any]) -> None:
    """Replace the draft with a run's config, keeping sections the run did not use."""
    d = copy.deepcopy(DEFAULT_DRAFT)
    d.update(copy.deepcopy(draft()))
    for k, v in cfg.items():
        if k in ("stage", "notes", "family_label"):
            continue
        if v is not None:
            d[k] = copy.deepcopy(v)
    st.session_state["draft"] = d
    _bump_form_version()


def _bump_form_version() -> None:
    """Widgets are keyed by a version so loading a draft re-seeds every form."""
    st.session_state["form_v"] = st.session_state.get("form_v", 0) + 1


def wkey(*parts: Any) -> str:
    return ".".join(str(p) for p in ("w", st.session_state.get("form_v", 0), *parts))


def spec_for(stage: str) -> RunSpec:
    d = draft()
    cfg = {"stage": stage, **{k: d[k] for k in ("purpose", "seed")}}
    for s in STAGE_SECTIONS[stage]:
        cfg[s] = d[s]
    return RunSpec.model_validate(cfg)


# =========================================================================== widgets


def param_form(kind: str, name: str, current: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Generated form for a registered callable's parameters."""
    out: Dict[str, Any] = {}
    fields = registry.param_fields(kind, name)
    if not fields:
        st.caption("No parameters.")
        return out
    cols = st.columns(min(3, len(fields)))
    for i, f in enumerate(fields):
        value = current.get(f.name, f.default)
        k = wkey(key, name, f.name)
        with cols[i % len(cols)]:
            if f.widget == "select":
                idx = f.choices.index(value) if value in f.choices else 0
                out[f.name] = st.selectbox(f.name, f.choices, index=idx, key=k, help=f.help or None)
            elif f.widget == "toggle":
                out[f.name] = st.toggle(f.name, value=bool(value), key=k)
            elif f.widget == "int":
                out[f.name] = st.number_input(
                    f.name, value=None if value is None else int(value), step=1,
                    min_value=None if f.min_value is None else int(f.min_value),
                    max_value=None if f.max_value is None else int(f.max_value), key=k,
                    placeholder="None")
                if out[f.name] is not None:
                    out[f.name] = int(out[f.name])
            elif f.widget == "float":
                out[f.name] = st.number_input(
                    f.name, value=None if value is None else float(value), format="%g",
                    min_value=None if f.min_value is None else float(f.min_value),
                    max_value=None if f.max_value is None else float(f.max_value), key=k,
                    placeholder="None")
            elif f.widget == "text":
                out[f.name] = st.text_input(f.name, value="" if value is None else str(value), key=k)
            else:
                raw = st.text_input(f"{f.name} (JSON)", value=json.dumps(value), key=k)
                try:
                    out[f.name] = json.loads(raw)
                except ValueError:
                    out[f.name] = raw  # bare strings such as "sqrt"
    return out


def callable_picker(kind: str, spec: Dict[str, Any], label: str, key: str) -> Dict[str, Any]:
    """Select a registered callable and edit its params. Returns {name, params}."""
    options = registry.names(kind)
    name = spec.get("name") if spec.get("name") in options else options[0]
    name = st.selectbox(label, options, index=options.index(name), key=wkey(key, "name"),
                        format_func=lambda n: f"{n} — {registry.get(kind, n).description}"[:90])
    params = spec.get("params", {}) if name == spec.get("name") else {}
    return {"name": name, "params": param_form(kind, name, params, key)}


def date_input(label: str, value: Any, key: str, max_value: Optional[date] = None) -> str:
    v = date.fromisoformat(str(value)[:10])
    if max_value and v > max_value:
        v = max_value
    return st.date_input(label, value=v, max_value=max_value, key=wkey(key)).isoformat()


# =========================================================================== launching


def launch(stage: str, label: Optional[str] = None) -> None:
    """Validate the draft for ``stage`` and show a launch button."""
    try:
        spec = spec_for(stage)
        h = spec.config_hash()
    except Exception as exc:
        st.error(f"Config is not valid for stage '{stage}': {exc}")
        return
    led = ledger()
    fam = led.get_family(spec.family_id())
    prior = led.query("SELECT COUNT(*) AS n FROM runs WHERE config_hash = ? AND status = 'done'", (h,))[0]["n"]
    cols = st.columns([2, 3])
    with cols[0]:
        clicked = st.button(label or f"Launch {stage} run", type="primary", key=wkey("launch", stage),
                            use_container_width=True)
    with cols[1]:
        st.caption(
            f"config `{h[:12]}` · family `{spec.family_id()}` "
            f"({'existing' if fam else 'new'}) · {prior} completed run(s) with this exact config. "
            "Every launch is logged as a trial."
        )
    if fam and fam["contaminated"]:
        st.error("This family is contaminated: its hold-out has been accessed more than once.")
    if clicked:
        try:
            run_id = jobs.submit(spec)
        except vault.VaultError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return
        st.session_state[f"selected_run.{stage}"] = run_id
        if not led.live_workers():
            st.warning(f"Queued {run_id}, but no worker is running. Start one in the Control room.")
        else:
            st.success(f"Queued {run_id}")


def worker_status() -> None:
    live = ledger().live_workers()
    if live:
        st.caption(f"Worker running (pid {live[0]['pid']}, {live[0]['status']})")
    else:
        st.caption(":orange[No worker running]")


# =========================================================================== results


@st.cache_data(show_spinner=False, max_entries=64)
def _read_table(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def table(run: Dict[str, Any], name: str) -> Optional[pd.DataFrame]:
    path = Path(run["run_dir"]) / "tables" / f"{name}.parquet"
    if not path.exists():
        return None
    return _read_table(str(path), path.stat().st_mtime)


def run_picker(stage: str, label: str = "Results from run") -> Optional[Dict[str, Any]]:
    """Pick a completed run of ``stage`` (or one that includes it) to view."""
    runs = ledger().list_runs(stage=stage, limit=100)
    if not runs:
        st.info(f"No {stage} runs yet.")
        return None
    ids = [r["run_id"] for r in runs]
    chosen = st.session_state.get(f"selected_run.{stage}")
    idx = ids.index(chosen) if chosen in ids else 0
    by_id = {r["run_id"]: r for r in runs}
    run_id = st.selectbox(
        label, ids, index=idx, key=f"pick.{stage}",
        format_func=lambda i: f"{i} · {by_id[i]['status']} · {by_id[i]['family_name']}",
    )
    st.session_state[f"selected_run.{stage}"] = run_id
    run = jobs.status(run_id)
    if run["status"] in ("queued", "running"):
        _live_status(run_id)
        return None
    if run["status"] != "done":
        st.error(f"Run {run['status']}: {run.get('error') or ''}")
        with st.expander("Log"):
            show_log(run)
        return None
    try:
        draft_hash = spec_for(stage).config_hash()
        if draft_hash != run["config_hash"]:
            st.caption("Showing a run whose config differs from the current draft.")
    except Exception:
        pass
    if run.get("contaminated"):
        st.error("Family contaminated (hold-out accessed more than once).")
    return run


@st.fragment(run_every="2s")
def _live_status(run_id: str) -> None:
    run = jobs.status(run_id)
    p = run["progress"]
    if run["status"] in ("queued", "running"):
        st.progress(float(p["progress"] or 0.0),
                    text=f"{run['status']} · {p['stage'] or '—'} · {p['message'] or ''}")
        if run.get("stale"):
            st.warning("Process is gone but the run is still marked running.")
    else:
        st.rerun()


def show_log(run: Dict[str, Any], tail: int = 60) -> None:
    recs = jobs.read_log(run["run_dir"], tail=tail)
    if not recs:
        p = Path(run["run_dir"]) / "process.log"
        st.code(p.read_text(errors="replace")[-4000:] if p.exists() else "(no log)")
        return
    st.code("\n".join(f"{r['time'][11:19]} {r['level']:<7} [{r['stage']}] {r['message']}" for r in recs))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def family_panel(family_id: str, score: Optional[float] = None, score_label: str = "Score",
                 metric: Optional[str] = None) -> Dict[str, Any]:
    """Every score sits beside its family's raw N, effective N and null threshold."""
    s = ledger().family_stats(family_id, metric)
    cols = st.columns(5)
    cols[0].metric(score_label, fmt(score))
    cols[1].metric("Trials N", s["n_raw"], help=f"{s['n_metric']} scored in {s['metric']}; every trial counts toward N")
    cols[2].metric("Effective N (eigen / Kish)", f"{fmt(s['n_eigen'], 1)} / {fmt(s['n_kish'], 1)}")
    cols[3].metric("Null threshold", fmt(s["null_threshold"]),
                   help="Expected best score among N_eff trials with no skill, from the "
                        "cross-trial score variance (expected_max_sharpe on scores).")
    cols[4].metric(f"Best in family ({s['metric']})", fmt(s["best_score"]))
    if s["rank_limited"]:
        st.warning("Rank-limited: more trials than fold scores, so effective N is capped. "
                   "Lengthen the score vector (afml.cross_validation.selection_overfit.criterion_variance_audit) before trusting it.")
    if score is not None and s["null_threshold"] is not None and score <= s["null_threshold"]:
        st.warning("This score does not beat the family's null threshold.")
    return s
