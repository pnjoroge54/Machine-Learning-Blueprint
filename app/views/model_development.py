"""5 · Model development: an Optuna study (or a grid search), one ledger trial per candidate."""

import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import studio_ui as ui
from afml.studio.adapters import pipelines as P
from afml.studio.stats import noise_floor, one_standard_error_selection

st.title("5 · Model development")
mod = ui.draft()["model"]
caps = P.capabilities()

st.subheader("Search")
modes = {"optuna": "Optuna study (TPE)", "grid": "Grid search"}
if not caps["optuna"]:
    st.info("Optuna is not installed in this environment, so only the grid search is available.")
    mod["search"] = "grid"
mod["search"] = st.radio("Search", list(modes), format_func=modes.get, horizontal=True,
                         index=list(modes).index(mod.get("search", "grid")), key=ui.wkey("mod.search"),
                         disabled=not caps["optuna"], label_visibility="collapsed")
optuna_mode = mod["search"] == "optuna"

prev_est = mod["estimator"]["name"]
mod["estimator"] = ui.callable_picker("estimator", mod["estimator"], "Estimator", "mod.est")
est = mod["estimator"]["name"]
supported = not optuna_mode or est in caps["optuna_estimators"]
if not supported:
    st.warning(f"The Optuna search fits estimators that take sample_weight directly: "
               f"{', '.join(caps['optuna_estimators'])}. Pick one of those, or use the grid search.")

if optuna_mode:
    if est != prev_est or not mod.get("search_space"):
        mod["search_space"] = P.default_search_space(est)
    st.caption("Each value is a list of choices, `{\"int\": [low, high]}` or "
               "`{\"float\": [low, high], \"log\": false}`. Fit weights (unweighted, uniqueness or return, "
               "with time decay) are searched too, so the fit weights on page 2 do not apply here. "
               "Every fold is still scored with the return-attribution weights `events['w']`.")
    raw = st.text_area("Search space (JSON; keys are estimator params)", json.dumps(mod["search_space"], indent=1),
                       key=ui.wkey("mod.space", est), height=160)
    try:
        space = json.loads(raw)
        assert isinstance(space, dict) and space
        mod["search_space"] = space
    except (ValueError, AssertionError):
        st.error("The search space must be a non-empty JSON object.")
    c1, c2, c3, c4 = st.columns(4)
    mod["n_trials"] = int(c1.number_input("Trials", 1, 5000, int(mod.get("n_trials", 50)), key=ui.wkey("mod.nt")))
    mod["timeout_s"] = int(c2.number_input("Time limit (s)", 10, 7 * 86400, int(mod.get("timeout_s", 3600)),
                                           step=60, key=ui.wkey("mod.to")))
    pruners = ["hyperband", "median"]
    mod["pruner"] = c3.selectbox("Pruner", pruners, index=pruners.index(mod.get("pruner", "hyperband")),
                                 key=ui.wkey("mod.pr"))
    metrics = ["neg_log_loss", "f1"]
    mod["metric"] = c4.selectbox("Metric", metrics, index=metrics.index(mod.get("metric", "neg_log_loss")),
                                 key=ui.wkey("mod.metric"), help="f1 needs 0/1 meta-labels")
    st.caption(f"Up to {mod['n_trials']} trials → up to {mod['n_trials']} ledger trials in this family. "
               "Pruned and failed trials count too; only complete trials have fold scores.")
else:
    raw = st.text_area("Param grid (JSON; keys are estimator params)", json.dumps(mod["param_grid"], indent=1),
                       key=ui.wkey("mod.grid"), height=120)
    try:
        grid = json.loads(raw)
        assert isinstance(grid, dict) and all(isinstance(v, list) for v in grid.values())
        mod["param_grid"] = grid
        n_cand = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        st.caption(f"{n_cand} candidates → {n_cand} trials in this family")
    except (ValueError, AssertionError):
        st.error("Param grid must be a JSON object of lists.")

c1, c2, c3 = st.columns(3)
mod["cv_splits"] = int(c1.number_input("PurgedKFold splits", 2, 20, int(mod["cv_splits"]), key=ui.wkey("mod.k")))
mod["pct_embargo"] = c2.number_input("Embargo", 0.0, 0.49, float(mod["pct_embargo"]), format="%g",
                                     key=ui.wkey("mod.emb"))
if not optuna_mode:
    mod["rnd_search_iter"] = int(c3.number_input("Random search iterations (0 = full grid)", 0, 10_000,
                                                 int(mod["rnd_search_iter"]), key=ui.wkey("mod.rnd")))
    with st.expander("Bagging"):
        c1, c2, c3 = st.columns(3)
        mod["bagging_n_estimators"] = int(c1.number_input("bagging_n_estimators", 0, 10_000,
                                                          int(mod["bagging_n_estimators"]), key=ui.wkey("mod.bn")))
        mod["bagging_max_samples"] = c2.number_input("bagging_max_samples", 0.01, 1.0,
                                                     float(mod["bagging_max_samples"]), key=ui.wkey("mod.bs"),
                                                     help="Set to the average uniqueness from the Labels page")
        mod["bagging_max_features"] = c3.number_input("bagging_max_features", 0.01, 1.0,
                                                      float(mod["bagging_max_features"]), key=ui.wkey("mod.bf"))
    st.caption("Folds are fit with the fit weights and scored with the return-attribution weights events['w'].")
if mod.get("window_fraction"):
    st.caption(f"Split mode: using window fraction {mod['window_fraction']} (the part importance did not see).")

if supported:
    ui.launch("model", "Run Optuna study" if optuna_mode else "Run grid search")

# --------------------------------------------------------------------------- results


def selection_check(rows, split_cols):
    """Warn when the winner is inside the noise floor, and name the one-SE choice."""
    if len(rows) < 2:
        return
    best, second = rows.iloc[0], rows.iloc[1]
    floor = noise_floor(best[split_cols].to_numpy(dtype=float))
    gap = best["mean_test_score"] - second["mean_test_score"]
    if not (np.isfinite(floor) and gap < floor):
        return
    st.warning(f"The best candidate beats the runner-up by {gap:.4f}, less than the noise floor "
               f"{floor:.4f} (fold standard error). The ranking is not informative.")
    results = [{"params": p, "fold_scores": r[split_cols].to_numpy(dtype=float)}
               for p, (_, r) in zip(rows["params_dict"], rows.iterrows())]
    try:
        chosen, diag = one_standard_error_selection(results, P.complexity)
        st.info(f"One-standard-error rule: {diag['n_within_1se']} candidates are within one standard error "
                f"of the best; the simplest is `{chosen['params']}`.")
    except ValueError as exc:
        st.caption(f"One-standard-error rule not applied: {exc}")


def parse_params(text):
    try:
        return json.loads(text.replace("'", '"').replace("True", "true").replace("False", "false")
                          .replace("None", "null"))
    except ValueError:
        return {}


st.divider()
st.subheader("Results")
run = ui.run_picker("model")
if run:
    s = run["summary"]["model"]
    ui.family_panel(run["family_id"], s["best_score"], f"Best CV {s['scoring']}", metric=s["scoring"])
    res = ui.table(run, "cv_results")
    split_cols = [c for c in res.columns if c.startswith("split")] if res is not None else []

    if s.get("search") == "optuna":
        st.caption(f"Best trial {s['best_trial']}: `{s['best_params']}` · {s['n_trials']} trials "
                   f"({s['n_complete']} complete, {s['n_pruned']} pruned, {s['n_failed']} failed) · "
                   f"{s['n_samples']:,} samples")
        hist = ui.table(run, "optuna_trials")
        if hist is not None:
            done = hist[hist["state"] == "COMPLETE"].sort_values("number")
            fig = go.Figure()
            fig.add_scatter(x=done["number"], y=done["value"], mode="markers", name="complete trial")
            fig.add_scatter(x=done["number"], y=done["value"].cummax(), mode="lines", name="best so far")
            fig.update_layout(title="Optimization history", xaxis_title="trial", yaxis_title=s["scoring"])
            st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        imp = ui.table(run, "param_importance")
        if imp is not None:
            c1.plotly_chart(px.bar(imp.reset_index().sort_values("importance"), x="importance", y="param",
                                   orientation="h", title="Parameter importance (fANOVA)"),
                            use_container_width=True)
        if hist is not None:
            c2.plotly_chart(px.histogram(hist, x="state", title="Trial states"), use_container_width=True)
        if res is not None:
            res = res.sort_values("mean_test_score", ascending=False)
            res["params_dict"] = [parse_params(p) for p in res["params"]]
            st.dataframe(res.drop(columns="params_dict"), hide_index=True, use_container_width=True)
            top = res.head(10)
            long = top.melt(id_vars=["number"], value_vars=split_cols, var_name="fold", value_name="score")
            st.plotly_chart(px.strip(long, x="number", y="score", color="fold",
                                     title="Per-fold CV scores, top 10 trials"), use_container_width=True)
            selection_check(res, split_cols)
        with st.expander("Optuna Dashboard"):
            st.markdown("The study is stored in the run folder. Open it in Optuna Dashboard "
                        "(only inside the VPN: the dashboard has no sign-in):")
            st.code(f"optuna-dashboard {s['storage']}", language="bash")
            if caps["optuna_dashboard"]:
                if st.button("Start dashboard on port 8080", key=f"dash.{run['run_id']}"):
                    try:
                        from afml.production.optuna_dashboard import launch_optuna_dashboard

                        launch_optuna_dashboard(s["storage"], host="127.0.0.1", port=8080, force_restart=True)
                        st.success("Dashboard running at http://127.0.0.1:8080")
                    except Exception as exc:
                        st.error(f"Could not start the dashboard: {exc}")
            else:
                st.caption("Install it with `pip install optuna-dashboard` to start it from here.")
    else:
        st.caption(f"Best params: `{s['best_params']}` · {s['n_candidates']} candidates · {s['n_samples']:,} samples")
        if res is not None:
            res = res.sort_values("rank_test_score")
            res["params_dict"] = [parse_params(p) for p in res["params"]]
            st.dataframe(res.drop(columns="params_dict"), hide_index=True, use_container_width=True)
            long = res.melt(id_vars=["params"], value_vars=split_cols, var_name="fold", value_name="score")
            st.plotly_chart(px.strip(long, x="params", y="score", color="fold", title="Per-fold CV scores"),
                            use_container_width=True)
            selection_check(res, split_cols)
