"""5 · Model development."""

import json

import numpy as np
import plotly.express as px
import streamlit as st

import studio_ui as ui
from afml.studio.adapters.pipelines import capabilities
from afml.studio.stats import noise_floor, one_standard_error_selection

st.title("5 · Model development")
mod = ui.draft()["model"]
caps = capabilities()
if not caps["optuna"]:
    st.info("Optuna is not installed in this environment, and FinancialModelSuggester is not on this branch, "
            "so this page runs a purged-CV grid search through clf_hyper_fit. Each grid candidate is one "
            "ledger trial. The embedded Optuna Dashboard arrives with the newer pipeline.")

st.subheader("Estimator and search space")
mod["estimator"] = ui.callable_picker("estimator", mod["estimator"], "Estimator", "mod.est")
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
mod["rnd_search_iter"] = int(c3.number_input("Random search iterations (0 = full grid)", 0, 10_000,
                                             int(mod["rnd_search_iter"]), key=ui.wkey("mod.rnd")))
with st.expander("Bagging"):
    c1, c2, c3 = st.columns(3)
    mod["bagging_n_estimators"] = int(c1.number_input("bagging_n_estimators", 0, 10_000,
                                                      int(mod["bagging_n_estimators"]), key=ui.wkey("mod.bn")))
    mod["bagging_max_samples"] = c2.number_input("bagging_max_samples", 0.01, 1.0, float(mod["bagging_max_samples"]),
                                                 key=ui.wkey("mod.bs"),
                                                 help="Set to the average uniqueness from the Labels page")
    mod["bagging_max_features"] = c3.number_input("bagging_max_features", 0.01, 1.0,
                                                  float(mod["bagging_max_features"]), key=ui.wkey("mod.bf"))
if mod.get("window_fraction"):
    st.caption(f"Split mode: using window fraction {mod['window_fraction']} (the part importance did not see).")
st.caption("On this branch clf_hyper_fit scores folds without the return-attribution weights; "
           "ModelDevelopmentPipeline in newer afml applies them.")

ui.launch("model", "Run model development")

# --------------------------------------------------------------------------- results

st.divider()
st.subheader("Results")
run = ui.run_picker("model")
if run:
    s = run["summary"]["model"]
    ui.family_panel(run["family_id"], s["best_score"], f"Best CV {s['scoring']}", metric=s["scoring"])
    st.caption(f"Best params: `{s['best_params']}` · {s['n_candidates']} candidates · {s['n_samples']:,} samples")

    res = ui.table(run, "cv_results")
    if res is not None:
        res = res.sort_values("rank_test_score")
        split_cols = [c for c in res.columns if c.startswith("split")]
        st.dataframe(res, hide_index=True, use_container_width=True)
        long = res.melt(id_vars=["params"], value_vars=split_cols, var_name="fold", value_name="score")
        st.plotly_chart(px.strip(long, x="params", y="score", color="fold", title="Per-fold CV scores"),
                        use_container_width=True)

        # Guardrail: is the winner distinguishable from the runner-up?
        if len(res) > 1:
            best, second = res.iloc[0], res.iloc[1]
            floor = noise_floor(best[split_cols].to_numpy(dtype=float))
            gap = best["mean_test_score"] - second["mean_test_score"]
            if np.isfinite(floor) and gap < floor:
                st.warning(f"The best candidate beats the runner-up by {gap:.4f}, less than the noise floor "
                           f"{floor:.4f} (fold standard error). The ranking is not informative.")
                means = res["mean_test_score"].to_numpy(dtype=float)
                ses = res[split_cols].std(axis=1, ddof=1).to_numpy() / np.sqrt(len(split_cols))
                order = np.argsort(res.index.to_numpy())  # grid order as a proxy for simplicity
                pick = one_standard_error_selection(means[order], ses[order])
                st.info("One-standard-error rule (grid order as the simplicity order): "
                        f"`{res.iloc[order[pick]]['params']}`")
