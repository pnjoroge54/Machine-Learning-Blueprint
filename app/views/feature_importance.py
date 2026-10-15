"""4 · Feature importance."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import studio_ui as ui
from afml.studio import jobs
from afml.studio.adapters.pipelines import capabilities

st.title("4 · Feature importance")
st.caption("AFML 11.4: feature importance is the research tool. Saving a feature set is logged as a trial.")
imp = ui.draft()["importance"]
caps = capabilities()

st.subheader("Methods and cross-validation")
c1, c2 = st.columns([2, 1])
imp["methods"] = c1.multiselect("Methods", caps["importance_methods"],
                                default=[m for m in imp["methods"] if m in caps["importance_methods"]],
                                key=ui.wkey("imp.methods"))
c1.caption("Not on this branch yet: " + ", ".join(caps["missing_importance_methods"]))
scorings = ["neg_log_loss", "accuracy"]
imp["scoring"] = c2.selectbox("Scoring", scorings, index=scorings.index(imp["scoring"]), key=ui.wkey("imp.sc"))
c1, c2, c3 = st.columns(3)
imp["n_splits"] = int(c1.number_input("PurgedKFold splits", 2, 20, int(imp["n_splits"]), key=ui.wkey("imp.k")))
imp["pct_embargo"] = c2.number_input("Embargo (fraction)", 0.0, 0.49, float(imp["pct_embargo"]), format="%g",
                                     key=ui.wkey("imp.emb"))
with c3:
    imp["split_mode"] = st.toggle("Split mode", imp["split_mode"], key=ui.wkey("imp.split"),
                                  help="Importance on the first part of the window, model development on the "
                                       "rest, so model CV scores do not reuse the folds that chose features.")
    if imp["split_mode"]:
        imp["importance_fraction"] = st.slider("Importance fraction", 0.2, 0.8, float(imp["importance_fraction"]),
                                               key=ui.wkey("imp.frac"))
imp["estimator"] = ui.callable_picker("estimator", imp["estimator"], "Estimator", "imp.est")

if imp["methods"]:
    ui.launch("importance", "Run feature importance")
else:
    st.warning("Choose at least one method.")

# --------------------------------------------------------------------------- results

st.divider()
st.subheader("Importance")
run = ui.run_picker("importance")
if run:
    s = run["summary"]["importance"]
    ui.family_panel(run["family_id"], s["cv_score_mean"], f"CV {s['scoring']}", metric=s["scoring"])
    st.caption(f"{s['n_samples']:,} samples · {s['n_features']} features · CV std err "
               f"{s['cv_score_std_err']:.4f}" + (" · split mode" if s["split_mode"] else ""))

    methods = [m for m in ("mdi", "mda", "sfi") if ui.table(run, m) is not None]
    cols = st.columns(len(methods) or 1)
    for col, m in zip(cols, methods):
        t = ui.table(run, m).reset_index().sort_values("mean")
        fig = go.Figure(go.Bar(x=t["mean"], y=t["feature"], orientation="h",
                               error_x=dict(type="data", array=t["std_err"])))
        fig.update_layout(title=m.upper(), height=max(300, 22 * len(t)), margin=dict(l=10, r=10))
        col.plotly_chart(fig, use_container_width=True)

    rc = ui.table(run, "rank_compare")
    if rc is not None:
        rc = rc.reset_index()
        fig = px.scatter(rc, x="mdi_rank", y="mda_rank", text="feature", title="MDI vs MDA rank")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Clustered MDI/MDA (and the substitution-effect flag that compares clustered with "
               "unclustered rankings) need ONC clustering from newer afml.")

    folds = ui.table(run, "mda_folds")
    if folds is not None:
        st.plotly_chart(px.box(folds, x="feature", y="importance", title="MDA per fold"),
                        use_container_width=True)

    st.subheader("Save a feature set")
    ranked = (ui.table(run, "mda") if ui.table(run, "mda") is not None else ui.table(run, methods[0])).index.tolist()
    k = st.slider("Keep top k (by the first available of MDA, MDI, SFI)", 1, len(ranked), len(ranked),
                  key=f"imp.topk.{run['run_id']}")
    chosen = st.multiselect("Columns", ranked, default=ranked[:k], key=f"imp.cols.{run['run_id']}.{k}")
    st.caption("Keeping top-k MDA features and then reporting a CV score on the same folds reuses the "
               "errors that chose them. Prefer split mode for the model run.")
    if st.button("Save feature set (logs a trial)", type="primary", disabled=not chosen):
        cfg = jobs.save_feature_set(run["run_id"], chosen)
        ui.load_into_draft(cfg)
        msg = f"Saved {len(chosen)} columns into the draft. The family now counts this as a trial."
        if s["split_mode"]:
            frac = cfg["importance"]["importance_fraction"]
            ui.draft()["model"]["window_fraction"] = [frac, 1.0]
            msg += f" Model development will use the remaining {1 - frac:.0%} of the window."
        st.success(msg)
