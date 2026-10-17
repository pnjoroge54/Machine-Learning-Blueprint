"""3 · Feature set."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import studio_ui as ui
from afml.studio import jobs, registry
from afml.studio.adapters.pipelines import capabilities

st.title("3 · Feature set")
fs = ui.draft()["features"]

st.subheader("Compose feature_configs")
steps = fs["steps"]
remove = None
for i, step in enumerate(steps):
    with st.expander(f"{i + 1}. {step['name']}" + (f"  (prefix {step['prefix']})" if step.get("prefix") else ""),
                     expanded=False):
        new = ui.callable_picker("feature", step, "Feature function", f"fs.{i}")
        new["prefix"] = st.text_input("Column prefix (optional)", step.get("prefix") or "",
                                      key=ui.wkey("fs", i, "prefix")) or None
        steps[i] = new
        if st.button("Remove step", key=ui.wkey("fs", i, "rm")):
            remove = i
if remove is not None:
    steps.pop(remove)
    ui._bump_form_version()
    st.rerun()

c1, c2 = st.columns([3, 1])
names = registry.names("feature")
to_add = c1.selectbox("Add a feature function", names, key=ui.wkey("fs.add"),
                      format_func=lambda n: f"{n} — {registry.get('feature', n).description}"[:90])
if c2.button("Add step", use_container_width=True):
    steps.append({"name": to_add, "params": registry.default_params("feature", to_add)})
    ui._bump_form_version()
    st.rerun()

if fs.get("selected_columns"):
    st.info(f"Saved feature set: keeping {len(fs['selected_columns'])} selected columns.")
    if st.button("Clear column selection"):
        fs["selected_columns"] = None
        st.rerun()

if steps:
    ui.launch("features", "Build feature matrix")
else:
    st.warning("Add at least one feature step.")

# --------------------------------------------------------------------------- results

st.divider()
st.subheader("Feature matrix")
run = ui.run_picker("features")
if run:
    s = run["summary"]["features"]
    m = st.columns(4)
    m[0].metric("Columns", s["n_columns"])
    m[1].metric("Rows", f"{s['n_rows']:,}")
    m[2].metric("Complete rows", f"{s['n_complete_rows']:,}")
    m[3].metric("Max NaN share", f"{s['max_nan_share']:.1%}")

    cat = ui.table(run, "catalog")
    if cat is not None:
        st.markdown("**Provenance catalog** (source to columns)")
        st.dataframe(cat, hide_index=True, use_container_width=True)
    nan = ui.table(run, "nan_share")
    if nan is not None:
        st.plotly_chart(px.bar(nan.reset_index(), x="column", y="nan_share", title="NaN share per column"),
                        use_container_width=True)
    desc = ui.table(run, "describe")
    if desc is not None:
        with st.expander("Describe"):
            st.dataframe(desc, use_container_width=True)

# --------------------------------------------------------------------------- screening

if capabilities()["screening"]:
    st.divider()
    st.subheader("Screen features")
    st.caption("Three gates in a fixed order: stationarity (ADF and KPSS), entropy, then mutual information "
               "with the page 2 labels under a permutation null, adjusted across the family by Westfall-Young. "
               "Set the thresholds once, before looking: tuning them until the screen selects what you hoped "
               "for makes its p-values worthless. Each screen is logged as a trial.")
    sc = ui.draft()["screening"]
    c1, c2, c3 = st.columns(3)
    targets = {"label": "Label (bin)", "event_return": "Event return (ret)"}
    sc["target"] = c1.selectbox("Target", list(targets), format_func=targets.get,
                                index=list(targets).index(sc["target"]), key=ui.wkey("sc.target"))
    schemes = {"circular_shift": "Circular shift", "stationary_bootstrap": "Stationary bootstrap"}
    sc["scheme"] = c2.selectbox("Null", list(schemes), format_func=schemes.get,
                                index=list(schemes).index(sc["scheme"]), key=ui.wkey("sc.scheme"),
                                help="Both keep the target's serial dependence. The naive shuffle is not "
                                     "offered: overlapping labels are autocorrelated, so it overstates evidence.")
    sc["n_perm"] = int(c3.number_input("Permutations", 50, 10_000, int(sc["n_perm"]), step=50,
                                       key=ui.wkey("sc.nperm")))
    c1, c2, c3, c4 = st.columns(4)
    sc["alpha"] = c1.number_input("alpha", 0.001, 0.49, float(sc["alpha"]), format="%g", key=ui.wkey("sc.alpha"))
    sc["min_entropy"] = c2.number_input("Min entropy", 0.0, 1.0, float(sc["min_entropy"]), key=ui.wkey("sc.ent"))
    sc["max_modal_occupancy"] = c3.number_input("Max modal share", 0.01, 1.0, float(sc["max_modal_occupancy"]),
                                                key=ui.wkey("sc.modal"))
    with c4:
        sc["skip_stationarity"] = st.toggle("Skip stationarity gate", sc["skip_stationarity"],
                                            key=ui.wkey("sc.skip"),
                                            help="For fractionally differenced features, where KPSS is the wrong test")
    if steps:
        ui.launch("screening", "Screen features")

    run = ui.run_picker("screening", "Screen from run")
    if run:
        s = run["summary"]["screening"]
        m = st.columns(4)
        m[0].metric("Features screened", s["n_features"])
        m[1].metric("Pass all three gates", s["n_selected"])
        m[2].metric("Effective tests", ui.fmt(s["effective_tests"], 1),
                    help="Independent-test equivalent of the MI family, from the joint null")
        m[3].metric("Events", f"{s['n_events']:,}")
        st.caption(f"Target {s['target']} · {s['scheme']} null"
                   + (f" (mean block {s['mean_block']:.0f} events)" if s.get("mean_block") else "")
                   + f" · {s['n_perm']} permutations · alpha {s['alpha']:g}")

        summ = ui.table(run, "screening_summary")
        if summ is not None:
            counts = summ["outcome"].map(lambda o: o.split(":")[0]).value_counts().rename_axis("outcome")
            c1, c2 = st.columns([1, 2])
            c1.plotly_chart(px.bar(counts.reset_index(), x="outcome", y="count", title="Where features stop"),
                            use_container_width=True)
            c2.dataframe(summ, use_container_width=True)
        mi = ui.table(run, "screening_mi")
        if mi is not None:
            mi = mi.sort_values("statistic")
            fig = go.Figure()
            fig.add_bar(x=mi["statistic"], y=mi.index, orientation="h", name="normalized MI")
            fig.add_scatter(x=mi["null_q95"], y=mi.index, mode="markers", name="null 95th percentile",
                            marker=dict(symbol="line-ns-open", size=14))
            fig.update_layout(title="Mutual information against its null", height=max(300, 26 * len(mi)))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Mutual-information detail"):
                st.dataframe(mi, use_container_width=True)

        if s["selected"]:
            if st.button(f"Keep the {len(s['selected'])} screened features (logs a trial)", type="primary",
                         key=f"sc.keep.{run['run_id']}"):
                ui.load_into_draft(jobs.save_feature_set(run["run_id"], s["selected"]))
                st.success(f"Saved {len(s['selected'])} columns into the draft. The family counts this as a trial.")
        else:
            st.info("No feature passed all three gates.")
