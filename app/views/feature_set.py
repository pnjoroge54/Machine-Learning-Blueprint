"""3 · Feature set."""

import plotly.express as px
import streamlit as st

import studio_ui as ui
from afml.studio import registry
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

if not capabilities()["screening"]:
    st.caption("Screening (stationarity, entropy, then mutual information under a circular-shift null "
               "with Westfall-Young adjustment) needs afml.screening.screen_features, which this branch "
               "does not have yet. The naive-shuffle null will not be offered when it lands.")

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
