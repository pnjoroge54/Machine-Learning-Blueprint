"""2 · Labels and weights."""


import plotly.express as px
import streamlit as st

import studio_ui as ui

st.title("2 · Labels and weights")
lab = ui.draft()["labels"]

st.subheader("Strategy and target")
c1, c2 = st.columns(2)
with c1:
    lab["strategy"] = ui.callable_picker("strategy", lab["strategy"], "Strategy (side)", "lab.strategy")
    st.caption("Chaining a completed primary run as the strategy needs LearnedStrategy.from_pipeline "
               "(newer afml); not available on this branch.")
with c2:
    lab["target"] = ui.callable_picker("target", lab["target"], "Target (barrier width)", "lab.target")

st.subheader("Triple barrier")
c1, c2, c3 = st.columns(3)
lab["profit_target"] = c1.number_input("profit_target (× target)", min_value=0.01,
                                       value=float(lab["profit_target"]), key=ui.wkey("lab.pt"))
lab["stop_loss"] = c2.number_input("stop_loss (× target)", min_value=0.01,
                                   value=float(lab["stop_loss"]), key=ui.wkey("lab.sl"))
lab["min_ret"] = c3.number_input("min_ret", min_value=0.0, value=float(lab["min_ret"]), format="%g",
                                 key=ui.wkey("lab.minret"))
c1, c2, c3 = st.columns(3)
units = ["days", "hours", "minutes", "num_bars"]
cur_unit, cur_val = next(iter(lab["max_holding_period"].items()))
unit = c1.selectbox("max_holding_period unit", units, index=units.index(cur_unit) if cur_unit in units else 0,
                    key=ui.wkey("lab.mhp.u"))
val = c2.number_input("max_holding_period", min_value=1, value=int(cur_val), step=1, key=ui.wkey("lab.mhp.v"))
lab["max_holding_period"] = {unit: int(val)}
with c3:
    lab["vertical_barrier_zero"] = st.toggle("vertical_barrier_zero", lab["vertical_barrier_zero"],
                                             key=ui.wkey("lab.vbz"))
    lab["filter_events"] = st.toggle("CUSUM filter at mean target", lab["filter_events"], key=ui.wkey("lab.cusum"))

st.subheader("Weights")
st.caption("Two weight series, never one: fit weights go to `estimator.fit`; "
           "score weights (return attribution, `events['w']`) are always used for CV scoring.")
c1, c2 = st.columns(2)
with c1:
    schemes = ["uniqueness", "time_decay", "none"]
    lab["train_weight_scheme"] = st.selectbox("Fit weights: train_weight_scheme", schemes,
                                              index=schemes.index(lab["train_weight_scheme"]),
                                              key=ui.wkey("lab.tws"))
    if lab["train_weight_scheme"] == "time_decay":
        lab["time_decay"] = st.slider("time_decay (last weight)", -1.0, 1.0, float(lab["time_decay"]),
                                      key=ui.wkey("lab.td"))
with c2:
    st.selectbox("Score weights: score_weight_source", ["return_attribution"], disabled=True,
                 key=ui.wkey("lab.sws"))

ui.launch("labels", "Build labels")

# --------------------------------------------------------------------------- results

st.divider()
st.subheader("Events")
run = ui.run_picker("labels")
if run:
    s = run["summary"]["labels"]
    m = st.columns(4)
    m[0].metric("Events", f"{s['n_events']:,}")
    m[1].metric("Model role (derived)", s["role"],
                help="Secondary (meta-labeling) when events carry a side from the strategy; never a toggle.")
    m[2].metric("Average uniqueness", ui.fmt(s["avg_uniqueness"], 3))
    m[3].metric("Median holding (h)", ui.fmt(s["holding_hours_median"], 1))

    c1, c2 = st.columns(2)
    bal = ui.table(run, "label_balance")
    if bal is not None:
        c1.plotly_chart(px.bar(bal.reset_index(), x="bin", y="count", title="Label balance"),
                        use_container_width=True)
    hold = ui.table(run, "holding_hours")
    if hold is not None:
        c2.plotly_chart(px.histogram(hold, x="hours", nbins=40, title="Holding time from t1 (hours)"),
                        use_container_width=True)
    conc = ui.table(run, "concurrency")
    if conc is not None:
        step = max(1, len(conc) // 5000)
        st.plotly_chart(px.line(conc.iloc[::step], y="concurrent", title="Concurrent events"),
                        use_container_width=True)

    w = ui.table(run, "weights")
    if w is not None:
        st.markdown("**Weights, by use** (separate axes on purpose)")
        c1, c2 = st.columns(2)
        fit_col, score_col = w.columns[0], w.columns[1]
        c1.plotly_chart(px.histogram(w, x=fit_col, nbins=40, title="Fit weights (train_weight_scheme)"),
                        use_container_width=True)
        c2.plotly_chart(px.histogram(w, x=score_col, nbins=40, title="Score weights (events['w'])"),
                        use_container_width=True)
