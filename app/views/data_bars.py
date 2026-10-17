"""1 · Data and bars."""

from datetime import date

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import studio_ui as ui
from afml.studio import registry, vault

st.title("1 · Data and bars")
d = ui.draft()["data"]
led = ui.ledger()

st.subheader("Data spec")
c1, c2, c3 = st.columns(3)
d["symbol"] = c1.text_input("Symbol", d["symbol"], key=ui.wkey("data.symbol")).strip()
sources = registry.names("source")
d["source"] = c2.selectbox("Source", sources, index=sources.index(d["source"]) if d["source"] in sources else 0,
                           key=ui.wkey("data.source"),
                           format_func=lambda n: f"{n} — {registry.get('source', n).description}"[:80])
d["account"] = c3.text_input("Account (MT5)", d.get("account") or "", key=ui.wkey("data.account")) or None

with st.expander("Source parameters", expanded=d["source"] != "mt5_ticks"):
    d["source_params"] = ui.param_form("source", d["source"], d.get("source_params", {}), "data.src")

c1, c2, c3 = st.columns(3)
bar_types = ["tick", "time", "volume", "dollar"]
d["bar_type"] = c1.selectbox("Bar type", bar_types, index=bar_types.index(d["bar_type"]), key=ui.wkey("data.bt"))
raw_size = c2.text_input("Bar size", str(d["bar_size"]), key=ui.wkey("data.bs"),
                         help="An integer (ticks, volume, dollars) or a timeframe such as M5 for dynamic tick bars")
d["bar_size"] = int(raw_size) if raw_size.isdigit() else raw_size
prices = ["bid", "ask", "mid_price", "bid_ask"]
d["price"] = c3.selectbox("Price", prices, index=prices.index(d["price"]), key=ui.wkey("data.px"))

# --------------------------------------------------------------------------- vault

boundary = vault.get_research_end(d["symbol"]) if d["symbol"] else None
st.subheader("Research window")
if boundary is None:
    st.error(f"No research_end for {d['symbol']}: the vault refuses every run on this symbol until one is set.")
    with st.form("set_research_end"):
        st.markdown("Pick a date after the latest bar you have **ever** examined for this symbol. "
                    "It is set once and never moved; bars after it are the hold-out.")
        new_end = st.date_input("research_end", value=date.today())
        sure = st.checkbox("I understand this date can never be moved")
        if st.form_submit_button("Set research_end", type="primary"):
            if not sure:
                st.warning("Tick the confirmation first.")
            else:
                try:
                    vault.set_research_end(d["symbol"], new_end)
                    st.rerun()
                except vault.VaultError as exc:
                    st.error(str(exc))
else:
    st.caption(f"research_end for {d['symbol']}: **{boundary}**. The date pickers cannot pass it.")

c1, c2 = st.columns(2)
with c1:
    d["start_date"] = ui.date_input("Start", d["start_date"], "data.start", max_value=boundary)
with c2:
    d["end_date"] = ui.date_input("End", d["end_date"], "data.end", max_value=boundary)

ui.launch("bars", "Build bars")

# --------------------------------------------------------------------------- results

st.divider()
st.subheader("Bars")
run = ui.run_picker("bars")
if run:
    s = run["summary"]["bars"]
    m = st.columns(4)
    m[0].metric("Bars", f"{s['n_bars']:,}")
    m[1].metric("Bars per day", f"{s['bars_per_day_mean']:.1f}")
    m[2].metric("Return std", f"{s['ret_std']:.2e}")
    m[3].metric("Min d with ADF p < 5%", ui.fmt(s["min_d_stationary"], 2))

    bars = ui.table(run, "bars")
    if bars is not None:
        step = max(1, len(bars) // 5000)
        st.plotly_chart(px.line(bars["close"].iloc[::step], title="Close"), use_container_width=True)
    c1, c2 = st.columns(2)
    hist = ui.table(run, "return_hist")
    if hist is not None:
        fig = go.Figure(go.Bar(x=(hist["left"] + hist["right"]) / 2, y=hist["count"]))
        fig.update_layout(title="Log-return distribution", xaxis_title="log return", yaxis_title="bars")
        c1.plotly_chart(fig, use_container_width=True)
    adf = ui.table(run, "adf_by_d")
    if adf is not None:
        fig = go.Figure()
        fig.add_scatter(x=adf["d"], y=adf["adf_stat"], name="ADF statistic", mode="lines+markers")
        fig.add_scatter(x=adf["d"], y=adf["corr_with_log_close"], name="corr with log close",
                        mode="lines+markers", yaxis="y2")
        fig.update_layout(title="ADF of fractionally differentiated log close",
                          xaxis_title="d", yaxis2=dict(overlaying="y", side="right", range=[0, 1.05]))
        c2.plotly_chart(fig, use_container_width=True)
        st.dataframe(adf, hide_index=True, use_container_width=True)
    per_day = ui.table(run, "bars_per_day")
    if per_day is not None:
        st.plotly_chart(px.bar(per_day, y="bars", title="Bars per day"), use_container_width=True)
