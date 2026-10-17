"""
AFML Studio entry point: ``python -m afml.studio app`` (or ``streamlit run app/Home.py``).

Expose it only inside a private mesh VPN such as Tailscale, never on a public port.
"""

import streamlit as st

st.set_page_config(page_title="AFML Studio", page_icon=":material/insights:", layout="wide")

import studio_ui as ui  # noqa: E402  (after set_page_config)
from afml.studio import paths  # noqa: E402

pages = {
    "Studio": [
        st.Page("views/control_room.py", title="Control room", icon=":material/dashboard:", default=True),
    ],
    "Research (AFML order)": [
        st.Page("views/data_bars.py", title="1 · Data and bars", icon=":material/candlestick_chart:"),
        st.Page("views/labels_weights.py", title="2 · Labels and weights", icon=":material/label:"),
        st.Page("views/feature_set.py", title="3 · Feature set", icon=":material/view_column:"),
        st.Page("views/feature_importance.py", title="4 · Feature importance", icon=":material/bar_chart:"),
        st.Page("views/model_development.py", title="5 · Model development", icon=":material/model_training:"),
    ],
    "After a config is frozen": [
        st.Page("views/validation.py", title="6 · Validation", icon=":material/verified:"),
        st.Page("views/holdout_export.py", title="7 · Hold-out and export", icon=":material/lock:"),
    ],
}

with st.sidebar:
    st.markdown("**AFML Studio**")
    ui.worker_status()
    st.caption(f"Home: `{paths.studio_home()}`")

st.navigation(pages).run()
