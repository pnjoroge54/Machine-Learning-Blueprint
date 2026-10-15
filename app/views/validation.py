"""6 · Validation (Phase 3): locked until a family's config is frozen."""

import streamlit as st

import studio_ui as ui

st.title("6 · Validation")
st.info("Phase 3. Validation can only reject: a backtest exists to discard models, not to improve them "
        "(AFML 11.4). This page stays locked until a family's config is frozen and pre-registered.")

frozen = [f for f in ui.ledger().list_families() if f["frozen_config_hash"]]
if not frozen:
    st.markdown(":lock: No family has a frozen config yet.")
else:
    st.dataframe([{"family": f["name"], "frozen config": f["frozen_config_hash"][:12],
                   "contaminated": bool(f["contaminated"])} for f in frozen], use_container_width=True)
    st.caption("The validation battery (signal permutation, CPCV paths, DSR, PBO, walk-forward, sequence risk, "
               "process permutation) lands in Phase 3, after the six defects in Section 10.1 are fixed.")
