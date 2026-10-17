"""7 · Hold-out and export (Phase 4)."""

import streamlit as st

import studio_ui as ui
from afml.studio import vault

st.title("7 · Hold-out and export")
st.info("Phase 4. One pre-registered hold-out evaluation per family, then ONNX export with "
        "validate_onnx_predictions parity before the MQL5 bundle is written.")

st.subheader("Hold-out boundaries")
ends = vault.all_research_ends()
if ends:
    st.dataframe([{"symbol": s, "research_end": str(d)} for s, d in ends.items()], use_container_width=True)
else:
    st.caption("No research_end dates set. Set them on page 1; until then the vault refuses every run.")

st.subheader("Families")
fams = ui.ledger().list_families()
st.dataframe([{"family": f["name"], "frozen": (f["frozen_config_hash"] or "")[:12],
               "contaminated": bool(f["contaminated"]), "reason": f["contaminated_reason"] or ""}
              for f in fams], use_container_width=True)
st.markdown(":lock: Hold-out runs and export are locked until Phase 4.")
