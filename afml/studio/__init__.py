"""
AFML Studio core: UI-agnostic config schemas, function registry, job runner,
research ledger and hold-out vault.

The Streamlit app in ``app/`` only edits configs, launches jobs and reads results;
every pipeline rule lives here, and nothing here imports Streamlit.
"""
