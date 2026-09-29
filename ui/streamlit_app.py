"""Linked dashboard: UMAP scatter <-> cluster panel <-> KG subgraph (PyVis) <-> explanation
card, plus a chat box routed to the harness agents.

Starting point: extend the harness's existing ui/streamlit_app.py rather than rewriting it.
"""

import streamlit as st

st.set_page_config(page_title="Explainable News Clustering", layout="wide")
st.title("Explainable News Clustering")
st.caption("Foundation UI stub — wire up scatter / cluster panel / KG view / explanation card here.")
