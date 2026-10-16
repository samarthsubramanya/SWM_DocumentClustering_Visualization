"""Linked dashboard: UMAP scatter <-> cluster panel <-> KG subgraph (PyVis) <-> explanation
card, plus a chat box routed to the harness agents.

Chat/API client is ported from the harness's ui/streamlit_app.py; the views are placeholders.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

AGENT_LABELS = {
    "cluster": "Cluster",
    "explain": "Explain",
    "kg": "Knowledge Graph (Text2Cypher)",
    "ingest": "Ingest",
    "none": "No Suitable Agent",
}

st.set_page_config(page_title="Explainable News Clustering", layout="wide")
st.title("Explainable News Clustering")

if "messages" not in st.session_state:
    st.session_state.messages = []


def backend_healthy() -> bool:
    try:
        r = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


def send_query(query: str) -> dict[str, Any]:
    r = httpx.post(QUERY_ENDPOINT, json={"query": query}, timeout=None)
    r.raise_for_status()
    return r.json()


def render_assistant(msg: dict[str, Any]) -> None:
    st.markdown(msg["content"])
    if msg.get("agent"):
        st.caption(f"Handled by {AGENT_LABELS.get(msg['agent'], msg['agent'])}")
        with st.expander("Execution details"):
            st.json(msg.get("metadata", {}))
    if msg.get("request_id"):
        st.caption(f"Request ID: {msg['request_id']}")


healthy = backend_healthy()
(st.success if healthy else st.error)("Backend connected" if healthy else "Backend unavailable")

with st.sidebar:
    st.header("Agents")
    st.markdown("\n".join(f"- **{v}**" for k, v in AGENT_LABELS.items() if k != "none"))
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

scatter_col, detail_col = st.columns([3, 2])
with scatter_col:
    st.subheader("Embedding map")
    st.info("TODO(ui): UMAP scatter coloured by cluster; click a point -> select doc.")
with detail_col:
    st.subheader("Cluster / explanation")
    st.info("TODO(ui): cluster panel (label, keywords), KG subgraph (PyVis), explanation card.")

st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_assistant(m) if m["role"] == "assistant" else st.markdown(m["content"])

if prompt := st.chat_input("Ask about clusters, explanations or the graph...", disabled=not healthy):
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Working on your request..."):
            try:
                resp = send_query(prompt)
            except httpx.HTTPStatusError as exc:
                msg = {"role": "assistant", "content": f"Backend request failed (HTTP {exc.response.status_code})."}
            except httpx.RequestError:
                msg = {"role": "assistant", "content": "Could not communicate with the API."}
            else:
                msg = {
                    "role": "assistant",
                    "content": resp.get("answer", ""),
                    "agent": resp.get("agent"),
                    "metadata": resp.get("metadata", {}),
                    "request_id": resp.get("request_id"),
                }
        st.session_state.messages.append(msg)
        render_assistant(msg)
