"""Linked dashboard: UMAP scatter <-> cluster panel <-> KG subgraph (PyVis) <-> explanation
card, plus a chat box routed to the harness agents.

Static views read Postgres / Neo4j (GraphML fallback) directly; only the "Generate rationale"
button and the chat call an LLM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # streamlit puts ui/ on the path, not the repo root
load_dotenv(ROOT / ".env")

from swm import db, explain, kg  # noqa: E402

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
TYPE_COLORS = {
    "Person": "#e15759", "Organization": "#4e79a7", "Location": "#59a14f",
    "Event": "#f28e2b", "Artifact": "#b07aa1", "Topic": "#9c755f",
}

st.set_page_config(page_title="Explainable News Clustering", layout="wide")
st.title("Explainable News Clustering")

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# Data access (cached)
# ==========================================================

@st.cache_data(ttl=60)
def load_runs() -> pd.DataFrame:
    rows = db.query("SELECT run_id, metrics FROM runs ORDER BY (metrics->>'nmi')::float DESC NULLS LAST")
    return pd.DataFrame([{"run_id": r["run_id"], **r["metrics"]} for r in rows])


@st.cache_data
def load_points(run_id: str) -> pd.DataFrame:
    return pd.DataFrame(db.query(
        """SELECT dc.doc_id, dc.cluster_id, dc.is_noise, dc.x, dc.y, d.label_name
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s""",
        (run_id,),
    ))


@st.cache_data(ttl=60)
def load_clusters(run_id: str) -> dict[int, dict]:
    return {r["cluster_id"]: r for r in db.query(
        "SELECT cluster_id, size, keywords, label FROM cluster_info WHERE run_id = %s ORDER BY cluster_id",
        (run_id,),
    )}


@st.cache_data(ttl=300)
def load_subgraph(run_id: str, cluster_id: int):
    return kg.cluster_subgraph(run_id, cluster_id)


def doc_text(doc_id: str) -> dict | None:
    rows = db.query("SELECT text, label_name FROM documents WHERE id = %s", (doc_id,))
    return rows[0] if rows else None


# ==========================================================
# Views
# ==========================================================

def cluster_name(c: int, info: dict) -> str:
    return f"{c} — {info['label'] or ', '.join(t for t, _ in info['keywords'][:3])}"


def pyvis_html(nodes: list[dict], edges: list[dict]) -> str:
    from pyvis.network import Network

    net = Network(height="460px", width="100%", directed=True, cdn_resources="in_line")
    # Many disconnected entities: strong central gravity keeps the layout compact so the
    # auto-fit zoom doesn't shrink labels to nothing.
    net.barnes_hut(gravity=-2500, central_gravity=1.2, spring_length=90)
    for n in nodes:
        net.add_node(n["id"], label=n["name"], title=f'{n["type"]} · {n["freq"]} docs',
                     color=TYPE_COLORS.get(n["type"], "#999999"),
                     # explicit size: vis.js hides labels of small `value`-scaled nodes
                     size=10 + 3 * min(n["freq"], 10), font={"size": 14})
    ids = {n["id"] for n in nodes}
    for e in edges:
        if e["source"] in ids and e["target"] in ids:
            net.add_edge(e["source"], e["target"], title=e["relation"], label=e["relation"])
    return net.generate_html()


def scatter(run_id: str, points: pd.DataFrame) -> None:
    st.subheader("Embedding map")
    df = points.assign(cluster=points["cluster_id"].astype(str))
    fig = px.scatter(
        df, x="x", y="y", color="cluster", custom_data=["doc_id"],
        hover_data={"doc_id": True, "label_name": True, "x": False, "y": False},
        render_mode="webgl", height=560,
    )
    fig.update_traces(marker={"size": 3})
    fig.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      xaxis={"visible": False}, yaxis={"visible": False})
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                            key=f"scatter-{run_id}", use_container_width=True)
    if event and event.selection.points:
        st.session_state.doc_id = event.selection.points[0]["customdata"][0]


def cluster_panel(run_id: str, clusters: dict[int, dict], cluster_id: int) -> None:
    info = clusters[cluster_id]
    st.markdown(f"**{info['label'] or 'Unlabeled (run 05_kg.py)'}** · {info['size']} docs")
    st.bar_chart(pd.DataFrame(info["keywords"][:10], columns=["term", "weight"]).set_index("term"),
                 horizontal=True, height=260, sort=False)
    nodes, edges, source = load_subgraph(run_id, cluster_id)
    if nodes:
        components.html(pyvis_html(nodes, edges), height=480)
        st.caption(f"Knowledge graph source: {source}")
    else:
        st.info("No knowledge graph for this cluster yet; run scripts/05_kg.py for this run.")


def explanation_card(run_id: str, doc_id: str) -> None:
    try:
        ev = explain.evidence(doc_id, run_id)
    except LookupError as exc:
        st.info(str(exc))
        return

    st.markdown(f"**{doc_id}** → cluster {ev['cluster_id']}"
                + (" *(noise, assigned to nearest centroid)*" if ev["is_noise"] else ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Cosine to centroid", f"{ev['own_centroid_sim']:.2f}")
    c2.metric("Margin vs runner-up", "–" if ev["margin"] is None else f"{ev['margin']:.2f}",
              help=f"Runner-up cluster: {ev['runner_up_cluster']}")
    c3.metric("Membership prob.", "–" if ev["membership_prob"] is None else f"{ev['membership_prob']:.2f}")

    hits = ", ".join(f"`{h['term']}`×{h['count']}" for h in ev["term_hits"][:10])
    st.markdown(f"**Cluster terms in this document:** {hits or 'none'}")
    st.markdown("**Nearest exemplars:** " + (", ".join(
        f"`{e['doc_id']}` ({e['similarity']:.2f})" for e in ev["exemplars"]) or "none"))

    entities = kg.doc_entities(doc_id)
    if entities:
        st.markdown("**KG entities:** " + ", ".join(entities))

    cached = explain.cached_rationale(doc_id, run_id)
    st.markdown("**Rationale**")
    st.write(cached or explain.template_rationale(ev, entities))
    if not cached and st.button("Generate LLM rationale", key=f"rationale-{doc_id}"):
        with st.spinner("Asking the LLM (validated against the evidence)..."):
            st.write(explain.rationale(doc_id, run_id, kg_entities=entities))

    if doc := doc_text(doc_id):
        with st.expander(f"Document text ({doc['label_name']})"):
            st.text(doc["text"][:5000])


def dashboard() -> None:
    runs = load_runs()
    if runs.empty:
        st.info("No clustering runs yet. Run scripts/01_ingest.py → 03_cluster.py.")
        return

    with st.sidebar:
        st.header("Run")
        run_ids = runs["run_id"].tolist()
        default = db.default_run()
        run_id = st.selectbox("Clustering run", run_ids,
                              index=run_ids.index(default) if default in run_ids else 0)
        with st.expander("All runs"):
            st.dataframe(runs.set_index("run_id").round(3))

    metrics = runs.set_index("run_id").loc[run_id]
    cols = st.columns(5)
    for col, key in zip(cols, ("nmi", "ari", "purity", "silhouette", "noise_rate")):
        value = metrics.get(key)
        col.metric(key.upper() if key != "noise_rate" else "Noise", "–" if pd.isna(value) else f"{value:.3f}")

    points, clusters = load_points(run_id), load_clusters(run_id)
    left, right = st.columns([3, 2])
    with left:
        scatter(run_id, points)
        doc_id = st.text_input("Document id", value=st.session_state.get("doc_id", ""),
                               placeholder="click a point or type e.g. train/sci.space/60151")
        st.session_state.doc_id = doc_id.strip()

    selected = points.loc[points["doc_id"] == st.session_state.doc_id, "cluster_id"]
    options = sorted(clusters)
    with right:
        st.subheader("Cluster")
        if options:
            cluster_id = st.selectbox(
                "Cluster", options, format_func=lambda c: cluster_name(c, clusters[c]),
                index=options.index(int(selected.iloc[0])) if not selected.empty else 0,
            )
            cluster_panel(run_id, clusters, cluster_id)

    st.subheader("Why is this document here?")
    if st.session_state.doc_id:
        explanation_card(run_id, st.session_state.doc_id)
    else:
        st.caption("Select a point in the map to see its explanation.")


# ==========================================================
# Chat (ported from the harness UI)
# ==========================================================

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


def chat() -> None:
    st.subheader("Ask the agents")
    healthy = backend_healthy()
    if not healthy:
        st.warning(f"Agent API unavailable at {API_BASE_URL}; start uvicorn src.api.app:app.")
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
                    msg = {"role": "assistant", "content": resp.get("answer", ""),
                           "agent": resp.get("agent"), "metadata": resp.get("metadata", {})}
            st.session_state.messages.append(msg)
            render_assistant(msg)


with st.sidebar:
    st.header("Agents")
    st.markdown("\n".join(f"- **{v}**" for k, v in AGENT_LABELS.items() if k != "none"))
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

try:
    dashboard()
except Exception as exc:  # Postgres down / schema missing: keep the chat usable
    st.error(f"Dashboard unavailable: {exc}")

chat()
