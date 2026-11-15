"""Linked dashboard: UMAP scatter <-> cluster panel <-> KG subgraph (PyVis) <-> explanation
card, plus a chat box routed to the harness agents.

Static views read Postgres / Neo4j (GraphML fallback) directly; only the "Generate rationale"
button and the chat call an LLM.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Postal addresses from signatures / catalogue listings: kept in the graph, hidden in the view.
ADDRESS_RE = re.compile(r"\b\d{5}(-\d{4})?\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b|,\s*[a-z]{2}\s+\d", re.I)
GRAPH_MAX_NODES = 25
REPORTS = ROOT / "eval" / "reports"
STAMP = re.compile(r"(\d{8}-\d{6})")
WEAK_MARGIN = 0.15

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


@st.cache_data
def load_mix(run_id: str) -> pd.DataFrame:
    """Gold newsgroup counts per cluster (shown only with the gold toggle on)."""
    return pd.DataFrame(db.query(
        """SELECT dc.cluster_id, d.label_name, count(*) AS n
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s GROUP BY 1, 2""",
        (run_id,),
    ))


@st.cache_data(ttl=300)
def load_subgraph(run_id: str, cluster_id: int):
    return kg.cluster_subgraph(run_id, cluster_id)


@st.cache_resource
def load_graph(run_id: str, cluster_id: int):
    return kg.load_graphml(run_id, cluster_id)


def show_gold() -> bool:
    return bool(st.session_state.get("show_gold", False))


def display_id(doc_id: str) -> str:
    """Opaque id for the UI: real ids (train/sci.space/60151) carry the gold newsgroup."""
    return doc_id if show_gold() else db.display_id(doc_id)


def doc_text(doc_id: str) -> dict | None:
    rows = db.query("SELECT text, label_name FROM documents WHERE id = %s", (doc_id,))
    return rows[0] if rows else None


# ==========================================================
# Views
# ==========================================================

def cluster_name(c: int, info: dict) -> str:
    return f"{c} — {info['label'] or ', '.join(t for t, _ in info['keywords'][:3])}"


def readable_subgraph(nodes: list[dict], edges: list[dict], keep=frozenset()) -> tuple[list[dict], list[dict], int]:
    """View-only filter: drop address-like entities, keep the most frequent ones and, when the
    graph has relations, drop entities without any. Nodes in `keep` (the selected post's
    entities) always stay. Returns (nodes, edges, n_hidden)."""
    kept = [n for n in nodes if not ADDRESS_RE.search(n["name"])]
    kept = sorted(kept, key=lambda n: -n["freq"])[:GRAPH_MAX_NODES]
    kept += [n for n in nodes if n["id"] in keep and n not in kept]
    ids = {n["id"] for n in kept}
    es = [e for e in edges if e["source"] in ids and e["target"] in ids]
    linked = {e["source"] for e in es} | {e["target"] for e in es}
    if len(linked) >= 5:  # enough structure to show; isolated nodes only add clutter
        kept = [n for n in kept if n["id"] in linked or n["freq"] >= 3 or n["id"] in keep]
    return kept, es, len(nodes) - len(kept)


def pyvis_html(nodes: list[dict], edges: list[dict], highlight=frozenset()) -> str:
    from pyvis.network import Network

    net = Network(height="460px", width="100%", directed=True, cdn_resources="in_line")
    # Many disconnected entities: strong central gravity keeps the layout compact so the
    # auto-fit zoom doesn't shrink labels to nothing.
    net.barnes_hut(gravity=-2500, central_gravity=1.2, spring_length=90)
    for n in nodes:
        color = TYPE_COLORS.get(n["type"], "#999999")
        mine = n["id"] in highlight   # named by the selected post: dark ring
        net.add_node(n["id"], label=n["name"], title=f'{n["type"]} · {n["freq"]} docs',
                     color={"background": color, "border": "#111111" if mine else color},
                     borderWidth=4 if mine else 1,
                     # explicit size: vis.js hides labels of small `value`-scaled nodes
                     size=10 + 3 * min(n["freq"], 10), font={"size": 16 if mine else 14})
    ids = {n["id"] for n in nodes}
    for e in edges:
        if e["source"] in ids and e["target"] in ids:
            net.add_edge(e["source"], e["target"], title=e["relation"], label=e["relation"])
    return net.generate_html()


def select_doc(run_id: str, points: pd.DataFrame, doc_id: str) -> None:
    """Select a post and its cluster. Call before the cluster selectbox is drawn."""
    st.session_state.doc_id = doc_id
    hit = points.loc[points["doc_id"] == doc_id, "cluster_id"]
    if not hit.empty:
        st.session_state[f"cluster-{run_id}"] = int(hit.iloc[0])


def apply_map_click(run_id: str, points: pd.DataFrame) -> None:
    """A click on the map selects that post and its cluster; read from the chart's state at the
    top of the run, so every view drawn below already reflects it."""
    state = st.session_state.get(f"scatter-{run_id}")
    picked = state.get("selection", {}).get("points", []) if state else []
    if picked and (doc := picked[0]["customdata"][0]) != st.session_state.get("map_pick"):
        st.session_state.map_pick = doc
        select_doc(run_id, points, doc)


def scatter(run_id: str, points: pd.DataFrame, clusters: dict[int, dict], cluster_id: int | None,
            doc_id: str) -> None:
    st.subheader("Embedding map")
    df = points.assign(cluster=points["cluster_id"].astype(str), doc=points["doc_id"].map(display_id))
    fig = px.scatter(
        df, x="x", y="y", color="cluster", custom_data=["doc_id"],
        hover_data={"doc": True, "cluster": True, "label_name": show_gold(), "doc_id": False, "x": False, "y": False},
        render_mode="webgl", height=560,
    )
    fig.update_traces(marker={"size": 3})
    if cluster_id is not None:  # the selected cluster stays bright, the rest fade
        fig.for_each_trace(lambda t: t.update(marker_opacity=1.0 if t.name == str(cluster_id) else 0.2))
        members = points.loc[points["cluster_id"] == cluster_id]
        if not members.empty and cluster_id in clusters:
            fig.add_annotation(x=members["x"].median(), y=members["y"].median(), showarrow=False,
                               text=cluster_name(cluster_id, clusters[cluster_id]),
                               bgcolor="rgba(255,255,255,0.85)", font={"size": 13, "color": "#111111"})
    row = points.loc[points["doc_id"] == doc_id]
    if not row.empty:
        fig.add_trace(go.Scattergl(
            x=row["x"], y=row["y"], mode="markers", customdata=[[doc_id]], hovertext=[display_id(doc_id)],
            hoverinfo="text", name="selected post", showlegend=False,
            # gold with a dark rim: visible on Streamlit's light and dark themes
            marker={"size": 18, "symbol": "star", "color": "#FFC627", "line": {"color": "#111111", "width": 1.5}},
        ))
    fig.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      xaxis={"visible": False}, yaxis={"visible": False})
    st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                    key=f"scatter-{run_id}", width="stretch")


def cluster_panel(run_id: str, clusters: dict[int, dict], cluster_id: int, highlight=frozenset()) -> None:
    info = clusters[cluster_id]
    st.markdown(f"**{info['label'] or 'Unlabeled (run 05_kg.py)'}** · {info['size']} docs")
    if show_gold() and not (mix := load_mix(run_id)).empty:
        top = mix[mix["cluster_id"] == cluster_id].nlargest(3, "n")
        st.caption("Newsgroups in this cluster (reference only): "
                   + " · ".join(f"{r.label_name} {r.n / info['size']:.0%}" for r in top.itertuples()))
    st.bar_chart(pd.DataFrame(info["keywords"][:10], columns=["term", "weight"]).set_index("term"),
                 horizontal=True, height=260, sort=False)
    nodes, edges, source = load_subgraph(run_id, cluster_id)
    if nodes:
        nodes, edges, hidden = readable_subgraph(nodes, edges, keep=highlight)
        components.html(pyvis_html(nodes, edges, highlight), height=480)
        legend = " ".join(f'<span style="color:{c}">●</span> {t}' for t, c in TYPE_COLORS.items())
        st.markdown(legend, unsafe_allow_html=True)
        st.caption(f"Top {len(nodes)} entities by document frequency ({hidden} more hidden: addresses, "
                   f"unconnected or rare). Node size = documents mentioning it. Source: {source}."
                   + (" Dark rings mark the entities the selected post names." if highlight else ""))
    else:
        st.info("No knowledge graph for this cluster yet; run scripts/05_kg.py for this run.")


def post_link(run_id: str, doc_id: str, cluster_id: int) -> tuple[Any, dict] | None:
    """(cluster graph, kg.path_to_core result) for the selected post, or None without a graph."""
    G, doc = load_graph(run_id, cluster_id), doc_text(doc_id)
    return (G, kg.path_to_core(G, doc_id, doc["text"])) if G is not None and doc else None


def path_markdown(G, path: list[str], doc_id: str) -> str:
    """'this post —mentions→ **nasa** ←mentions— doc-1a2b3c4d —mentions→ **ssrt**'"""
    def name(n):
        a = G.nodes[n]
        if a["kind"] == "document":
            return "this post" if a["name"] == doc_id else f"post `{display_id(a['name'])}`"
        return f"**{a['name']}**"

    out = [name(path[0]) if G.nodes[path[0]]["kind"] == "document" else f"this post names {name(path[0])}"]
    for u, v in zip(path, path[1:]):
        if G.has_edge(u, v):
            out.append(f" —{G.edges[u, v].get('relation', '')}→ {name(v)}")
        else:
            out.append(f" ←{G.edges[v, u].get('relation', '')}— {name(v)}")
    return "".join(out)


def link_markdown(G, link: dict, doc_id: str) -> str:
    core = ", ".join(G.nodes[n]["name"] for n in link["core"]) or "none"
    if link["path"]:
        tail = (" (a core entity itself)" if len(link["path"]) == 1 else "")
        return f"**Path to the cluster's core** ({core}): {path_markdown(G, link['path'], doc_id)}{tail}"
    if link["mentions"]:
        return f"**Path to the cluster's core** ({core}): none; the entities this post names are not connected to them."
    return f"**Path to the cluster's core** ({core}): this post names none of the cluster's graph entities."


def explanation_card(run_id: str, doc_id: str, link: tuple[Any, dict] | None = None) -> None:
    try:
        ev = explain.evidence(doc_id, run_id)
    except LookupError as exc:
        st.info(str(exc))
        return

    st.markdown(f"**{display_id(doc_id)}** → cluster {ev['cluster_id']}"
                + (" *(noise, assigned to nearest centroid)*" if ev["is_noise"] else ""))
    hits_n = len(ev["term_hits"])
    weak = ev["margin"] is not None and ev["margin"] < WEAK_MARGIN or hits_n <= 1
    strength = "weak" if weak else "strong"
    margin_txt = "-" if ev["margin"] is None else ("0.00" if abs(ev["margin"]) < 0.005 else f"{ev['margin']:.2f}")
    plural = "s" if hits_n != 1 else ""
    (st.warning if weak else st.success)(
        f"Evidence: **{strength}** ({hits_n} cluster term{plural} in the text, "
        f"margin {margin_txt}; weak = margin < {WEAK_MARGIN} or ≤ 1 term). "
        "Weak assignments are the ones most likely to change when evidence is removed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cosine to centroid", f"{ev['own_centroid_sim']:.2f}")
    c2.metric("Margin vs runner-up", margin_txt,
              help=f"Runner-up cluster: {ev['runner_up_cluster']}")
    if ev["membership_prob"] is not None:
        c3.metric("Membership prob.", f"{ev['membership_prob']:.2f}")
    else:
        c3.metric("Membership prob.", "n/a", help="Only density-based (HDBSCAN) runs give membership probabilities.")

    hits = ", ".join(f"`{h['term']}`×{h['count']}" for h in ev["term_hits"][:10])
    st.markdown(f"**Cluster terms in this document:** {hits or 'none'}")
    st.markdown("**Nearest exemplars:** " + (", ".join(
        f"`{display_id(e['doc_id'])}` ({e['similarity']:.2f})" for e in ev["exemplars"]) or "none"))

    entities = kg.doc_entities(doc_id)
    if entities:
        st.markdown("**KG entities:** " + ", ".join(entities))
    if link and link[1]["core"]:
        st.markdown(link_markdown(link[0], link[1], doc_id))

    cached = explain.cached_rationale(doc_id, run_id)
    st.markdown("**Rationale**")
    st.write(cached or explain.template_rationale(ev, entities))
    if not cached and st.button("Generate LLM rationale", key=f"rationale-{doc_id}"):
        with st.spinner("Asking the LLM (validated against the evidence)..."):
            st.write(explain.rationale(doc_id, run_id, kg_entities=entities))

    if doc := doc_text(doc_id):
        with st.expander("Document text"):
            if show_gold():
                st.caption(f"Gold newsgroup (reference only, never used by the pipeline): {doc['label_name']}")
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
        st.toggle("Show gold newsgroups", key="show_gold",
                  help="Reveal the 20 Newsgroups labels in ids and hovers. Off by default: the pipeline "
                       "never sees them, and the views should not either.")

    st.caption(f"Run: `{run_id}` · labels are used only for the scores below")
    metrics = runs.set_index("run_id").loc[run_id]
    keys = ("nmi", "ari", "purity", "silhouette", "dbcv", "noise_rate")
    names = {"noise_rate": "Noise", "dbcv": "DBCV"}
    helps = {"silhouette": "Assumes convex clusters; favours k-means / agglomerative.",
             "dbcv": "Density-based validity (no labels); favours HDBSCAN-style clusters."}
    for col, key in zip(st.columns(len(keys)), keys):
        value = metrics.get(key)
        col.metric(names.get(key, key.upper()), "–" if value is None or pd.isna(value) else f"{value:.3f}",
                   help=helps.get(key))

    points, clusters = load_points(run_id), load_clusters(run_id)
    options = sorted(clusters)
    key = f"cluster-{run_id}"   # the selectbox's state: set by map clicks and typed ids too
    st.session_state.setdefault("doc_id", "")
    if options and st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    apply_map_click(run_id, points)

    left, right = st.columns([3, 2])
    with left:
        shown = display_id(st.session_state["doc_id"]) if st.session_state.get("doc_id") else ""
        typed = st.text_input("Document", value=shown, placeholder="click a point, or paste a doc-… id").strip()
        if typed != shown:
            by_display = dict(zip(points["doc_id"].map(display_id), points["doc_id"]))
            select_doc(run_id, points, by_display.get(typed, typed))
        scatter(run_id, points, clusters, st.session_state.get(key), st.session_state.doc_id)

    doc_id = st.session_state.doc_id
    home = points.loc[points["doc_id"] == doc_id, "cluster_id"]
    link = post_link(run_id, doc_id, int(home.iloc[0])) if not home.empty else None
    with right:
        st.subheader("Cluster")
        if options:
            cluster_id = st.selectbox("Cluster", options, key=key,
                                      format_func=lambda c: cluster_name(c, clusters[c]))
            mine = set(link[1]["mentions"]) | set(link[1]["path"] or []) if link else set()
            cluster_panel(run_id, clusters, cluster_id,
                          frozenset(mine) if not home.empty and cluster_id == int(home.iloc[0]) else frozenset())

    st.subheader("Why is this document here?")
    if doc_id:
        explanation_card(run_id, doc_id, link)
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


def render_cypher(meta: dict[str, Any]) -> None:
    """The KG agent's query, its validation trail and the rows the answer was built from."""
    repairs = meta.get("validation") or []
    ok = "rows" in meta
    st.caption(f"Generated Cypher · {'valid' if ok else 'rejected'} after {meta.get('attempts', 1)} "
               f"attempt{'s' if meta.get('attempts', 1) != 1 else ''} · runs in a read-only transaction")
    st.code(meta["cypher"], language="cypher")
    if repairs:
        with st.expander(f"Rejected by validation ({len(repairs)})"):
            for r in repairs:
                st.code(r["cypher"], language="cypher")
                st.markdown("\n".join(f"- {p}" for p in r["problems"]))
    if ok:
        st.dataframe(pd.DataFrame(meta["rows"]), hide_index=True) if meta["rows"] else st.caption("No rows.")


def render_assistant(msg: dict[str, Any]) -> None:
    st.markdown(msg["content"])
    if msg.get("metadata", {}).get("cypher"):
        render_cypher(msg["metadata"])
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
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []

def evaluation() -> None:
    """The saved evaluation reports (eval/reports), newest first; no database needed."""
    runs = sorted(REPORTS.glob("text2cypher-*.json"), key=lambda p: STAMP.search(p.name).group(1))
    if runs:
        r = json.loads(runs[-1].read_text())
        st.subheader("Text2Cypher")
        st.caption(f"{runs[-1].name} · {r['n_questions']} gold questions, {r['n_informative']} scored "
                   "(gold answers that are empty or over 50 rows are excluded)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid on first try", f"{r['validity_first_try']:.0%}")
        c2.metric("Valid after repair", f"{r['validity']:.0%}")
        c3.metric("Execution accuracy", f"{r['execution_accuracy']:.3f}", help="Same rows as the gold Cypher.")
        st.dataframe(pd.DataFrame(r["results"])[["id", "pattern", "question", "attempts", "valid_first_try", "correct"]],
                     hide_index=True)

    reports = sorted(REPORTS.glob("*.md"), key=lambda p: STAMP.search(p.name).group(1) if STAMP.search(p.name) else "",
                     reverse=True)
    if not reports:
        st.info(f"No reports in {REPORTS}; run scripts/06_eval.py.")
        return
    st.subheader("Reports")
    chosen = st.selectbox("Report", reports, format_func=lambda p: p.name)
    raw = chosen.read_bytes()
    try:
        st.markdown(raw.decode("utf-8"))
    except UnicodeDecodeError:   # some reports were written on Windows (cp1252: "±" = 0xb1)
        st.markdown(raw.decode("cp1252", errors="replace"))


explore, ask, results = st.tabs(["Explore", "Ask the agents", "Evaluation"])
with explore:
    try:
        dashboard()
    except Exception as exc:  # Postgres down / schema missing: keep the other tabs usable
        st.error(f"Dashboard unavailable: {exc}")
with ask:
    chat()
with results:
    evaluation()