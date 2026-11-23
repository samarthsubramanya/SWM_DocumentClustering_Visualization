# src/api/routes/explore.py
"""Read-only data for the web dashboard (web/): runs, the map, clusters, posts, reports.

The Streamlit app (ui/) reads Postgres and Neo4j directly; a browser app cannot, so this
router serves the same views over HTTP. Posts are always addressed by their opaque
dashboard id (doc-3f9a1c2e); real ids and newsgroup names, which give the gold label away,
are only returned when the caller asks for them with ?gold=true.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from pipeline import db, explain, kg

router = APIRouter(prefix="/api/v1/explore", tags=["explore"])

REPORTS = Path(__file__).resolve().parents[4] / "EVALUATIONS"
STAMP = re.compile(r"(\d{8}-\d{6})")
# Postal addresses from signatures and catalogue listings: kept in the graph, hidden in the view.
ADDRESS = re.compile(r"\b\d{5}(-\d{4})?\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b|,\s*[a-z]{2}\s+\d", re.I)
GRAPH_NODES = 25


# ==========================================================
# Runs and the map
# ==========================================================

@router.get("/runs")
def runs() -> dict[str, Any]:
    rows = db.query("SELECT run_id, metrics FROM runs ORDER BY (metrics->>'nmi')::float DESC NULLS LAST")
    return {"default": db.default_run(), "runs": [{"run_id": r["run_id"], "metrics": r["metrics"]} for r in rows]}


@router.get("/runs/{run_id}/points")
def points(run_id: str, gold: bool = False) -> dict[str, Any]:
    """Every post's 2-D position and cluster, as parallel arrays (17k rows stay small)."""
    rows = db.query(
        """SELECT dc.doc_id, dc.cluster_id, dc.is_noise, dc.x, dc.y, d.label_name
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s ORDER BY dc.doc_id""",
        (run_id,),
    )
    if not rows:
        raise HTTPException(404, f"No posts for run {run_id!r}.")
    out = {
        "ids": [db.display_id(r["doc_id"]) for r in rows],
        "x": [round(r["x"], 4) for r in rows],
        "y": [round(r["y"], 4) for r in rows],
        "cluster": [r["cluster_id"] for r in rows],
        "noise": [bool(r["is_noise"]) for r in rows],
    }
    if gold:
        out["newsgroup"] = [r["label_name"] for r in rows]
    return out


@router.get("/runs/{run_id}/clusters")
def clusters(run_id: str, gold: bool = False) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT cluster_id, size, keywords, label FROM cluster_info WHERE run_id = %s ORDER BY cluster_id",
        (run_id,),
    )
    mix = _newsgroup_mix(run_id) if gold else {}
    return [{"cluster_id": r["cluster_id"], "size": r["size"], "label": r["label"],
             "keywords": r["keywords"][:15], "newsgroups": mix.get(r["cluster_id"], [])} for r in rows]


def _newsgroup_mix(run_id: str) -> dict[int, list[dict]]:
    rows = db.query(
        """SELECT dc.cluster_id, d.label_name, count(*) AS n
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s
           GROUP BY 1, 2 ORDER BY 1, 3 DESC""",
        (run_id,),
    )
    mix: dict[int, list[dict]] = {}
    for r in rows:
        mix.setdefault(r["cluster_id"], []).append({"newsgroup": r["label_name"], "posts": r["n"]})
    return {c: groups[:3] for c, groups in mix.items()}


# ==========================================================
# A cluster's knowledge graph
# ==========================================================

@router.get("/runs/{run_id}/clusters/{cluster_id}/graph")
def cluster_graph(run_id: str, cluster_id: int, post: str | None = None) -> dict[str, Any]:
    """The cluster's most mentioned entities and the relations among them. With ?post=doc-…,
    the entities that post names are kept and flagged, so the view can ring them."""
    nodes, edges, source = kg.cluster_subgraph(run_id, cluster_id)
    named = set()
    if post and (doc_id := db.resolve_display_id(post)):
        link = _link(run_id, cluster_id, doc_id)
        named = set(link["mentions"]) | set(link["path"] or []) if link else set()

    kept = sorted((n for n in nodes if not ADDRESS.search(n["name"])), key=lambda n: -n["freq"])[:GRAPH_NODES]
    kept += [n for n in nodes if n["id"] in named and n not in kept]
    ids = {n["id"] for n in kept}
    links = [e for e in edges if e["source"] in ids and e["target"] in ids]
    connected = {e["source"] for e in links} | {e["target"] for e in links}
    if len(connected) >= 5:  # enough structure to show: isolated, rare entities only add clutter
        kept = [n for n in kept if n["id"] in connected or n["freq"] >= 3 or n["id"] in named]
    return {
        "source": source,
        "hidden": len(nodes) - len(kept),
        "nodes": [{**n, "named_by_post": n["id"] in named} for n in kept],
        "edges": [e for e in links if e["source"] in {n["id"] for n in kept} and e["target"] in {n["id"] for n in kept}],
    }


@lru_cache(maxsize=64)
def _graph(run_id: str, cluster_id: int):
    return kg.load_graphml(run_id, cluster_id)


def _link(run_id: str, cluster_id: int, doc_id: str) -> dict | None:
    G = _graph(run_id, cluster_id)
    rows = db.query("SELECT text FROM documents WHERE id = %s", (doc_id,))
    return kg.path_to_core(G, doc_id, rows[0]["text"]) if G is not None and rows else None


# ==========================================================
# One post: evidence, path to the cluster's core, rationale
# ==========================================================

def _doc(post: str) -> str:
    if not (doc_id := db.resolve_display_id(post)):
        raise HTTPException(404, f"No post {post!r}.")
    return doc_id


@router.get("/runs/{run_id}/posts/{post}")
def post_detail(run_id: str, post: str, gold: bool = False) -> dict[str, Any]:
    doc_id = _doc(post)
    try:
        ev = explain.evidence(doc_id, run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc).replace(doc_id, post)) from exc

    entities = kg.doc_entities(doc_id)
    link = _link(run_id, ev["cluster_id"], doc_id)
    doc = db.query("SELECT text, label_name FROM documents WHERE id = %s", (doc_id,))[0]
    rationale = explain.cached_rationale(doc_id, run_id)
    return {
        "post": post,
        "cluster_id": ev["cluster_id"],
        "is_noise": ev["is_noise"],
        "own_centroid_sim": ev["own_centroid_sim"],
        "margin": ev["margin"],
        "runner_up_cluster": ev["runner_up_cluster"],
        "membership_prob": ev["membership_prob"],
        "term_hits": ev["term_hits"][:10],
        "exemplars": [{"post": db.display_id(e["doc_id"]), "similarity": e["similarity"]} for e in ev["exemplars"]],
        "entities": entities,
        "path": _path_steps(run_id, ev["cluster_id"], doc_id, link) if link else None,
        "rationale": rationale or explain.template_rationale(ev, entities),
        "rationale_from_llm": rationale is not None,
        "text": doc["text"][:5000],
        "newsgroup": doc["label_name"] if gold else None,
    }


def _path_steps(run_id: str, cluster_id: int, doc_id: str, link: dict) -> dict[str, Any]:
    """kg.path_to_core as display steps: [{kind, name, relation, direction}, ...]."""
    G = _graph(run_id, cluster_id)
    core = [G.nodes[n]["name"] for n in link["core"]]

    def step(node: str) -> dict:
        a = G.nodes[node]
        if a["kind"] == "document":
            return {"kind": "post", "name": "this post" if a["name"] == doc_id else db.display_id(a["name"])}
        return {"kind": "entity", "name": a["name"], "type": a.get("type", "")}

    path = link["path"]
    steps = [step(n) for n in path] if path else []
    for i, (u, v) in enumerate(zip(path or [], (path or [])[1:]), start=1):
        forward = G.has_edge(u, v)
        steps[i]["relation"] = (G.edges[u, v] if forward else G.edges[v, u]).get("relation", "")
        steps[i]["direction"] = "forward" if forward else "backward"
    starts_from_text = bool(path) and G.nodes[path[0]]["kind"] != "document"
    return {"core": core, "steps": steps, "named_in_text": starts_from_text,
            "names_any_entity": bool(link["mentions"])}


@router.post("/runs/{run_id}/posts/{post}/rationale")
def generate_rationale(run_id: str, post: str) -> dict[str, Any]:
    """Ask the LLM for a rationale; it is validated against the evidence and cached."""
    doc_id = _doc(post)
    text = explain.rationale(doc_id, run_id, kg_entities=kg.doc_entities(doc_id))
    return {"rationale": text, "rationale_from_llm": explain.cached_rationale(doc_id, run_id) is not None}


# ==========================================================
# Evaluation reports (EVALUATIONS/), no database needed
# ==========================================================

def _stamp(path: Path) -> str:
    m = STAMP.search(path.name)
    return m.group(1) if m else ""


@router.get("/reports")
def reports() -> list[dict[str, str]]:
    files = [p for p in REPORTS.rglob("*") if p.suffix in (".md", ".json")]
    return [{"name": p.relative_to(REPORTS).as_posix(), "kind": p.suffix[1:], "stamp": _stamp(p)}
            for p in sorted(files, key=_stamp, reverse=True)]


@router.get("/reports/{name:path}")
def report(name: str) -> dict[str, Any]:
    path = (REPORTS / name).resolve()
    if REPORTS.resolve() not in path.parents or not path.is_file():   # no path tricks: only files in EVALUATIONS/
        raise HTTPException(404, f"No report {name!r}.")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:   # some reports were written on Windows (cp1252)
        text = raw.decode("cp1252", errors="replace")
    return {"name": name, "kind": path.suffix[1:], "content": json.loads(text) if path.suffix == ".json" else text}
