"""ClusterAgent: "show cluster 3", "describe the braves cluster", "why is year in the braves
cluster?", "compare HDBSCAN vs k-means".

Wraps swm.cluster; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from swm import db

CLUSTER_REF = re.compile(r"cluster\s*#?\s*(\d+)", re.IGNORECASE)
COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|best run|which run|all runs|runs)\b", re.IGNORECASE)
# "why is year in the braves cluster", "why does 'nasa' appear in cluster 8"
WHY_TERM = re.compile(
    r"why\s+(?:is|are|does|do)\s+['\"“]?(?P<term>[\w .'’-]+?)['\"”]?\s+"
    r"(?:in|part of|appear in|show up in|belong to|a term (?:of|in))\s+(?P<ref>.+)",
    re.IGNORECASE,
)
# Words that name no cluster: matching "the braves cluster" should look at "braves" only.
NOISE = {"the", "a", "an", "of", "in", "on", "about", "cluster", "clusters", "group", "topic", "one",
         "show", "describe", "tell", "me", "what", "which", "is", "are", "why", "this", "that"}


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    return await asyncio.to_thread(_answer, query)


def _answer(query: str) -> dict[str, Any]:
    # ponytail: regex intents (term, cluster, comparison); add an LLM sub-router if questions
    # outgrow these shapes.
    run_id = db.default_run()
    clusters = _clusters(run_id)
    if m := WHY_TERM.search(query):
        cid = find_cluster(m.group("ref"), clusters)
        if cid is not None:
            return _why_term(run_id, clusters, cid, m.group("term").strip().lower())
    cid = find_cluster(query, clusters)
    if cid is not None:
        return _describe(run_id, clusters, cid)
    if COMPARE.search(query):
        return _compare_runs()
    names = "\n".join(f"- cluster {c}: {_name(info)}" for c, info in sorted(clusters.items()))
    return {"answer": f"Which cluster do you mean? Name it by number or by a word from its label:\n{names}",
            "metadata": {"run_id": run_id}}


def _clusters(run_id: str) -> dict[int, dict]:
    return {r["cluster_id"]: r for r in db.query(
        "SELECT cluster_id, size, keywords, label FROM cluster_info WHERE run_id = %s", (run_id,))}


def _name(info: dict) -> str:
    return info["label"] or ", ".join(t for t, _ in info["keywords"][:3])


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in NOISE}


def find_cluster(text: str, clusters: dict[int, dict]) -> int | None:
    """Cluster named in `text`: "cluster 3", or the cluster whose label (weight 2) and top terms
    (weight 1) share the most words with it. None when nothing matches or two clusters tie."""
    if m := CLUSTER_REF.search(text):
        cid = int(m.group(1))
        return cid if cid in clusters else None
    asked = _words(text)
    scores = {}
    for cid, info in clusters.items():
        label = _words(info["label"] or "")
        terms = {t.lower() for t, _ in info["keywords"][:10]}
        scores[cid] = 2 * len(asked & label) + len(asked & terms)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0 or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
        return None
    return ranked[0][0]


def _describe(run_id: str, clusters: dict[int, dict], cluster_id: int) -> dict[str, Any]:
    info = clusters[cluster_id]
    groups = db.query(
        """SELECT d.label_name, count(*) AS n FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id
           WHERE dc.run_id = %s AND dc.cluster_id = %s GROUP BY 1 ORDER BY 2 DESC LIMIT 3""",
        (run_id, cluster_id),
    )
    terms = ", ".join(t for t, _ in info["keywords"][:10])
    mix = ", ".join(f'{g["label_name"]} ({g["n"]})' for g in groups)
    answer = (
        f"**Cluster {cluster_id}**" + (f" — {info['label']}" if info["label"] else "") + "\n\n"
        f"- Size: {info['size']} documents\n- Top terms: {terms}\n- Dominant newsgroups: {mix}"
    )
    return {"answer": answer, "metadata": {"run_id": run_id, "cluster_id": cluster_id}}


def _why_term(run_id: str, clusters: dict[int, dict], cluster_id: int, term: str) -> dict[str, Any]:
    """Why a word shows up in a cluster's terms or label: its c-TF-IDF rank and weight, and how
    much more often the cluster's posts use it than the rest of the corpus."""
    info = clusters[cluster_id]
    ranks = {t.lower(): (i, w) for i, (t, w) in enumerate(info["keywords"], 1)}
    # labels read "entity · entity · entity (top term)"
    head, _, tail = (info["label"] or "").lower().partition("(")
    entities, label_term = [p.strip() for p in head.split("·") if p.strip()], tail.rstrip(") ").strip()
    pattern = r"\m" + re.escape(term).replace(r"\ ", r"\s+") + r"\M"   # whole word(s), case-insensitive
    row = db.query(
        """SELECT count(*) FILTER (WHERE dc.cluster_id = %s) AS hits, count(*) AS hits_all
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id
           WHERE dc.run_id = %s AND d.text ~* %s""",
        (cluster_id, run_id, pattern),
    )[0]
    total = sum(c["size"] for c in clusters.values())
    inside = row["hits"] / max(info["size"], 1)
    outside = (row["hits_all"] - row["hits"]) / max(total - info["size"], 1)

    lines = [f"**“{term}” in cluster {cluster_id}** — {_name(info)}", ""]
    if term in ranks:
        rank, weight = ranks[term]
        lines.append(f"- It is the cluster's #{rank} term by c-TF-IDF (weight {weight:.3f}): frequent in this "
                     "cluster's posts and rarer in the others.")
    else:
        lines.append(f"- It is not among the cluster's top {len(ranks)} c-TF-IDF terms.")
    if term in entities:
        lines.append("- It is in the cluster label: one of the top PageRank entities of the cluster's "
                     "knowledge graph.")
    elif term == label_term:
        lines.append("- It is in the cluster label as the top term, in parentheses after the KG entities.")
    lines.append(f"- {inside:.0%} of this cluster's {info['size']} posts contain it, against {outside:.0%} "
                 "of posts in other clusters"
                 + (f" ({inside / outside:.1f}× as often)." if outside and inside > outside else "."))
    if term not in ranks and term not in entities:
        lines.append("- So it does not characterize this cluster; it only appears in some of its posts.")
    return {"answer": "\n".join(lines),
            "metadata": {"run_id": run_id, "cluster_id": cluster_id, "term": term,
                         "rank": ranks.get(term, (None,))[0], "share_in_cluster": round(inside, 3),
                         "share_elsewhere": round(outside, 3)}}


def _compare_runs() -> dict[str, Any]:
    runs = db.query(
        "SELECT run_id, metrics FROM runs ORDER BY (metrics->>'nmi')::float DESC NULLS LAST"
    )
    if not runs:
        return {"answer": "No clustering runs yet; run scripts/03_cluster.py.", "metadata": {}}
    cols = ("nmi", "ari", "purity", "silhouette", "n_clusters", "noise_rate")
    lines = ["| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in runs:
        m = r["metrics"]
        lines.append(f"| {r['run_id']} | " + " | ".join(
            f"{m[c]:.3f}" if isinstance(m.get(c), float) else str(m.get(c, "")) for c in cols) + " |")
    return {"answer": "\n".join(lines), "metadata": {"runs": [r["run_id"] for r in runs]}}
