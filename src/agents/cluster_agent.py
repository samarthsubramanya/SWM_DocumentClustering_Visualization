"""ClusterAgent: "show cluster 3", "compare HDBSCAN vs k-means".

Wraps swm.cluster; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from swm import db

CLUSTER_REF = re.compile(r"cluster\s*#?\s*(\d+)", re.IGNORECASE)


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    return await asyncio.to_thread(_answer, query)


def _answer(query: str) -> dict[str, Any]:
    # ponytail: regex intent split (one cluster vs. run comparison); add an LLM sub-router if
    # questions outgrow these two shapes.
    if m := CLUSTER_REF.search(query):
        return _describe(db.default_run(), int(m.group(1)))
    return _compare_runs()


def _describe(run_id: str, cluster_id: int) -> dict[str, Any]:
    info = db.query(
        "SELECT size, keywords, label FROM cluster_info WHERE run_id = %s AND cluster_id = %s",
        (run_id, cluster_id),
    )
    if not info:
        return {"answer": f"Cluster {cluster_id} does not exist in run `{run_id}`.", "metadata": {"run_id": run_id}}
    info = info[0]
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
