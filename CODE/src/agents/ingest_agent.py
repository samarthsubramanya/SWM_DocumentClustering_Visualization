"""IngestAgent (Optional A): "add this article: <text>" -> place it in a cluster or flag it as emerging.

Wraps pipeline.stream; registered with the harness's LangGraph orchestrator. The post is placed
in the run named by SWM_INGEST_RUN_ID (default: the dashboard's run) and explained with the same
evidence as every other post. Nothing is saved.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from pipeline import db, stream

MIN_WORDS = 20      # the corpus drops shorter posts too
WEAK_MARGIN = 0.15  # same rule as the dashboards' evidence banner
# "add this article: ...", "ingest: ...", "where does this post belong? ..."
COMMAND = re.compile(r"^\s*(?:please\s+)?(?:add|ingest|assign|place|classify|cluster|where (?:does|would))\b[^:?\n]*[:?]\s*",
                     re.IGNORECASE)


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    return await asyncio.to_thread(_answer, query)


def article_text(query: str) -> str:
    """The pasted article, without the request in front of it."""
    return COMMAND.sub("", query, count=1).strip().strip('"“”')


def _answer(query: str) -> dict[str, Any]:
    text = article_text(query)
    if len(text.split()) < MIN_WORDS:
        return {"answer": f"Paste the article after \"add this article:\". It needs at least {MIN_WORDS} words "
                          "to be placed reliably.", "metadata": {}}

    run_id = os.getenv("SWM_INGEST_RUN_ID") or db.default_run()
    ev = stream.assigner(run_id).assign([text])[0]
    label = db.query("SELECT label, keywords FROM cluster_info WHERE run_id = %s AND cluster_id = %s",
                     (run_id, ev["cluster_id"]))
    name = (label[0]["label"] or ", ".join(t for t, _ in label[0]["keywords"][:3])) if label else ""
    terms = [h["term"] for h in ev["term_hits"][:8]]
    weak = ev["margin"] is not None and ev["margin"] < WEAK_MARGIN or len(terms) <= 1
    nearest = [f"`{db.display_id(e['doc_id'])}` ({e['similarity']:.2f})" for e in ev["exemplars"]]

    lines = [f"**New post → cluster {ev['cluster_id']}**" + (f" — {name}" if name else ""), ""]
    if ev["emerging"]:
        lines.append("- **Emerging:** it is less similar to every cluster than 95% of the run's own posts are "
                     "to theirs, so it may be a topic this run has no cluster for.")
    lines += [
        f"- **{'Weak' if weak else 'Strong'} evidence:** similarity {ev['own_centroid_sim']:.2f} to this cluster's "
        f"center, margin {ev['margin']:.2f} over cluster {ev['runner_up_cluster']}.",
        "- **Cluster terms in it:** " + (", ".join(f"`{t}`" for t in terms) if terms else "none"),
        "- **Nearest posts in the cluster:** " + ", ".join(nearest),
        "",
        f"Placed by a vote of its {stream.VOTERS} nearest posts in run `{run_id}`; not saved.",
    ]
    shown = {**ev, "exemplars": [{**e, "doc_id": db.display_id(e["doc_id"])} for e in ev["exemplars"]]}
    return {"answer": "\n".join(lines), "metadata": {"run_id": run_id, "evidence": shown}}
