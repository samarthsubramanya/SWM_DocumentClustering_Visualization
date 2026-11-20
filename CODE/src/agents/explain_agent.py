"""ExplainAgent: "why is doc 42 in cluster 3?" -> evidence + LLM rationale.

Wraps pipeline.explain; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from pipeline import db, explain, kg

DOC_ID = re.compile(r"\b(?:train|test)/[\w.+-]+/\d+\b")
DOC_NUMBER = re.compile(r"\bdoc(?:ument)?\s*#?\s*(\d+)", re.IGNORECASE)


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    return await asyncio.to_thread(_answer, query)


def _find_doc(query: str) -> str | None:
    if m := db.DISPLAY_ID.search(query):          # the dashboard's opaque ids (doc-3f9a1c2e)
        return db.resolve_display_id(m.group(0))
    if m := DOC_ID.search(query):
        return m.group(0)
    if m := DOC_NUMBER.search(query):
        rows = db.query("SELECT id FROM documents WHERE id LIKE %s ORDER BY id LIMIT 1", (f"%/{m.group(1)}",))
        return rows[0]["id"] if rows else None
    return None


def _answer(query: str) -> dict[str, Any]:
    doc_id = _find_doc(query)
    if not doc_id:
        return {
            "answer": "Which document? Give its id as shown in the dashboard (e.g. `doc-3f9a1c2e`).",
            "metadata": {},
        }
    run_id = db.default_run()
    try:
        ev = explain.evidence(doc_id, run_id)
    except LookupError as exc:  # no evidence computed for this run: say so instead of failing the request
        return {"answer": str(exc).replace(doc_id, db.display_id(doc_id)), "metadata": {"run_id": run_id}}
    text = explain.rationale(doc_id, run_id, kg_entities=kg.doc_entities(doc_id))
    # Answers use opaque ids: real ids name the gold newsgroup, which the views keep hidden.
    shown = {**ev, "exemplars": [{**e, "doc_id": db.display_id(e["doc_id"])} for e in ev["exemplars"]]}
    return {
        "answer": f"**{db.display_id(doc_id)}** → cluster {ev['cluster_id']}\n\n{text}",
        "metadata": {"run_id": run_id, "doc_id": db.display_id(doc_id), "evidence": shown},
    }
