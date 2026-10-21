"""ExplainAgent: "why is doc 42 in cluster 3?" -> evidence + LLM rationale.

Wraps swm.explain; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from swm import db, explain, kg

DOC_ID = re.compile(r"\b(?:train|test)/[\w.+-]+/\d+\b")
DOC_NUMBER = re.compile(r"\bdoc(?:ument)?\s*#?\s*(\d+)", re.IGNORECASE)


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    return await asyncio.to_thread(_answer, query)


def _find_doc(query: str) -> str | None:
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
            "answer": "Which document? Give its id (e.g. `train/sci.space/60151`) or number (`doc 60151`).",
            "metadata": {},
        }
    run_id = db.default_run()
    ev = explain.evidence(doc_id, run_id)
    text = explain.rationale(doc_id, run_id, kg_entities=kg.doc_entities(doc_id))
    return {
        "answer": f"**{doc_id}** → cluster {ev['cluster_id']}\n\n{text}",
        "metadata": {"run_id": run_id, "doc_id": doc_id, "evidence": ev},
    }
