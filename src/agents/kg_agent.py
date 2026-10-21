"""KGAgent: Text2Cypher, mirroring the harness's Text2SQL agent.

Bounded schema discovery over the Neo4j graph -> LLM writes Cypher -> execute -> summarize.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.core.call_llm import get_llm
from src.core.events import emit
from swm import kg

# Defense in depth: regex screen here, READ transaction in kg.read_query.
WRITE_CLAUSE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL)\b", re.IGNORECASE)
MAX_ROWS = 50
ATTEMPTS = 2

PROMPT = """Write ONE read-only Cypher query for Neo4j 5 that answers the question.

Graph schema:
{schema}

Notes:
- Entity names are stored lowercase; match user text with toLower(...) and CONTAINS.
- Cluster membership lives on Document nodes (Document.cluster_id, integer).
- Return at most {max_rows} rows (use LIMIT).
- Never write to the graph.
{feedback}
Question: {question}
"""


class CypherQuery(BaseModel):
    cypher: str = Field(description="One read-only Cypher query")


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    # ponytail: single generate -> execute -> repair loop; port the harness Text2SQL
    # discovery/verifier nodes if accuracy on the held-out questions falls short.
    schema = kg.schema_text(kg.load_ontology())
    llm = get_llm().with_structured_output(CypherQuery)
    feedback, cypher, rows = "", "", None

    for _ in range(ATTEMPTS):
        cypher = (await llm.ainvoke(PROMPT.format(
            schema=schema, max_rows=MAX_ROWS, feedback=feedback, question=query,
        ))).cypher.strip()
        if WRITE_CLAUSE.search(cypher):
            feedback = f"\nYour previous query was rejected because it writes to the graph:\n{cypher}\n"
            continue
        try:
            rows = await asyncio.to_thread(kg.read_query, cypher, MAX_ROWS)
            break
        except Exception as exc:  # Cypher syntax / runtime error -> one repair attempt
            emit(component="kg_agent", event="cypher_failed", message="Cypher execution failed.",
                 data={"error": str(exc)})
            feedback = f"\nYour previous query failed:\n{cypher}\nError: {exc}\nFix it.\n"

    if rows is None:
        return {"answer": "I couldn't build a valid graph query for that question.",
                "metadata": {"cypher": cypher, "error": feedback.strip()}}

    summary = await get_llm().ainvoke(
        "Answer the question using only these Neo4j results. "
        "If there are no rows, say no matching data was found.\n"
        f"Question: {query}\nCypher: {cypher}\nRows: {json.dumps(rows, default=str)}"
    )
    return {"answer": summary.content, "metadata": {"cypher": cypher, "rows": rows}}
