"""KGAgent: Text2Cypher, mirroring the harness's Text2SQL agent.

Generate Cypher from the ontology schema -> validate (read-only, EXPLAIN, schema names) ->
repair with the validator's findings -> execute in a READ transaction -> summarize.
Evaluated by scripts/06_eval.py --text2cypher against eval/text2cypher.yaml.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from src.agents.core.call_llm import get_llm
from src.core.events import emit
from swm import kg

MAX_ROWS = 50
ATTEMPTS = 3

PROMPT = """Write ONE read-only Cypher query for Neo4j 5 that answers the question.

Graph schema:
{schema}

Notes:
- Entity names are stored lowercase; match user text with toLower(...) and CONTAINS.
- Cluster membership lives on Document nodes (Document.cluster_id, integer).
- Use only the labels, relationship types and properties listed in the schema.
- Return at most {max_rows} rows (use LIMIT).
- Never write to the graph.
{feedback}
Question: {question}
"""


class CypherQuery(BaseModel):
    cypher: str = Field(description="One read-only Cypher query")


async def text2cypher(question: str) -> dict[str, Any]:
    """Generate -> validate -> repair loop. Returns {cypher, rows, attempts, problems}.

    rows is None when no attempt produced a valid, executable query. problems lists the
    validator/runtime findings of every attempt (the repair trail).
    """
    schema = kg.schema_text(kg.load_ontology())
    llm = get_llm().with_structured_output(CypherQuery)
    feedback, cypher, problems = "", "", []

    for attempt in range(1, ATTEMPTS + 1):
        cypher = (await llm.ainvoke(PROMPT.format(
            schema=schema, max_rows=MAX_ROWS, feedback=feedback, question=question,
        ))).cypher.strip()
        found = await asyncio.to_thread(kg.validate_cypher, cypher)
        if not found:
            try:
                rows = await asyncio.to_thread(kg.read_query, cypher, MAX_ROWS)
                return {"cypher": cypher, "rows": rows, "attempts": attempt, "problems": problems}
            except Exception as exc:  # runtime error EXPLAIN can't catch (e.g. type mismatch)
                found = [f"execution failed: {exc}"]
        problems.append({"attempt": attempt, "cypher": cypher, "problems": found})
        emit(component="kg_agent", event="cypher_rejected", message="Cypher failed validation.",
             data={"error": "; ".join(found)})
        feedback = ("\nYour previous query was rejected:\n" + cypher + "\nProblems:\n- "
                    + "\n- ".join(found) + "\nWrite a corrected query.\n")

    return {"cypher": cypher, "rows": None, "attempts": ATTEMPTS, "problems": problems}


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    result = await text2cypher(query)
    meta = {"cypher": result["cypher"], "attempts": result["attempts"], "validation": result["problems"]}
    if result["rows"] is None:
        return {"answer": "I couldn't build a valid graph query for that question.", "metadata": meta}

    summary = await get_llm().ainvoke(
        "Answer the question using only these Neo4j results. "
        "If there are no rows, say no matching data was found.\n"
        f"Question: {query}\nCypher: {result['cypher']}\nRows: {json.dumps(result['rows'], default=str)}"
    )
    return {"answer": summary.content, "metadata": {**meta, "rows": result["rows"]}}
