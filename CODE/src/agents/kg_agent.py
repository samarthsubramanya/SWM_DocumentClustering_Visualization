"""KGAgent: Text2Cypher, mirroring the harness's Text2SQL agent.

Generate Cypher from the ontology schema -> validate (read-only, EXPLAIN, schema names) ->
repair with the validator's findings -> execute in a READ transaction -> summarize.
Evaluated by scripts/06_eval.py --text2cypher against eval/text2cypher.yaml.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.core.call_llm import get_llm
from src.core.events import emit
from pipeline import db, kg

MAX_ROWS = 50
ATTEMPTS = 3

PROMPT = """Write ONE read-only Cypher query for Neo4j 5 that answers the question.

Graph schema:
{schema}

Rules:
- An entity's only name property is `name`, in lowercase. Match a named entity exactly, with the
  user's spelling lowercased: e.name = 'nasa'.
- If the question does not say what type an entity is, match it without a label:
  (d:Document)-[:mentions]->(e) WHERE e.name = 'israel'.
- Documents reach entities through `mentions` (and Topic nodes through `about`). To find documents
  about an entity, follow that edge; never search Document.id, which is an opaque identifier.
- Cluster membership lives on Document nodes (Document.cluster_id, integer).
- Count what the question asks about: "how many organizations" counts DISTINCT organization nodes.
- Use only the labels, relationship types and properties listed in the schema.
- Return at most {max_rows} rows (use LIMIT). Never write to the graph.

Examples:
Q: How many distinct people are mentioned in cluster 3?
A: MATCH (:Document {{cluster_id: 3}})-[:mentions]->(p:Person) RETURN count(DISTINCT p) AS people
Q: Which documents mention apple?
A: MATCH (d:Document)-[:mentions]->(e) WHERE e.name = 'apple' RETURN DISTINCT d.id LIMIT {max_rows}
Q: Which organizations appear in both cluster 2 and cluster 9?
A: MATCH (:Document {{cluster_id: 2}})-[:mentions]->(o:Organization)<-[:mentions]-(:Document {{cluster_id: 9}})
   RETURN DISTINCT o.name LIMIT {max_rows}
Q: Which people are affiliated with the nsa?
A: MATCH (p:Person)-[:affiliatedWith]->(:Organization {{name: 'nsa'}}) RETURN p.name LIMIT {max_rows}
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
        try:
            cypher = (await llm.ainvoke(PROMPT.format(
                schema=schema, max_rows=MAX_ROWS, feedback=feedback, question=question,
            ))).cypher.strip()
        except RuntimeError as exc:  # no parsable query (e.g. output budget spent thinking): a failed attempt
            cypher, found = "", [f"no query produced: {str(exc)[:200]}"]
        else:
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


REAL_DOC_ID = re.compile(r"^(?:train|test)/[\w.+-]+/\d+$")


def opaque_rows(rows: list[dict]) -> list[dict]:
    """Replace real document ids (they name the gold newsgroup) with the dashboard's opaque ids."""
    def swap(v):
        return db.display_id(v) if isinstance(v, str) and REAL_DOC_ID.match(v) else v
    return [{k: [swap(x) for x in v] if isinstance(v, list) else swap(v) for k, v in r.items()} for r in rows]


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node."""
    result = await text2cypher(query)
    meta = {"cypher": result["cypher"], "attempts": result["attempts"], "validation": result["problems"]}
    if result["rows"] is None:
        return {"answer": "I couldn't build a valid graph query for that question.", "metadata": meta}

    rows = opaque_rows(result["rows"])
    summary = await get_llm().ainvoke(
        "Answer the question using only these Neo4j results. "
        "If there are no rows, say no matching data was found.\n"
        f"Question: {query}\nCypher: {result['cypher']}\nRows: {json.dumps(rows, default=str)}"
    )
    # A reasoning model can spend its whole output budget thinking and return no text; the rows
    # are already the answer, so list them rather than fail the request.
    return {"answer": summary.content.strip() or rows_answer(rows), "metadata": {**meta, "rows": rows}}


def rows_answer(rows: list[dict], shown: int = 20) -> str:
    if not rows:
        return "No matching data was found."
    lines = [", ".join(str(v) for v in r.values()) for r in rows[:shown]]
    more = f"\n- … and {len(rows) - shown} more" if len(rows) > shown else ""
    return f"{len(rows)} result{'s' if len(rows) != 1 else ''}:\n- " + "\n- ".join(lines) + more
