"""KGAgent: Text2Cypher, mirroring the harness's Text2SQL agent.

Bounded schema discovery over the Neo4j graph -> LLM writes Cypher -> execute -> summarize.
"""

from __future__ import annotations

from typing import Any


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node.

    Return {"answer": str, "metadata": dict}; metadata is exposed via the API.
    """
    # TODO(kg): call swm.kg + Neo4j; scope: Text2Cypher: schema discovery -> LLM writes Cypher -> execute -> summarize. Mirror harness/Agentic-harnesss/src/agents/text2sql/sql_agent.
    return {
        "answer": "kg agent is not implemented yet.",
        "metadata": {"stub": True},
    }
