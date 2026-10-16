"""ExplainAgent: "why is doc 42 in cluster 3?" -> evidence + LLM rationale.

Wraps swm.explain; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

from typing import Any


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node.

    Return {"answer": str, "metadata": dict}; metadata is exposed via the API.
    """
    # TODO(explain): call swm.explain; scope: "why is doc 42 in cluster 3?" -> evidence + LLM rationale.
    return {
        "answer": "explain agent is not implemented yet.",
        "metadata": {"stub": True},
    }
