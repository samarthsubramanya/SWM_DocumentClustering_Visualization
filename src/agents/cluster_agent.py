"""ClusterAgent: "show cluster 3", "compare HDBSCAN vs k-means".

Wraps swm.cluster; registered with the harness's LangGraph orchestrator.
"""

from __future__ import annotations

from typing import Any


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node.

    Return {"answer": str, "metadata": dict}; metadata is exposed via the API.
    """
    # TODO(cluster): call swm.cluster; scope: "show cluster 3", "compare HDBSCAN vs k-means".
    return {
        "answer": "cluster agent is not implemented yet.",
        "metadata": {"stub": True},
    }
