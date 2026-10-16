"""IngestAgent (Optional A): "add this article" / RSS -> embed -> assign or flag emerging.

Not required for the foundation; stub kept so the orchestrator wiring is ready.
"""

from __future__ import annotations

from typing import Any


async def run(query: str) -> dict[str, Any]:
    """Entry point called by the Base Agent handoff node.

    Return {"answer": str, "metadata": dict}; metadata is exposed via the API.
    """
    # TODO(ingest): call swm.ingest + swm.cluster (Optional A); scope: new article -> embed -> assign or flag emerging.
    return {
        "answer": "ingest agent is not implemented yet.",
        "metadata": {"stub": True},
    }
