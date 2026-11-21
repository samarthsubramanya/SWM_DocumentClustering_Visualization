"""Base-agent graph routes to the ingest agent end-to-end (LLM mocked; no Ollama or database needed)."""

import asyncio
import importlib
from types import SimpleNamespace

from src.agents.base.base_agent import graph as _graph  # noqa: F401

orch = importlib.import_module("src.agents.base.base_agent.nodes.orchestrator_node")


def test_routes_to_ingest_agent(monkeypatch):
    class FakeStructured:
        async def ainvoke(self, _):
            return orch.RoutingDecision(intent="add article", selected_agent="ingest", confidence=0.9)

    monkeypatch.setattr(
        orch, "get_llm", lambda: SimpleNamespace(with_structured_output=lambda _s: FakeStructured())
    )
    result = asyncio.run(
        _graph.base_agent_graph.ainvoke({"request_id": "t1", "user_query": "add this article", "metadata": {}})
    )
    assert result["status"] == "SUCCESS"
    assert result["execution"]["selected_agent"] == "ingest"
    assert "Paste the article" in result["final_response"]   # no article text: it asks for one
