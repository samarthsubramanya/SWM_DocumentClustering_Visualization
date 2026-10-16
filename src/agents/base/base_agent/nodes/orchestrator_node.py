# src/agents/base/base_agent/nodes/orchestrator_node.py

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.core.call_llm import get_llm
from src.core.events import emit

from ..state import AgentName, GlobalState


class RoutingDecision(BaseModel):
    """
    Structured routing decision produced by the Base Agent.
    """

    intent: str = Field(
        description=(
            "A short classification of the user's request."
        )
    )

    selected_agent: AgentName = Field(
        description=(
            "The agent best suited to handle the user's request."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the selected routing decision."
        ),
    )


async def orchestrator_node(
    state: GlobalState,
) -> dict:
    """
    Select the agent best suited to handle the user request.

    This node performs routing only.

    It does not invoke the selected agent, use tools,
    answer the user, or perform agent-specific reasoning.
    """

    # ======================================================
    # Routing started
    # ======================================================

    emit(
        component="base_agent",
        event="routing_started",
        message="Selecting an agent for the request.",
        data={
            "request_id": state["request_id"],
        },
    )

    # ======================================================
    # Build routing prompt
    # ======================================================

    prompt = f"""
You are the routing component of an agent system.

Select exactly one available agent for the user's request.

User request:
{state["user_query"]}

Available agents:

1. cluster
   Inspect or compare clusterings of the news corpus.
   Examples: "show cluster 3", "compare HDBSCAN vs k-means",
   "what are the top terms of cluster 7?"

2. explain
   Explain why a document belongs to its cluster.
   Examples: "why is doc 42 in cluster 3?"

3. kg
   Questions answered from the ontology-anchored knowledge graph (Neo4j).
   Examples: "which organizations appear in clusters 3 and 7?"

4. ingest
   Add a new article (pasted text or RSS) and assign it to a cluster
   or flag it as emerging.

5. none
   Use when no available agent is suitable for the request.

Rules:

- Select only one agent.
- Do not answer the user's request.
- Do not generate Cypher or SQL.
- Do not invoke any agent or tool.
- Base the decision only on which agent should handle the request.
- intent must be a short classification.
- confidence must be between 0 and 1.
- Use none when no agent can appropriately handle the request.
"""
    
    # ======================================================
    # Ask LLM for routing decision
    # ======================================================

    llm = get_llm()

    routing_llm = llm.with_structured_output(
        RoutingDecision
    )

    decision = await routing_llm.ainvoke(
        prompt
    )

    # ======================================================
    # Routing completed
    # ======================================================

    emit(
        component="base_agent",
        event="agent_selected",
        message="Agent routing decision completed.",
        data={
            "request_id": state["request_id"],
            "selected_agent": decision.selected_agent,
            "intent": decision.intent,
            "confidence": decision.confidence,
        },
    )

    # ======================================================
    # Update GlobalState
    # ======================================================

    return {
        "execution": {
            "intent": decision.intent,
            "selected_agent": decision.selected_agent,
            "confidence": decision.confidence,
        },

        # A suitable agent will transition this to actual
        # execution through the graph's conditional routing.
        "status": (
            "RUNNING"
            if decision.selected_agent != "none"
            else "ROUTING"
        ),

        "execution_log": [
            {
                "agent": "base_agent",
                "step": "agent_selection",
                "status": "SUCCESS",
                "response": {
                    "selected_agent": decision.selected_agent,
                },
                "metadata": {
                    "intent": decision.intent,
                    "confidence": decision.confidence,
                },
            }
        ],
    }
