# src/agents/base/base_agent/nodes/handoff_node.py

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.cluster_agent import run as run_cluster
from src.agents.explain_agent import run as run_explain
from src.agents.ingest_agent import run as run_ingest
from src.agents.kg_agent import run as run_kg
from src.core.events import emit

from ..state import GlobalState


# ==========================================================
# Agent registry
# ==========================================================

# Each agent is `async run(query) -> {"answer": str, "metadata": dict}`.
# Replace a stub's body in src/agents/<name>_agent.py; nothing here changes.
AGENTS = {
    "cluster": run_cluster,
    "explain": run_explain,
    "kg": run_kg,
    "ingest": run_ingest,
}


async def handoff_node(
    state: GlobalState,
) -> dict:
    """
    Hand the request to the agent selected by the Base Agent.

    This node is the boundary between shared GlobalState and
    agent-private execution state.

    Responsibilities:

    1. Inspect the orchestrator's selected agent.
    2. Translate GlobalState into that agent's private input.
    3. Invoke the private agent graph.
    4. Translate the private result back into GlobalState.
    5. Handle the no-suitable-agent path.

    Private agent state is never copied directly into GlobalState.
    """

    selected_agent = state["execution"]["selected_agent"]
    request_id = state["request_id"]
    user_query = state["user_query"]

    # ======================================================
    # Preconditions
    # ======================================================

    if selected_agent is None:
        raise ValueError(
            "Handoff attempted before an agent was selected."
        )

    # ======================================================
    # No suitable agent
    # ======================================================

    if selected_agent == "none":
        response = (
            "No suitable agent is currently available "
            "to handle this request."
        )

        emit(
            component="base_agent",
            event="no_agent_available",
            message="No suitable agent was found for the request.",
            data={
                "request_id": request_id,
            },
        )

        return {
            "messages": [
                AIMessage(
                    content=response
                )
            ],
            "final_response": response,
            "status": "SUCCESS",
            "error": None,
            "execution_log": [
                {
                    "agent": "base_agent",
                    "step": "handoff",
                    "status": "INFO",
                    "response": {
                        "selected_agent": "none",
                        "message": response,
                    },
                    "metadata": {
                        "intent": state["execution"]["intent"],
                        "confidence": state["execution"]["confidence"],
                    },
                }
            ],
        }

    # ======================================================
    # Handoff started
    # ======================================================

    emit(
        component="base_agent",
        event="handoff_started",
        message=f"Handing request to {selected_agent}.",
        data={
            "request_id": request_id,
        },
    )

    # ======================================================
    # Invoke selected agent
    # ======================================================

    run_agent = AGENTS.get(selected_agent)

    if run_agent is None:
        raise ValueError(
            f"Unsupported selected agent: {selected_agent}"
        )

    result = await run_agent(user_query)

    answer = result.get("answer")

    if not answer:
        raise ValueError(
            f"{selected_agent} agent response does not contain an answer."
        )

    execution_metadata = result.get("metadata", {})

    # ======================================================
    # Handoff completed
    # ======================================================

    emit(
        component="base_agent",
        event="handoff_completed",
        message=f"{selected_agent} completed successfully.",
        data={
            "request_id": request_id,
        },
    )

    # ======================================================
    # Translate private result -> GlobalState
    # ======================================================

    return {
        "messages": [
            AIMessage(
                content=answer
            )
        ],

        "final_response": answer,

        "status": "SUCCESS",

        "error": None,

        "execution_log": [
            {
                "agent": selected_agent,
                "step": "execution",
                "status": "SUCCESS",
                "response": {
                    "answer": answer,
                },
                "metadata": execution_metadata,
            }
        ],
    }
