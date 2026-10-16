# src/agents/base/base_agent/nodes/init_node.py

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.core.events import emit

from ..state import GlobalState


async def init_node(
    state: GlobalState,
) -> dict:
    """
    Initialize request-scoped global state for the Base Agent.

    The Base Agent acts as the orchestrator for the harness.
    This node prepares the shared state before agent selection.

    Agent-specific private state is initialized only after
    the Base Agent routes the request to a selected agent.
    """

    # ======================================================
    # Validate request input
    # ======================================================

    request_id = state["request_id"]
    user_query = state["user_query"]

    if not request_id.strip():
        raise ValueError(
            "request_id cannot be empty."
        )

    if not user_query.strip():
        raise ValueError(
            "user_query cannot be empty."
        )

    # ======================================================
    # Emit request initialization event
    # ======================================================

    emit(
        component="base_agent",
        event="request_initialized",
        message="Request initialized for orchestration.",
        data={
            "request_id": request_id,
            "user_query": user_query,
        },
    )

    # ======================================================
    # Initialize shared global state
    # ======================================================

    return {
        # Public conversation history.
        "messages": [
            HumanMessage(
                content=user_query
            )
        ],

        # No routing decision has been made yet.
        "execution": {
            "intent": None,
            "selected_agent": None,
            "confidence": None,
        },

        # The next stage is agent selection.
        "status": "ROUTING",

        # No final result exists yet.
        "final_response": None,
        "error": None,

        # Public execution milestones.
        "execution_log": [],

        # Preserve optional caller-provided metadata.
        "metadata": state.get(
            "metadata",
            {},
        ),
    }