# src/agents/base/base_agent/state.py

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# Agents the orchestrator can route to. "none" = no suitable agent.
# Add a name here, in AGENTS (handoff_node) and in the routing prompt.
AgentName = Literal[
    "cluster",
    "explain",
    "kg",
    "ingest",
    "none",
]


# ==========================================================
# Execution Context
# ==========================================================

class ExecutionContext(TypedDict):
    """
    Routing decision produced by the orchestrator.

    The orchestrator determines which available agent should
    handle the current user request.
    """

    # Short classification of the user's request.
    # Useful for routing observability and debugging.
    intent: str | None

    # Agent selected to handle the request.
    #
    # None means no routing decision has been made yet.
    selected_agent: AgentName | None

    # Confidence assigned by the orchestrator to its
    # routing decision.
    confidence: float | None


# ==========================================================
# Execution Record
# ==========================================================

class ExecutionRecord(TypedDict):
    """
    Public execution record emitted by harness components
    and agents.

    Agent-private state is intentionally excluded.
    """

    # Component responsible for this execution step.
    agent: str

    # Public description of the execution step.
    step: str

    status: Literal[
        "SUCCESS",
        "FAILED",
        "RETRY",
        "INFO",
    ]

    # Public result associated with this step.
    response: dict[str, Any]

    # Additional structured observability information.
    metadata: dict[str, Any]


# ==========================================================
# Global State
# ==========================================================

class GlobalState(TypedDict):
    """
    Shared state for one complete Harness invocation.

    The Harness, orchestrator, and public agent adapters operate
    on this state.

    Agent-specific implementation details remain inside each
    agent's private graph and private state.
    """

    # ======================================================
    # Request
    # ======================================================

    # Unique identifier for the current request.
    request_id: str

    # Original user request.
    user_query: str

    # Shared conversation history.
    #
    # Kept at the harness level so future multi-turn behavior
    # can reuse the same public conversation state.
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    # ======================================================
    # Routing
    # ======================================================

    # Routing decision populated by the orchestrator.
    execution: ExecutionContext

    # ======================================================
    # Response
    # ======================================================

    # Final user-facing response produced by the selected
    # agent or by the harness when no suitable agent exists.
    final_response: str | None

    # Current lifecycle state of the complete request.
    status: Literal[
        "CREATED",
        "ROUTING",
        "RUNNING",
        "SUCCESS",
        "FAILED",
    ]

    # Latest public request-level error.
    error: str | None

    # ======================================================
    # Observability
    # ======================================================

    # Public execution records accumulated across the
    # orchestrator and selected agent.
    execution_log: Annotated[
        list[ExecutionRecord],
        add,
    ]

    # ======================================================
    # Additional Context
    # ======================================================

    # Extensible request-level metadata.
    #
    # This should contain shared/public context only.
    # Private agent implementation state does not belong here.
    metadata: dict[str, Any]
