# src/api/models.py

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.base.base_agent.state import AgentName


# ==========================================================
# Query Request
# ==========================================================

class QueryRequest(BaseModel):
    """
    Public HTTP request accepted by the Agent Harness.

    The API is responsible for converting this request into
    the GlobalState input expected by the Base Agent.
    """

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language request to be handled by "
            "the Agent Harness."
        ),
    )


# ==========================================================
# Query Response
# ==========================================================

class QueryResponse(BaseModel):
    """
    Public HTTP response returned after the Base Agent
    completes execution.

    Agent-private state and internal LangGraph state are
    intentionally excluded from this contract.
    """

    request_id: str = Field(
        ...,
        description=(
            "Unique identifier assigned to this request."
        ),
    )

    status: Literal[
        "SUCCESS",
        "FAILED",
    ] = Field(
        ...,
        description=(
            "Final public execution status."
        ),
    )

    agent: AgentName = Field(
        ...,
        description=(
            "Agent selected by the orchestrator."
        ),
    )

    answer: str = Field(
        ...,
        description=(
            "Final user-facing response."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Public execution metadata associated with "
            "the selected agent."
        ),
    )


# ==========================================================
# Health Response
# ==========================================================

class HealthResponse(BaseModel):
    """
    Response returned by the lightweight health endpoint.

    This represents HTTP service availability only and does
    not perform expensive dependency or agent checks.
    """

    status: Literal["ok"] = "ok"