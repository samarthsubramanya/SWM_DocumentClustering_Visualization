# src/api/routes/query.py

from __future__ import annotations

import uuid
from typing import Any, get_args

from fastapi import APIRouter, HTTPException

from src.agents.base.base_agent.graph import base_agent_graph
from src.agents.base.base_agent.state import AgentName
from src.api.models import QueryRequest, QueryResponse
from src.core.events import emit


# ==========================================================
# Router
# ==========================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["query"],
)


# ==========================================================
# Public Metadata Extraction
# ==========================================================

def _extract_agent_metadata(
    *,
    result: dict[str, Any],
    selected_agent: str,
) -> dict[str, Any]:
    """
    Extract public metadata produced by the selected agent.

    Agent-private LangGraph state is intentionally not exposed
    through the HTTP API.

    The Base Agent's execution_log is the public boundary
    between private agent execution and external consumers.
    """

    metadata: dict[str, Any] = {
        "intent": result["execution"]["intent"],
        "confidence": result["execution"]["confidence"],
    }

    execution_log = result.get(
        "execution_log",
        [],
    )

    # ======================================================
    # Find selected agent execution record
    # ======================================================

    for record in reversed(execution_log):

        if (
            record.get("agent") == selected_agent
            and record.get("step") == "execution"
        ):
            agent_metadata = record.get(
                "metadata",
                {},
            )

            metadata.update(
                agent_metadata
            )

            break

    return metadata


# ==========================================================
# Query Endpoint
# ==========================================================

@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
) -> QueryResponse:
    """
    Execute one user request through the complete Agent Harness.

    The API does not select or invoke private agents directly.

    Every request enters through the Base Agent, which owns
    orchestration and handoff.
    """

    request_id = str(
        uuid.uuid4()
    )

    user_query = request.query.strip()

    # ======================================================
    # Validate normalized query
    # ======================================================

    if not user_query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty.",
        )

    # ======================================================
    # Request received
    # ======================================================

    emit(
        component="api",
        event="request_received",
        message="HTTP query request received.",
        data={
            "request_id": request_id,
            "user_query": user_query,
        },
    )

    try:
        # ==================================================
        # Invoke complete agent system
        # ==================================================

        result = await base_agent_graph.ainvoke(
            {
                "request_id": request_id,
                "user_query": user_query,
                "metadata": {
                    "source": "api",
                },
            }
        )

        # ==================================================
        # Validate completed GlobalState
        # ==================================================

        status = result.get(
            "status"
        )

        if status != "SUCCESS":
            error = (
                result.get("error")
                or "Agent execution failed."
            )

            emit(
                component="api",
                event="request_failed",
                message="Agent request failed.",
                data={
                    "request_id": request_id,
                    "error": error,
                },
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "The request could not be completed."
                ),
            )

        execution = result.get(
            "execution"
        )

        if not execution:
            raise RuntimeError(
                "Base Agent completed without "
                "an execution context."
            )

        selected_agent = execution.get(
            "selected_agent"
        )

        if selected_agent not in get_args(AgentName):
            raise RuntimeError(
                "Base Agent returned an invalid "
                f"selected_agent: {selected_agent!r}"
            )

        answer = result.get(
            "final_response"
        )

        if not answer:
            raise RuntimeError(
                "Base Agent completed without "
                "a final response."
            )

        # ==================================================
        # Build public metadata
        # ==================================================

        metadata = _extract_agent_metadata(
            result=result,
            selected_agent=selected_agent,
        )

        # ==================================================
        # Build HTTP response
        # ==================================================

        response = QueryResponse(
            request_id=request_id,
            status="SUCCESS",
            agent=selected_agent,
            answer=answer,
            metadata=metadata,
        )

        # ==================================================
        # Request completed
        # ==================================================

        emit(
            component="api",
            event="request_completed",
            message="HTTP query request completed.",
            data={
                "request_id": request_id,
                "selected_agent": selected_agent,
                "status": "SUCCESS",
            },
        )

        return response

    # ======================================================
    # Preserve intentional HTTP errors
    # ======================================================

    except HTTPException:
        raise

    # ======================================================
    # Unexpected execution failure
    # ======================================================

    except Exception as exc:

        emit(
            component="api",
            event="request_failed",
            message="HTTP query request failed.",
            data={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )

        # Do not expose internal graph/database/LLM errors
        # directly to the HTTP client.
        raise HTTPException(
            status_code=500,
            detail=(
                "The request could not be completed."
            ),
        ) from exc