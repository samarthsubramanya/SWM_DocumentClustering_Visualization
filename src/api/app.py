# src/api/app.py

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


# ==========================================================
# Environment
# ==========================================================

# Resolve the repository root independently of the process
# working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Environment configuration must be loaded before importing
# routers, graphs, LLM clients, or other LangChain components.
load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=False,
)


# ==========================================================
# Imports
# ==========================================================

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.models import HealthResponse
from src.api.routes import query_router
from src.core.events import emit


# ==========================================================
# Application Lifecycle
# ==========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Manage FastAPI application lifecycle.

    Lifecycle events are emitted through the shared event
    system so startup and shutdown can later be consumed by
    the observability layer without coupling the API directly
    to a telemetry backend.
    """

    # ======================================================
    # Startup
    # ======================================================

    emit(
        component="api",
        event="application_started",
        message="SWM Agent API started.",
        data={},
    )

    yield

    # ======================================================
    # Shutdown
    # ======================================================

    emit(
        component="api",
        event="application_stopped",
        message="SWM Agent API stopped.",
        data={},
    )


# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="SWM Agent API",
    description=(
        "HTTP interface for the SWM agent orchestration system."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ==========================================================
# Routers
# ==========================================================

app.include_router(
    query_router
)


# ==========================================================
# Health
# ==========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
async def health() -> HealthResponse:
    """
    Return lightweight HTTP service health.

    This endpoint intentionally does not invoke Ollama,
    PostgreSQL, LangGraph, or any private agent.

    Dependency readiness checks can be introduced separately
    later if required.
    """

    return HealthResponse(
        status="ok"
    )