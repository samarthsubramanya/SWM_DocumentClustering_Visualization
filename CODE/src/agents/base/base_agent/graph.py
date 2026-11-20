# src/agents/base/base_agent/graph.py

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    handoff_node,
    init_node,
    orchestrator_node,
)
from .state import GlobalState


def build_base_agent_graph():
    """
    Build and compile the Base Agent orchestration graph.

    The Base Agent performs three steps:

    1. Initialize shared GlobalState.
    2. Select the most suitable available agent.
    3. Hand the request to the selected private agent.

    The handoff node invokes the selected private agent,
    translates its result back into GlobalState, and handles
    the no-suitable-agent path.

    The Base Agent itself contains no reasoning loop.
    """

    graph = StateGraph(GlobalState)

    # ======================================================
    # Nodes
    # ======================================================

    graph.add_node(
        "init",
        init_node,
    )

    graph.add_node(
        "orchestrator",
        orchestrator_node,
    )

    graph.add_node(
        "handoff",
        handoff_node,
    )

    # ======================================================
    # Flow
    # ======================================================

    graph.add_edge(
        START,
        "init",
    )

    graph.add_edge(
        "init",
        "orchestrator",
    )

    graph.add_edge(
        "orchestrator",
        "handoff",
    )

    graph.add_edge(
        "handoff",
        END,
    )

    # ======================================================
    # Compile
    # ======================================================

    return graph.compile()


base_agent_graph = build_base_agent_graph()
