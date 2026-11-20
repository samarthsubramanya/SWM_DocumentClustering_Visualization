# src/core/events.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    """
    Structured event emitted by agents during execution.

    Events describe what happened inside the system without
    coupling the emitting component to a specific observer.

    The same event structure can later be consumed by console
    logging, LangSmith, metrics, tracing, or other telemetry.
    """

    component: str
    event: str
    message: str

    data: dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the event into a serializable dictionary.
        """

        return asdict(self)


def emit(
    *,
    component: str,
    event: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> AgentEvent:
    """
    Emit a structured agent event.

    For the current implementation, events are rendered directly
    to stdout so execution progress is visible during development.

    The returned AgentEvent can later also be forwarded to
    observers, tracing systems, or telemetry backends without
    changing the code that emits it.
    """

    agent_event = AgentEvent(
        component=component,
        event=event,
        message=message,
        data=data or {},
    )

    _print_event(agent_event)

    return agent_event


def _print_event(
    event: AgentEvent,
) -> None:
    """
    Render an AgentEvent to stdout.

    Console formatting intentionally lives here rather than
    inside individual agent nodes.
    """

    component = event.component.replace(
        "_",
        " ",
    ).title()

    print(
        f"\n[{component}] {event.message}",
        flush=True,
    )

    if not event.data:
        return

    # ======================================================
    # Reasoning / action summary
    # ======================================================

    reasoning = event.data.get(
        "reasoning_summary"
    )

    if reasoning:
        print(
            f"AI: {reasoning}",
            flush=True,
        )

    # ======================================================
    # User query
    # ======================================================

    user_query = event.data.get(
        "user_query"
    )

    if user_query:
        print(
            "\nUser query:",
            flush=True,
        )
        print(
            user_query,
            flush=True,
        )

    # ======================================================
    # Web search query
    # ======================================================

    search_query = event.data.get(
        "search_query"
    )

    if search_query:
        print(
            "\nSearch query:",
            flush=True,
        )
        print(
            search_query,
            flush=True,
        )

    # ======================================================
    # SQL query
    # ======================================================

    sql = event.data.get(
        "sql"
    )

    if sql:
        print(
            "\nSQL:",
            flush=True,
        )
        print(
            sql,
            flush=True,
        )

    # ======================================================
    # SQL query batch
    # ======================================================

    sql_queries = event.data.get(
        "sql_queries"
    )

    if sql_queries:
        for index, query in enumerate(
            sql_queries,
            start=1,
        ):
            print(
                f"\nSQL {index}:",
                flush=True,
            )
            print(
                query,
                flush=True,
            )

    # ======================================================
    # Missing information
    # ======================================================

    missing_information = event.data.get(
        "missing_information"
    )

    if missing_information:
        print(
            "\nMissing information:",
            flush=True,
        )

        for item in missing_information:
            print(
                f"  - {item}",
                flush=True,
            )

    # ======================================================
    # Result count
    # ======================================================

    row_count = event.data.get(
        "row_count"
    )

    if row_count is not None:
        print(
            f"Rows: {row_count}",
            flush=True,
        )

    # ======================================================
    # Error
    # ======================================================

    error = event.data.get(
        "error"
    )

    if error:
        print(
            f"Error: {error}",
            flush=True,
        )