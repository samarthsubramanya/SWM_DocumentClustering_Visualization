# src/agents/core/call_llm.py

from __future__ import annotations

import os
from typing import Any

from langchain_ollama import ChatOllama

from src.core.events import emit


# ==========================================================
# Configuration
# ==========================================================

DEFAULT_STRUCTURED_ATTEMPTS = 2


# ==========================================================
# Structured Output Wrapper
# ==========================================================

class StructuredLLMWrapper:
    """
    Reliability wrapper around a structured-output runnable.

    Existing agent code can continue using:

        llm = get_llm()

        structured_llm = llm.with_structured_output(MySchema)

        result = await structured_llm.ainvoke(...)

    without implementing retry behavior inside individual nodes.
    """

    def __init__(
        self,
        runnable: Any,
        *,
        component: str = "llm",
        max_attempts: int = DEFAULT_STRUCTURED_ATTEMPTS,
    ) -> None:
        self._runnable = runnable
        self._component = component
        self._max_attempts = max_attempts

    async def ainvoke(
        self,
        input_data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke structured output with bounded retry behavior.

        include_raw=True is used internally so parsing failures
        can be inspected before deciding whether to retry.
        """

        last_error: Exception | None = None
        last_raw_content: str | None = None

        for attempt in range(
            1,
            self._max_attempts + 1,
        ):
            try:
                result = await self._runnable.ainvoke(
                    input_data,
                    *args,
                    **kwargs,
                )

            except Exception as exc:
                last_error = exc

                emit(
                    component=self._component,
                    event="structured_llm_invocation_failed",
                    message="Structured LLM invocation failed.",
                    data={
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )

                if attempt < self._max_attempts:
                    emit(
                        component=self._component,
                        event="structured_llm_retry",
                        message="Retrying structured LLM invocation.",
                        data={
                            "attempt": attempt + 1,
                        },
                    )

                    continue

                break

            # ==================================================
            # include_raw=True result
            # ==================================================

            parsed = result.get("parsed")
            raw = result.get("raw")
            parsing_error = result.get(
                "parsing_error"
            )

            if parsed is not None:
                return parsed

            # ==================================================
            # Inspect raw response
            # ==================================================

            raw_content = ""

            if raw is not None:
                raw_content = str(
                    getattr(
                        raw,
                        "content",
                        "",
                    )
                ).strip()

            last_raw_content = (
                raw_content or None
            )

            # ==================================================
            # Classify failure
            # ==================================================

            if parsing_error is not None:
                last_error = parsing_error
                failure_reason = (
                    "structured_output_parse_failure"
                )

            elif not raw_content:
                last_error = RuntimeError(
                    "Model returned empty structured output."
                )

                failure_reason = (
                    "empty_structured_output"
                )

            else:
                last_error = RuntimeError(
                    "Model returned structured output "
                    "that could not be parsed."
                )

                failure_reason = (
                    "unparsed_structured_output"
                )

            emit(
                component=self._component,
                event="structured_llm_output_failed",
                message="Structured LLM output was invalid.",
                data={
                    "attempt": attempt,
                    "reason": failure_reason,
                    "error": str(last_error),
                },
            )

            # ==================================================
            # Retry
            # ==================================================

            if attempt < self._max_attempts:
                emit(
                    component=self._component,
                    event="structured_llm_retry",
                    message="Retrying structured LLM invocation.",
                    data={
                        "attempt": attempt + 1,
                        "reason": failure_reason,
                    },
                )

        # ======================================================
        # Attempts exhausted
        # ======================================================

        error_message = (
            "Structured LLM failed to produce valid output "
            f"after {self._max_attempts} attempt(s)."
        )

        if last_error is not None:
            error_message += (
                f" Last error: {last_error}"
            )

        if last_raw_content:
            error_message += (
                f" Raw output: {last_raw_content!r}"
            )

        raise RuntimeError(
            error_message
        )


# ==========================================================
# Shared LLM Wrapper
# ==========================================================

class SharedLLM:
    """
    Application-level wrapper around ChatOllama.

    Plain invocations are forwarded directly to ChatOllama.

    Structured-output invocations are automatically wrapped
    with bounded retry and raw-output inspection.
    """

    def __init__(
        self,
        llm: ChatOllama,
    ) -> None:
        self._llm = llm

    async def ainvoke(
        self,
        input_data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Perform a normal unstructured LLM invocation.
        """

        return await self._llm.ainvoke(
            input_data,
            *args,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: Any,
        **kwargs: Any,
    ) -> StructuredLLMWrapper:
        """
        Create a hardened structured-output runnable.

        include_raw is controlled here so every structured
        invocation receives the same reliability behavior.
        """

        # The wrapper requires the raw response, parsed result,
        # and parsing error to inspect failures safely.
        kwargs["include_raw"] = True

        runnable = self._llm.with_structured_output(
            schema,
            **kwargs,
        )

        return StructuredLLMWrapper(
            runnable,
        )


# ==========================================================
# Shared Model
# ==========================================================

_chat_llm = ChatOllama(
    model=os.getenv(
        "OLLAMA_MODEL",
        "llama3.1:8b",
    ),
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ),
    temperature=0,

    # Do not spend large reasoning budgets on small
    # routing/state-transition decisions.
    reasoning=False,

    # Explicitly provide enough output capacity rather than
    # relying on the runtime/model default.
    num_predict=2048,

    # Text2Cypher / rationale prompts include schema and evidence.
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
)

_llm = SharedLLM(
    _chat_llm
)


# ==========================================================
# Public Interface
# ==========================================================

def get_llm() -> SharedLLM:
    """
    Return the shared application LLM.

    Existing callers may use either:

        await get_llm().ainvoke(...)

    or:

        structured_llm = get_llm().with_structured_output(...)
        await structured_llm.ainvoke(...)

    Structured calls automatically receive bounded retry
    handling.
    """

    return _llm


async def call_llm(
    messages: Any,
) -> Any:
    """
    Invoke the shared LLM asynchronously.
    """

    return await _llm.ainvoke(
        messages
    )