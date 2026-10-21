"""Sync Ollama chat model for swm's batch stages (KG extraction, rationales).

The async harness agents in src/ use src/agents/core/call_llm.py; swm stays independent of src/.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache


def model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


@lru_cache(maxsize=1)
def _chat():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name(),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        # 4096 is enough for extraction (~1.2k prompt + 1k output) and frees KV-cache memory
        # for more parallel slots; keep it equal to the harness agents' value to avoid reloads.
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        # Without a cap, JSON-format output can loop forever on odd inputs and block Ollama.
        num_predict=1024,
        client_kwargs={"timeout": 180},
    )


def chat(prompt: str) -> str:
    return str(_chat().invoke(prompt).content)


class BadOutput(RuntimeError):
    """The model answered, but not with valid structured output (e.g. truncated at the
    num_predict cap). Deterministic at temperature 0, so retrying the same prompt won't help."""


def structured(prompt: str, schema, attempts: int = 3):
    """Pydantic-structured output. Retries only transient Ollama errors (down, overloaded, timeout);
    raises BadOutput for invalid output and RuntimeError when Ollama stays unreachable."""
    import httpx
    from ollama import ResponseError

    llm = _chat().with_structured_output(schema)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            out = llm.invoke(prompt)
        except (httpx.HTTPError, ConnectionError, ResponseError) as exc:
            error = exc
            time.sleep(2 ** attempt)
            continue
        except Exception as exc:  # JSON / pydantic parse failure
            raise BadOutput(str(exc)) from exc
        if out is None:
            raise BadOutput("empty structured output")
        return out
    raise RuntimeError(f"Ollama unreachable after {attempts} attempts: {error}")
