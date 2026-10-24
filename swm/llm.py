"""Sync chat model for swm's batch stages (KG extraction, rationales).

Two providers, chosen by LLM_PROVIDER:
  ollama  (default)  local / Sol Ollama server            OLLAMA_BASE_URL, OLLAMA_MODEL
  openai             any OpenAI-compatible endpoint,      LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
                     e.g. ASU RC (Voyager keys): https://openai.rc.asu.edu/v1

The async harness agents in src/ build their model through build_chat() too, so one .env
switches both. swm stays independent of src/.
"""

from __future__ import annotations

import os
import threading
import time


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def model_name() -> str:
    """Model id; also names the KG extraction cache dir, so switching models never mixes caches."""
    if provider() == "openai":
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ConfigError("LLM_PROVIDER=openai needs LLM_MODEL (list ids with GET <LLM_BASE_URL>/models).")
        return model
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class ConfigError(Exception):
    """Wrong key / model / endpoint. Deliberately not a RuntimeError: it must stop the run
    instead of being cached as a failed extraction or skipped as 'unreachable'."""


class BadOutput(RuntimeError):
    """The model answered, but not with valid structured output (e.g. truncated at the
    output cap). Deterministic at temperature 0, so retrying the same prompt won't help."""


def build_chat(num_predict: int = 1024, num_ctx: int | None = None, timeout: float = 180,
               **ollama_kwargs):
    """LangChain chat model for the configured provider. ollama_kwargs (e.g. reasoning=False)
    are passed to ChatOllama only."""
    num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    if provider() == "openai":
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("LLM_BASE_URL", "https://openai.rc.asu.edu/v1")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("LLM_PROVIDER=openai needs LLM_API_KEY (create one in Voyager > Keys).")
        extra_body = None
        if os.getenv("LLM_DISABLE_THINKING", "1") == "1":
            # Qwen3-style reasoning models otherwise spend the output budget on <think> tokens.
            # Servers that don't know the field ignore it.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        return ChatOpenAI(
            model=model_name(),
            base_url=base_url,
            api_key=api_key,
            temperature=0,
            max_tokens=num_predict,   # same cap as Ollama's num_predict: stops runaway JSON
            timeout=timeout,
            max_retries=6,            # SDK backs off on 429 (token budget) / 5xx / connection errors
            extra_body=extra_body,
        )
    if provider() != "ollama":
        raise ConfigError(f"Unknown LLM_PROVIDER={provider()!r}; use 'ollama' or 'openai'.")

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name(),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        num_ctx=num_ctx,
        # Without a cap, JSON-format output can loop forever on odd inputs and block Ollama.
        num_predict=num_predict,
        client_kwargs={"timeout": timeout},
        **ollama_kwargs,
    )


_CHAT = None
_CHAT_LOCK = threading.Lock()


def _chat():
    """Shared client, built once. The lock matters: KG extraction calls this from many threads
    at once, and concurrent first imports of httpx/openai fail ('partially initialized module')."""
    global _CHAT
    if _CHAT is None:
        with _CHAT_LOCK:
            if _CHAT is None:
                _CHAT = build_chat(num_predict=1024)
    return _CHAT


def chat(prompt: str) -> str:
    return str(_chat().invoke(prompt).content)


def _error_classes() -> tuple[tuple, tuple]:
    """(transient, fatal) exception classes for the configured provider."""
    import httpx

    if provider() == "openai":
        import openai

        transient = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                     openai.InternalServerError, httpx.HTTPError, ConnectionError)
        fatal = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
                 openai.BadRequestError)
        return transient, fatal
    from ollama import ResponseError

    return (httpx.HTTPError, ConnectionError, ResponseError), ()


def structured(prompt: str, schema, attempts: int = 3):
    """Pydantic-structured output. Retries only transient errors (down, overloaded, rate-limited,
    timeout); raises ConfigError for bad key/model/request, BadOutput for invalid output, and
    RuntimeError when the endpoint stays unreachable."""
    method = os.getenv("LLM_STRUCTURED_METHOD") or None  # e.g. json_schema | function_calling | json_mode
    kwargs = {"method": method} if method else {}
    llm = _chat().with_structured_output(schema, **kwargs)
    transient, fatal = _error_classes()
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            out = llm.invoke(prompt)
        except fatal as exc:
            raise ConfigError(f"{type(exc).__name__}: {exc}") from exc
        except transient as exc:
            error = exc
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        except Exception as exc:  # JSON / pydantic parse failure
            raise BadOutput(str(exc)) from exc
        if out is None:
            raise BadOutput("empty structured output")
        return out
    raise RuntimeError(f"LLM endpoint unreachable after {attempts} attempts: {error}")
