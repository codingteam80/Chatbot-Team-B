from __future__ import annotations

from typing import Dict
import time

# Lazy adapter symbol kept for backward-compatible monkeypatching in tests.
Ollama = None

from config.settings import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_FAST_MODEL,
    OLLAMA_COMPLEX_MODEL,
    OLLAMA_FORCED_MODEL,
    OLLAMA_FAST_CONTEXT_WINDOW,
    OLLAMA_COMPLEX_CONTEXT_WINDOW,
    OLLAMA_KEEP_ALIVE_OVERRIDE,
    OLLAMA_FAST_KEEP_ALIVE,
    OLLAMA_COMPLEX_KEEP_ALIVE,
    POST_COMPLEX_FAST_RECOVERY_WAIT_SECONDS,
)


# =====================================================
# Cached Ollama clients, one lightweight client per model
# =====================================================

_ollama_clients: Dict[str, object] = {}


def _normalize_model_name(model_name: str | None) -> str:
    clean = str(model_name or "").strip()
    return clean or OLLAMA_MODEL


def _is_transient_connection_error(error: Exception) -> bool:
    """Return True only for quick local connection/session interruptions.

    Read/generation timeouts are intentionally not retried because repeating a
    long model call would make the user wait twice as long.
    """

    name = type(error).__name__.casefold()
    detail = str(error or "").casefold()
    if "timeout" in name or "timed out" in detail or "timeout" in detail:
        return False
    markers = (
        "connecterror",
        "connectionerror",
        "connection refused",
        "connection reset",
        "server disconnected",
        "remoteprotocolerror",
        "failed to establish",
    )
    return any(marker in name or marker in detail for marker in markers)


def _model_runtime_policy(model_name: str) -> tuple[int, str]:
    """Return context-window and keep-alive settings for one routed model."""

    resolved = _normalize_model_name(model_name)

    if resolved == OLLAMA_COMPLEX_MODEL and OLLAMA_COMPLEX_MODEL != OLLAMA_FAST_MODEL:
        context_window = OLLAMA_COMPLEX_CONTEXT_WINDOW
        keep_alive = OLLAMA_COMPLEX_KEEP_ALIVE
    else:
        context_window = OLLAMA_FAST_CONTEXT_WINDOW
        keep_alive = OLLAMA_FAST_KEEP_ALIVE

    if OLLAMA_KEEP_ALIVE_OVERRIDE:
        keep_alive = OLLAMA_KEEP_ALIVE_OVERRIDE

    return int(context_window), str(keep_alive)


def get_ollama(model_name: str | None = None):
    """Return a cached LlamaIndex Ollama client for one model name.

    Creating the Python client does not force every configured model to load
    into RAM/VRAM at DocuBot startup. Ollama loads a model only when that model
    is actually used for generation.
    """

    resolved_model = _normalize_model_name(model_name)

    client = _ollama_clients.get(resolved_model)

    if client is None:
        # Import the LlamaIndex adapter only when generation is actually needed.
        # This keeps a new Streamlit process lighter when the first request can
        # be answered deterministically from structured company knowledge.
        global Ollama
        if Ollama is None:
            from llama_index.llms.ollama import Ollama as _Ollama
            Ollama = _Ollama

        context_window, keep_alive = _model_runtime_policy(resolved_model)

        print(
            f"[OLLAMA] Preparing model client: {resolved_model} "
            f"(context={context_window}, keep_alive={keep_alive})"
        )

        client = Ollama(
            model=resolved_model,
            request_timeout=OLLAMA_TIMEOUT,
            temperature=0.0,
            context_window=context_window,
            keep_alive=keep_alive,
        )

        _ollama_clients[resolved_model] = client

        print(f"[OLLAMA] Client ready: {resolved_model}")

    return client


class OllamaClient:
    """Small lazy wrapper around a specific Ollama model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = _normalize_model_name(model_name)
        self.context_window, self.keep_alive = _model_runtime_policy(self.model_name)

    @property
    def llm(self):
        return get_ollama(self.model_name)

    def generate(self, prompt: str):
        # If a completed complex answer is already restoring the fast model in
        # the background, reuse that one load instead of racing a duplicate
        # Ollama model-switch request.  This affects residency only.
        if self.model_name == OLLAMA_FAST_MODEL and not OLLAMA_FORCED_MODEL:
            try:
                from runtime.model_residency import wait_for_fast_model_recovery

                wait_for_fast_model_recovery(
                    timeout=float(POST_COMPLEX_FAST_RECOVERY_WAIT_SECONDS)
                )
            except Exception:
                # Residency optimization must never block normal generation.
                pass

        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as error:
            if not _is_transient_connection_error(error):
                raise

            # Ollama can briefly reset a local keep-alive connection while a
            # model is being reloaded. Drop the cached Python adapter and retry
            # exactly once. This does not retry long read/generation timeouts.
            _ollama_clients.pop(self.model_name, None)
            time.sleep(0.4)
            response = get_ollama(self.model_name).complete(prompt)
            return str(response)
