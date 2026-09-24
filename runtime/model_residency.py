from __future__ import annotations

import json
import threading
import time
from typing import Any

from config.settings import (
    OLLAMA_COMPLEX_MODEL,
    OLLAMA_FAST_KEEP_ALIVE,
    OLLAMA_FAST_MODEL,
    OLLAMA_FORCED_MODEL,
    POST_COMPLEX_FAST_RECOVERY_DELAY_SECONDS,
    POST_COMPLEX_FAST_RECOVERY_MIN_AVAILABLE_RAM_MB,
    POST_COMPLEX_FAST_RECOVERY_MIN_RAM_MB,
    POST_COMPLEX_FAST_RECOVERY_MODE,
)
from runtime.prewarm import _physical_memory_mb, _post_ollama_generate

_LOCK = threading.RLock()
_EVENT = threading.Event()
_EVENT.set()
_STATUS: dict[str, Any] = {
    "state": "idle",
    "seconds": 0.0,
    "error": "",
    "trigger_model": "",
}


def post_complex_fast_recovery_enabled(
    *,
    total_ram_mb: float | None = None,
    available_ram_mb: float | None = None,
) -> bool:
    """Return whether automatic fast-model recovery is safe on this machine."""

    mode = str(POST_COMPLEX_FAST_RECOVERY_MODE or "auto").strip().lower()
    if mode in {"0", "false", "off", "no", "disabled"}:
        return False
    if mode in {"1", "true", "on", "yes", "enabled"}:
        return True

    detected_total, detected_available = _physical_memory_mb()
    total = detected_total if total_ram_mb is None else float(total_ram_mb)
    available = (
        detected_available
        if available_ram_mb is None
        else float(available_ram_mb)
    )
    return (
        total >= float(POST_COMPLEX_FAST_RECOVERY_MIN_RAM_MB)
        and available >= float(POST_COMPLEX_FAST_RECOVERY_MIN_AVAILABLE_RAM_MB)
    )


def get_fast_model_recovery_status() -> dict[str, Any]:
    with _LOCK:
        return json.loads(json.dumps(_STATUS, default=str))


def _set_status(**values: Any) -> None:
    with _LOCK:
        _STATUS.update(values)


def _recovery_worker(trigger_model: str) -> None:
    delay = max(0.0, float(POST_COMPLEX_FAST_RECOVERY_DELAY_SECONDS))
    if delay:
        time.sleep(delay)

    started = time.perf_counter()
    _set_status(state="running", error="", trigger_model=trigger_model)

    try:
        response = _post_ollama_generate(
            {
                "model": OLLAMA_FAST_MODEL,
                "prompt": "",
                "stream": False,
                "keep_alive": OLLAMA_FAST_KEEP_ALIVE,
            }
        )
        elapsed = time.perf_counter() - started
        _set_status(
            state="ready",
            seconds=round(elapsed, 4),
            error="",
            done_reason=str(response.get("done_reason") or ""),
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        _set_status(
            state="failed",
            seconds=round(elapsed, 4),
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        _EVENT.set()


def schedule_fast_model_recovery_after_complex(
    trigger_model: str | None,
) -> dict[str, Any]:
    """Restore the fast Ollama model after a completed complex-model answer.

    The work is deliberately asynchronous.  It starts only after AnswerService
    has finished all complex-model generation/verification, so it cannot evict
    Qwen while that answer is still being produced.
    """

    resolved = str(trigger_model or "").strip()
    if (
        OLLAMA_FORCED_MODEL
        or not resolved
        or resolved != OLLAMA_COMPLEX_MODEL
        or OLLAMA_COMPLEX_MODEL == OLLAMA_FAST_MODEL
        or not post_complex_fast_recovery_enabled()
    ):
        return {"scheduled": False, "reason": "not_applicable"}

    with _LOCK:
        state = str(_STATUS.get("state") or "idle")
        if state in {"scheduled", "running"}:
            return {"scheduled": False, "reason": "already_in_progress"}

        _STATUS.update(
            {
                "state": "scheduled",
                "seconds": 0.0,
                "error": "",
                "trigger_model": resolved,
            }
        )
        _EVENT.clear()

    threading.Thread(
        target=_recovery_worker,
        args=(resolved,),
        name="docubot-fast-model-recovery",
        daemon=True,
    ).start()
    return {"scheduled": True, "trigger_model": resolved}


def wait_for_fast_model_recovery(timeout: float | None = None) -> bool:
    """Reuse an in-flight background fast-model load instead of duplicating it."""

    with _LOCK:
        state = str(_STATUS.get("state") or "idle")

    if state not in {"scheduled", "running"}:
        return state == "ready"

    _EVENT.wait(timeout=timeout)
    with _LOCK:
        return str(_STATUS.get("state") or "") == "ready"


def _reset_for_tests() -> None:
    global _STATUS
    with _LOCK:
        _STATUS = {
            "state": "idle",
            "seconds": 0.0,
            "error": "",
            "trigger_model": "",
        }
        _EVENT.set()
