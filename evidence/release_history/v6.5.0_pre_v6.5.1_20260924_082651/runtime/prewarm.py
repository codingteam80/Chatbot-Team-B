from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from config.settings import (
    OLLAMA_FAST_KEEP_ALIVE,
    OLLAMA_FAST_MODEL,
    OLLAMA_FORCED_MODEL,
    STARTUP_PREWARM_DELAY_SECONDS,
    STARTUP_PREWARM_MIN_AVAILABLE_RAM_MB,
    STARTUP_PREWARM_MIN_RAM_MB,
    STARTUP_PREWARM_MODE,
    STARTUP_PREWARM_OLLAMA_URL,
)

_LOCK = threading.RLock()
_STARTED = False
_RETRIEVAL_EVENT = threading.Event()
_FAST_LLM_EVENT = threading.Event()
_STATUS: dict[str, Any] = {
    "enabled": False,
    "started": False,
    "retrieval": {"state": "not_started", "seconds": 0.0, "error": ""},
    "fast_llm": {"state": "not_started", "seconds": 0.0, "error": ""},
}


def _physical_memory_mb() -> tuple[float, float]:
    """Best-effort (total, available) physical RAM detection."""

    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                scale = 1024.0 * 1024.0
                return (
                    float(status.ullTotalPhys) / scale,
                    float(status.ullAvailPhys) / scale,
                )
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        scale = 1024.0 * 1024.0
        return (
            float(page_size * page_count) / scale,
            float(page_size * available_pages) / scale,
        )
    except Exception:
        return 0.0, 0.0


def background_prewarm_enabled(
    *,
    total_ram_mb: float | None = None,
    available_ram_mb: float | None = None,
) -> bool:
    """Resolve on/off/auto startup prewarm behavior with memory headroom."""

    mode = str(STARTUP_PREWARM_MODE or "auto").strip().lower()

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
        total >= float(STARTUP_PREWARM_MIN_RAM_MB)
        and available >= float(STARTUP_PREWARM_MIN_AVAILABLE_RAM_MB)
    )


def _update_component(name: str, **values: Any) -> None:
    with _LOCK:
        current = dict(_STATUS.get(name) or {})
        current.update(values)
        _STATUS[name] = current


def get_prewarm_status() -> dict[str, Any]:
    with _LOCK:
        return json.loads(json.dumps(_STATUS, default=str))


def prewarm_retrieval_sync() -> dict[str, Any]:
    """Load the exact cached vector/embed/reranker resources used in production."""

    started = time.perf_counter()
    _update_component("retrieval", state="running", error="")

    try:
        # Lazy imports are intentional: deterministic structured-only sessions
        # should remain able to skip all semantic ML resources when prewarm is off.
        from embeddings.embedding_model import get_embedding_model
        from retrieval.reranker import get_reranker_model

        from retrieval.qdrant_search import qdrant_collection_count
        qdrant_collection_count()

        embed_model = get_embedding_model()
        # Warm one tiny inference so first real semantic query does not also pay
        # framework/kernel initialization. This vector is never stored.
        embed_model.get_text_embedding("query: docubot runtime readiness")

        # v6.5.0: build/load the small Rule-level semantic cache while startup
        # prewarm is already paying the embedding-model readiness cost.  The
        # production Qdrant/BM25 stores are never rebuilt or modified.
        from retrieval.semantic_rule_resolver import build_semantic_rule_index
        semantic_index = build_semantic_rule_index(embedding_model=embed_model)

        reranker = get_reranker_model()
        reranker.predict(
            [["docubot readiness", "docubot readiness"]],
            batch_size=1,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - started
        _update_component("retrieval", state="ready", seconds=round(elapsed, 4))
        return {
            "ok": True,
            "seconds": round(elapsed, 4),
            "semantic_rule_index": semantic_index,
        }

    except Exception as error:
        elapsed = time.perf_counter() - started
        message = f"{type(error).__name__}: {error}"
        _update_component(
            "retrieval",
            state="failed",
            seconds=round(elapsed, 4),
            error=message,
        )
        return {"ok": False, "seconds": round(elapsed, 4), "error": message}

    finally:
        _RETRIEVAL_EVENT.set()


def _post_ollama_generate(payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    url = str(STARTUP_PREWARM_OLLAMA_URL or "http://127.0.0.1:11434").rstrip("/")
    request = urllib.request.Request(
        url + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def prewarm_fast_llm_sync() -> dict[str, Any]:
    """Ask local Ollama to load the configured fast model without generation."""

    started = time.perf_counter()
    _update_component("fast_llm", state="running", error="")

    if OLLAMA_FORCED_MODEL:
        _update_component(
            "fast_llm",
            state="skipped",
            seconds=0.0,
            error="forced-model diagnostic override is active",
        )
        _FAST_LLM_EVENT.set()
        return {
            "ok": True,
            "skipped": True,
            "seconds": 0.0,
            "reason": "forced-model diagnostic override is active",
        }

    try:
        payload = {
            "model": OLLAMA_FAST_MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": OLLAMA_FAST_KEEP_ALIVE,
        }
        response = _post_ollama_generate(payload)
        elapsed = time.perf_counter() - started
        _update_component("fast_llm", state="ready", seconds=round(elapsed, 4))
        return {
            "ok": True,
            "seconds": round(elapsed, 4),
            "done_reason": str(response.get("done_reason") or ""),
        }

    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        elapsed = time.perf_counter() - started
        message = f"{type(error).__name__}: {error}"
        _update_component(
            "fast_llm",
            state="failed",
            seconds=round(elapsed, 4),
            error=message,
        )
        return {"ok": False, "seconds": round(elapsed, 4), "error": message}

    finally:
        _FAST_LLM_EVENT.set()


def prewarm_all_sync(*, force: bool = False) -> dict[str, Any]:
    """Run both startup warmups concurrently and wait for validation output."""

    enabled = force or background_prewarm_enabled()
    with _LOCK:
        _STATUS["enabled"] = bool(enabled)
        _STATUS["started"] = bool(enabled)

    if not enabled:
        return {"enabled": False, "retrieval": {}, "fast_llm": {}}

    results: dict[str, Any] = {}

    def run_retrieval():
        results["retrieval"] = prewarm_retrieval_sync()

    def run_fast_llm():
        results["fast_llm"] = prewarm_fast_llm_sync()

    retrieval_thread = threading.Thread(target=run_retrieval, daemon=True)
    fast_thread = threading.Thread(target=run_fast_llm, daemon=True)
    retrieval_thread.start()
    fast_thread.start()
    retrieval_thread.join()
    fast_thread.join()

    return {
        "enabled": True,
        "retrieval": results.get("retrieval", {}),
        "fast_llm": results.get("fast_llm", {}),
    }


def _retrieval_worker(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)
    result = prewarm_retrieval_sync()
    print(f"[PREWARM] Retrieval: {result}")


def _fast_llm_worker(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)
    result = prewarm_fast_llm_sync()
    print(f"[PREWARM] Fast LLM: {result}")


def start_background_prewarm() -> dict[str, Any]:
    """Start one bounded process-local startup warmup and return immediately."""

    global _STARTED

    enabled = background_prewarm_enabled()

    with _LOCK:
        _STATUS["enabled"] = bool(enabled)
        if not enabled:
            return get_prewarm_status()
        if _STARTED:
            return get_prewarm_status()
        _STARTED = True
        _STATUS["started"] = True

    delay = max(0.0, float(STARTUP_PREWARM_DELAY_SECONDS))

    # Retrieval and Ollama are separate runtimes. Running the two warmups in
    # parallel reduces wall-clock readiness while the 12-GB auto RAM guard
    # prevents this eager path on genuinely memory-constrained PCs.
    threading.Thread(
        target=_retrieval_worker,
        args=(delay,),
        name="docubot-retrieval-prewarm",
        daemon=True,
    ).start()
    threading.Thread(
        target=_fast_llm_worker,
        args=(delay,),
        name="docubot-fast-llm-prewarm",
        daemon=True,
    ).start()

    return get_prewarm_status()


def wait_for_retrieval_prewarm(timeout: float | None = None) -> bool:
    """Allow a real semantic query to reuse an in-flight warmup safely."""

    with _LOCK:
        state = str((_STATUS.get("retrieval") or {}).get("state") or "")

    if state != "running":
        return state == "ready"

    _RETRIEVAL_EVENT.wait(timeout=timeout)
    with _LOCK:
        return str((_STATUS.get("retrieval") or {}).get("state") or "") == "ready"
