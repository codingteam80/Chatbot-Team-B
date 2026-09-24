from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "environment_setup"
OLLAMA_MODELS = ("qwen2.5:7b", "qwen3-embedding:8b")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run(cmd, *, check=False, capture=False, cwd=None):
    kwargs = {
        "cwd": cwd or ROOT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": check,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, **kwargs)


def _find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    prog = os.environ.get("ProgramFiles")
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    if prog:
        candidates.append(Path(prog) / "Ollama" / "ollama.exe")
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _install_ollama_with_winget() -> str:
    winget = shutil.which("winget")
    if not winget:
        raise RuntimeError(
            "Ollama is not installed and WinGet is unavailable. "
            "Install Ollama for Windows from https://ollama.com/download/windows, "
            "then run Setup_DocuBot_Production_Environment.bat again."
        )
    print("[SETUP] Ollama is missing. Installing Ollama.Ollama with WinGet...")
    result = _run(
        [
            winget, "install", "--exact", "--id", "Ollama.Ollama",
            "--accept-package-agreements", "--accept-source-agreements",
        ],
        check=False,
    )
    if result.returncode not in (0,):
        # WinGet sometimes returns nonzero when the package is already present;
        # always re-detect before deciding it failed.
        print(f"[SETUP] WinGet returned {result.returncode}; checking installation...")
    exe = _find_ollama()
    if not exe:
        raise RuntimeError(
            "Ollama installation did not become available. "
            "Open a new terminal or install from https://ollama.com/download/windows, "
            "then run the setup BAT again."
        )
    return exe


def _api_ready() -> bool:
    try:
        import requests
        from config import settings
        url = settings.EMBED_OLLAMA_URL.rstrip("/") + "/api/tags"
        response = requests.get(url, timeout=3)
        return response.ok
    except Exception:
        return False


def _ensure_ollama_service(ollama_exe: str) -> None:
    if _api_ready():
        print("[OK] Ollama API is already running.")
        return

    print("[SETUP] Starting local Ollama service...")
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    subprocess.Popen(
        [ollama_exe, "serve"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=(os.name != "nt"),
    )
    for _ in range(60):
        if _api_ready():
            print("[OK] Ollama API is ready.")
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama was started but its local API did not become ready.")


def _ollama_list(ollama_exe: str) -> str:
    result = _run([ollama_exe, "list"], capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"'ollama list' failed:\n{result.stdout or ''}")
    return result.stdout or ""


def _ensure_ollama_model(ollama_exe: str, model: str) -> str:
    current = _ollama_list(ollama_exe).lower()
    if model.lower() in current:
        print(f"[OK] Ollama model present: {model}")
        return "present"
    print(f"[SETUP] Pulling Ollama model: {model}")
    result = _run([ollama_exe, "pull", model])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull Ollama model: {model}")
    current = _ollama_list(ollama_exe).lower()
    if model.lower() not in current:
        raise RuntimeError(f"Ollama pull completed but model is not listed: {model}")
    print(f"[OK] Ollama model ready: {model}")
    return "downloaded"


def _verify_python_runtime() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"DocuBot production baseline uses Python 3.11; current venv is "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
        )
    print(f"[OK] Python {sys.version.split()[0]} (production 3.11 line)")


def _verify_imports() -> dict[str, str]:
    modules = {
        "streamlit": "streamlit",
        "qdrant_client": "qdrant_client",
        "rank_bm25": "rank_bm25",
        "torch": "torch",
        "sentence_transformers": "sentence_transformers",
        "transformers": "transformers",
        "llama_index_ollama": "llama_index.llms.ollama",
    }
    versions = {}
    for label, mod_name in modules.items():
        mod = importlib.import_module(mod_name)
        version = getattr(mod, "__version__", "installed")
        versions[label] = str(version)
        print(f"[OK] Python dependency: {label} ({version})")
    return versions


def _verify_canonical_settings() -> dict[str, object]:
    from config import settings
    checks = {
        "knowledge_profile": settings.KNOWLEDGE_PROFILE,
        "vector_backend": settings.VECTOR_BACKEND,
        "chunking_profile": settings.CHUNKING_PROFILE,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "embedding_model": settings.EMBED_MODEL_NAME,
        "fast_model": settings.FAST_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
        "reranker": settings.RERANKER_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "vector_top_k": settings.VECTOR_TOP_K,
        "bm25_top_k": settings.BM25_TOP_K,
        "final_top_k": settings.FINAL_TOP_K,
    }
    expected = {
        "knowledge_profile": "technical",
        "vector_backend": "qdrant",
        "chunking_profile": "v4",
        "embedding_backend": "ollama",
        "embedding_model": "qwen3-embedding:8b",
        "fast_model": "qwen2.5:7b",
        "complex_model": "qwen2.5:7b",
        "reranker": RERANKER_MODEL,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "vector_top_k": 10,
        "bm25_top_k": 10,
        "final_top_k": 3,
    }
    errors = []
    for key, wanted in expected.items():
        actual = checks[key]
        if actual != wanted:
            errors.append(f"{key}: actual={actual!r} expected={wanted!r}")
    if errors:
        raise RuntimeError("Canonical production settings mismatch: " + "; ".join(errors))
    print("[OK] Canonical production settings match v6.4.83 manifest.")
    return checks


def _ensure_reranker_cache() -> dict[str, str]:
    print(f"[SETUP] Verifying/downloading reranker: {RERANKER_MODEL}")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=RERANKER_MODEL,
            local_files_only=True,
        )
        source = "already_cached"
        print("[OK] Reranker weights already cached.")
    except Exception:
        snapshot = snapshot_download(repo_id=RERANKER_MODEL)
        source = "downloaded"
        print("[OK] Reranker weights downloaded.")

    # Load from the resolved local snapshot to prove the cache is usable
    # without changing production reranker behavior.
    import torch
    from sentence_transformers import CrossEncoder

    _ = CrossEncoder(
        str(snapshot),
        device="cpu",
        max_length=512,
        activation_fn=torch.nn.Sigmoid(),
    )
    print("[OK] BGE reranker loaded successfully from local cache.")
    return {"model": RERANKER_MODEL, "cache_path": str(snapshot), "status": source}


def _run_kb_health() -> dict[str, object]:
    cmd = [sys.executable, "-X", "utf8", "-u", "-m", "scripts.kb_health"]
    print("[SETUP] Running read-only production KB health...")
    result = _run(cmd, capture=True)
    output = result.stdout or ""
    print(output, end="" if output.endswith("\n") else "\n")
    return {"exit_code": int(result.returncode), "output_tail": output[-6000:]}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "version": "v6.4.83",
        "phase": "production_environment_setup",
        "created": _now(),
        "status": "FAIL",
        "python": {},
        "ollama": {},
        "reranker": {},
        "canonical_settings": {},
        "kb_health": {},
        "error": None,
    }

    try:
        free = shutil.disk_usage(ROOT).free / (1024 ** 3)
        print("=" * 78)
        print("DocuBot v6.4.83 - Production Environment Setup")
        print("=" * 78)
        print(f"Project root : {ROOT}")
        print(f"Free disk    : {free:.1f} GB")
        if free < 20:
            print("[WARN] Less than 20 GB free disk. Model downloads may not fit.")

        _verify_python_runtime()
        report["python"]["version"] = sys.version.split()[0]
        report["python"]["dependencies"] = _verify_imports()
        report["canonical_settings"] = _verify_canonical_settings()

        ollama_exe = _find_ollama() or _install_ollama_with_winget()
        report["ollama"]["exe"] = ollama_exe
        print(f"[OK] Ollama executable: {ollama_exe}")
        _ensure_ollama_service(ollama_exe)

        model_status = {}
        for model in OLLAMA_MODELS:
            model_status[model] = _ensure_ollama_model(ollama_exe, model)
        report["ollama"]["models"] = model_status
        report["ollama"]["api_ready"] = True

        report["reranker"] = _ensure_reranker_cache()

        kb = _run_kb_health()
        report["kb_health"] = kb
        if kb["exit_code"] != 0:
            raise RuntimeError(
                "Python/models are installed, but the read-only KB health check failed. "
                "Do not rebuild automatically; review the KB-health output."
            )

        report["status"] = "PASS"

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print()
        print(f"[FAIL] {report['error']}")

    report_path = LOG_DIR / f"environment_setup_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path = LOG_DIR / f"environment_setup_{ts}.txt"
    lines = [
        "DocuBot v6.4.83 Production Environment Setup",
        "=" * 72,
        f"Created: {report['created']}",
        f"Status : {report['status']}",
        f"Python : {report.get('python', {}).get('version', '')}",
        f"Error  : {report.get('error') or ''}",
        "",
        "JSON report:",
        str(report_path),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    print("ENVIRONMENT SETUP", report["status"])
    print("=" * 78)
    print("Report:", report_path)
    if report["status"] == "PASS":
        print("No KB rebuild was performed.")
        print("Normal use: open the project in VS Code and run run.py.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
