from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from scripts import smart_build

checks = []


def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


# Architecture/profile lock: this recovery patch must not tune retrieval quality.
check("generation model unchanged", settings.OLLAMA_FAST_MODEL == "qwen2.5:7b" and settings.OLLAMA_COMPLEX_MODEL == "qwen2.5:7b", f"{settings.OLLAMA_FAST_MODEL}/{settings.OLLAMA_COMPLEX_MODEL}")
check("embedding model unchanged", settings.EMBED_MODEL_NAME == "qwen3-embedding:8b", settings.EMBED_MODEL_NAME)
check("reranker model unchanged", settings.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3", settings.RERANKER_MODEL)
check("chunk size unchanged", settings.CHUNK_SIZE == 900, settings.CHUNK_SIZE)
check("chunk overlap unchanged", settings.CHUNK_OVERLAP == 150, settings.CHUNK_OVERLAP)
check("global threshold unchanged", abs(float(settings.MIN_RETRIEVAL_SCORE) - 0.55) < 1e-12, settings.MIN_RETRIEVAL_SCORE)
check("vector top-k unchanged", settings.VECTOR_TOP_K == 10, settings.VECTOR_TOP_K)
check("bm25 top-k unchanged", settings.BM25_TOP_K == 10, settings.BM25_TOP_K)
check("final top-k unchanged", settings.FINAL_TOP_K == 3, settings.FINAL_TOP_K)

# Error classification keeps live locks/permissions fail-closed while allowing
# non-lock read failures to enter source-rebuild recovery.
class FakeError(Exception):
    pass

check(
    "Qdrant concurrent-owner error is classified locked",
    smart_build._qdrant_storage_error_kind(
        FakeError("Storage folder is already accessed by another instance of Qdrant client")
    ) == "locked",
)
check(
    "Windows sharing error is classified locked",
    smart_build._qdrant_storage_error_kind(FakeError("[WinError 32] file is being used by another process")) == "locked",
)
check(
    "permission failure stays fail-closed",
    smart_build._qdrant_storage_error_kind(FakeError("[WinError 5] Access is denied")) == "permission",
)
check(
    "corrupt/incompatible open failure is recoverable kind",
    smart_build._qdrant_storage_error_kind(FakeError("invalid local storage metadata")) == "unreadable",
)

# Planner proof: an unreadable active store with source documents must request
# a full rebuild recovery instead of pretending the collection merely has 0 rows.
orig_state = smart_build._qdrant_storage_state
try:
    smart_build._qdrant_storage_state = lambda: {
        "path": str(settings.QDRANT_DIR),
        "exists": True,
        "readable": False,
        "count": None,
        "error": "FakeError: invalid local storage metadata",
        "error_kind": "unreadable",
    }
    plan = smart_build.get_update_plan()
    check("unreadable active store plans full rebuild", plan.get("mode") == "full_rebuild", plan.get("mode"))
    check("planner records Qdrant recovery requirement", bool(plan.get("qdrant_recovery_required")), plan.get("reason"))
    check("planner preserves exact storage error evidence", "invalid local storage metadata" in str(plan.get("collection_error") or ""), plan.get("collection_error"))
finally:
    smart_build._qdrant_storage_state = orig_state

# Preflight policy proof without touching active indexes or Ollama.
orig_fs = smart_build._filesystem_preflight
orig_module = smart_build._module_available
orig_qp = smart_build._qdrant_storage_preflight
orig_ollama = smart_build._ollama_embedding_preflight
try:
    smart_build._filesystem_preflight = lambda docs: ([], [], {"simulated": True})
    smart_build._module_available = lambda name: True
    smart_build._ollama_embedding_preflight = lambda: (True, "")
    docs = list(smart_build.get_all_documents())

    smart_build._qdrant_storage_preflight = lambda: (
        False,
        "simulated unreadable active Qdrant",
        "unreadable",
    )
    recovery = smart_build.get_kb_update_preflight(
        docs,
        requires_embedding=True,
        allow_unreadable_qdrant_recovery=True,
    )
    check("full rebuild preflight allows unreadable-store recovery", recovery.get("ok"), recovery.get("issues"))
    check("recovery use is explicitly recorded", bool((recovery.get("qdrant_recovery") or {}).get("used")), recovery.get("qdrant_recovery"))
    check("recovery still requires embedding readiness", any("bootstrap a new staging vector store" in str(x) for x in recovery.get("notes") or []), recovery.get("notes"))

    no_recovery = smart_build.get_kb_update_preflight(
        docs,
        requires_embedding=True,
        allow_unreadable_qdrant_recovery=False,
    )
    check("ordinary update still blocks unreadable active store", not no_recovery.get("ok"), no_recovery.get("issues"))

    smart_build._qdrant_storage_preflight = lambda: (
        False,
        "simulated live Qdrant owner",
        "locked",
    )
    locked = smart_build.get_kb_update_preflight(
        docs,
        requires_embedding=True,
        allow_unreadable_qdrant_recovery=True,
    )
    check("full rebuild recovery does not bypass live Qdrant lock", not locked.get("ok"), locked.get("issues"))
    check("locked store never marked recovery-used", not bool((locked.get("qdrant_recovery") or {}).get("used")), locked.get("qdrant_recovery"))

    smart_build._qdrant_storage_preflight = lambda: (
        False,
        "simulated access denied",
        "permission",
    )
    denied = smart_build.get_kb_update_preflight(
        docs,
        requires_embedding=True,
        allow_unreadable_qdrant_recovery=True,
    )
    check("full rebuild recovery does not bypass permission failure", not denied.get("ok"), denied.get("issues"))
finally:
    smart_build._filesystem_preflight = orig_fs
    smart_build._module_available = orig_module
    smart_build._qdrant_storage_preflight = orig_qp
    smart_build._ollama_embedding_preflight = orig_ollama

smart_source = (ROOT / "scripts" / "smart_build.py").read_text(encoding="utf-8")
runner_source = (ROOT / "scripts" / "kb_update_runner.py").read_text(encoding="utf-8")
full_source = (ROOT / "scripts" / "build_qdrant_index.py").read_text(encoding="utf-8")
probe_source = (ROOT / "scripts" / "test_v6_5_5_16_1_isolated_storage_recovery.py").read_text(encoding="utf-8")

check("web/BAT worker authorizes recovery only for full rebuild", 'plan.get("mode") == "full_rebuild"' in runner_source and 'plan.get("qdrant_recovery_required")' in runner_source)
check("smart build authorizes recovery only for full rebuild", 'plan.get("mode") == "full_rebuild"' in smart_source and 'allow_unreadable_qdrant_recovery' in smart_source)
check("full rebuild embeds into sibling staging before activation", "__staging__" in full_source and "Embedding with Qwen3 and writing staging Qdrant index" in full_source)
check("full rebuild activation retains rollback directory until commit", "_activate_staging_directory" in full_source and "_rollback_active_directory" in full_source and "_commit_active_directory" in full_source)
check("transaction cleanup tolerates malformed file path", "path.is_symlink() or path.is_file()" in full_source and "path.unlink(missing_ok=True)" in full_source)
check("isolated live recovery probe uses storage override", "DOCUBOT_OPTION_C_STORAGE_DIR" in probe_source)
check("isolated live recovery probe limits source to one existing test document", "Employee_Leave_Test.txt" in probe_source and "only_probe = lambda" in probe_source and "smart_build.get_all_documents = only_probe" in probe_source)
check("isolated live recovery probe intentionally starts from unreadable Qdrant path", "INTENTIONALLY_UNREADABLE_QDRANT_STORAGE" in probe_source)
check("production code contains no manual copy-storage workaround", "copy paste" not in smart_source.casefold() and "copy another storage" not in smart_source.casefold())

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.5.5.16.1",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": [item["name"] for item in failed],
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if not failed else 1)
