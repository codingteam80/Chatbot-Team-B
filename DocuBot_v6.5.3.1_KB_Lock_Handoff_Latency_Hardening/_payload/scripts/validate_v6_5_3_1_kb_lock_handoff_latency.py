from __future__ import annotations

import ast
import importlib.util
import json
import os
import py_compile
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "kb_lock_handoff_latency"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.5.3.1_kb_lock_handoff_latency_validation_latest.json"
checks: list[dict] = []


def check(name: str, passed: bool, detail="") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})


def source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        # Some settings are assigned from env expressions first
                        # and then given a literal fallback in a later assignment.
                        continue
    return None


changed = [
    "app.py",
    "retrieval/retriever.py",
    "scripts/kb_update_runner.py",
    "scripts/test_kb_update_portability.py",
]
for rel in changed:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, exc)

app_source = source("app.py")
retriever_source = source("retrieval/retriever.py")
runner_source = source("scripts/kb_update_runner.py")
portability_source = source("scripts/test_kb_update_portability.py")

# Architecture/config invariants are checked from the real settings file without
# importing the application or model stack.
settings_path = ROOT / "config" / "settings.py"
if settings_path.is_file():
    check("Chunk size remains 900", literal_assignment(settings_path, "CHUNK_SIZE") == 900, literal_assignment(settings_path, "CHUNK_SIZE"))
    check("Chunk overlap remains 150", literal_assignment(settings_path, "CHUNK_OVERLAP") == 150, literal_assignment(settings_path, "CHUNK_OVERLAP"))
    check("Minimum retrieval score remains 0.55", literal_assignment(settings_path, "MIN_RETRIEVAL_SCORE") == 0.55, literal_assignment(settings_path, "MIN_RETRIEVAL_SCORE"))
    check("Vector Top-K remains 10", literal_assignment(settings_path, "VECTOR_TOP_K") == 10, literal_assignment(settings_path, "VECTOR_TOP_K"))
    check("BM25 Top-K remains 10", literal_assignment(settings_path, "BM25_TOP_K") == 10, literal_assignment(settings_path, "BM25_TOP_K"))
    check("Final Top-K remains 3", literal_assignment(settings_path, "FINAL_TOP_K") == 3, literal_assignment(settings_path, "FINAL_TOP_K"))
    check("MultiQuery variant count remains 2", literal_assignment(settings_path, "MULTI_QUERY_VARIANT_COUNT") == 2, literal_assignment(settings_path, "MULTI_QUERY_VARIANT_COUNT"))
else:
    check("Settings file exists", False, settings_path)

# Parent/worker ownership invariants.
check("Streamlit never imports update-lock internals", "_update_lock" not in app_source)
check("Streamlit uses session double-click guard", "kb_update_in_progress" in app_source and "disabled=(not rebuild_confirmed or update_in_progress)" in app_source)
check("Streamlit launches isolated worker", "launch_kb_update_subprocess" in app_source)
check("Runner records worker owner role", '"owner_role": "worker"' in runner_source)
check("Runner records run id", '"run_id": str(run_id or "")' in runner_source)
check("Runner records parent pid", '"parent_pid": int(parent_pid or 0)' in runner_source)
check("Runner records process start identity", "process_start_token" in runner_source and "_process_start_token" in runner_source)
check("Launcher does not acquire worker lock", "with _update_lock" not in runner_source[runner_source.find("def launch_kb_update_subprocess"):runner_source.find("def main()")])
check("Launcher performs explicit lock handoff", "_prepare_launcher_lock_handoff" in runner_source)
check("Legacy v6.5.3 parent self-lock cleanup is explicit", "removed_legacy_parent_self_lock" in runner_source)
check("Live worker lock remains blocking", "owner_role" in runner_source and "action\": \"blocked" in runner_source)
check("Worker lock uses atomic exclusive create", "os.O_CREAT | os.O_EXCL | os.O_WRONLY" in runner_source)
check("Worker releases only its own run-id lock", "same_owner" in runner_source and "current.get(\"run_id\"" in runner_source)

# Latency hardening/observability: v6.5.3 acceptance behavior is preserved;
# this hotfix adds explicit proof of whether the expensive MultiQuery path ran.
check("Semantic latency decision event exists", "MISRA SEMANTIC LATENCY DECISION" in retriever_source)
check("Single-query timing is recorded", "single_query_seconds" in retriever_source)
check("MultiQuery generation timing is recorded", "multi_query_generation_seconds" in retriever_source)
check("Widened semantic timing is recorded", "widened_resolution_seconds" in retriever_source)
check("Single-query proven path explicitly skips MultiQuery", "multi_query_skipped_single_query_proven" in retriever_source)
check("Ambiguous path explicitly records MultiQuery use", "multi_query_used_for_ambiguous_single_query" in retriever_source)
latency_method = retriever_source[
    retriever_source.find("def resolve_misra_rule_semantics"):
    retriever_source.find("def _generate_multi_query_searches")
]
check(
    "No Rule/question mapping added to semantic latency orchestration",
    all(token not in latency_method for token in [
        "Rule 17.7", "Rule 21.17", "Rule 16.4",
        "localtime", "strerror", "write_status", "update_state",
    ]),
)

# Unit-test lock handoff in isolation, with no KB/model imports and no production lock.
scripts_pkg = sys.modules.get("scripts")
if scripts_pkg is None:
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = [str(ROOT / "scripts")]
    sys.modules["scripts"] = scripts_pkg

kb_health_stub = types.ModuleType("scripts.kb_health")
kb_health_stub.run_health_check = lambda: 0
sys.modules["scripts.kb_health"] = kb_health_stub

smart_stub = types.ModuleType("scripts.smart_build")
smart_stub.get_kb_update_preflight = lambda docs: {"ok": True}
smart_stub.get_update_plan = lambda: {"mode": "noop", "documents": [], "changes": {}}
smart_stub.smart_build = lambda: True
sys.modules["scripts.smart_build"] = smart_stub

try:
    spec = importlib.util.spec_from_file_location(
        "docubot_kb_update_runner_hotfix_validation",
        ROOT / "scripts" / "kb_update_runner.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory(prefix="docubot_lock_test_") as temp:
        lock_path = Path(temp) / "kb_update.lock"

        no_lock = module._prepare_launcher_lock_handoff(
            source="streamlit_button",
            parent_pid=os.getpid(),
            lock_path=lock_path,
        )
        check("No-lock handoff is a no-op", no_lock.get("action") == "none", no_lock)

        legacy = {
            "pid": os.getpid(),
            "source": "streamlit_button",
            "created": "legacy-v6.5.3",
            "python": sys.executable,
        }
        lock_path.write_text(json.dumps(legacy), encoding="utf-8")
        handoff = module._prepare_launcher_lock_handoff(
            source="streamlit_button",
            parent_pid=os.getpid(),
            lock_path=lock_path,
        )
        check(
            "Field-observed live parent self-lock is removed",
            handoff.get("action") == "removed_legacy_parent_self_lock"
            and not lock_path.exists(),
            handoff,
        )

        with module._update_lock(
            "worker_test",
            run_id="unit-worker-run",
            parent_pid=os.getpid(),
            lock_path=lock_path,
        ):
            payload = module._read_lock(lock_path)
            check("Worker lock is written", lock_path.is_file(), payload)
            check("Worker lock role is worker", payload.get("owner_role") == "worker", payload)
            check("Worker lock run-id is preserved", payload.get("run_id") == "unit-worker-run", payload)
            blocked = module._prepare_launcher_lock_handoff(
                source="streamlit_button",
                parent_pid=os.getpid(),
                lock_path=lock_path,
            )
            check("Live worker cannot be stolen by launcher", blocked.get("action") == "blocked" and lock_path.exists(), blocked)

        check("Worker lock releases after context", not lock_path.exists(), lock_path)

        # A definitely invalid/dead PID must be treated as stale.
        lock_path.write_text(json.dumps({
            "pid": 999999999,
            "source": "streamlit_button",
            "owner_role": "worker",
            "run_id": "dead",
            "process_start_token": "dead",
        }), encoding="utf-8")
        stale = module._prepare_launcher_lock_handoff(
            source="streamlit_button",
            parent_pid=os.getpid(),
            lock_path=lock_path,
        )
        check("Dead worker lock is cleaned safely", stale.get("action") == "removed_stale_lock" and not lock_path.exists(), stale)

except Exception as exc:
    check("Lock handoff unit tests execute", False, f"{type(exc).__name__}: {exc}")

# Portability dry-run itself must be non-mutating and test the real failure mode.
try:
    tree = ast.parse(portability_source)
    direct_smart_build_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "smart_build"
    ]
    check("Portability dry-run never calls smart_build", not direct_smart_build_calls, len(direct_smart_build_calls))
except Exception as exc:
    check("Portability dry-run parses", False, exc)
check("Portability dry-run reproduces legacy parent self-lock", "Legacy Streamlit parent self-lock is handed off safely" in portability_source)
check("Portability dry-run verifies live-worker blocking", "Live worker lock still blocks a second update" in portability_source)

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.3.1.json"
check("v6.5.3.1 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check("Manifest parses", True)
        check("Manifest base is v6.5.3", "v6.5.3" in str(manifest.get("base", "")))
        check("Manifest records worker-only lock ownership", manifest.get("kb_update_worker_only_lock_owner") is True)
        check("Manifest records legacy parent-lock handoff", manifest.get("kb_update_legacy_parent_self_lock_handoff") is True)
        check("Manifest records PID-reuse identity guard", manifest.get("kb_update_process_identity_guard") is True)
        check("Manifest records latency diagnostics", manifest.get("semantic_latency_decision_diagnostics") is True)
        check("Manifest preserves early-accept behavior", manifest.get("single_query_semantic_bge_early_accept_preserved") is True)
        check("Manifest says no patch-time KB rebuild", manifest.get("kb_rebuild_performed_by_patch") is False)
        check("Manifest preserves chunking", manifest.get("chunk_size") == 900 and manifest.get("chunk_overlap") == 150)
        check("Manifest preserves threshold", float(manifest.get("minimum_retrieval_score", -1)) == 0.55)
    except Exception as exc:
        check("Manifest parses", False, exc)

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.5.3.1",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(checks),
    "checks": checks,
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("version", "validation", "failed_checks", "check_count")}, indent=2))
if failed:
    for item in failed:
        print("[FAIL]", item["name"], "::", item.get("detail", ""))
    raise SystemExit(1)
print(f"[PASS] v6.5.3.1 validator: {len(checks)} checks, 0 failures.")
raise SystemExit(0)
