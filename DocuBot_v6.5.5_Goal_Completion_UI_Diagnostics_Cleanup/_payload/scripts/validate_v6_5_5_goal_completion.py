from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "logs" / "v6_5_5_completion" / "validation_latest.json"
CHECKS: list[dict[str, Any]] = []


def check(name: str, passed: bool, detail: Any = "") -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def syntax_ok(rel: str) -> tuple[bool, str]:
    try:
        ast.parse(read(rel), filename=rel)
        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


PYTHON_FILES = [
    "app.py",
    "config/settings.py",
    "ingestion/metadata.py",
    "retrieval/bm25_index.py",
    "retrieval/retriever.py",
    "retrieval/semantic_rule_resolver.py",
    "scripts/kb_update_runner.py",
    "scripts/smart_build.py",
    "scripts/update_qdrant_incremental.py",
    "scripts/test_retrieval_pipeline.py",
    "scripts/test_llm_from_verified_context.py",
    "scripts/test_incremental_kb_lifecycle_live.py",
    "scripts/generate_v6_5_5_completion_benchmark.py",
    "scripts/run_v6_5_5_completion_suite.py",
    "ui/streamlit_ui.py",
]
for rel in PYTHON_FILES:
    ok, detail = syntax_ok(rel)
    check(f"Python syntax: {rel}", ok, detail)

settings = read("config/settings.py")
app = read("app.py")
styles = read("ui/styles.css")
runner = read("scripts/kb_update_runner.py")
smart_build = read("scripts/smart_build.py")
incremental = read("scripts/update_qdrant_incremental.py")
retriever = read("retrieval/retriever.py")
retrieval_test = read("scripts/test_retrieval_pipeline.py")
llm_test = read("scripts/test_llm_from_verified_context.py")
lifecycle = read("scripts/test_incremental_kb_lifecycle_live.py")
suite = read("scripts/run_v6_5_5_completion_suite.py")
deploy = read("docs/deployment/LAN_DEPLOYMENT_GUIDE.txt")
components = read("docs/PRODUCTION_COMPONENTS.md")
manifest = json.loads(read("FINAL_COMPONENT_MANIFEST_v6.5.5.json"))
cleanup = json.loads(read("cleanup_plan_v6.5.5.json"))
bank = json.loads(read("qa/v6_5_5_completion_benchmark_bank.json"))

# Production architecture invariants.
check("Generation model remains qwen2.5:7b", '"qwen2.5:7b"' in settings)
check("Embedding model remains qwen3-embedding:8b", '"qwen3-embedding:8b"' in settings)
check("Reranker remains BAAI/bge-reranker-v2-m3", '"BAAI/bge-reranker-v2-m3"' in settings)
check("Chunk size remains 900", re.search(r"^CHUNK_SIZE\s*=\s*900\s*$", settings, re.M) is not None)
check("Chunk overlap remains 150", re.search(r"^CHUNK_OVERLAP\s*=\s*150\s*$", settings, re.M) is not None)
check("Minimum retrieval score remains 0.55", re.search(r"^MIN_RETRIEVAL_SCORE\s*=\s*0\.55\s*$", settings, re.M) is not None)
check("Vector Top-K remains 10", re.search(r"^VECTOR_TOP_K\s*=\s*10\s*$", settings, re.M) is not None)
check("BM25 Top-K remains 10", re.search(r"^BM25_TOP_K\s*=\s*10\s*$", settings, re.M) is not None)
check("Final Top-K remains 3", re.search(r"^FINAL_TOP_K\s*=\s*3\s*$", settings, re.M) is not None)
check("Exactly two MultiQuery alternatives remain configured by default", 'DOCUBOT_MULTI_QUERY_VARIANTS", "2"' in settings and "MULTI_QUERY_VARIANT_COUNT = 2" in settings)

# UI contract from the user's screenshots/instruction.
check("Incremental confirmation checkbox exists", "process only new, modified" in app and 'key="kb_smart_update_confirm"' in app)
check("Full rebuild uses the same explicit confirmation control", "safely rebuild all source documents" in app and app.count('key="kb_smart_update_confirm"') == 1)
check("Confirmation resets when change-set signature changes", "kb_update_confirm_plan" in app and "_kb_plan_signature(kb_plan)" in app)
check("Update button is disabled until confirmed", "disabled=(not update_confirmed or update_in_progress)" in app)
check("KB worker request triggers a rerun before long work", "kb_update_requested = True" in app and "st.rerun()" in app)
footer_index = app.find("StreamlitUI.render_footer_note()")
deferred_index = app.find("# DEFERRED KNOWLEDGE-BASE UPDATE WORKER")
worker_call_index = app.find("launch_kb_update_subprocess", deferred_index)
check("Long KB worker is deferred until after chat/welcome render", footer_index >= 0 and deferred_index > footer_index and worker_call_index > deferred_index, (footer_index, deferred_index, worker_call_index))
check("Maintenance panel has a dedicated stable keyed region", 'key="kb_maintenance_panel"' in app)
check("KB update does not insert a spinner/status row that shifts layout", "st.spinner(" not in app and 'key="kb_update_status_area"' not in app)
check("Maintenance panel remains in normal flow", ".st-key-kb_maintenance_panel" in styles and "position: fixed" not in styles[styles.rfind("v6.5.5 — KB MAINTENANCE LAYOUT STABILITY"):])
check("Chat submit is disabled while KB update runs", 'st.session_state.get("kb_update_in_progress", False)' in app)

# Working incremental update invariants must remain untouched.
check("Planner still exposes incremental mode", '"incremental"' in smart_build)
check("Incremental updater still snapshots changed sources", "source snapshot" in incremental.casefold())
check("Incremental updater still embeds changed chunks only", "Embedding/upserting only new/modified chunks" in incremental)
check("Incremental updater still removes modified/deleted vectors only", "Removing only modified/deleted document vectors" in incremental)
check("Incremental updater retains source stability guard", "source_stability_guard" in incremental)
check("KB worker version reports v6.5.5", 'VERSION = "v6.5.5"' in runner)
check("Worker-only lock ownership preserved", '"owner_role": "worker"' in runner and "_prepare_launcher_lock_handoff" in runner)
check("Semantic latency decision diagnostics preserved", "MISRA SEMANTIC LATENCY DECISION" in retriever)
check("Single-query proven path can skip MultiQuery", "multi_query_skipped_single_query_proven" in retriever)
check("Ambiguous path still uses MultiQuery", "multi_query_used_for_ambiguous_single_query" in retriever)

# Retrieval-only / generation-only / latency completion diagnostics.
check("Retrieval test never calls answer generation", '"llm_answer_generation_called": False' in retrieval_test and "OllamaClient" not in retrieval_test)
check("Retrieval test reports recall", "recall_passes" in retrieval_test)
check("Retrieval test reports precision", "precision_passes" in retrieval_test)
check("Retrieval test reports semantic consistency", "concept_consistency_passes" in retrieval_test)
check("Retrieval test records MultiQuery used/skipped decisions", "multi_query_used_count" in retrieval_test and "multi_query_skipped_single_query_proven_count" in retrieval_test)
check("Retrieval test reports latency distribution", '"median"' in retrieval_test and '"p95"' in retrieval_test)
check("LLM diagnostic blocks failed retrieval contexts", "retrieval gate failed; LLM generation intentionally not executed" in llm_test)
check("LLM diagnostic supports all verified cases", 'default=0, help="1-based case number; 0 means all retrieval-pass cases"' in llm_test)
check("LLM diagnostic verifies structured reference grounding", "reference_pass" in llm_test and "wrong_references" in llm_test)
check("LLM diagnostic verifies Yes/No polarity where expected", "polarity_pass" in llm_test and "expected_prefix" in llm_test)
check("Completion suite gates retrieval before LLM", suite.find('"retrieval_only"') < suite.find('"llm_from_verified_context"'))
check("Completion suite evaluates 25-second combined target", "TARGET_SECONDS = 25.0" in suite and "combined_seconds" in suite)

# Actual delete lifecycle is available as an explicit safe live probe, never at install time.
check("Live lifecycle requires clean noop baseline", 'initial.get("mode") != "noop"' in lifecycle)
check("Live lifecycle covers add", '_run_stage("add", "added")' in lifecycle)
check("Live lifecycle covers modify", '_run_stage("modify", "updated")' in lifecycle)
check("Live lifecycle covers delete", '_run_stage("delete", "deleted")' in lifecycle)
check("Live lifecycle delete does not embed changed chunks", 'expected_change == "deleted"' in lifecycle and 'embedded_changed_chunks' in lifecycle)
check("Live lifecycle has fail-safe recovery", "v6.5.5_lifecycle_recovery" in lifecycle and "finally:" in lifecycle)

# Test-only benchmark bank remains outside production runtime.
cases = list(bank.get("cases") or [])
concepts = {str(case.get("concept_id") or "") for case in cases if isinstance(case, dict)}
check("Completion benchmark bank contains 24 test-only cases", len(cases) == 24, len(cases))
check("Completion benchmark bank contains 12 semantic concepts", len(concepts) == 12, len(concepts))
production_sources = "\n".join(read(rel) for rel in ["app.py", "retrieval/retriever.py", "services/answer_service.py", "services/misra_compliance.py"])
check("Production runtime does not import completion benchmark bank", "v6_5_5_completion_benchmark_bank" not in production_sources)

# Cleanup/finalization: active docs describe only the current stack; historical
# evidence is preserved rather than deleted by the installer.
active_doc_text = (deploy + "\n" + components).casefold()
check("Active deployment docs contain no Llama3 production reference", "llama3" not in active_doc_text)
check("Active deployment docs contain no Chroma production reference", "chroma" not in active_doc_text)
check("Active deployment docs name current Qdrant store", "option_c_qwen3_qdrant_v4" in deploy)
check("Active component doc names Qwen2.5 generation", "qwen2.5:7b" in components)
check("Cleanup plan archives rather than destroys historical files", all(item.get("action") == "archive_if_exact" for item in cleanup.get("items") or []))
check("Cleanup plan preserves history under evidence/release_history", str(cleanup.get("archive_root", "")).startswith("evidence/release_history/"))
check("Final manifest says historical test evidence preserved", manifest.get("cleanup", {}).get("historical_test_evidence_preserved") is True)
check("Final manifest forbids install-time KB rebuild", manifest.get("install_time_production_kb_rebuild") is False)

# Genericity guard: test expected references must not become production mappings.
for forbidden in ("rule_17_7_return_value_used", "rule_21_17_string_bounds", "rule_16_4_switch_default"):
    check(f"Production runtime has no benchmark concept mapping: {forbidden}", forbidden not in production_sources)

failed = [item for item in CHECKS if not item["passed"]]
report = {
    "version": "v6.5.5",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(CHECKS),
    "checks": CHECKS,
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("version", "validation", "failed_checks", "check_count")}, indent=2))
for item in failed:
    print(f"[FAIL] {item['name']} :: {item['detail']}")
raise SystemExit(0 if not failed else 1)
