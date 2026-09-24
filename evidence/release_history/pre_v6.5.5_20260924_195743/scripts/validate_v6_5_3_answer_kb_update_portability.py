from __future__ import annotations

import ast
import json
import py_compile
import sys
import types
from pathlib import Path

import numpy as np

# Lightweight import shims for sandbox/static validation only.
if "streamlit" not in sys.modules:
    st = types.ModuleType("streamlit")
    def _identity_cache(*args, **kwargs):
        def decorator(fn):
            fn.clear = lambda: None
            return fn
        return decorator
    st.cache_resource = _identity_cache
    st.cache_data = _identity_cache
    st.session_state = {}
    sys.modules["streamlit"] = st
if "rank_bm25" not in sys.modules:
    rb = types.ModuleType("rank_bm25")
    class BM25Okapi:
        def __init__(self, *args, **kwargs):
            pass
    rb.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = rb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from retrieval.semantic_rule_resolver import SemanticRuleResolver, _load_bm25_records, extract_rule_profiles
from services.answer_service import AnswerService
from services.misra_compliance import MisraComplianceMode
import scripts.smart_build as smart_build_module

OUTDIR = ROOT / "logs" / "answer_kb_update_portability"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.5.3_answer_kb_update_portability_validation_latest.json"
checks = []

def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})

def source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

changed = [
    "app.py",
    "services/answer_service.py",
    "services/misra_compliance.py",
    "retrieval/semantic_rule_resolver.py",
    "runtime/prewarm.py",
    "scripts/smart_build.py",
    "scripts/kb_update_runner.py",
    "scripts/update_server_kb.py",
    "scripts/test_retrieval_pipeline.py",
    "scripts/test_llm_from_verified_context.py",
    "scripts/test_kb_update_portability.py",
]
for rel in changed:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, exc)

# Production architecture locks.
check("Generation model remains qwen2.5:7b", settings.OLLAMA_FAST_MODEL == "qwen2.5:7b" and settings.OLLAMA_COMPLEX_MODEL == "qwen2.5:7b")
check("Embedding model remains qwen3-embedding:8b", settings.EMBED_MODEL_NAME == "qwen3-embedding:8b")
check("Reranker remains BAAI/bge-reranker-v2-m3", settings.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3")
check("Minimum retrieval score remains 0.55", abs(float(settings.MIN_RETRIEVAL_SCORE) - 0.55) < 1e-12)
check("Vector/BM25/Final Top-K remain 10/10/3", (settings.VECTOR_TOP_K, settings.BM25_TOP_K, settings.FINAL_TOP_K) == (10, 10, 3))
check("Chunk size/overlap remain 900/150", (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP) == (900, 150))
check("MultiQuery remains original + two alternatives", settings.MULTI_QUERY_RETRIEVAL_ENABLED is True and settings.MULTI_QUERY_VARIANT_COUNT == 2)
check("RRF K remains 60", settings.MULTI_QUERY_RRF_K == 60)

records = _load_bm25_records()
profiles = extract_rule_profiles(records)
check("Production BM25 corpus remains 951 chunks", len(records) == 951, len(records))
check("Corpus-derived semantic inventory remains 173 profiles", len(profiles) == 173, len(profiles))

answer_source = source("services/answer_service.py")
misra_source = source("services/misra_compliance.py")
semantic_source = source("retrieval/semantic_rule_resolver.py")
app_source = source("app.py")
runner_source = source("scripts/kb_update_runner.py")
smart_source = source("scripts/smart_build.py")
server_source = source("scripts/update_server_kb.py")
bat_source = source("Update_DocuBot_Knowledge_Base.bat")
prewarm_source = source("runtime/prewarm.py")

# Exact v6.5.2 field failures.
semantic_item = {
    "_misra_semantic_resolver": True,
    "metadata": {"section_type": "rule", "rule_id": "16.4"},
    "text": "Rule 16.4\nEvery switch statement shall have a default label",
}
scope_guard_value = MisraComplianceMode.deterministic_scope_guard_answer(
    "A switch handles every known enum value but has no default label. Is that acceptable under MISRA C?",
    [semantic_item],
)
check("Semantic scope guard preserves string return contract", isinstance(scope_guard_value, str), type(scope_guard_value))
check("Semantic scope guard no longer emits tuple sentinel", scope_guard_value == "", repr(scope_guard_value))
check("Removed tuple sentinel source", "return \"\", {\"reason\": \"corpus-semantic Rule resolution" not in misra_source)

records_for_yes_no = MisraComplianceMode.load_authoritative_bm25_records()
rule_16_4 = MisraComplianceMode.authoritative_semantic_reference_evidence(
    records_for_yes_no, kind="rule", identifier="16.4", diagnostics={}
)
tagalog_switch = "Kung kumpleto naman ang listed cases ng switch pero walang default, compliant ba iyon sa MISRA?"
english_switch = "A switch handles every currently known enum value but has no default label. Is that acceptable under MISRA C?"
check("Semantic required-presence compliance finalizes No without tuple", MisraComplianceMode.deterministic_yes_no_answer(tagalog_switch, rule_16_4).startswith("No. Rule 16.4"), MisraComplianceMode.deterministic_yes_no_answer(tagalog_switch, rule_16_4))
check("Semantic required-presence permission finalizes No without LLM", MisraComplianceMode.deterministic_yes_no_answer(english_switch, rule_16_4).startswith("No. Rule 16.4"), MisraComplianceMode.deterministic_yes_no_answer(english_switch, rule_16_4))

state = {
    "context": "accepted prior context",
    "misra": True,
    "anchor_question": "previous unrelated MISRA question",
    "references": ["Rule 17.7"],
}
self_contained = "A string-handling library call may write beyond the destination object's bounds. Which MISRA requirement addresses that risk?"
deictic = "Which MISRA rule applies to that?"
explain = "Can you explain that?"
check("Self-contained current-turn antecedent does not reuse stale MISRA state", AnswerService._is_grounded_followup_candidate(self_contained, state) is False)
check("True deictic Rule follow-up still reuses grounded state", AnswerService._is_grounded_followup_candidate(deictic, state) is True)
check("True explanation follow-up still reuses grounded state", AnswerService._is_grounded_followup_candidate(explain, state) is True)
check("Follow-up fix is identifier-free", "Rule 21.17" not in answer_source and "Rule 17.7" not in answer_source)

# Synthetic Rule-level resolver tests. No production Rule identifiers or benchmark phrases.
class FakeEmbed:
    def get_query_embedding_batch(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
    def rerank(self, query, candidates):
        out = []
        for item in candidates:
            clone = dict(item)
            ident = str((clone.get("profile") or {}).get("identifier", ""))
            clone["rerank_score"] = float(self.scores.get(ident, 0.0))
            out.append(clone)
        return sorted(out, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

fixture_profiles = [
    {"kind":"rule","identifier":str(i),"display_name":f"Rule {i}","requirement_text":f"Requirement {i}","profile_text":f"Profile {i}"}
    for i in range(1,5)
]

def unit_vector(sim):
    return [float(sim), float(max(0.0, 1.0 - sim * sim) ** 0.5), 0.0]

vectors = np.asarray([unit_vector(0.82), unit_vector(0.70), unit_vector(0.50), unit_vector(0.30)], dtype=np.float32)
resolver = SemanticRuleResolver(embedding_model=FakeEmbed(), reranker=FakeReranker({"1":0.05,"2":0.03,"3":0.01,"4":0.005}))
resolver._metadata = {"profiles": fixture_profiles}
resolver._vectors = vectors
resolver._ensure_loaded = lambda: None
single = resolver.resolve("generic natural standards question", semantic_top_k=4, rerank_top_k=4)
check("Strong original-query semantic+BGE agreement can early-accept", single.get("accepted") is True and single.get("acceptance_basis") == "single_query_semantic_agreement", single)
check("Single-query early accept keeps authoritative selection at semantic rank 1", single.get("reference") == "Rule 1", single)

narrow_vectors = np.asarray([unit_vector(0.70), unit_vector(0.66), unit_vector(0.50), unit_vector(0.30)], dtype=np.float32)
resolver2 = SemanticRuleResolver(embedding_model=FakeEmbed(), reranker=FakeReranker({"1":0.05,"2":0.03,"3":0.01,"4":0.005}))
resolver2._metadata = {"profiles": fixture_profiles}
resolver2._vectors = narrow_vectors
resolver2._ensure_loaded = lambda: None
narrow = resolver2.resolve("generic natural standards question", semantic_top_k=4, rerank_top_k=4)
check("Small original-query semantic margin remains ambiguous", narrow.get("accepted") is False, narrow)

resolver3 = SemanticRuleResolver(embedding_model=FakeEmbed(), reranker=FakeReranker({"1":0.02,"2":0.01,"3":0.005,"4":0.001}))
resolver3._metadata = {"profiles": fixture_profiles}
resolver3._vectors = vectors
resolver3._ensure_loaded = lambda: None
wide = resolver3.resolve("generic natural standards question", query_variants=["variant one", "variant two"], semantic_top_k=4, rerank_top_k=4)
check("MultiQuery semantic-consensus fallback remains available", wide.get("accepted") is True and wide.get("acceptance_basis") in {"semantic_consensus", "strict_bge"}, wide)
check("Global chunk threshold not lowered by Rule-level early accept", "MIN_RETRIEVAL_SCORE =" not in semantic_source)

# No question-specific benchmark mapping in resolver.
for forbidden in ["write_status", "update_state", "localtime", "strerror", "switch handles every", "Rule 17.7", "Rule 21.17", "Rule 16.4"]:
    check(f"No benchmark mapping in semantic resolver: {forbidden}", forbidden.casefold() not in semantic_source.casefold())

# Prewarm improvement remains quality-neutral.
check("Prewarm initializes Qwen3 query-embedding path", "get_query_embedding" in prewarm_source and "MISRA requirement lookup readiness" in prewarm_source)
check("Prewarm only loads semantic metadata and does not run MultiQuery", "SemanticRuleResolver" in prewarm_source and "generate_multi_query" not in prewarm_source)

# Update Knowledge Base button portability and safety.
check("Web button uses isolated KB update worker", "launch_kb_update_subprocess" in app_source and "smart_build()" not in app_source)
check("Web status is rechecked instead of one-time kb_checked latch", 'kb_checked' not in app_source)
check("LAN read-only update guard remains", "LAN_SERVER_MODE" in app_source and "ALLOW_WEB_KB_UPDATE" in app_source)
check("Worker launches with exact serving Python", "sys.executable" in runner_source and '"-m",\n        "scripts.kb_update_runner"' in runner_source)
check("Worker pins project working directory", 'cwd=str(ROOT)' in runner_source)
check("Cross-process KB update lock exists", "kb_update.lock" in runner_source and "O_EXCL" in runner_source)
check("KB update worker verifies semantic Rule index", "build_semantic_rule_index" in runner_source)
check("KB update worker runs KB Health", "run_health_check" in runner_source)
check("Web/BAT share the same worker engine", "run_kb_update_worker" in server_source and "launch_kb_update_subprocess" in app_source)
check("Smart-build preflight probes writable directories", "_writable_directory_probe" in smart_source)
check("Smart-build preflight checks transactional free space", "required_free_bytes" in smart_source and "shutil.disk_usage" in smart_source)
check("Smart-build preflight detects local Qdrant lock/access failures early", "_qdrant_storage_preflight" in smart_source and "Close other DocuBot/Python processes" in smart_source)
check("BAT supports venv, .venv and env", all(token in bat_source for token in ['venv\\Scripts\\python.exe','.venv\\Scripts\\python.exe','env\\Scripts\\python.exe']))

# Probe only filesystem safety; this never rebuilds Qdrant/BM25 or contacts Ollama.
try:
    fs_issues, fs_notes, fs_diag = smart_build_module._filesystem_preflight([])
    check("Filesystem portability probe executes without mutation failure", isinstance(fs_diag, dict) and "write_probes" in fs_diag, {"issues":fs_issues,"notes":fs_notes})
except Exception as exc:
    check("Filesystem portability probe executes without mutation failure", False, exc)

# Diagnostic separation requested for retrieval vs LLM generation.
retrieval_diag = source("scripts/test_retrieval_pipeline.py")
llm_diag = source("scripts/test_llm_from_verified_context.py")
check("Retrieval-only diagnostic explicitly skips answer generation", '"llm_answer_generation_called": False' in retrieval_diag)
check("Retrieval-only diagnostic records final chunks/references", '"final_chunks"' in retrieval_diag and '"final_references"' in retrieval_diag)
check("LLM diagnostic blocks when retrieval gate failed", "retrieval gate failed" in llm_diag)
check("LLM diagnostic consumes verified_context", "verified_context" in llm_diag)
portability_diag = source("scripts/test_kb_update_portability.py")
try:
    portability_tree = ast.parse(portability_diag)
    direct_smart_build_calls = [
        node for node in ast.walk(portability_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "smart_build"
    ]
    check("KB button portability dry run never calls smart_build", not direct_smart_build_calls, len(direct_smart_build_calls))
except Exception as exc:
    check("KB button portability dry run never calls smart_build", False, exc)
check("KB button portability dry run checks same child Python dependencies", "qdrant_client,streamlit,rank_bm25" in portability_diag)
check("KB button portability BAT exists", (ROOT / "Test_DocuBot_KB_Update_Portability.bat").is_file())

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.3.json"
check("v6.5.3 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check("v6.5.3 manifest parses", True)
        check("Manifest says no patch-time KB rebuild", manifest.get("kb_rebuild_performed_by_patch") is False)
        check("Manifest preserves chunking", manifest.get("chunk_size") == 900 and manifest.get("chunk_overlap") == 150)
        check("Manifest preserves threshold", float(manifest.get("minimum_retrieval_score", -1)) == 0.55)
        check("Manifest records KB button isolation", manifest.get("update_knowledge_base_button_isolated_worker") is True)
        check("Manifest records answer tuple fix", manifest.get("yes_no_tuple_contract_fixed") is True)
    except Exception as exc:
        check("v6.5.3 manifest parses", False, exc)

failed = [x for x in checks if not x["passed"]]
result = {
    "version": "v6.5.3",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(checks),
    "profile_count": len(profiles),
    "checks": checks,
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("version","validation","failed_checks","check_count","profile_count")}, indent=2))
if failed:
    for item in failed:
        print("[FAIL]", item["name"], "::", item.get("detail", ""))
    raise SystemExit(1)
print(f"[PASS] v6.5.3 validator: {len(checks)} checks, 0 failures.")
raise SystemExit(0)
