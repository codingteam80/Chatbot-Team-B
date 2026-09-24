from __future__ import annotations

import ast
import copy
import importlib.util
import json
import py_compile
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "deterministic_evidence_finalization_language_guard"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "v6.4.102_deterministic_evidence_finalization_language_guard_validation_latest.json"
checks: list[dict] = []


def check(name, passed, detail=""):
    item = {"name": name, "passed": bool(passed), "detail": str(detail or "")}
    checks.append(item)
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" :: {detail}" if detail else ""))


def text(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


for rel in [
    "services/misra_compliance.py",
    "services/answer_service.py",
    "chat/query_enricher.py",
    "retrieval/retriever.py",
    "retrieval/multi_query.py",
    "config/settings.py",
    "config/prompts.py",
    "llm/ollama_client.py",
]:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as error:
        check(f"Python syntax: {rel}", False, repr(error))

settings_src = text("config/settings.py")
retriever_src = text("retrieval/retriever.py")
mq_src = text("retrieval/multi_query.py")
misra_src = text("services/misra_compliance.py")
answer_src = text("services/answer_service.py")

# Architecture locks.
check("Generation model remains qwen2.5:7b", '"qwen2.5:7b"' in settings_src)
check("Embedding model remains qwen3-embedding:8b", '"qwen3-embedding:8b"' in settings_src)
check("Reranker remains BAAI/bge-reranker-v2-m3", '"BAAI/bge-reranker-v2-m3"' in settings_src)
check("Vector Top-K remains 10", bool(re.search(r"^VECTOR_TOP_K\s*=\s*10\s*$", settings_src, re.M)))
check("BM25 Top-K remains 10", bool(re.search(r"^BM25_TOP_K\s*=\s*10\s*$", settings_src, re.M)))
check("Final Top-K remains 3", bool(re.search(r"^FINAL_TOP_K\s*=\s*3\s*$", settings_src, re.M)))
check("Minimum retrieval score remains 0.55", bool(re.search(r"^MIN_RETRIEVAL_SCORE\s*=\s*0\.55\s*$", settings_src, re.M)))
check("Chunk size remains 900", bool(re.search(r"^CHUNK_SIZE\s*=\s*900\s*$", settings_src, re.M)))
check("Chunk overlap remains 150", bool(re.search(r"^CHUNK_OVERLAP\s*=\s*150\s*$", settings_src, re.M)))
check("MultiQuery remains enabled", 'MULTI_QUERY_RETRIEVAL_ENABLED = _env_bool("DOCUBOT_MULTI_QUERY", True)' in settings_src)
check("MultiQuery remains two alternatives", '"DOCUBOT_MULTI_QUERY_VARIANTS", "2"' in settings_src)
check("Structured MultiQuery remains present", "def multi_query_json_schema" in mq_src and "parse_multi_query_variants" in mq_src)
check("Original query retained before variants", "searches = [original]" in retriever_src)
check("RRF fusion remains present", "_multi_query_rrf_fuse" in retriever_src)

# Implementation shape.
check("Unused parameter use-family includes access semantics", '"access", "accessed", "accesses", "accessing"' in misra_src)
check("English modal May language disambiguation exists", 'first_word == "may"' in answer_src and 'r"^may\\s+(?:a|an|the|this|that|these|those)\\b"' in answer_src)
check("Tagalog May remains multilingual", '"paano", "alin", "ilang", "gaano", "may"' in answer_src)
check("Post-retrieval MISRA authoritative finalizer exists", "def _post_retrieval_misra_authoritative_promotion" in answer_src)
check("Post-retrieval finalizer requires reference overlap", "accepted_refs.intersection(rescued_refs)" in answer_src)
check("Post-retrieval finalizer runs before generation", "MISRA POST-RETRIEVAL AUTHORITATIVE FINALIZATION" in answer_src)
check("v6.4.101 duplicate retry guard preserved", "SKIPPED DUPLICATE MISRA QUERY" in answer_src)
check("v6.4.101 evidence-aware empty-context promotion preserved", "MISRA EVIDENCE-AWARE PROMOTION" in answer_src)

try:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("v64102_misra", ROOT / "services" / "misra_compliance.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    M = mod.MisraComplianceMode

    rule27 = "there should be no unused parameters in functions"
    rule153 = "any label referenced by a goto statement shall be declared in the same block, or in any block enclosing the goto statement"

    runtime_cases = [
        ("Does MISRA allow a callback to receive a parameter that is never accessed inside the callback?", rule27, "Rule 2.7", "No. Rule 2.7"),
        ("Ano ang applicable MISRA rule kung may parameter ang function pero walang code na gumagamit ng value nito?", rule27, "Rule 2.7", "Rule 2.7"),
        ("A function accepts an argument, but the matching parameter is deliberately left untouched and unused. What MISRA guideline applies?", rule27, "Rule 2.7", "Rule 2.7"),
        ("May a goto jump to a label declared inside a more deeply nested block?", rule153, "Rule 15.3", "No. Rule 15.3"),
        ("Kung ang destination label ng goto ay nasa inner scope kaysa sa goto statement, compliant ba ito sa MISRA?", rule153, "Rule 15.3", "No. Rule 15.3"),
        ("Which MISRA rule defines whether a goto label must be in the same or an enclosing scope?", rule153, "Rule 15.3", "Rule 15.3"),
    ]

    for q, expected_cue, expected_ref, expected_prefix in runtime_cases:
        cues = M.semantic_cues(q)
        check(f"v6.4.101 retest concept converges: {q}", cues == [expected_cue], cues)
        check(f"v6.4.101 retest enters deterministic MISRA route: {q}", M.is_request(q, current_topic=""), M.yes_no_intent(q))
        rescued, _ = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in rescued]
        check(f"v6.4.101 retest authoritative evidence is narrow: {q}", refs == [expected_ref], refs)
        answer = M.deterministic_yes_no_answer(q, rescued) if M.yes_no_intent(q) else M.deterministic_requirement_lookup(q, rescued)
        check(f"v6.4.101 retest deterministic answer: {q}", answer.replace("**", "").startswith(expected_prefix), answer)

    extra_unused = [
        "Under MISRA, is a parameter acceptable if the function never accesses it?",
        "Which requirement covers a callback parameter that is not consumed anywhere?",
        "Anong rule kung parameter ay hindi nire-reference sa function?",
    ]
    for q in extra_unused:
        check(f"Generalized unused-input relation: {q}", M.semantic_cues(q) == [rule27], M.semantic_cues(q))

    precision_controls = [
        "Can a parameter be accessed after it is modified?",
        "What does MISRA say about a parameter that is never modified?",
        "Can an unused return value be ignored in a function that also has parameters?",
        "Which MISRA rule covers an unused macro parameter?",
    ]
    for q in precision_controls:
        check(f"Unused-input precision guard: {q}", rule27 not in M.semantic_cues(q), M.semantic_cues(q))

    # Execute the pure English-language detector without importing Streamlit.
    tree = ast.parse(answer_src)
    lang_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_is_clearly_english_query")
    lang_ns = {"re": re}
    exec(textwrap.dedent(ast.get_source_segment(answer_src, lang_node)), lang_ns)
    is_english = lang_ns["_is_clearly_english_query"]
    check("English modal May is not translated", is_english(None, "May a goto jump to a nested label?") is True)
    check("Tagalog existential May remains multilingual", is_english(None, "May parameter ang function pero hindi ginagamit?") is False)
    check("Tagalog May issue remains multilingual", is_english(None, "May issue ba sa goto label?") is False)

    # Execute the pure post-retrieval promotion helper.  Simulate a normal
    # accepted Rule 15.3 chunk by removing rescue-only annotations first.
    promote_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_post_retrieval_misra_authoritative_promotion")
    promote_ns = {"re": re, "MisraComplianceMode": M}
    exec(textwrap.dedent(ast.get_source_segment(answer_src, promote_node)), promote_ns)
    promote = promote_ns["_post_retrieval_misra_authoritative_promotion"]

    q_goto = "May a goto jump to a label declared inside a more deeply nested block?"
    goto_rescued, _ = M.rule_body_cue_rescue_from_authoritative_corpus(q_goto, top_k=6)
    generic_accepted = [copy.deepcopy(goto_rescued[0])]
    for item in generic_accepted:
        for key in ["_misra_rule_body_rescue", "_misra_cue", "_misra_cue_coverage", "_misra_assessment_state", "_misra_observation"]:
            item.pop(key, None)
    promoted_context, promoted_results, details = promote(q_goto, generic_accepted)
    check("Accepted Rule 15.3 is promoted before generic generation", bool(promoted_context) and M.available_references(promoted_results) == {"rule:15.3"}, details)

    q_unused = "Does MISRA allow a callback to receive a parameter that is never accessed inside the callback?"
    unused_rescued, _ = M.rule_body_cue_rescue_from_authoritative_corpus(q_unused, top_k=6)
    mismatch_context, mismatch_results, mismatch_details = promote(q_goto, unused_rescued)
    check("Post-retrieval promotion rejects mismatched Rule evidence", not mismatch_context and not mismatch_results, mismatch_details)

except Exception as error:
    check("v6.4.102 behavioral validation", False, repr(error))

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.102.json"
check("v6.4.102 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = manifest.get("architecture_lock", {})
        hardening = manifest.get("hardening", {})
        check("Manifest base is v6.4.101", manifest.get("base_runtime") == "verified v6.4.101 Evidence-Aware Routing + Retry Latency Guard")
        check("Manifest preserves threshold 0.55", lock.get("minimum_retrieval_score") == 0.55)
        check("Manifest preserves Top-K 10/10/3", (lock.get("vector_top_k"), lock.get("bm25_top_k"), lock.get("final_top_k")) == (10, 10, 3))
        check("Manifest preserves MultiQuery two alternatives", manifest.get("multi_query", {}).get("alternative_count") == 2)
        check("Manifest says no KB rebuild", manifest.get("kb_rebuild") is False)
        check("Manifest records language detection guard", hardening.get("english_modal_may_language_guard") is True)
        check("Manifest records post-retrieval deterministic finalization", hardening.get("post_retrieval_authoritative_finalization") is True)
    except Exception as error:
        check("v6.4.102 manifest parses", False, repr(error))

failed = [item for item in checks if not item["passed"]]
summary = {
    "version": "v6.4.102",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "checks": checks,
}
OUTFILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print()
print("Validation result:", summary["validation"])
print("Failed checks:", len(failed))
print("Output:", OUTFILE)
raise SystemExit(0 if not failed else 1)
