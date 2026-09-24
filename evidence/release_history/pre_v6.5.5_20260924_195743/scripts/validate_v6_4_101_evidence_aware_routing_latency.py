from __future__ import annotations

import importlib.util
import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "evidence_aware_routing_latency"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "v6.4.101_evidence_aware_routing_latency_validation_latest.json"
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
mq_src = text("retrieval/multi_query.py")
retriever_src = text("retrieval/retriever.py")
answer_src = text("services/answer_service.py")
misra_src = text("services/misra_compliance.py")

# Architecture locks: v6.4.101 fixes routing/selection, not thresholds or models.
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
check("RRF K remains 60", '"DOCUBOT_MULTI_QUERY_RRF_K", "60"' in settings_src)
check("Structured MultiQuery remains present", "def multi_query_json_schema" in mq_src and "parse_multi_query_variants" in mq_src)
check("Original query retained before variants", "searches = [original]" in retriever_src)
check("RRF fusion remains present", "_multi_query_rrf_fuse" in retriever_src)

# v6.4.101 implementation shape.
check(
    "Source-backed catalog/reference lookup can enter MISRA evidence path",
    "if not ((explicit_misra or misra_context) and natural_requirement):" in misra_src,
)
check(
    "Unused-parameter relation includes read/Tagalog gumagamit variants",
    '"read", "reads", "reading"' in misra_src and '"gumamit", "gumagamit"' in misra_src,
)
check(
    "Goto target/destination scope concept is explicit",
    're.search(r"\\b(?:target|destination)\\b", clean)' in misra_src
    and 'child|nested|inner|outer' in misra_src,
)
check(
    "Specific goto-scope cue suppresses broad Rule 15.1 cue",
    "or re.search(r\"\\b(?:target|destination)\\b\", clean)" in misra_src,
)
check(
    "Evidence-aware authoritative promotion is integrated",
    "MISRA EVIDENCE-AWARE PROMOTION" in answer_src
    and "rule_body_cue_rescue_from_authoritative_corpus" in answer_src,
)
check(
    "Duplicate MISRA multilingual retry guard is integrated",
    "duplicate_misra_retry" in answer_src
    and "SKIPPED DUPLICATE MISRA QUERY" in answer_src,
)
check(
    "Rule 15.3 directional permission guard is integrated",
    "directional scope permission" in misra_src
    and 'has_rule_15_3' in misra_src,
)

try:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "v64101_misra", ROOT / "services" / "misra_compliance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    M = mod.MisraComplianceMode

    rule27 = "there should be no unused parameters in functions"
    rule153 = "any label referenced by a goto statement shall be declared in the same block, or in any block enclosing the goto statement"

    # Exact six unseen runtime questions that exposed the v6.4.100 regression.
    cases = [
        (
            "In MISRA C, is a callback parameter that is declared but never read considered acceptable?",
            rule27, "Rule 2.7", "No. Rule 2.7",
        ),
        (
            "Kung may input parameter ang function pero wala ni isang statement na gumagamit dito, anong MISRA requirement ang applicable?",
            rule27, "Rule 2.7", "Rule 2.7",
        ),
        (
            "Which MISRA rule covers a function argument whose corresponding parameter is intentionally ignored?",
            rule27, "Rule 2.7", "Rule 2.7",
        ),
        (
            "Can a goto target be placed in a child block under MISRA C?",
            rule153, "Rule 15.3", "No. Rule 15.3",
        ),
        (
            "Sa MISRA, valid ba kung ang label ng goto ay nasa nested block sa loob ng pinanggalingang block?",
            rule153, "Rule 15.3", "No. Rule 15.3",
        ),
        (
            "What are the block-scope constraints for labels used as goto destinations in MISRA C?",
            rule153, "Rule 15.3", "Rule 15.3",
        ),
    ]

    for q, expected_cue, expected_ref, answer_prefix in cases:
        cues = M.semantic_cues(q)
        check(f"Unseen concept cue: {q}", cues == [expected_cue], cues)
        check(f"Unseen query enters deterministic MISRA path: {q}", M.is_request(q, current_topic=""), M.yes_no_intent(q))
        search = M.build_search_query(q, q)
        check(f"Search pins authoritative wording: {q}", expected_cue in search, search[:360])
        results, diagnostics = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        check(f"Authoritative rescue is narrow: {q}", refs == [expected_ref], refs)
        if M.yes_no_intent(q):
            answer = M.deterministic_yes_no_answer(q, results)
        else:
            answer = M.deterministic_requirement_lookup(q, results)
        normalized_answer = answer.replace("**", "")
        check(f"Deterministic answer quality: {q}", normalized_answer.startswith(answer_prefix), answer)

    # Additional unseen variations to ensure the fix is relational, not copied sentences.
    extra_unused = [
        "Under MISRA C, what applies when a handler receives a parameter but never reads it?",
        "Anong MISRA rule kung may argument ang callback pero wala talagang gumagamit sa parameter?",
        "Which MISRA guideline applies to a deliberately neglected callback parameter?",
    ]
    for q in extra_unused:
        cues = M.semantic_cues(q)
        results, _ = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        check(f"Extra unused paraphrase -> Rule 2.7: {q}", rule27 in cues and refs == ["Rule 2.7"], {"cues": cues, "refs": refs})

    extra_goto = [
        "Under MISRA C, may the goto destination live in an inner block?",
        "Can the goto target label be inside a nested child block?",
        "Which MISRA requirement controls the scope of a goto destination label?",
    ]
    for q in extra_goto:
        cues = M.semantic_cues(q)
        results, _ = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        check(f"Extra goto paraphrase -> Rule 15.3: {q}", cues == [rule153] and refs == ["Rule 15.3"], {"cues": cues, "refs": refs})

    # Precision controls: broadened relation vocabulary must not hijack unrelated concepts.
    negatives = [
        "What does MISRA say about reading a function parameter before modifying it?",
        "Can an unused return value be ignored in a function that also has parameters?",
        "Which MISRA rule covers an unused macro parameter?",
        "Can a goto statement be used at all under MISRA C?",
    ]
    for q in negatives:
        cues = M.semantic_cues(q)
        if "goto statement be used at all" in q:
            passed = rule153 not in cues
        else:
            passed = rule27 not in cues
        check(f"Precision control: {q}", passed, cues)

except Exception as error:
    check("v6.4.101 semantic/routing unit checks", False, repr(error))

# Preservation checks from prior releases.
check("v6.4.100 concept matcher preserved", "def _unused_function_parameter_concept" in misra_src)
check("v6.4.99 post-normalization rescue preserved", "MisraComplianceMode.post_normalization_rescue_query" in answer_src and "MISRA POST-NORMALIZATION RESCUE" in answer_src)
check("v6.4.98 rerank canonicalization preserved", "MisraComplianceMode.build_rerank_query" in answer_src and "MISRA RERANK INTENT" in answer_src)
check("v6.4.98 strict follow-up guard preserved", "explicit_anaphor" in answer_src and "stale Rule/topic reuse" in answer_src)
check("v6.4.98 deterministic requirement lookup preserved", "deterministic_requirement_lookup" in answer_src)
check("v6.4.98 evidence pinning preserved", 'item.get("_structured_near_threshold_rescue")' in retriever_src and "pinned_evidence" in retriever_src)
check("v6.4.98 exact concept intersection preserved", "exact concept intersection" in retriever_src and "if broad_pointer_query and not chosen_ids" in retriever_src)

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.101.json"
check("v6.4.101 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = manifest.get("architecture_lock", {})
        check("Manifest base is v6.4.100", manifest.get("base_runtime") == "verified v6.4.100 Concept-Level Semantic Matcher Hardening")
        check("Manifest preserves threshold 0.55", lock.get("minimum_retrieval_score") == 0.55)
        check("Manifest preserves Top-K 10/10/3", (lock.get("vector_top_k"), lock.get("bm25_top_k"), lock.get("final_top_k")) == (10, 10, 3))
        check("Manifest preserves original + two MultiQuery alternatives", manifest.get("multi_query", {}).get("alternative_count") == 2)
        check("Manifest says MultiQuery algorithm unchanged", manifest.get("multi_query_algorithm_changed") is False)
        check("Manifest says no KB rebuild", manifest.get("kb_rebuild") is False)
        check("Manifest records evidence-aware promotion", manifest.get("hardening", {}).get("evidence_aware_authoritative_promotion") is True)
        check("Manifest records duplicate retry guard", manifest.get("hardening", {}).get("duplicate_multilingual_multiquery_guard") is True)
    except Exception as error:
        check("v6.4.101 manifest parses", False, repr(error))

failed = [item for item in checks if not item["passed"]]
summary = {
    "version": "v6.4.101",
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
