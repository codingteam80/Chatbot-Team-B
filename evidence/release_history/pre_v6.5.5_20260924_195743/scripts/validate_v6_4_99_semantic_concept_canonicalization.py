from __future__ import annotations

import importlib.util
import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "semantic_concept_canonicalization"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "v6.4.99_semantic_concept_canonicalization_validation_latest.json"
checks: list[dict] = []


def check(name, passed, detail=""):
    item = {"name": name, "passed": bool(passed), "detail": str(detail or "")}
    checks.append(item)
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" :: {detail}" if detail else ""))


def text(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


# Syntax / import-surface checks.
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

# Architecture locks. v6.4.99 must not solve the issue by relaxing retrieval.
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

try:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "v6499_misra", ROOT / "services" / "misra_compliance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    M = mod.MisraComplianceMode

    rule27 = "there should be no unused parameters in functions"
    rule153 = "any label referenced by a goto statement shall be declared in the same block, or in any block enclosing the goto statement"

    unused_questions = [
        "Is it acceptable in MISRA C for a function parameter to never be used?",
        "Ano ang sinasabi ng MISRA tungkol sa parameter na dineclare pero hindi ginagamit sa function?",
        "May restriction ba sa MISRA kapag may function argument na unused?",
    ]
    goto_questions = [
        "Where is a label targeted by a goto allowed to be declared under MISRA C?",
        "Sa MISRA, saan dapat naka-declare ang label na tina-target ng goto?",
        "May restriction ba sa location ng label na nire-reference ng goto statement?",
    ]

    unused_results = {}
    for q in unused_questions:
        cues = M.semantic_cues(q)
        check(f"Unused concept canonicalizes: {q}", cues == [rule27], cues)
        check(f"Unused concept enters MISRA mode: {q}", M.is_request(q, current_topic="MISRA"), M.yes_no_intent(q))
        search = M.build_search_query(q, q)
        check(f"Unused search includes canonical source cue: {q}", rule27 in search, search[:300])
        results, diagnostics = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        unused_results[q] = results
        check(f"Unused authoritative rescue -> Rule 2.7: {q}", refs == ["Rule 2.7"], refs)

    first_answer = M.deterministic_yes_no_answer(unused_questions[0], unused_results[unused_questions[0]])
    check("Unused English permission finalizes direct No + Rule 2.7", first_answer.startswith("No. Rule 2.7"), first_answer)

    for q in unused_questions[1:]:
        answer = M.deterministic_requirement_lookup(q, unused_results[q])
        check(f"Unused lookup finalizes Rule 2.7: {q}", "Rule 2.7" in answer and "Rule 17.8" not in answer, answer)

    for q in goto_questions:
        cues = M.semantic_cues(q)
        check(f"Goto scope regression cue: {q}", cues == [rule153], cues)
        results, diagnostics = M.rule_body_cue_rescue_from_authoritative_corpus(q, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        check(f"Goto authoritative rescue -> Rule 15.3: {q}", refs == ["Rule 15.3"], refs)

    normalized = "MISRA restriction for unused function argument"
    promoted = M.post_normalization_rescue_query(
        original_question="May restriction ba sa MISRA kapag may function argument na hindi nagagamit?",
        normalized_query=normalized,
        current_topic="MISRA",
    )
    check("Post-normalization MISRA rescue promotes source-backed concept", promoted == normalized, promoted)
    blocked = M.post_normalization_rescue_query(
        original_question="What is the leave policy?",
        normalized_query="unused function parameter",
        current_topic="",
    )
    check("Post-normalization rescue does not hijack non-MISRA queries", blocked == "", blocked)

except Exception as error:
    check("Semantic canonicalization + authoritative rescue unit checks", False, repr(error))

# Integration / preservation checks without initializing heavy runtime components.
check(
    "Answer service integrates post-normalization MISRA promotion",
    "MisraComplianceMode.post_normalization_rescue_query" in answer_src
    and "MISRA POST-NORMALIZATION RESCUE" in answer_src
    and "post_normalized_misra_query" in answer_src,
)
check(
    "Post-normalized query becomes MISRA evidence query before retrieval",
    "post_normalized_misra_query\n            or str(question or \"\").strip()" in answer_src,
)
check("v6.4.98 rerank canonicalization preserved", "MisraComplianceMode.build_rerank_query" in answer_src and "MISRA RERANK INTENT" in answer_src)
check("v6.4.98 strict follow-up guard preserved", "explicit_anaphor" in answer_src and "stale Rule/topic reuse" in answer_src)
check("v6.4.98 deterministic requirement lookup preserved", "deterministic_requirement_lookup" in answer_src)
check("v6.4.98 evidence pinning preserved", 'item.get("_structured_near_threshold_rescue")' in retriever_src and "pinned_evidence" in retriever_src)
check("v6.4.98 exact concept intersection preserved", "exact concept intersection" in retriever_src and "if broad_pointer_query and not chosen_ids" in retriever_src)

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.99.json"
check("v6.4.99 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = manifest.get("architecture_lock", {})
        check("Manifest base is v6.4.98", manifest.get("base_runtime") == "verified v6.4.98 Runtime Intent + Evidence Selection Hardening")
        check("Manifest preserves threshold 0.55", lock.get("minimum_retrieval_score") == 0.55)
        check("Manifest preserves Top-K 10/10/3", (lock.get("vector_top_k"), lock.get("bm25_top_k"), lock.get("final_top_k")) == (10, 10, 3))
        check("Manifest preserves original + two MultiQuery alternatives", manifest.get("multi_query", {}).get("alternative_count") == 2)
        check("Manifest says MultiQuery algorithm unchanged", manifest.get("multi_query_algorithm_changed") is False)
        check("Manifest says no KB rebuild", manifest.get("kb_rebuild") is False)
    except Exception as error:
        check("v6.4.99 manifest parses", False, repr(error))

failed = [item for item in checks if not item["passed"]]
summary = {
    "version": "v6.4.99",
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
