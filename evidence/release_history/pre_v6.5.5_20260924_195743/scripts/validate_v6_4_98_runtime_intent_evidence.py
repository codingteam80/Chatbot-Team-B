from __future__ import annotations

import importlib
import importlib.util
import json
import py_compile
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "runtime_intent_evidence"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "v6.4.98_runtime_intent_evidence_validation_latest.json"
checks: list[dict] = []


def check(name, passed, detail=""):
    item = {"name": name, "passed": bool(passed), "detail": str(detail or "")}
    checks.append(item)
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" :: {detail}" if detail else ""))


def text(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


for rel in [
    "chat/query_enricher.py",
    "services/misra_compliance.py",
    "services/answer_service.py",
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
enricher_src = text("chat/query_enricher.py")

# Architecture locks: this patch must not solve intent/evidence bugs by changing
# the certified stack, threshold, Top-K, chunking, or MultiQuery count/fusion.
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
check("Structured MultiQuery from v6.4.97 remains present", "def multi_query_json_schema" in mq_src and "parse_multi_query_variants" in mq_src)
check("Original query still retained before variants", "searches = [original]" in retriever_src)
check("RRF fusion still present", "_multi_query_rrf_fuse" in retriever_src)

try:
    # Load the lightweight modules directly so importing chat/__init__.py does
    # not require Streamlit just to validate deterministic intent rules.
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("v6498_query_enricher", ROOT / "chat" / "query_enricher.py")
    qe_mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(qe_mod)
    QueryEnricher = qe_mod.QueryEnricher

    spec = importlib.util.spec_from_file_location("v6498_misra", ROOT / "services" / "misra_compliance.py")
    mc_mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mc_mod)
    MisraComplianceMode = mc_mod.MisraComplianceMode

    enricher = QueryEnricher()
    enriched = enricher.enrich(
        "Are unused function parameters allowed?",
        intent_question="Are unused function parameters allowed?",
    )
    noisy_terms = ["permitted roles", "allowed roles", "access requirements", "authorization"]
    check(
        "Technical allowed/permitted question avoids authorization expansion",
        not any(term in enriched.casefold() for term in noisy_terms),
        enriched,
    )

    check(
        "Switch omission is classified as permission Yes/No",
        MisraComplianceMode.yes_no_intent("Can a switch statement omit the default label?") == "permission",
    )

    expected_cues = {
        "Hindi ginagamit na parameter": "there should be no unused parameters in functions",
        "Bakit bawal side effect sa kanan ng &&?": "the right hand operand of a logical && or || operator shall not contain persistent side effects",
        "What restrictions apply to goto labels?": "any label referenced by a goto statement shall be declared in the same block, or in any block enclosing the goto statement",
        "What MISRA rules are relevant to string handling and pointer bounds?": "use of the string handling functions from string h shall not result in accesses beyond the bounds of the objects referenced by their pointer parameters",
        "Can a switch statement omit the default label?": "every switch statement shall have a default label",
    }
    for question, wanted in expected_cues.items():
        cues = MisraComplianceMode.semantic_cues(question)
        check(f"Semantic cue: {question}", wanted in cues, cues)

    goto_cues = MisraComplianceMode.semantic_cues("What restrictions apply to goto labels?")
    check("Specific goto-label scope does not broaden to generic goto ban", goto_cues == [expected_cues["What restrictions apply to goto labels?"]], goto_cues)

    check(
        "Why/Bakit side-effect question routes to rationale",
        MisraComplianceMode.answer_focus("Bakit bawal side effect sa kanan ng &&?").startswith("MISRA RATIONALE:"),
    )
    check(
        "Tagalog unused-parameter phrase routes to requirement lookup",
        MisraComplianceMode.answer_focus("Hindi ginagamit na parameter").startswith("MISRA REQUIREMENT LOOKUP:"),
    )

    rescue_expectations = {
        "Hindi ginagamit na parameter": ["Rule 2.7"],
        "Bakit bawal side effect sa kanan ng &&?": ["Rule 13.5"],
        "What restrictions apply to goto labels?": ["Rule 15.3"],
        "Can a switch statement omit the default label?": ["Rule 16.4"],
        "Are the features of <stdarg.h> allowed?": ["Rule 17.1"],
        "What MISRA rules are relevant to string handling and pointer bounds?": ["Rule 21.17"],
        "Pointer galing Standard Library": ["Rule 21.19", "Rule 21.20"],
    }
    rescue_results = {}
    for question, expected_refs in rescue_expectations.items():
        results, diagnostics = MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(question, top_k=6)
        refs = [str((item.get("metadata", {}) or {}).get("section_title", "")).strip() for item in results]
        rescue_results[question] = results
        check(f"Authoritative Rule rescue: {question}", refs == expected_refs, refs)

    yes_unused = MisraComplianceMode.deterministic_yes_no_answer(
        "Are unused function parameters allowed?",
        rescue_results.get("Hindi ginagamit na parameter", []),
    )
    check("Unused parameter Yes/No finalizes Rule 2.7", yes_unused.startswith("No. Rule 2.7"), yes_unused)

    yes_switch = MisraComplianceMode.deterministic_yes_no_answer(
        "Can a switch statement omit the default label?",
        rescue_results.get("Can a switch statement omit the default label?", []),
    )
    check("Switch default Yes/No finalizes Rule 16.4", yes_switch.startswith("No. Rule 16.4"), yes_switch)

    rationale = MisraComplianceMode.deterministic_followup_detail(
        "Bakit bawal side effect sa kanan ng &&?",
        rescue_results.get("Bakit bawal side effect sa kanan ng &&?", []),
    )
    check("Rule 13.5 rationale is source-backed", "Rule 13.5" in rationale and "right-hand operand" in rationale, rationale[:280])

    pointer_answer = MisraComplianceMode.deterministic_requirement_lookup(
        "Pointer galing Standard Library",
        rescue_results.get("Pointer galing Standard Library", []),
    )
    check("Standard Library pointer lookup retains 21.19 and 21.20", "Rule 21.19" in pointer_answer and "Rule 21.20" in pointer_answer, pointer_answer[:320])

    string_answer = MisraComplianceMode.deterministic_requirement_lookup(
        "What MISRA rules are relevant to string handling and pointer bounds?",
        rescue_results.get("What MISRA rules are relevant to string handling and pointer bounds?", []),
    )
    check("String/pointer/bounds lookup narrows to Rule 21.17", "Rule 21.17" in string_answer and "Rule 21.19" not in string_answer, string_answer)

except Exception as error:
    check("Intent + authoritative Rule rescue unit checks", False, repr(error))

# Source integration checks for the service layer. These are intentionally
# static so validation does not initialize Ollama, Qdrant, or the reranker.
check("Answer service canonicalizes MISRA rerank intent", "MisraComplianceMode.build_rerank_query" in answer_src and "MISRA RERANK INTENT" in answer_src)
check("Answer service has strict anaphora-only standalone MISRA guard", "explicit_anaphor" in answer_src and "MisraComplianceMode.semantic_cues(question)" in answer_src and "stale Rule/topic reuse" in answer_src)
check("Answer service finalizes deterministic requirement lookup", "precomputed_misra_requirement_lookup_answer" in answer_src and "deterministic_requirement_lookup" in answer_src)
check("MISRA answer focus supports source rationale", "MISRA RATIONALE:" in misra_src)
check("MISRA answer focus supports requirement lookup", "MISRA REQUIREMENT LOOKUP:" in misra_src)
check("Technical permission classifier is present", "def _is_technical_permission_question" in enricher_src)
check("Near-threshold rescued evidence is pinned through diversity", 'item.get("_structured_near_threshold_rescue")' in retriever_src and "pinned_evidence" in retriever_src)
check("Exact concept intersection precedes broad pointer family", "exact concept intersection" in retriever_src and "if broad_pointer_query and not chosen_ids" in retriever_src)

# Exercise the two retriever hardenings without loading any model. The sandbox
# used for build validation may not have the UI/BM25 dependencies, so tiny
# import-only stubs are used only when those packages are absent.
try:
    if importlib.util.find_spec("streamlit") is None:
        st = types.ModuleType("streamlit")
        def cache_resource(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]
            return lambda fn: fn
        st.cache_resource = cache_resource
        st.session_state = {}
        sys.modules["streamlit"] = st
    if importlib.util.find_spec("rank_bm25") is None:
        rb = types.ModuleType("rank_bm25")
        rb.BM25Okapi = type("BM25Okapi", (), {})
        sys.modules["rank_bm25"] = rb

    retriever_mod = importlib.import_module("retrieval.retriever")
    CompanyRetriever = retriever_mod.CompanyRetriever
    probe = CompanyRetriever.__new__(CompanyRetriever)

    sample = [
        {"text": "first", "metadata": {"file_name": "MISRA.pdf", "chunk_id": 1}, "score": 0.90},
        {"text": "second", "metadata": {"file_name": "MISRA.pdf", "chunk_id": 2}, "score": 0.80},
        {"text": "rescued", "metadata": {"file_name": "MISRA.pdf", "chunk_id": 3}, "score": 0.54, "_structured_near_threshold_rescue": True},
    ]
    diversified = probe._apply_diversity_filter(sample, max_chunks_per_file=2)
    check("Pinned rescue survives same-file diversity cap", [item["text"] for item in diversified] == ["first", "second", "rescued"], [item["text"] for item in diversified])

    records = MisraComplianceMode.load_authoritative_bm25_records()
    probe.bm25 = type("Corpus", (), {"records": records})()
    exact = probe._retrieve_structured_rule_catalog(
        "What MISRA rules are relevant to string handling and pointer bounds?",
        intent_query="What MISRA rules are relevant to string handling and pointer bounds?",
    )
    exact_ids = [str((item.get("metadata", {}) or {}).get("rule_id", "")) for item in exact]
    check("Catalog concept intersection selects Rule 21.17 only", exact_ids == ["21.17"], exact_ids)

    broad = probe._retrieve_structured_rule_catalog(
        "Which MISRA rules apply to pointers?",
        intent_query="Which MISRA rules apply to pointers?",
    )
    broad_ids = [str((item.get("metadata", {}) or {}).get("rule_id", "")) for item in broad]
    check("Broad pointer inventory behavior is preserved", len(broad_ids) > 3 and "21.17" in broad_ids, broad_ids[:8])
except Exception as error:
    check("Retriever pinning + concept-intersection unit checks", False, repr(error))

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.98.json"
check("v6.4.98 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = manifest.get("architecture_lock", {})
        check("Manifest base is v6.4.97", manifest.get("base_runtime") == "verified v6.4.97 Structured MultiQuery Output Hardening")
        check("Manifest preserves threshold 0.55", lock.get("minimum_retrieval_score") == 0.55)
        check("Manifest preserves Top-K 10/10/3", (lock.get("vector_top_k"), lock.get("bm25_top_k"), lock.get("final_top_k")) == (10, 10, 3))
        check("Manifest preserves original + two MultiQuery alternatives", manifest.get("multi_query", {}).get("alternative_count") == 2)
        check("Manifest says no KB rebuild", manifest.get("kb_rebuild") is False)
    except Exception as error:
        check("v6.4.98 manifest parses", False, repr(error))

failed = [item for item in checks if not item["passed"]]
summary = {
    "version": "v6.4.98",
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
