from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Offline validation shims only. Production PCs use the real packages.
if importlib.util.find_spec("streamlit") is None:
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_resource = lambda *a, **k: (lambda fn: fn)
    streamlit.session_state = {}
    sys.modules["streamlit"] = streamlit

if importlib.util.find_spec("rank_bm25") is None:
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = type("BM25Okapi", (), {})
    sys.modules["rank_bm25"] = rank_bm25

import retrieval.retriever as retriever_module
from config.settings import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    MIN_RETRIEVAL_SCORE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL_NAME,
    OLLAMA_MODEL,
    RERANKER_MODEL,
)
from retrieval.retriever import CompanyRetriever
from services.misra_compliance import MisraComplianceMode

checks: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail not in (None, "") else ""))


def fresh_retriever():
    obj = CompanyRetriever.__new__(CompanyRetriever)
    obj._multi_query_search_cache = {}
    obj._multi_query_variant_cache = {}
    return obj


def refs(results):
    return [
        str((item.get("metadata", {}) or {}).get("section_title", "") or "")
        for item in (results or [])
    ]


# ---------------------------------------------------------------------------
# 1) Explicit either/or uncertainty is decomposed generically into exactly two
#    user-derived variants. No MISRA Rule vocabulary is injected here.
# ---------------------------------------------------------------------------
ambiguity_probes = [
    "Under MISRA C, a function call changes persistent state, but I'm unsure whether the concern is evaluation order or using that call inside a logical expression. What should I check?",
    "I'm not sure whether the issue is ownership transfer or object lifetime. What should I review?",
    "We are uncertain if the risk is overflow handling or signedness conversion. What should we inspect?",
]
for idx, question in enumerate(ambiguity_probes, 1):
    variants = CompanyRetriever._explicit_binary_ambiguity_variants(question)
    check(f"explicit ambiguity probe {idx} yields exactly two variants", len(variants) == 2, variants)
    check(f"explicit ambiguity probe {idx} variants are distinct", len({v.casefold() for v in variants}) == 2, variants)

non_ambiguity = [
    "Which MISRA rule covers persistent side effects in && or ||?",
    "Should I use a pointer or an index here?",
    "Explain evaluation order for this expression.",
]
for idx, question in enumerate(non_ambiguity, 1):
    variants = CompanyRetriever._explicit_binary_ambiguity_variants(question)
    check(f"non-explicit ambiguity probe {idx} does not force split", variants == [], variants)

# ---------------------------------------------------------------------------
# 2) Production MultiQuery uses the explicit split before any LLM variant call,
#    while preserving original + exactly two alternatives and cache reuse.
# ---------------------------------------------------------------------------
probe = ambiguity_probes[0]
original_generator = retriever_module.generate_multi_query_variants
retriever_module.generate_multi_query_variants = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM generator should not be called"))
try:
    obj = fresh_retriever()
    started = time.perf_counter()
    searches = obj._generate_multi_query_searches(probe, intent_query=probe, structured_reference=None)
    elapsed = time.perf_counter() - started
finally:
    retriever_module.generate_multi_query_variants = original_generator
check("explicit ambiguity MultiQuery keeps original + exactly two variants", len(searches) == 3 and searches[0] == probe, searches)
check("explicit ambiguity MultiQuery avoids variant-generation LLM", elapsed < 0.25, round(elapsed, 6))
check("explicit ambiguity variants cached by semantic intent", len(obj._multi_query_variant_cache.get(probe.casefold(), ())) == 2, obj._multi_query_variant_cache.get(probe.casefold()))

# ---------------------------------------------------------------------------
# 3) A genuine semantic tie can be finalized as a small source-verified bundle
#    only when every retained candidate is strong in every MultiQuery arm.
# ---------------------------------------------------------------------------
records = MisraComplianceMode.load_authoritative_bm25_records()
observed_tie = {
    "status": "ambiguous",
    "accepted": False,
    "query_count": 3,
    "multi_query_rule_resolution": True,
    "candidates": [
        {"reference": "Rule 13.2", "semantic_rank": 1, "semantic_score": 1.0, "best_similarity": 0.703694, "multi_query_hits": 3, "rerank_score": 0.009562},
        {"reference": "Rule 13.5", "semantic_rank": 2, "semantic_score": 0.994595, "best_similarity": 0.703634, "multi_query_hits": 3, "rerank_score": 0.002031},
        {"reference": "Rule 13.1", "semantic_rank": 3, "semantic_score": 0.973488, "best_similarity": 0.611922, "multi_query_hits": 3, "rerank_score": 0.000126},
    ],
}
bundle = MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, observed_tie)
check("observed semantic tie materializes two authoritative Rules", refs(bundle) == ["Rule 13.2", "Rule 13.5"], refs(bundle))
check("weaker third semantic candidate is excluded", "Rule 13.1" not in refs(bundle), refs(bundle))
check("ambiguity bundle carries source-verification marker", all(item.get("_misra_semantic_ambiguity_bundle") is True for item in bundle), refs(bundle))

generic_tie = {
    "status": "ambiguous",
    "accepted": False,
    "query_count": 3,
    "multi_query_rule_resolution": True,
    "candidates": [
        {"reference": "Rule 8.7", "semantic_rank": 1, "semantic_score": 1.0, "best_similarity": 0.72, "multi_query_hits": 3},
        {"reference": "Rule 10.6", "semantic_rank": 2, "semantic_score": 0.98, "best_similarity": 0.705, "multi_query_hits": 3},
        {"reference": "Rule 14.4", "semantic_rank": 3, "semantic_score": 0.95, "best_similarity": 0.60, "multi_query_hits": 3},
    ],
}
generic_bundle = MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, generic_tie)
check("ambiguity bundle is generic across unrelated source Rules", refs(generic_bundle) == ["Rule 8.7", "Rule 10.6"], refs(generic_bundle))

low_similarity = dict(observed_tie)
low_similarity["candidates"] = [dict(item) for item in observed_tie["candidates"]]
low_similarity["candidates"][1]["best_similarity"] = 0.60
check("low-similarity runner is not promoted as ambiguity bundle", MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, low_similarity) == [])

missing_arm = dict(observed_tie)
missing_arm["candidates"] = [dict(item) for item in observed_tie["candidates"]]
missing_arm["candidates"][1]["multi_query_hits"] = 2
check("candidate missing one MultiQuery arm is not promoted", MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, missing_arm) == [])

single_query = dict(observed_tie)
single_query["query_count"] = 1
check("single-query ambiguity cannot use MultiQuery bundle path", MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, single_query) == [])

already_accepted = dict(observed_tie)
already_accepted["status"] = "accepted"
already_accepted["accepted"] = True
check("accepted semantic winner bypasses ambiguity bundle path", MisraComplianceMode.authoritative_semantic_ambiguity_evidence(records, already_accepted) == [])

# ---------------------------------------------------------------------------
# 4) Multi-rule ambiguity evidence renders as guidance rather than a guessed
#    verdict and every visible identifier comes from the verified source bodies.
# ---------------------------------------------------------------------------
guidance = MisraComplianceMode.deterministic_guidance(probe, bundle)
check("ambiguity bundle renders deterministic multi-rule guidance", "Rule 13.2" in guidance and "Rule 13.5" in guidance, guidance)
check("ambiguity guidance does not guess violation/compliance", "violation exists" in guidance.casefold() and "actual code" in guidance.casefold(), guidance)
check("ambiguity guidance references remain grounded", MisraComplianceMode.references_are_grounded(guidance, bundle), guidance)

# ---------------------------------------------------------------------------
# 5) Integration and anti-hardcoding checks.
# ---------------------------------------------------------------------------
answer_source = (ROOT / "services" / "answer_service.py").read_text(encoding="utf-8")
retriever_source = (ROOT / "retrieval" / "retriever.py").read_text(encoding="utf-8")
misra_source = (ROOT / "services" / "misra_compliance.py").read_text(encoding="utf-8")
check("AnswerService consumes authoritative semantic ambiguity bundle", "authoritative_semantic_ambiguity_evidence" in answer_source and "MISRA CORPUS SEMANTIC AMBIGUITY EVIDENCE" in answer_source)
check("production retriever contains no exact manual probe hardcoding", ambiguity_probes[0].casefold() not in retriever_source.casefold())
check("production MISRA service contains no exact manual probe hardcoding", ambiguity_probes[0].casefold() not in misra_source.casefold())
check("production AnswerService contains no exact manual probe hardcoding", ambiguity_probes[0].casefold() not in answer_source.casefold())

# ---------------------------------------------------------------------------
# 6) Certified production profile remains unchanged.
# ---------------------------------------------------------------------------
check("generation model unchanged", OLLAMA_MODEL == "qwen2.5:7b", OLLAMA_MODEL)
check("embedding model unchanged", EMBED_MODEL_NAME == "qwen3-embedding:8b", EMBED_MODEL_NAME)
check("reranker model unchanged", RERANKER_MODEL == "BAAI/bge-reranker-v2-m3", RERANKER_MODEL)
check("chunk size unchanged", CHUNK_SIZE == 900, CHUNK_SIZE)
check("chunk overlap unchanged", CHUNK_OVERLAP == 150, CHUNK_OVERLAP)
check("global threshold unchanged", float(MIN_RETRIEVAL_SCORE) == 0.55, MIN_RETRIEVAL_SCORE)
check("vector top-k unchanged", VECTOR_TOP_K == 10, VECTOR_TOP_K)
check("bm25 top-k unchanged", BM25_TOP_K == 10, BM25_TOP_K)
check("final top-k unchanged", FINAL_TOP_K == 3, FINAL_TOP_K)

failures = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.15",
    "validation": "PASS" if not failures else "FAIL",
    "checks": len(checks),
    "failures": len(failures),
    "failed_checks": [item["name"] for item in failures],
}
print(json.dumps(summary, indent=2))
raise SystemExit(0 if not failures else 1)
