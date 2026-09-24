from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from retrieval.semantic_rule_resolver import SemanticRuleResolver, _load_bm25_records, extract_rule_profiles
from services.misra_compliance import MisraComplianceMode

OUTDIR = ROOT / "logs" / "corpus_semantic_rule_resolver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.5.0_corpus_semantic_rule_resolver_validation_latest.json"

checks = []

def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})


def source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

# Syntax checks first: fail-safe against shipping a partially edited runtime.
changed = [
    "retrieval/semantic_rule_resolver.py",
    "retrieval/retriever.py",
    "embeddings/ollama_embedding.py",
    "runtime/prewarm.py",
    "services/misra_compliance.py",
    "services/answer_service.py",
    "scripts/build_misra_semantic_rule_index.py",
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
check("Vector Top-K remains 10", settings.VECTOR_TOP_K == 10)
check("BM25 Top-K remains 10", settings.BM25_TOP_K == 10)
check("Final Top-K remains 3", settings.FINAL_TOP_K == 3)
check("Minimum retrieval score remains 0.55", abs(float(settings.MIN_RETRIEVAL_SCORE) - 0.55) < 1e-12)
check("Chunk size remains 900", settings.CHUNK_SIZE == 900)
check("Chunk overlap remains 150", settings.CHUNK_OVERLAP == 150)
check("MultiQuery remains enabled", settings.MULTI_QUERY_RETRIEVAL_ENABLED is True)
check("MultiQuery remains two alternatives", settings.MULTI_QUERY_VARIANT_COUNT == 2)
check("RRF K remains 60", settings.MULTI_QUERY_RRF_K == 60)

# Corpus-derived profile extraction: no new KB rebuild required.
records = _load_bm25_records()
profiles = extract_rule_profiles(records)
refs = {p.display_name for p in profiles}
check("BM25 corpus remains available", len(records) == 951, len(records))
check("Corpus-derived Rule/Directive profiles extracted", len(profiles) >= 170, len(profiles))
check("Expected current profile inventory is 173", len(profiles) == 173, len(profiles))
for ref in ["Rule 2.7", "Rule 13.5", "Rule 15.3", "Rule 16.4", "Rule 21.19"]:
    check(f"Corpus profile exists: {ref}", ref in refs)

semantic_source = source("retrieval/semantic_rule_resolver.py")
# Runtime resolver must not contain test-specific Rule identifiers or query phrase mappings.
for forbidden in ["Rule 2.7", "Rule 15.3", "never touches", "unused parameter", "nested block"]:
    check(f"Semantic resolver has no hardcoded mapping token: {forbidden}", forbidden.casefold() not in semantic_source.casefold())
check("Semantic index is derived from structured source roles", "extract_rule_profiles" in semantic_source and "requirement_text" in semantic_source and "supporting_roles" in semantic_source)
check("Rule-level MultiQuery uses RRF", "MULTI_QUERY_RRF_K" in semantic_source and "rrf" in semantic_source.casefold())
check("Semantic resolver preserves global threshold floor", "MIN_RETRIEVAL_SCORE" in semantic_source)

# Scope detection should generalize without a concept-to-Rule dictionary.
unseen = "What MISRA requirement applies when a callback receives an input it never touches anywhere in its implementation?"
check("Unseen natural MISRA wording enters semantic Rule scope", MisraComplianceMode.semantic_rule_resolution_scope(unseen))
check("Unseen wording does not require a legacy semantic cue", not bool(MisraComplianceMode.semantic_cues(unseen)), MisraComplianceMode.semantic_cues(unseen))
check("Generic non-MISRA policy question stays outside semantic Rule scope", not MisraComplianceMode.semantic_rule_resolution_scope("What company policy applies when an employee is late?"))
check("Code-like MISRA review stays outside single-Rule semantic scope", not MisraComplianceMode.semantic_rule_resolution_scope("MISRA check: int f(int x) { return x + 1; }"))
check("Unseen question is recognized as requirement lookup", MisraComplianceMode.semantic_rule_lookup_intent(unseen))

# Independently source-verify a semantic winner from the authoritative corpus.
evidence = MisraComplianceMode.authoritative_semantic_reference_evidence(records, kind="rule", identifier="2.7", diagnostics={"validator": True})
check("Semantic reference can be independently verified in corpus", len(evidence) == 1)
if evidence:
    ev = evidence[0]
    check("Verified semantic evidence is marked source-resolved", ev.get("_misra_semantic_resolver") is True)
    check("Verified semantic evidence is exact Rule 2.7", str((ev.get("metadata", {}) or {}).get("rule_id", "")) == "2.7")
    check("Verified semantic evidence carries exact source requirement", "There should be no unused parameters in functions" in str(ev.get("text", "")))
    check("Legacy deterministic Yes/No is disabled for semantic evidence", MisraComplianceMode.deterministic_yes_no_answer("Is this acceptable under MISRA?", evidence) == "")
check("Nonexistent semantic reference cannot be fabricated", MisraComplianceMode.authoritative_semantic_reference_evidence(records, kind="rule", identifier="999.999") == [])

# Validate source-resolved generation contract semantics.
if evidence:
    contract = MisraComplianceMode.build_generation_contract("Is this acceptable under MISRA?", evidence)
    check("Generation contract labels semantic evidence SOURCE-RESOLVED", "SOURCE-RESOLVED REQUIREMENT" in contract, contract[:800])
    check("Generation contract forbids information-not-found after source resolution", "Never answer 'Information not found'" in contract)

# Synthetic Rule-level resolver test: query variants are batch-embedded, fused by RRF,
# then a clear BGE winner is accepted; close winners remain ambiguous.
class FakeEmbed:
    def get_query_embedding_batch(self, texts):
        rows = []
        for i, _ in enumerate(texts):
            rows.append([1.0, 0.05 * i, 0.0])
        return rows

class FakeReranker:
    def __init__(self, scores): self.scores = list(scores)
    def rerank(self, query, candidates):
        for item, score in zip(candidates, self.scores):
            item["rerank_score"] = score
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

synthetic_profiles = [
    {"kind": "rule", "identifier": str(i+1), "display_name": f"Rule {i+1}", "requirement_text": f"Requirement {i+1}", "profile_text": f"Rule {i+1}. Requirement {i+1}", "file_name": "MISRA.pdf", "file_path": "MISRA.pdf", "page_start": i+1, "page_end": i+1, "supporting_roles": []}
    for i in range(4)
]
synthetic_vectors = np.asarray([[1.0,0.0,0.0],[0.8,0.6,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]], dtype=np.float32)
synthetic_vectors /= np.linalg.norm(synthetic_vectors, axis=1, keepdims=True)
resolver = SemanticRuleResolver(embedding_model=FakeEmbed(), reranker=FakeReranker([0.93, 0.55, 0.20, 0.10]))
resolver._metadata = {"profiles": synthetic_profiles}
resolver._vectors = synthetic_vectors
resolver._ensure_loaded = lambda: None
rrf_result = resolver.resolve("original semantic query", query_variants=["variant one", "variant two"])
check("Rule-level MultiQuery resolver executes three semantic queries", rrf_result.get("query_count") == 3, rrf_result)
check("Rule-level MultiQuery fusion reports RRF", rrf_result.get("multi_query_fusion") == "RRF", rrf_result)
check("Clear precision winner can be accepted", rrf_result.get("accepted") is True, rrf_result)

ambiguous = SemanticRuleResolver(embedding_model=FakeEmbed(), reranker=FakeReranker([0.70, 0.68, 0.20, 0.10]))
ambiguous._metadata = {"profiles": synthetic_profiles}
ambiguous._vectors = synthetic_vectors
ambiguous._ensure_loaded = lambda: None
ambiguous_result = ambiguous.resolve("original semantic query")
check("Close Rule candidates remain ambiguous instead of guessed", ambiguous_result.get("accepted") is False and ambiguous_result.get("status") == "ambiguous", ambiguous_result)

# Runtime integration checks by source inspection avoid loading Qdrant/Ollama during validator.
retriever_source = source("retrieval/retriever.py")
answer_source = source("services/answer_service.py")
prewarm_source = source("runtime/prewarm.py")
embedding_source = source("embeddings/ollama_embedding.py")
check("Retriever widens ambiguous Rule resolution with MultiQuery", "query_variants=searches[1:]" in retriever_source and "multi_query_rule_resolution" in retriever_source)
check("Retriever caches MultiQuery variants to prevent duplicate LLM expansion", "_multi_query_search_cache" in retriever_source and "REUSED CACHED VARIANTS" in retriever_source)
check("Qwen3 query variants embed in one batch", "get_query_embedding_batch" in embedding_source)
check("Startup prewarm builds/loads derived semantic Rule index", "build_semantic_rule_index" in prewarm_source)
check("Answer service uses semantic Rule resolver before generic retrieval", "resolve_misra_rule_semantics" in answer_source and "authoritative_semantic_reference_evidence" in answer_source)
check("Semantic Yes/No uses compact structured source verifier", "misra_semantic_relation_verifier" in answer_source and "generate_structured_json" in answer_source)
check("Semantic relation verifier supports uncertainty", '"needs_context"' in answer_source)
check("No Qdrant/BM25 rebuild is invoked by semantic resolver", "rebuild" not in semantic_source.casefold())

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.0.json"
check("v6.5.0 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        check("v6.5.0 manifest parses", False, exc)
    else:
        check("v6.5.0 manifest parses", True)
        check("Manifest base is v6.4.102", manifest.get("base") == "v6.4.102")
        check("Manifest records corpus-driven Rule resolver", manifest.get("corpus_driven_semantic_rule_resolver") is True)
        check("Manifest records Rule-level MultiQuery RRF", manifest.get("rule_level_multiquery_rrf") is True)
        check("Manifest records no KB rebuild", manifest.get("kb_rebuild_performed") is False)
        check("Manifest preserves threshold", float(manifest.get("minimum_retrieval_score", -1)) == 0.55)

failed = [item for item in checks if not item["passed"]]
result = {"version": "v6.5.0", "validation": "PASS" if not failed else "FAIL", "failed_checks": len(failed), "profile_count": len(profiles), "checks": checks}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"version": result["version"], "validation": result["validation"], "failed_checks": len(failed), "profile_count": len(profiles)}, indent=2))
if failed:
    for item in failed:
        print("[FAIL]", item["name"], "::", item.get("detail", ""))
    raise SystemExit(1)
print(f"[PASS] v6.5.0 validator: {len(checks)} checks, 0 failures.")
raise SystemExit(0)
