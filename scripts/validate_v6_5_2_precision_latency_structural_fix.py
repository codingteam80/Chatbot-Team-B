from __future__ import annotations

import json
import py_compile
import sys
import types
from pathlib import Path

import numpy as np

# The validator exercises pure retriever helper methods without launching the
# Streamlit UI. Provide the tiny decorator surface required by bm25_index at
# import time when Streamlit is unavailable in the validation environment.
if "streamlit" not in sys.modules:
    st = types.ModuleType("streamlit")
    def _identity_cache(*args, **kwargs):
        def decorator(fn):
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
from retrieval.retriever import CompanyRetriever
from services.misra_compliance import MisraComplianceMode
import retrieval.retriever as retriever_module

OUTDIR = ROOT / "logs" / "precision_latency_structural_fix"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.5.2_precision_latency_structural_fix_validation_latest.json"

checks = []

def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})

def source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

changed = [
    "retrieval/semantic_rule_resolver.py",
    "retrieval/retriever.py",
    "services/misra_compliance.py",
    "services/answer_service.py",
]
for rel in changed:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, exc)

# Architecture locks: this release must not hide benchmark failures by changing
# the production models, global chunk gate, or splitter geometry.
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

semantic_source = source("retrieval/semantic_rule_resolver.py")
retriever_source = source("retrieval/retriever.py")
misra_source = source("services/misra_compliance.py")
answer_source = source("services/answer_service.py")

check("Rule resolver reranks concise authoritative requirements", "_rerank_rule_candidates" in semantic_source and 'clone["text"] = f"{reference}. {requirement}"' in semantic_source)
check("Rule resolver has query-vector cache", "_query_vector_cache" in semantic_source)
check("Rule resolver has Rule-rerank score cache", "_rerank_score_cache" in semantic_source)
check("Semantic consensus gate is MultiQuery-only", "len(queries) > 1" in semantic_source and 'acceptance_basis = "semantic_consensus"' in semantic_source)
check("Global chunk threshold is not lowered", "MIN_RETRIEVAL_SCORE" in semantic_source and "MIN_RETRIEVAL_SCORE =" not in semantic_source)
check("Retriever has intent-scoped MultiQuery variant cache", "_multi_query_variant_cache" in retriever_source and 'cache_scope": "semantic intent"' in retriever_source)
check("Retriever has role-aware MISRA precision pool", "_prepare_misra_role_aware_reranker_pool" in retriever_source and "MISRA ROLE-AWARE PRECISION POOL" in retriever_source)
check("Trailing Yes/No question clause is recognized", "permission_candidates" in misra_source and "sentence_parts" in misra_source)
check("Answer evidence log records semantic acceptance basis", "acceptance_basis" in answer_source and "full_multi_query_consensus" in answer_source)

# No benchmark phrase or expected Rule mapping may be added to the semantic resolver.
for forbidden in [
    "return value unused",
    "call chain",
    "localtime",
    "strerror",
    "write_status",
    "update_state",
    "switch handles every",
    "Rule 17.7",
    "Rule 17.2",
    "Rule 13.5",
    "Rule 16.4",
    "Rule 21.20",
    "Rule 21.17",
]:
    check(f"No benchmark mapping in semantic resolver: {forbidden}", forbidden.casefold() not in semantic_source.casefold())

# Generic trailing-sentence yes/no behavior; no Rule identifier is involved.
trailing = "The described construct satisfies every listed case. Is that acceptable under MISRA C?"
check("Trailing scenario + question is permission intent", MisraComplianceMode.yes_no_intent(trailing) == "permission", MisraComplianceMode.yes_no_intent(trailing))
check("Trailing scenario + question receives MISRA Yes/No focus", MisraComplianceMode.answer_focus(trailing).startswith("MISRA YES OR NO:"), MisraComplianceMode.answer_focus(trailing))
check("Plain requirement lookup remains non-binary", MisraComplianceMode.yes_no_intent("Which MISRA requirement covers this construct?") == "")

# Synthetic semantic resolver fixtures test architecture, not benchmark text.
class CountingEmbed:
    def __init__(self):
        self.calls = []
    def get_query_embedding_batch(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

class CountingReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []
    def rerank(self, query, candidates):
        self.calls.append([str((item.get("profile") or {}).get("display_name", "")) for item in candidates])
        # scores are indexed by Rule number, not call position, so cache/missing
        # candidate subsets remain deterministic.
        for item in candidates:
            identifier = int(str((item.get("profile") or {}).get("identifier", "0") or "0"))
            item["rerank_score"] = float(self.scores[identifier - 1])
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

profiles_fixture = [
    {
        "kind": "rule",
        "identifier": str(i + 1),
        "display_name": f"Rule {i + 1}",
        "requirement_text": f"Requirement {i + 1}",
        "profile_text": f"Rule {i + 1}. Requirement: Requirement {i + 1}. Rationale: supporting text that is deliberately longer.",
        "file_name": "MISRA.pdf",
        "file_path": "MISRA.pdf",
        "page_start": i + 1,
        "page_end": i + 1,
        "supporting_roles": ["rationale"],
    }
    for i in range(4)
]
vectors = np.asarray([
    [1.0, 0.0, 0.0],
    [0.65, 0.76, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

def make_resolver(scores):
    embed = CountingEmbed()
    reranker = CountingReranker(scores)
    resolver = SemanticRuleResolver(embedding_model=embed, reranker=reranker)
    resolver._metadata = {"profiles": profiles_fixture}
    resolver._vectors = vectors
    resolver._ensure_loaded = lambda: None
    return resolver, embed, reranker

resolver, embed, reranker = make_resolver([0.03, 0.01, 0.005, 0.001])
first = resolver.resolve("natural query")
wide = resolver.resolve("natural query", query_variants=["variant one", "variant two"])
check("Single-query low BGE result remains ambiguous", first.get("accepted") is False and first.get("status") == "ambiguous", first)
check("Widened semantic consensus can accept a stable Rule leader", wide.get("accepted") is True and wide.get("reference") == "Rule 1" and wide.get("acceptance_basis") == "semantic_consensus", wide)
check("Original query embedding is reused during widening", len(embed.calls) == 2 and embed.calls[0] == ["natural query"] and embed.calls[1] == ["variant one", "variant two"], embed.calls)
# First call reranks five-or-fewer profiles; widened pass should reuse existing
# query/profile scores for repeated candidates and only score genuinely new ones.
check("Rule rerank cache avoids full repeated CrossEncoder work", len(reranker.calls) <= 2, reranker.calls)

resolver2, _, _ = make_resolver([0.15, 0.17, 0.01, 0.005])
slight_disagreement = resolver2.resolve("natural query", query_variants=["variant one", "variant two"])
check("Small BGE disagreement cannot erase a strong semantic leader", slight_disagreement.get("accepted") is True and slight_disagreement.get("reference") == "Rule 1", slight_disagreement)

resolver3, _, _ = make_resolver([0.10, 0.40, 0.01, 0.005])
large_disagreement = resolver3.resolve("natural query", query_variants=["variant one", "variant two"])
check("Large independent BGE disagreement remains ambiguous", large_disagreement.get("accepted") is False and large_disagreement.get("status") == "ambiguous", large_disagreement)

# Boundary case: a narrow but real semantic lead plus only a small independent
# BGE disagreement is accepted after MultiQuery consensus; a smaller semantic
# margin must remain ambiguous. This models calibration behavior without any
# production Rule identifiers or benchmark phrases.
def unit_vector(similarity):
    return [float(similarity), float(max(0.0, 1.0 - similarity * similarity) ** 0.5), 0.0]
edge_vectors = np.asarray([
    unit_vector(0.686),
    unit_vector(0.659),
    unit_vector(0.55),
    unit_vector(0.45),
], dtype=np.float32)
edge = SemanticRuleResolver(embedding_model=CountingEmbed(), reranker=CountingReranker([0.169, 0.200, 0.02, 0.01]))
edge._metadata = {"profiles": profiles_fixture}
edge._vectors = edge_vectors
edge._ensure_loaded = lambda: None
edge_result = edge.resolve("natural query", query_variants=["variant one", "variant two"])
check("Narrow semantic lead survives small BGE disagreement", edge_result.get("accepted") is True and edge_result.get("reference") == "Rule 1", edge_result)

narrow_vectors = np.asarray([
    unit_vector(0.686),
    unit_vector(0.670),
    unit_vector(0.55),
    unit_vector(0.45),
], dtype=np.float32)
narrow = SemanticRuleResolver(embedding_model=CountingEmbed(), reranker=CountingReranker([0.169, 0.180, 0.02, 0.01]))
narrow._metadata = {"profiles": profiles_fixture}
narrow._vectors = narrow_vectors
narrow._ensure_loaded = lambda: None
narrow_result = narrow.resolve("natural query", query_variants=["variant one", "variant two"])
check("Too-small semantic margin remains ambiguous", narrow_result.get("accepted") is False, narrow_result)

# Role-aware pool: broad appendix/summary chunks may remain as support, but they
# cannot crowd structured Rule roles out when enough structured evidence exists.
probe = object.__new__(CompanyRetriever)
synthetic_pool = []
for i in range(6):
    synthetic_pool.append({"text": f"Appendix summary {i}", "metadata": {"section_type": "section", "section_role": "document_section", "file_name": "MISRA.pdf"}})
for rid in ["1.1", "1.2", "1.3"]:
    synthetic_pool.append({"text": f"Rule {rid}\nRequirement", "metadata": {"section_type": "rule", "section_role": "requirement", "rule_id": rid, "file_name": "MISRA.pdf"}})
    synthetic_pool.append({"text": f"Rule {rid}\nRationale", "metadata": {"section_type": "section", "section_role": "rationale", "parent_type": "rule", "parent_identifier": rid, "parent_rule_id": rid, "file_name": "MISRA.pdf"}})
role_pool = probe._prepare_misra_role_aware_reranker_pool(synthetic_pool, limit=6)
broad_count = sum(1 for item in role_pool if str((item.get("metadata", {}) or {}).get("section_role", "")) == "document_section")
structured_refs = [probe._misra_role_reference(item)[0] for item in role_pool if probe._misra_role_reference(item)[0]]
check("Role-aware pool caps broad document sections", broad_count <= 2, broad_count)
check("Role-aware pool preserves structured Rule evidence", len(structured_refs) >= 4, structured_refs)
check("Role-aware pool is metadata-driven", all(ref.startswith("Rule ") for ref in structured_refs), structured_refs)

# Intent-only MultiQuery reuse: a decorated fallback query must reuse variants
# generated for the same semantic intent rather than invoke the LLM again.
cache_probe = object.__new__(CompanyRetriever)
cache_probe._multi_query_search_cache = {}
cache_probe._multi_query_variant_cache = {}
call_count = {"n": 0}
original_generator = retriever_module.generate_multi_query_variants
try:
    def fake_generator(client, question):
        call_count["n"] += 1
        return ["variant alpha", "variant beta"]
    retriever_module.generate_multi_query_variants = fake_generator
    first_searches = cache_probe._generate_multi_query_searches("plain query", intent_query="same semantic intent")
    second_searches = cache_probe._generate_multi_query_searches("MISRA C | plain query | normalized", intent_query="same semantic intent")
finally:
    retriever_module.generate_multi_query_variants = original_generator
check("MultiQuery variants generated only once per semantic intent", call_count["n"] == 1, call_count)
check("Decorated fallback reuses the same two alternatives", first_searches[1:] == second_searches[1:] == ["variant alpha", "variant beta"], {"first": first_searches, "second": second_searches})

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.2.json"
check("v6.5.2 manifest exists", manifest_path.is_file())
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        check("v6.5.2 manifest parses", False, exc)
    else:
        check("v6.5.2 manifest parses", True)
        check("Manifest base is v6.5.0 production runtime", manifest.get("base") == "v6.5.0 production runtime + v6.5.1.1 benchmark tooling")
        check("Manifest says no KB rebuild", manifest.get("kb_rebuild_performed") is False)
        check("Manifest preserves global retrieval threshold", float(manifest.get("minimum_retrieval_score", -1)) == 0.55)
        check("Manifest preserves chunking", manifest.get("chunk_size") == 900 and manifest.get("chunk_overlap") == 150)
        check("Manifest records role-aware precision", manifest.get("misra_role_aware_precision_pool") is True)
        check("Manifest records duplicate MultiQuery elimination", manifest.get("intent_scoped_multiquery_variant_reuse") is True)
        check("Manifest records semantic consensus calibration", manifest.get("semantic_rule_consensus_precision_gate") is True)

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.5.2",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(checks),
    "profile_count": len(profiles),
    "checks": checks,
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("version", "validation", "failed_checks", "check_count", "profile_count")}, indent=2))
if failed:
    for item in failed:
        print("[FAIL]", item["name"], "::", item.get("detail", ""))
    raise SystemExit(1)
print(f"[PASS] v6.5.2 validator: {len(checks)} checks, 0 failures.")
raise SystemExit(0)
