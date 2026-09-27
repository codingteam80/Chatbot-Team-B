from __future__ import annotations

import importlib.util
import json
import sys
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

from retrieval.retriever import CompanyRetriever
from config import settings

checks: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail not in (None, "") else ""))


probe = CompanyRetriever.__new__(CompanyRetriever)
probe._multi_query_variant_cache = {}
probe._multi_query_search_cache = {}

# v6.5.5.12 direct-fact behavior must remain intact: focused direct facts stay
# single-query, while ambiguous semantic questions are not forced into the gate.
direct_fact_holdouts = [
    ("Who won the Coastal Automation Prize in 2044?", "winner"),
    ("When was the Orion maintenance standard approved?", "approver"),
    ("How many training hours are granted each year?", "quantity"),
    ("Which supervisor approved the updated access procedure?", "approver"),
]
for idx, (question, expected_kind) in enumerate(direct_fact_holdouts, 1):
    kind = probe._direct_relation_kind(question)
    check(f"direct-fact holdout {idx} relation recognized", kind == expected_kind, kind)
    searches = probe._generate_multi_query_searches(question, question, None)
    check(f"direct-fact holdout {idx} stays single-query", searches == [question], searches)

semantic_holdouts = [
    "How should I reason about an unclear pointer lifetime concern?",
    "What guidance is relevant to this ambiguous control-flow situation?",
]
for idx, question in enumerate(semantic_holdouts, 1):
    check(
        f"ambiguous semantic holdout {idx} remains outside direct-fact gate",
        probe._direct_relation_kind(question) == "",
        probe._direct_relation_kind(question),
    )


class AbsentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {"query_tokens": tokens, "present_tokens": [], "missing_tokens": tokens}


class LegacySingleTokenBM25:
    """Compatibility shim: no coverage_search means preserve normal retrieval."""

    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {
            "query_tokens": tokens,
            "present_tokens": tokens[:1],
            "missing_tokens": tokens[1:],
        }


class WeakSingleTokenBM25(LegacySingleTokenBM25):
    def coverage_search(self, query: str, top_k=5, minimum_score=0.0):
        return []


class MultiAnchorCoverageBM25(LegacySingleTokenBM25):
    def coverage_search(self, query: str, top_k=5, minimum_score=0.0):
        return [{"text": "subject anchors co-occur", "metadata": {}, "score": 1.0}]


class TwoPresentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {
            "query_tokens": tokens,
            "present_tokens": tokens[:2],
            "missing_tokens": tokens[2:],
        }

    def coverage_search(self, query: str, top_k=5, minimum_score=0.0):
        return []


# Completely absent subjects still fail closed.
oos = "Who won the Aurora Systems Championship in 2088?"
probe.bm25 = AbsentBM25()
fast_fail, details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("fully absent direct-fact subject fast-fails", fast_fail is True, details)

# v6.5.5.13 closure: a 3+ anchor subject with one isolated corpus token is not
# enough when the existing lexical-coverage view proves there is no record with
# two subject anchors together. This prevents incidental words from paying for
# the expensive vector/reranker path.
probe.bm25 = WeakSingleTokenBM25()
weak_fail, weak_details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("weak single-token footprint can fast-fail", weak_fail is True, weak_details)
check(
    "weak single-token fast-fail records coverage proof",
    bool(weak_details.get("weak_single_token_footprint")) and weak_details.get("coverage_candidates") == 0,
    weak_details,
)

# Compatibility/safety: custom or older BM25 implementations that cannot prove
# co-occurrence continue down normal retrieval instead of being force-failed.
probe.bm25 = LegacySingleTokenBM25()
legacy_fail, legacy_details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("legacy single-token footprint preserves retrieval", legacy_fail is False, legacy_details)

# A corpus record with multi-anchor lexical coverage is a real subject footprint
# and must preserve the normal retrieval path.
probe.bm25 = MultiAnchorCoverageBM25()
coverage_fail, coverage_details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("multi-anchor coverage preserves retrieval", coverage_fail is False, coverage_details)

# Two globally present anchors are also sufficient to preserve retrieval even
# when the compact coverage helper is unavailable or returns nothing.
probe.bm25 = TwoPresentBM25()
two_fail, two_details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("two present subject anchors preserve retrieval", two_fail is False, two_details)

# Short subjects stay conservative: one token out of two must not be force-failed.
short_subject = "Who won the Orion Cup?"
probe.bm25 = WeakSingleTokenBM25()
short_fail, short_details = probe._bm25_direct_relation_ood_fast_fail(short_subject, [], short_subject)
check("short two-anchor subject remains conservative", short_fail is False, short_details)

# Anti-overfit: production retriever must not contain exact manual/validator probes.
source = (ROOT / "retrieval" / "retriever.py").read_text(encoding="utf-8").casefold()
manual_probes = [
    "who won the fifa world cup in 2022",
    "aurora systems championship in 2088",
    "coastal automation prize in 2044",
    "orion maintenance standard approved",
    "training hours are granted each year",
]
for idx, phrase in enumerate(manual_probes, 1):
    check(f"no exact OOD probe hardcoding {idx}", phrase not in source)

# Architecture lock.
check("generation model unchanged", settings.OLLAMA_FAST_MODEL == "qwen2.5:7b", settings.OLLAMA_FAST_MODEL)
check("embedding model unchanged", settings.EMBED_MODEL_NAME == "qwen3-embedding:8b", settings.EMBED_MODEL_NAME)
check("reranker model unchanged", settings.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3", settings.RERANKER_MODEL)
check("chunk size unchanged", int(settings.CHUNK_SIZE) == 900, settings.CHUNK_SIZE)
check("chunk overlap unchanged", int(settings.CHUNK_OVERLAP) == 150, settings.CHUNK_OVERLAP)
check("global threshold unchanged", float(settings.MIN_RETRIEVAL_SCORE) == 0.55, settings.MIN_RETRIEVAL_SCORE)
check("vector top-k unchanged", int(settings.VECTOR_TOP_K) == 10, settings.VECTOR_TOP_K)
check("bm25 top-k unchanged", int(settings.BM25_TOP_K) == 10, settings.BM25_TOP_K)
check("final top-k unchanged", int(settings.FINAL_TOP_K) == 3, settings.FINAL_TOP_K)

failed = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.13",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
