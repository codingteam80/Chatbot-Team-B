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

# Focused direct relations should stay single-query. These are deliberately
# different subjects/wording from the manual calibration prompts.
direct_fact_holdouts = [
    ("Who won the Northern Robotics Challenge in 2041?", "winner"),
    ("When was the Atlas safety procedure approved?", "approver"),
    ("How many annual leave days are provided?", "quantity"),
    ("Which manager approved the revised travel procedure?", "approver"),
]
for idx, (question, expected_kind) in enumerate(direct_fact_holdouts, 1):
    kind = probe._direct_relation_kind(question)
    check(f"direct-fact holdout {idx} relation recognized", kind == expected_kind, kind)
    searches = probe._generate_multi_query_searches(question, question, None)
    check(f"direct-fact holdout {idx} stays single-query", searches == [question], searches)

# A semantic question with no narrow direct relation must remain eligible for
# the existing ambiguous-query MultiQuery flow rather than being swallowed by
# this new gate. We only assert that the new classifier does not mark it as a
# direct fact; the retained v6.5.5.10/v6.5.5.11 validators cover MQ behavior.
semantic_holdouts = [
    "How should I reason about a vague pointer safety concern?",
    "What MISRA guidance is relevant to this ambiguous control-flow situation?",
]
for idx, question in enumerate(semantic_holdouts, 1):
    check(
        f"ambiguous semantic holdout {idx} not forced into direct-fact gate",
        probe._direct_relation_kind(question) == "",
        probe._direct_relation_kind(question),
    )

# Existing OOD guard semantics remain unchanged; this patch only avoids LLM
# rewrite expansion for already-focused direct facts.
class AbsentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {"query_tokens": tokens, "present_tokens": [], "missing_tokens": tokens}


class PresentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {"query_tokens": tokens, "present_tokens": tokens[:1], "missing_tokens": tokens[1:]}


oos = "Who won the Example Engineering Cup in 2099?"
probe.bm25 = AbsentBM25()
fast_fail, details = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("fully absent direct-fact subject still fast-fails", fast_fail is True, details)
probe.bm25 = PresentBM25()
fast_fail_present, details_present = probe._bm25_direct_relation_ood_fast_fail(oos, [], oos)
check("single lexical footprint still preserves normal retrieval", fast_fail_present is False, details_present)

# Anti-overfit: production retriever must not contain exact manual probe text.
source = (ROOT / "retrieval" / "retriever.py").read_text(encoding="utf-8").casefold()
manual_probes = [
    "who won the fifa world cup in 2022",
    "northern robotics challenge in 2041",
    "atlas safety procedure approved",
    "annual leave days are provided",
]
for idx, phrase in enumerate(manual_probes, 1):
    check(f"no exact direct-fact probe hardcoding {idx}", phrase not in source)

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
    "version": "v6.5.5.12",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
