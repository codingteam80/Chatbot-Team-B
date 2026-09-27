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
from services.answer_service import AnswerService
from services.misra_compliance import MisraComplianceMode

checks: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail not in (None, "") else ""))


def rescue(question: str):
    results, details = MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(
        question,
        top_k=6,
    )
    return results, details


def refs(results):
    return [
        str((item.get("metadata", {}) or {}).get("section_title", "") or "")
        for item in (results or [])
    ]


# ---------------------------------------------------------------------------
# 1) Self-contained natural lookups must not borrow a stale previous scenario.
# ---------------------------------------------------------------------------
prior_non_misra_state = {"context": "accepted prior context", "misra": False}
prior_misra_state = {"context": "accepted prior context", "misra": True}
self_contained = [
    "May function na may return value pero tinawag lang siya at hindi ginamit ang ibinalik na value. Anong MISRA rule ang dapat tingnan?",
    "A string-handling library call may write beyond the destination object's bounds. Which MISRA requirement addresses that risk?",
]
for idx, question in enumerate(self_contained, 1):
    check(
        f"self-contained lookup {idx} ignores stale non-MISRA anchor",
        AnswerService._is_grounded_followup_candidate(question, prior_non_misra_state) is False,
    )
    check(
        f"self-contained lookup {idx} ignores stale MISRA anchor",
        AnswerService._is_grounded_followup_candidate(question, prior_misra_state) is False,
    )
check(
    "true deictic follow-up remains a follow-up",
    AnswerService._is_grounded_followup_candidate("Can you explain that?", prior_misra_state) is True,
)

# ---------------------------------------------------------------------------
# 2) Source-language concept rescue for the slow/failing natural scenarios.
#    These tests assert source convergence, never phrase->Rule production maps.
# ---------------------------------------------------------------------------
scenario_expectations = [
    (
        "May function na may return value pero tinawag lang siya at hindi ginamit ang ibinalik na value. Anong MISRA rule ang dapat tingnan?",
        "Rule 17.7",
        "value returned by a function",
    ),
    (
        "A string-handling library call may write beyond the destination object's bounds. Which MISRA requirement addresses that risk?",
        "Rule 21.17",
        "string handling functions",
    ),
    (
        "Anong MISRA rule ang applicable kapag bumabalik sa parehong function ang call chain kahit dumaan muna sa ibang function?",
        "Rule 17.2",
        "directly or indirectly",
    ),
    (
        "A function reaches itself again through another helper function. Which MISRA rule covers that call pattern?",
        "Rule 17.2",
        "directly or indirectly",
    ),
    (
        "May pointer mula sa strerror, tapos tumawag ulit sa strerror bago gamitin ulit ang lumang pointer. Anong MISRA rule ang applicable?",
        "Rule 21.20",
        "subsequent call to the same function",
    ),
]
for idx, (question, expected_ref, expected_cue_fragment) in enumerate(scenario_expectations, 1):
    results, details = rescue(question)
    check(f"scenario {idx} resolves one authoritative source Rule", refs(results) == [expected_ref], refs(results))
    cues = [str(cue).casefold() for cue in details.get("semantic_cues", [])]
    check(
        f"scenario {idx} uses source requirement wording",
        any(expected_cue_fragment.casefold() in cue for cue in cues),
        details.get("semantic_cues", []),
    )

# Holdout wording: same behavior, different wording.
holdouts = [
    ("A non-void call is made only for side effects and the returned result is thrown away. Which requirement applies?", "Rule 17.7"),
    ("A string library function can overrun the destination buffer. What MISRA requirement covers that?", "Rule 21.17"),
    ("The call path eventually gets back to the starting function through a helper. Which MISRA rule is relevant?", "Rule 17.2"),
    ("I keep the pointer from localtime, invoke localtime again, then read through the old pointer. Which rule applies?", "Rule 21.20"),
]
for idx, (question, expected_ref) in enumerate(holdouts, 1):
    results, _ = rescue(question)
    check(f"holdout scenario {idx} converges on source Rule", refs(results) == [expected_ref], refs(results))

# ---------------------------------------------------------------------------
# 3) Broad structured inventories should bypass semantic MultiQuery/LLM synthesis.
# ---------------------------------------------------------------------------
retriever = CompanyRetriever()
pointer_question = "Which MISRA rules apply to pointers?"
pointer_results = retriever._retrieve_structured_rule_catalog(
    query="MISRA C | Which MISRA rules apply to pointers? | which misra rules apply to pointers",
    intent_query=pointer_question,
)
check("pointer inventory is source-structured", len(pointer_results) >= 10, len(pointer_results))
check(
    "pointer inventory carries structured topic anchors",
    bool(pointer_results) and all(item.get("_structured_topic_anchor") for item in pointer_results),
)
service = AnswerService.__new__(AnswerService)
check("pointer question is multi-answer/list intent", service._is_multi_answer_question(pointer_question) is True)
pointer_answer = service._deterministic_structured_topic_list_answer(pointer_results)
check(
    "pointer inventory renders deterministically",
    "Pointer-related MISRA rules" in pointer_answer and "11.1–11.9" in pointer_answer and "18.1–18.5" in pointer_answer,
    pointer_answer[:500],
)

# ---------------------------------------------------------------------------
# 4) Clear direct-fact OOD questions can fail closed before MultiQuery only
#    when all meaningful subject tokens are absent from the corpus.
# ---------------------------------------------------------------------------
class AbsentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {"query_tokens": tokens, "present_tokens": [], "missing_tokens": tokens}


class PresentBM25:
    def meaningful_token_presence(self, query: str):
        tokens = [token for token in str(query).split() if token]
        return {
            "query_tokens": tokens,
            "present_tokens": tokens[:1],
            "missing_tokens": tokens[1:],
        }


probe = CompanyRetriever.__new__(CompanyRetriever)
probe.bm25 = AbsentBM25()
oos_question = "Who won the Example Engineering Cup in 2099?"
fast_fail, details = probe._bm25_direct_relation_ood_fast_fail(oos_question, [], oos_question)
check("absent direct-fact subject fast-fails before ML retrieval", fast_fail is True, details)
probe.bm25 = PresentBM25()
fast_fail_present, details_present = probe._bm25_direct_relation_ood_fast_fail(oos_question, [], oos_question)
check("any corpus footprint preserves normal retrieval", fast_fail_present is False, details_present)

# ---------------------------------------------------------------------------
# 5) Regression + anti-overfit + architecture lock.
# ---------------------------------------------------------------------------
for question, expected_ref in [
    ("Can I have an empty else block?", "Rule 15.7"),
    ("Can I write if (counter) under MISRA?", "Rule 14.4"),
    ("I add two uint16_t values and store the result in a uint32_t. Is the wider destination enough?", "Rule 10.6"),
]:
    results, _ = rescue(question)
    check(f"prior calibrated behavior preserved: {expected_ref}", refs(results) == [expected_ref], refs(results))

production_files = [
    ROOT / "services" / "answer_service.py",
    ROOT / "services" / "misra_compliance.py",
    ROOT / "retrieval" / "retriever.py",
]
production_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in production_files)
manual_probes = [
    *self_contained,
    *(question for question, _, _ in scenario_expectations),
    *(question for question, _ in holdouts),
    pointer_question,
    oos_question,
]
for idx, sentence in enumerate(dict.fromkeys(manual_probes), 1):
    check(f"no exact calibration-probe hardcoding {idx}", sentence not in production_text)

settings_text = (ROOT / "config" / "settings.py").read_text(encoding="utf-8-sig")
for needle, label in [
    ('qwen2.5:7b', "generation model unchanged"),
    ('qwen3-embedding:8b', "embedding model unchanged"),
    ('BAAI/bge-reranker-v2-m3', "reranker model unchanged"),
    ('CHUNK_SIZE = 900', "chunk size unchanged"),
    ('CHUNK_OVERLAP = 150', "chunk overlap unchanged"),
    ('MIN_RETRIEVAL_SCORE = 0.55', "global threshold unchanged"),
    ('VECTOR_TOP_K = 10', "vector top-k unchanged"),
    ('BM25_TOP_K = 10', "bm25 top-k unchanged"),
    ('FINAL_TOP_K = 3', "final top-k unchanged"),
]:
    check(label, needle in settings_text)

failed = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.11",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
