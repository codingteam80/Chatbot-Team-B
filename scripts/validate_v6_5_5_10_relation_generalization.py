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


# A relation may appear in a trailing sentence after a scenario description.
# This is generic intent handling and must not depend on a Rule identifier.
trailing_relation = (
    "I add two uint16_t values and store the result in a uint32_t. "
    "Is the wider destination enough?"
)
check(
    "trailing relation sentence is recognized",
    MisraComplianceMode.yes_no_intent(trailing_relation) == "relation",
    MisraComplianceMode.yes_no_intent(trailing_relation),
)
check(
    "source-backed trailing relation enters MISRA guarded mode",
    MisraComplianceMode.is_request(trailing_relation, "") is True,
    MisraComplianceMode.is_request(trailing_relation, ""),
)

# The authoritative source cue should resolve before generic MultiQuery/rerank.
trailing_results, trailing_details = rescue(trailing_relation)
check(
    "trailing widening scenario resolves to one source Rule body",
    refs(trailing_results) == ["Rule 10.6"],
    refs(trailing_results),
)
check(
    "trailing widening scenario uses corpus requirement wording",
    any(
        "composite expression" in str(cue).casefold()
        and "wider essential type" in str(cue).casefold()
        for cue in trailing_details.get("semantic_cues", [])
    ),
    trailing_details.get("semantic_cues", []),
)

# Source-expanded examples must prove the relation without asking Qwen for a
# polarity judgment. This closes both the wrong-Yes and cold-start latency issue.
relation_question = (
    "Does assigning a narrow arithmetic expression to a wider variable make "
    "the calculation itself wider?"
)
relation_results, _ = rescue(relation_question)
relation_answer = MisraComplianceMode.deterministic_semantic_relation_answer(
    relation_question,
    relation_results,
)
check(
    "source examples resolve calculation-width relation as No",
    relation_answer.startswith("No.") and "Rule 10.6" in relation_answer,
    relation_answer,
)
check(
    "relation answer explains destination versus operation timing",
    "wider destination" in relation_answer.casefold()
    and "before the operation" in relation_answer.casefold(),
    relation_answer,
)

trailing_answer = MisraComplianceMode.deterministic_semantic_relation_answer(
    trailing_relation,
    trailing_results,
)
check(
    "source examples resolve wider-destination sufficiency as No",
    trailing_answer.startswith("No.") and "Rule 10.6" in trailing_answer,
    trailing_answer,
)

# Holdout wording proves behavior generalization rather than exact-probe tuning.
holdouts = [
    "Will storing a 16-bit sum in a 32-bit destination automatically make the addition 32-bit?",
    "A narrow integer sum is assigned to a wider object. Does that make the operation itself wider?",
    "I calculate with smaller unsigned operands and save into a larger integer. Is the larger target sufficient?",
]
for idx, question in enumerate(holdouts, 1):
    intent = MisraComplianceMode.yes_no_intent(question)
    results, _details = rescue(question)
    answer = MisraComplianceMode.deterministic_semantic_relation_answer(question, results)
    check(f"holdout relation {idx} classified generically", intent == "relation", intent)
    check(f"holdout relation {idx} stays in guarded MISRA mode", MisraComplianceMode.is_request(question, "") is True)
    check(f"holdout relation {idx} converges on source Rule", refs(results) == ["Rule 10.6"], refs(results))
    check(f"holdout relation {idx} gets source-proven No", answer.startswith("No.") and "Rule 10.6" in answer, answer)

# Do not over-apply this specialized source-example proof to unrelated binary
# relations that do not have the composite-expression widening cue.
unrelated = [
    "Does a wider destination guarantee that a pointer is valid?",
    "Does using goto make a function recursive?",
]
for idx, question in enumerate(unrelated, 1):
    results, _details = rescue(question)
    answer = MisraComplianceMode.deterministic_semantic_relation_answer(question, results)
    check(f"unrelated relation {idx} is not force-answered", answer == "", answer)

# Existing v6.5.5.9 semantics must stay intact.
for question, expected_prefix, expected_ref in [
    ("Can I have an empty else block?", "Needs more context.", "Rule 15.7"),
    ("Can I write if (counter) under MISRA?", "Needs more context.", "Rule 14.4"),
]:
    results, _details = rescue(question)
    answer = MisraComplianceMode.deterministic_yes_no_answer(question, results)
    check(
        f"existing semantics preserved: {expected_ref}",
        answer.startswith(expected_prefix) and expected_ref in answer,
        answer,
    )

# Exact statement no-LLM behavior must remain unchanged.
service = AnswerService.__new__(AnswerService)
context = """===== DOCUMENT 1 =====
Rule 22.2
A block of memory shall only be freed if it was allocated by means of a
Standard Library function
C90 [Undefined 92], C99 [Undefined 169]
Category
Mandatory
Analysis
Undecidable, System
Applies to
C90, C99"""
statement_question = "What does Rule 22.2 state?"
focus = service._detect_answer_focus(statement_question, "")
statement_answer = service._deterministic_structured_answer(
    context,
    statement_question,
    statement_question,
    focus,
)
check(
    "exact Rule statement remains deterministic",
    statement_answer.startswith("Rule 22.2:") and "Standard Library function" in statement_answer,
    statement_answer,
)

# Anti-overfit: exact manual probes belong only in QA/validator files.
production_files = [
    ROOT / "services" / "answer_service.py",
    ROOT / "services" / "misra_compliance.py",
]
production_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in production_files)
manual_probes = [relation_question, trailing_relation, *holdouts]
for idx, sentence in enumerate(manual_probes, 1):
    check(f"no exact relation-probe hardcoding {idx}", sentence not in production_text)

# Stable production parameters remain untouched.
settings_text = (ROOT / "config" / "settings.py").read_text(encoding="utf-8-sig")
for needle, label in [
    ('qwen2.5:7b', "generation model unchanged"),
    ('qwen3-embedding:8b', "embedding model unchanged"),
    ('BAAI/bge-reranker-v2-m3', "reranker model unchanged"),
    ('CHUNK_SIZE = 900', "chunk size unchanged"),
    ('CHUNK_OVERLAP = 150', "chunk overlap unchanged"),
    ('MIN_RETRIEVAL_SCORE = 0.55', "global threshold unchanged"),
]:
    check(label, needle in settings_text)

failed = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.10",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
