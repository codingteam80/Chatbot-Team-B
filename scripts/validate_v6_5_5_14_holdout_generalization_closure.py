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


def rule_ids(results):
    return [
        str((item.get("metadata", {}) or {}).get("rule_id", "") or "")
        for item in (results or [])
    ]


# ---------------------------------------------------------------------------
# 1) One-source-file linkage wording: English/Taglish paraphrases converge on
#    the authoritative source statement without phrase->Rule hardcoding.
# ---------------------------------------------------------------------------
linkage_probes = [
    "Kung sa iisang .c file lang talaga ginagamit ang function, ano ang MISRA expectation tungkol sa linkage nito?",
    "A helper function is used only inside one source file. What does MISRA expect for its linkage?",
    "Isang C file lang gumagamit ng helper function. Anong guidance ang applicable sa visibility/linkage nito?",
]
for idx, question in enumerate(linkage_probes, 1):
    results, details = rescue(question)
    check(f"single-file linkage probe {idx} enters guarded MISRA mode", MisraComplianceMode.is_request(question, "") is True)
    check(f"single-file linkage probe {idx} resolves source Rule 8.7", refs(results) == ["Rule 8.7"], refs(results))
    check(
        f"single-file linkage probe {idx} uses authoritative linkage wording",
        any(
            "external linkage" in str(cue).casefold()
            and "one translation unit" in str(cue).casefold()
            for cue in details.get("semantic_cues", [])
        ),
        details.get("semantic_cues", []),
    )


# ---------------------------------------------------------------------------
# 2) Relation verbs such as change/alter/affect should classify proposition
#    questions without widening unrelated prose into MISRA mode.
# ---------------------------------------------------------------------------
width_relation_probes = [
    "A calculation is performed with smaller unsigned operands and only afterward stored in a larger unsigned object. Does the larger object change the width of the earlier calculation?",
    "A narrow arithmetic result is saved to a wider integer. Does that alter the width used for the earlier operation?",
    "Smaller unsigned operands are evaluated first and the result is assigned to a larger target. Does the larger target affect the calculation width?",
]
for idx, question in enumerate(width_relation_probes, 1):
    intent = MisraComplianceMode.yes_no_intent(question)
    results, _details = rescue(question)
    answer = MisraComplianceMode.deterministic_semantic_relation_answer(question, results)
    check(f"width relation probe {idx} classified as relation", intent == "relation", intent)
    check(f"width relation probe {idx} enters guarded MISRA mode", MisraComplianceMode.is_request(question, "") is True)
    check(f"width relation probe {idx} resolves source Rule 10.6", refs(results) == ["Rule 10.6"], refs(results))
    check(f"width relation probe {idx} gets source-proven No", answer.startswith("No.") and "Rule 10.6" in answer, answer)

unrelated_change = "Does storing a large image in a smaller folder change the image width?"
check(
    "unrelated change relation does not enter MISRA mode",
    MisraComplianceMode.is_request(unrelated_change, "") is False,
    {
        "intent": MisraComplianceMode.yes_no_intent(unrelated_change),
        "cues": MisraComplianceMode.semantic_cues(unrelated_change),
    },
)


# ---------------------------------------------------------------------------
# 3) Repeated Standard Library provider calls: frequency wording such as twice
#    and second call is the same source relation as subsequent call.
# ---------------------------------------------------------------------------
repeated_provider_probes = [
    "After calling localtime twice, code still uses the pointer returned by the first call. What MISRA issue does this raise?",
    "The pointer from localtime is kept, then a second localtime call happens before that old pointer is used. Which MISRA requirement applies?",
    "I call strerror two times and later read through the pointer from the first call. What MISRA rule should I inspect?",
]
for idx, question in enumerate(repeated_provider_probes, 1):
    results, details = rescue(question)
    check(f"repeated-provider probe {idx} enters guarded MISRA mode", MisraComplianceMode.is_request(question, "") is True)
    check(f"repeated-provider probe {idx} resolves source Rule 21.20", refs(results) == ["Rule 21.20"], refs(results))
    check(
        f"repeated-provider probe {idx} uses subsequent-call source wording",
        any("subsequent call to the same function" in str(cue).casefold() for cue in details.get("semantic_cues", [])),
        details.get("semantic_cues", []),
    )


# ---------------------------------------------------------------------------
# 4) Family precision: family + distinctive construct narrows within the
#    authoritative family; broad family queries remain broad.
# ---------------------------------------------------------------------------
retriever = CompanyRetriever()
goto_probes = [
    "Which MISRA requirements deal with goto-based control flow?",
    "List goto-related rules in MISRA control flow.",
]
for idx, question in enumerate(goto_probes, 1):
    results = retriever._retrieve_structured_rule_catalog(
        query=f"MISRA C | {question} | {question.casefold()}",
        intent_query=question,
    )
    check(f"goto family probe {idx} narrows to source members 15.1-15.4", rule_ids(results) == ["15.1", "15.2", "15.3", "15.4"], rule_ids(results))

label_question = "What MISRA rules are related to labels and goto control flow?"
label_results = retriever._retrieve_structured_rule_catalog(
    query=f"MISRA C | {label_question} | {label_question.casefold()}",
    intent_query=label_question,
)
check("goto+label family intersection stays source-driven", rule_ids(label_results) == ["15.2", "15.3"], rule_ids(label_results))

broad_control = "Which MISRA rules cover control flow?"
broad_results = retriever._retrieve_structured_rule_catalog(
    query=f"MISRA C | {broad_control} | {broad_control.casefold()}",
    intent_query=broad_control,
)
check(
    "broad control-flow family remains complete",
    rule_ids(broad_results) == ["15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7"],
    rule_ids(broad_results),
)


# ---------------------------------------------------------------------------
# 5) Retain representative prior behavior and architecture parameters.
# ---------------------------------------------------------------------------
regressions = [
    ("Can I write if (counter) under MISRA?", "Rule 14.4"),
    ("A non-void call is made only for side effects and the returned result is thrown away. Which requirement applies?", "Rule 17.7"),
    ("A string library function can overrun the destination buffer. What MISRA requirement covers that?", "Rule 21.17"),
    ("The call path eventually gets back to the starting function through a helper. Which MISRA rule is relevant?", "Rule 17.2"),
]
for question, expected_ref in regressions:
    results, _ = rescue(question)
    check(f"prior behavior preserved: {expected_ref}", refs(results) == [expected_ref], refs(results))

# Anti-overfit: exact holdout probes belong only to validator/evidence, never
# production source. We check the files modified by this patch plus answer layer.
production_files = [
    ROOT / "services" / "misra_compliance.py",
    ROOT / "services" / "answer_service.py",
    ROOT / "retrieval" / "retriever.py",
]
production_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in production_files)
manual_probes = [*linkage_probes, *width_relation_probes, *repeated_provider_probes, *goto_probes, label_question, broad_control]
for idx, sentence in enumerate(dict.fromkeys(manual_probes), 1):
    check(f"no exact holdout-probe hardcoding {idx}", sentence not in production_text)

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
    "version": "v6.5.5.14",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
