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
    results, _diag = MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(
        question,
        top_k=6,
    )
    return results


service = AnswerService.__new__(AnswerService)

# 1) Exact statement questions should be classified as structured and stay on
# the no-LLM exact-source formatter. This is relation-driven, not Rule-specific.
statement_probes = [
    "What does Rule 22.2 state?",
    "What does Rule 8.7 state?",
    "What does Directive 4.1 state?",
]
for idx, question in enumerate(statement_probes, 1):
    focus = service._detect_answer_focus(question, "")
    check(
        f"exact statement wording {idx} uses structured statement route",
        focus.startswith("STRUCTURED STATEMENT:"),
        focus,
    )

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
question = "What does Rule 22.2 state?"
focus = service._detect_answer_focus(question, "")
exact_answer = service._deterministic_structured_answer(context, question, question, focus)
check(
    "exact Rule statement renders directly without generation",
    exact_answer == (
        "Rule 22.2: A block of memory shall only be freed if it was allocated by means of a "
        "Standard Library function"
    ),
    exact_answer,
)

# 2) Generic empty-else wording must remain scope-aware. The accepted Rule is
# correct, but a generic 'else' question does not prove the if/else-if scope.
empty_generic = "Can I have an empty else block?"
empty_results = rescue(empty_generic)
empty_answer = MisraComplianceMode.deterministic_yes_no_answer(empty_generic, empty_results)
check(
    "generic empty-else answer remains conditional",
    empty_answer.startswith("Needs more context.") and "Rule 15.7" in empty_answer,
    empty_answer,
)

empty_scoped = "Can the final else of an if/else-if chain be empty?"
empty_scoped_results = rescue(empty_scoped)
empty_scoped_answer = MisraComplianceMode.deterministic_yes_no_answer(
    empty_scoped,
    empty_scoped_results,
)
check(
    "explicitly scoped empty-else question gets source-grounded No",
    empty_scoped_answer.startswith("No.") and "Rule 15.7" in empty_scoped_answer,
    empty_scoped_answer,
)

# 3) A positive source requirement must not be treated as automatic permission
# when the user's facts do not establish the required property.
if_generic = "Can I write if (counter) under MISRA?"
if_results = rescue(if_generic)
if_answer = MisraComplianceMode.deterministic_yes_no_answer(if_generic, if_results)
check(
    "unknown if-condition type is not guessed Yes",
    if_answer.startswith("Needs more context.") and "Rule 14.4" in if_answer,
    if_answer,
)

# Explicit source prohibitions still retain the fast deterministic No path.
trigraph_question = "Under MISRA C, can I use trigraphs?"
trigraph_results = rescue(trigraph_question)
trigraph_answer = MisraComplianceMode.deterministic_yes_no_answer(
    trigraph_question,
    trigraph_results,
)
check(
    "explicit source prohibition still gives deterministic No",
    trigraph_answer.startswith("No.") and "Rule 4.2" in trigraph_answer,
    trigraph_answer,
)

# 4) Relation questions are classified separately from permission/lookup and
# use the compact source-grounded relation verifier. The verifier sees the full
# accepted source evidence, including source examples/rationale when present.
relation_question = (
    "Does assigning a narrow arithmetic expression to a wider variable make "
    "the calculation itself wider?"
)
check(
    "semantic relation wording is classified as relation",
    MisraComplianceMode.yes_no_intent(relation_question) == "relation",
    MisraComplianceMode.yes_no_intent(relation_question),
)
relation_results = rescue(relation_question)
check(
    "relation question does not pre-guess deterministic polarity",
    MisraComplianceMode.deterministic_yes_no_answer(relation_question, relation_results) == "",
    MisraComplianceMode.deterministic_yes_no_answer(relation_question, relation_results),
)
check(
    "relation question does not collapse to requirement lookup",
    MisraComplianceMode.deterministic_requirement_lookup(relation_question, relation_results) == "",
    MisraComplianceMode.deterministic_requirement_lookup(relation_question, relation_results),
)

class FakeRelationClient:
    model_name = "validator-fake"
    context_window = 8192
    keep_alive = "0"

    def __init__(self):
        self.prompt = ""

    def generate_structured_json(self, prompt, schema):
        self.prompt = str(prompt)
        return '{"verdict":"no"}'


fake_client = FakeRelationClient()
service._get_llm_client = lambda _model: fake_client
relation_answer = service._semantic_misra_relation_answer(
    relation_question,
    relation_results,
)
check(
    "relation verifier accepts authoritative source-rescue evidence",
    relation_answer.startswith("No.") and "Rule 10.6" in relation_answer,
    relation_answer,
)
check(
    "relation verifier receives full accepted source evidence",
    "SOURCE EVIDENCE:" in fake_client.prompt
    and "Cast causes addition in uint32_t" in fake_client.prompt,
    "full_evidence=" + str("SOURCE EVIDENCE:" in fake_client.prompt),
)
check(
    "relation verifier explicitly forbids missing-fact assumptions",
    "Never assume an undeclared variable type" in fake_client.prompt,
)

# 5) Anti-overfit: manual probe sentences are allowed in QA only; they must not
# appear in production source. The production changes describe behaviors and
# source relations, never question -> Rule mappings.
production_files = [
    ROOT / "services" / "answer_service.py",
    ROOT / "services" / "misra_compliance.py",
]
production_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in production_files)
manual_probes = [empty_generic, if_generic, relation_question]
for idx, sentence in enumerate(manual_probes, 1):
    check(f"no manual-probe hardcoding {idx}", sentence not in production_text)

# Stable production parameters remain untouched by this patch.
settings_text = (ROOT / "config" / "settings.py").read_text(encoding="utf-8-sig")
for needle, label in [
    ("qwen3-embedding:8b", "embedding model unchanged"),
    ("BAAI/bge-reranker-v2-m3", "reranker model unchanged"),
]:
    check(label, needle in settings_text)

failed = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.9",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
