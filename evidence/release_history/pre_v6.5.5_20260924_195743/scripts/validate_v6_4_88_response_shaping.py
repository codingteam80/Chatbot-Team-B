from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from services.answer_service import AnswerService
from services.query_service import QueryService


def ck(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def exact(qs, svc, question, search=None):
    context, results = qs.retrieve_context(
        search or question,
        intent_question=question,
        source_family="misra",
    )
    focus = svc._detect_answer_focus(question, question)
    answer = svc._deterministic_structured_answer(
        context=context,
        question=question,
        resolved_question=question,
        answer_focus=focus,
    )
    return focus, answer, results


def main():
    checks = []

    excluded = {
        "logs","evidence","venv",".venv","env",".git","__pycache__",
        ".pytest_cache",".mypy_cache",".ruff_cache","node_modules"
    }
    errors = []
    py_count = 0
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0].casefold() in excluded:
            continue
        if "__pycache__" in rel.parts:
            continue
        py_count += 1
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except SyntaxError as exc:
            errors.append(f"{rel}: {exc}")
    checks.append(ck("Active DocuBot Python syntax", not errors, f"{py_count} files"))

    expected = {
        "VECTOR_BACKEND": "qdrant",
        "CHUNKING_PROFILE": "v4",
        "EMBED_MODEL_NAME": "qwen3-embedding:8b",
        "OLLAMA_FAST_MODEL": "qwen2.5:7b",
        "OLLAMA_COMPLEX_MODEL": "qwen2.5:7b",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "CHUNK_SIZE": 900,
        "CHUNK_OVERLAP": 150,
        "VECTOR_TOP_K": 10,
        "BM25_TOP_K": 10,
        "FINAL_TOP_K": 3,
        "MIN_RETRIEVAL_SCORE": 0.55,
    }
    for attr, wanted in expected.items():
        actual = getattr(settings, attr)
        checks.append(ck(f"Architecture lock: {attr}", actual == wanted, f"actual={actual!r}"))

    qs = QueryService()
    svc = AnswerService.__new__(AnswerService)

    # 1) Simple terms: no standards/reference dump.
    focus, ans, _ = exact(
        qs, svc,
        "Explain MISRA Rule 2.2 in simple terms.",
        "misra rule 2.2 in simple terms",
    )
    checks.append(ck(
        "Simple terms contains only requirement + plain explanation",
        "There shall be no dead code" in ans
        and "In simple terms" in ans
        and "IEC 61508" not in ans
        and "ISO 26262" not in ans
        and "DO-178C" not in ans
        and "Analysis" not in ans
        and "Applies to" not in ans
        and "Example:" not in ans,
        ans[:500],
    ))

    # 2) Multi-rule explanation: no categories unless asked; meaningful rationale.
    q = "Explain Rules 8.7, 10.1 and 14.3."
    context, results = qs.retrieve_context(
        "rules 8.7 10.1 and 14.3",
        intent_question=q,
        source_family="misra",
    )
    ans = svc._deterministic_structured_multi_reference_answer(results, question=q)
    checks.append(ck(
        "Multi-rule explanation is concise and metadata-free",
        all(f"Rule {rid}" in ans for rid in ("8.7","10.1","14.3"))
        and "— Advisory" not in ans
        and "— Required" not in ans
        and "C90 [" not in ans
        and "C99 [" not in ans
        and "Rationale:** 1." not in ans
        and "floating type" in ans,
        ans[:900],
    ))

    # 3) Explanation + example: do not dump Category/Analysis/Applies-to.
    focus, ans, _ = exact(
        qs, svc,
        "Explain Rule 8.7 then give an example.",
        "rule 8.7 then give an example",
    )
    checks.append(ck(
        "Rule 8.7 explanation hides unrequested metadata",
        "Rule 8.7" in ans
        and "Rationale" in ans
        and "No explicit Example section is provided" in ans
        and "\nCategory\n" not in ans
        and "\nAnalysis\n" not in ans
        and "\nApplies to\n" not in ans
        and "[Koenig" not in ans,
        ans[:800],
    ))

    # 4) Explicit classification: Category is shown because user asked for it,
    # but unrelated Analysis/Applies-to metadata stays hidden.
    focus, ans, _ = exact(
        qs, svc,
        "Discuss Rule 10.1 and explain prior classification.",
        "discuss rule 10.1 and explain prior classification",
    )
    checks.append(ck(
        "Explicit classification shows Category only with concise rationale",
        "Rule 10.1 — Required" in ans
        and "Rationale" in ans
        and "floating type" in ans
        and "Analysis" not in ans
        and "Applies to" not in ans
        and "C90 [" not in ans
        and "C99 [" not in ans,
        ans[:700],
    ))

    # 5) Summary: no Category because it was not asked for.
    focus, ans, _ = exact(
        qs, svc,
        "Summarize Rule 13.5.",
        "rule 13.5",
    )
    checks.append(ck(
        "Summary returns requirement + rationale only",
        "**Rule 13.5**" in ans
        and "— Required" not in ans
        and "persistent side effects" in ans
        and "Rationale" in ans
        and "Example" not in ans
        and "Analysis" not in ans
        and "Applies to" not in ans,
        ans[:600],
    ))

    # 6) Topic list: rule identifiers + clean short statements only.
    pointer_q = "Which MISRA rules apply to pointers?"
    _, pointer_results = qs.retrieve_context(
        "which misra rules apply to pointers policy standard rules requirements compliance scope exception responsibility",
        intent_question=pointer_q,
        source_family="misra",
    )
    ans = svc._deterministic_structured_topic_list_answer(pointer_results)
    checks.append(ck(
        "Pointer inventory hides technical mapping metadata",
        "Rule 11.1" in ans
        and "Rule 11.9" in ans
        and "C90 [" not in ans
        and "C99 [" not in ans
        and "Undefined" not in ans
        and "Implementation" not in ans,
        ans[:900],
    ))

    # 7) Metadata is hidden by default, not deleted. Explicit detail requests
    # must still return it from the same source block.
    for question, search, expected_piece, label in [
        ("What is the category of Rule 8.7?", "rule 8.7 category", "Advisory", "Category"),
        ("What is the analysis of Rule 8.7?", "rule 8.7 analysis", "Decidable, System", "Analysis"),
        ("What does Rule 8.7 apply to?", "rule 8.7 applies to", "C90, C99", "Applies to"),
    ]:
        focus, detail_ans, _ = exact(qs, svc, question, search)
        checks.append(ck(
            f"Explicit {label} remains queryable",
            expected_piece in detail_ans,
            detail_ans[:300],
        ))

    # 8) Explicit example still returns actual source example formatting.
    focus, ans, _ = exact(
        qs, svc,
        "Show compliant and non-compliant examples for Rule 14.4.",
        "rule 14.4 examples",
    )
    checks.append(ck(
        "Explicit Rule 14.4 example remains available",
        "Example" in ans
        and "Non-compliant" in ans
        and "Compliant" in ans
        and "while" in ans,
        ans[:700],
    ))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "response_shaping_hotfix"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.88",
        "phase": "final_manual_qa_response_shaping_validation",
        "overall": overall,
        "checks": checks,
        "model_stack_changed": False,
        "retrieval_parameters_changed": False,
        "kb_rebuild_performed": False,
        "manual_retest": "FOCUSED RESPONSE-SHAPING RETEST REQUIRED" if overall == "PASS" else "BLOCKED",
    }
    (outdir / "v6.4.88_response_shaping_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
