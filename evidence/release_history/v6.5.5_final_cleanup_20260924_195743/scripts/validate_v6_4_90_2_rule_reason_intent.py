from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from services.answer_service import AnswerService
from services.query_service import QueryService


def ck(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def exact_reason(svc, question, compliance_mode):
    context, results = svc._retrieve_context_for_request(
        search_question=question,
        semantic_target_question=question,
        misra_compliance_mode=compliance_mode,
    )
    focus = svc._detect_answer_focus(question, question)
    answer = svc._deterministic_structured_answer(
        context=context,
        question=question,
        resolved_question=question,
        answer_focus=focus,
    )
    ids = [
        str((item.get("metadata", {}) or {}).get("section_id", "") or "")
        for item in results
        if isinstance(item, dict)
    ]
    return focus, answer, ids


def main():
    checks = []

    excluded = {
        "logs", "evidence", "venv", ".venv", "env", ".git", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
    }
    errors = []
    count = 0
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0].casefold() in excluded:
            continue
        if "__pycache__" in rel.parts:
            continue
        count += 1
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except SyntaxError as exc:
            errors.append(f"{rel}: {exc}")
    checks.append(ck("Active DocuBot Python syntax", not errors, f"{count} files"))

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
        checks.append(
            ck(f"Architecture lock: {attr}", actual == wanted, f"actual={actual!r}")
        )

    qs = QueryService()
    svc = AnswerService.__new__(AnswerService)
    svc.query_service = qs

    # Different surface forms for one semantic intent family. Rule 10.6 is a
    # useful validator fixture because its own Rationale explicitly points to
    # Section 8.10.3; runtime behavior itself contains no Rule 10.6 hardcoding.
    causal_queries = [
        "Why does Rule 10.6 exist?",
        "What is the rationale behind Rule 10.6?",
        "What is the reason for Rule 10.6?",
        "Explain the purpose of Rule 10.6.",
        "What is the justification for Rule 10.6?",
        "What is the basis for Rule 10.6?",
        "What motivated Rule 10.6?",
    ]

    for q in causal_queries:
        for mode in (False, True):
            focus, answer, section_ids = exact_reason(svc, q, mode)
            checks.append(
                ck(
                    f"Generalized Rule causal intent :: {q} :: compliance_mode={mode}",
                    focus.startswith("REASON:")
                    and "8.10.3" in section_ids
                    and "Section 8.10.3" in answer
                    and "confusion" in answer.lower()
                    and "misconception" in answer.lower()
                    and not answer.strip().startswith("The rationale is described")
                    and not answer.strip().startswith("Rationale: The rationale is described"),
                    f"focus={focus[:80]!r}; answer={answer[:950]}",
                )
            )

    # Regression: a normal explanation must remain an explanation, not causal.
    normal_q = "Explain Rule 13.5."
    normal_focus = svc._detect_answer_focus(normal_q, normal_q)
    checks.append(
        ck(
            "Ordinary Rule explanation regression",
            normal_focus.startswith("STRUCTURED EXPLANATION:"),
            normal_focus,
        )
    )

    # Regression: explicit metadata retains priority over causal vocabulary.
    category_q = "Explain the classification and rationale of Rule 10.1."
    category_focus = svc._detect_answer_focus(category_q, category_q)
    checks.append(
        ck(
            "Explicit classification regression",
            category_focus.startswith("STRUCTURED EXPLANATION WITH CATEGORY:"),
            category_focus,
        )
    )

    # v6.4.90 generalized regressions.
    iso_q = "ISO C Portability issue references?"
    _, iso_results = qs.retrieve_context(
        "iso c portability issue references",
        intent_question=iso_q,
    )
    iso_answer = svc._deterministic_named_section_answer(iso_results)
    checks.append(
        ck(
            "Section-title fast path regression",
            "Section 6.10.1" in iso_answer
            and "ISO C portability issue references" in iso_answer,
            iso_answer[:500],
        )
    )

    compiler_q = "What are the rules for compiler switch?"
    _, compiler_results = qs.retrieve_context(
        compiler_q,
        intent_question=compiler_q,
        source_family="misra",
    )
    compiler_answer = svc._deterministic_structured_topic_list_answer(compiler_results)
    checks.append(
        ck(
            "Compiler switch disambiguation regression",
            "Rule 20.9" in compiler_answer
            and "Rule 16.1" not in compiler_answer,
            compiler_answer[:700],
        )
    )

    switch_q = "What are the rules for switch statement?"
    _, switch_results = qs.retrieve_context(
        switch_q,
        intent_question=switch_q,
        source_family="misra",
    )
    switch_ids = [
        str((item.get("metadata", {}) or {}).get("rule_id", "") or "")
        for item in switch_results
        if isinstance(item, dict)
    ]
    checks.append(
        ck(
            "C switch family regression",
            switch_ids == [f"16.{i}" for i in range(1, 8)],
            switch_ids,
        )
    )

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "rule_reason_intent_hotfix"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.90.2",
        "phase": "generalized_rule_reason_intent_validation",
        "overall": overall,
        "checks": checks,
        "runtime_file_changed": "services/answer_service.py",
        "retriever_changed": False,
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "chunking_changed": False,
        "kb_rebuild_performed": False,
        "focused_manual_retest": (
            "READY_FOR_GENERALIZATION_RETEST" if overall == "PASS" else "BLOCKED"
        ),
    }
    (outdir / "v6.4.90.2_rule_reason_intent_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
