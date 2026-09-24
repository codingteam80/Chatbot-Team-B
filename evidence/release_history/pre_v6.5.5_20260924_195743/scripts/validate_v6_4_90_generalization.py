from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService
from services.query_service import QueryService


def ck(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def ids(results):
    out = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) or {}
        rid = str(metadata.get("rule_id", "") or "").strip()
        if rid and rid not in out:
            out.append(rid)
    return out


def main():
    checks = []

    excluded = {
        "logs", "evidence", "venv", ".venv", "env", ".git", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
    }
    syntax_errors = []
    py_count = 0
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0].casefold() in excluded:
            continue
        if "__pycache__" in rel.parts:
            continue
        py_count += 1
        try:
            ast.parse(
                p.read_text(encoding="utf-8", errors="replace"),
                filename=str(p),
            )
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}: {exc}")
    checks.append(
        ck("Active DocuBot Python syntax", not syntax_errors, f"{py_count} files")
    )

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
            ck(
                f"Architecture lock: {attr}",
                actual == wanted,
                f"actual={actual!r}",
            )
        )

    svc = AnswerService.__new__(AnswerService)

    checks.append(
        ck(
            "Technical English noun phrase stays on English path",
            svc._is_clearly_english_query(
                "ISO C Portability issue references?"
            )
            and svc._is_clearly_english_query(
                "System configuration reference"
            )
            and not svc._is_clearly_english_query(
                "Kailan nilagdaan ang dokumento?"
            ),
        )
    )

    qs = QueryService()
    svc.query_service = qs

    # Near-exact Section-title lookup must stay BM25/structured only.
    iso_q = "ISO C Portability issue references?"
    iso_context, iso_results = qs.retrieve_context(
        "iso c portability issue references",
        intent_question=iso_q,
    )
    iso_ids = [
        str((item.get("metadata", {}) or {}).get("section_id", "") or "")
        for item in iso_results
        if isinstance(item, dict)
    ]
    iso_answer = svc._deterministic_named_section_answer(iso_results)
    retriever = qs._get_retriever()
    checks.append(
        ck(
            "Near-exact Section title resolves Section 6.10.1",
            iso_ids == ["6.10.1"]
            and bool(iso_results)
            and bool(iso_results[0].get("_structured_section_title_anchor")),
            iso_ids,
        )
    )
    checks.append(
        ck(
            "Section-title fast path avoids vector/reranker",
            retriever.vector_searcher is None and retriever.reranker is None,
            (
                f"vector={'loaded' if retriever.vector_searcher is not None else 'not-loaded'}, "
                f"reranker={'loaded' if retriever.reranker is not None else 'not-loaded'}"
            ),
        )
    )
    checks.append(
        ck(
            "Section-title answer is deterministic and source-grounded",
            "Section 6.10.1" in iso_answer
            and "ISO C portability issue references" in iso_answer
            and "original standard" in iso_answer,
            iso_answer[:800],
        )
    )

    # Compiler/toolchain option wording must never become C switch Rule 16.x.
    compiler_queries = [
        "What are the rules for compiler switch?",
        "Which MISRA rules mention compiler command-line options?",
        "List MISRA rules related to compiler flags.",
    ]
    for q in compiler_queries:
        context, results = qs.retrieve_context(
            q,
            intent_question=q,
            source_family="misra",
        )
        got = ids(results)
        rendered = svc._deterministic_structured_topic_list_answer(results)
        checks.append(
            ck(
                f"Compiler-option intent stays separate from C switch :: {q}",
                got == ["20.9"]
                and "Rule 20.9" in rendered
                and "Rule 16.1" not in rendered
                and "separate from C switch-statement rules" in rendered,
                rendered[:900],
            )
        )

    switch_q = "What are the rules for switch statement?"
    _, switch_results = qs.retrieve_context(
        switch_q,
        intent_question=switch_q,
        source_family="misra",
    )
    switch_ids = ids(switch_results)
    checks.append(
        ck(
            "C switch-statement family remains Rules 16.1-16.7",
            switch_ids == [f"16.{i}" for i in range(1, 8)],
            switch_ids,
        )
    )

    # Exact rationale cross-reference follow: no model synthesis required.
    reason_queries = [
        "Why does Rule 10.6 exist?",
        "What is the rationale behind Rule 10.6?",
    ]
    for q in reason_queries:
        context, results = svc._retrieve_context_for_request(
            search_question=q,
            semantic_target_question=q,
            misra_compliance_mode=False,
        )
        section_ids = [
            str((item.get("metadata", {}) or {}).get("section_id", "") or "")
            for item in results
            if isinstance(item, dict)
        ]
        focus = svc._detect_answer_focus(q, q)
        answer = svc._deterministic_structured_answer(
            context=context,
            question=q,
            resolved_question=q,
            answer_focus=focus,
        )
        checks.append(
            ck(
                f"Rationale cross-reference follows exact cited Section :: {q}",
                "8.10.3" in section_ids
                and "Section 8.10.3" in answer
                and "confusion" in answer.lower()
                and "misconception" in answer.lower()
                and not answer.strip().startswith(
                    "The rationale is described in the introduction"
                ),
                answer[:1200],
            )
        )

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"

    outdir = ROOT / "logs" / "generalization_hotfix"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.90",
        "phase": "generalized_intent_crossref_fastpath_validation",
        "overall": overall,
        "checks": checks,
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "chunking_changed": False,
        "kb_rebuild_performed": False,
        "focused_manual_retest": (
            "READY_AFTER_VALIDATOR_PASS" if overall == "PASS" else "BLOCKED"
        ),
    }
    (outdir / "v6.4.90_generalization_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
