from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from services.answer_service import AnswerService
from services.query_service import QueryService

EXPECTED_POINTER_IDS = [
    "7.4", "8.13",
    "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9",
    "18.1", "18.2", "18.3", "18.4", "18.5",
    "21.15", "21.16", "21.17", "21.19", "21.20",
    "22.5", "22.6",
]


def ck(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def rule_ids(results):
    output = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        rid = str((item.get("metadata", {}) or {}).get("rule_id", "") or "").strip()
        if rid and rid not in output:
            output.append(rid)
    return output


def main():
    checks = []

    excluded = {
        "logs", "evidence", "venv", ".venv", "env", ".git", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
    }
    errors = []
    count = 0
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0].casefold() in excluded:
            continue
        if "__pycache__" in rel.parts:
            continue
        count += 1
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
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
        checks.append(ck(f"Architecture lock: {attr}", actual == wanted, f"actual={actual!r}"))

    qs = QueryService()
    svc = AnswerService.__new__(AnswerService)

    q = "Discuss Rule 10.1 and explain prior classification."
    context, results = qs.retrieve_context(
        "discuss rule 10.1 and explain prior classification",
        intent_question=q,
        source_family="misra",
    )
    focus = svc._detect_answer_focus(q, q)
    answer = svc._deterministic_structured_answer(
        context=context,
        question=q,
        resolved_question=q,
        answer_focus=focus,
    )
    checks.append(ck(
        "Rule 10.1 multi-item rationale is qualified",
        "Rule 10.1 — Required" in answer
        and "multiple numbered rationale points" in answer
        and "the first states:" in answer
        and "floating type" in answer
        and "Rationale:** 1." not in answer,
        answer[:850],
    ))

    q2 = "Explain Rules 8.7, 10.1 and 14.3."
    context2, results2 = qs.retrieve_context(
        "rules 8.7 10.1 and 14.3",
        intent_question=q2,
        source_family="misra",
    )
    answer2 = svc._deterministic_structured_multi_reference_answer(results2, question=q2)
    checks.append(ck(
        "Grouped Rule 10.1 rationale is not overstated",
        all(f"Rule {rid}" in answer2 for rid in ("8.7", "10.1", "14.3"))
        and "multiple numbered rationale points" in answer2
        and "Rationale:** 1." not in answer2,
        answer2[:1000],
    ))

    pointer_q = "Which MISRA rules apply to pointers?"
    pointer_context, pointer_results = qs.retrieve_context(
        "which misra rules apply to pointers",
        intent_question=pointer_q,
        source_family="misra",
    )
    got_ids = rule_ids(pointer_results)
    pointer_answer = svc._deterministic_structured_topic_list_answer(pointer_results)
    checks.append(ck("Broad pointer inventory exact Rule IDs", got_ids == EXPECTED_POINTER_IDS, got_ids))
    checks.append(ck(
        "Broad pointer answer is complete but compact",
        "Pointer-related MISRA rules" in pointer_answer
        and "Scope:" in pointer_answer
        and "Rules 11.1–11.9" in pointer_answer
        and "Rules 18.1–18.5" in pointer_answer
        and "Rules 21.15–21.17, 21.19–21.20" in pointer_answer
        and "Rules 22.5–22.6" in pointer_answer
        and "Rule 7.4" in pointer_answer
        and "Rule 8.13" in pointer_answer
        and "C90 [" not in pointer_answer
        and "Undefined" not in pointer_answer,
        pointer_answer[:1200],
    ))

    narrow_q = "List MISRA rules for pointer type conversions."
    narrow_context, narrow_results = qs.retrieve_context(
        "misra rules for pointer type conversions",
        intent_question=narrow_q,
        source_family="misra",
    )
    narrow_ids = rule_ids(narrow_results)
    checks.append(ck(
        "Narrow pointer-conversion query remains Rule 11.1–11.9",
        narrow_ids == [f"11.{i}" for i in range(1, 10)],
        narrow_ids,
    ))

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    outdir = ROOT / "logs" / "semantic_completeness_hotfix"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.89",
        "phase": "semantic_completeness_final_qa_candidate",
        "overall": overall,
        "checks": checks,
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "kb_rebuild_performed": False,
        "full_manual_qa": "READY_AFTER_VALIDATOR_PASS" if overall == "PASS" else "BLOCKED",
    }
    (outdir / "v6.4.89_semantic_completeness_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
