from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from services.answer_service import AnswerService
from services.query_service import QueryService
from utils.structured_reference import extract_structured_references


def ck(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def rule_ids(results):
    out = []
    for item in results or []:
        md = item.get("metadata", {}) if isinstance(item, dict) else {}
        rid = str(md.get("rule_id", "") or "").strip()
        if rid and rid not in out:
            out.append(rid)
    return out


def main() -> int:
    checks = []

    # 1) Source syntax, excluding environment/history trees.
    excluded = {"logs","evidence","venv",".venv","env",".git","__pycache__",".pytest_cache","node_modules"}
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
            ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}: {exc}")
    checks.append(ck("Active DocuBot Python syntax", not syntax_errors, f"{py_count} files"))

    # 2) Architecture lock.
    expected = {
        "KNOWLEDGE_PROFILE": "technical",
        "VECTOR_BACKEND": "qdrant",
        "CHUNKING_PROFILE": "v4",
        "EMBEDDING_BACKEND": "ollama",
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

    # 3) Grouped references must survive punctuation-stripped normalized queries.
    refs = extract_structured_references("rules 8.7 10.1 and 14.3")
    got = [(r.kind, r.identifier) for r in refs]
    checks.append(ck(
        "Grouped Rule parsing after normalization",
        got == [("rule","8.7"),("rule","10.1"),("rule","14.3")],
        got,
    ))

    # Use real read-only structured retrieval paths. These do not invoke Ollama.
    qs = QueryService()
    svc = AnswerService.__new__(AnswerService)

    # 4) Multi-rule query retrieves the three exact requested anchors.
    multi_context, multi_results = qs.retrieve_context(
        "rules 8.7 10.1 and 14.3",
        intent_question="Explain Rules 8.7, 10.1 and 14.3.",
        source_family="misra",
    )
    multi_ids = rule_ids(multi_results)
    checks.append(ck(
        "Multi-rule exact retrieval 8.7/10.1/14.3",
        multi_ids == ["8.7","10.1","14.3"],
        multi_ids,
    ))
    multi_answer = svc._deterministic_structured_multi_reference_answer(multi_results)
    checks.append(ck(
        "Multi-rule deterministic answer",
        all(f"Rule {rid}" in multi_answer for rid in ("8.7","10.1","14.3")),
        multi_answer[:240],
    ))

    # 5) Pointer catalog query must remain a LIST and render retrieved Rule 11 family.
    pointer_q = "Which MISRA rules apply to pointers?"
    pointer_focus = svc._detect_answer_focus(pointer_q, pointer_q)
    checks.append(ck("Pointer catalog answer focus = LIST", pointer_focus.startswith("LIST:"), pointer_focus[:80]))
    pointer_context, pointer_results = qs.retrieve_context(
        "which misra rules apply to pointers policy standard rules requirements compliance scope exception responsibility",
        intent_question=pointer_q,
        source_family="misra",
    )
    pointer_ids = rule_ids(pointer_results)
    pointer_answer = svc._deterministic_structured_topic_list_answer(pointer_results)
    checks.append(ck(
        "Pointer structured family retrieved",
        pointer_ids == [f"11.{i}" for i in range(1,10)],
        pointer_ids,
    ))
    checks.append(ck(
        "Pointer list deterministic finalization",
        bool(pointer_answer) and "Rule 11.1" in pointer_answer and "Rule 11.9" in pointer_answer,
        pointer_answer[:220],
    ))

    # 6) Simple terms should be concise and not dump full source sections.
    q = "Explain MISRA Rule 2.2 in simple terms."
    ctx, res = qs.retrieve_context("misra rule 2.2 in simple terms", intent_question=q, source_family="misra")
    focus = svc._detect_answer_focus(q, q)
    ans = svc._deterministic_structured_answer(ctx, q, q, focus)
    checks.append(ck(
        "Rule 2.2 simple-terms concise rendering",
        focus.startswith("STRUCTURED SIMPLE EXPLANATION:")
        and "**In simple terms:**" in ans
        and "Analysis" not in ans
        and "Example:" not in ans
        and len(ans) < 900,
        ans[:300],
    ))

    # 7) Rule 13.5 summary must preserve the full requirement and rationale,
    # without interpreting code punctuation/comments as labels.
    q = "Summarize Rule 13.5."
    ctx, res = qs.retrieve_context("rule 13.5", intent_question=q, source_family="misra")
    focus = svc._detect_answer_focus(q, q)
    ans = svc._deterministic_structured_answer(ctx, q, q, focus)
    checks.append(ck(
        "Rule 13.5 clean summary",
        "contain persistent side effects" in ans
        and "**Rationale:**" in ans
        and "**{:" not in ans
        and "This side effect is persistent" not in ans,
        ans[:320],
    ))

    # 8) Rule 8.7 + example: exact source has no Example block; say so directly,
    # rather than invoking generation and falling back.
    q = "Explain Rule 8.7 then give an example."
    ctx, res = qs.retrieve_context("rule 8.7 then give an example", intent_question=q, source_family="misra")
    focus = svc._detect_answer_focus(q, q)
    ans = svc._deterministic_structured_answer(ctx, q, q, focus)
    checks.append(ck(
        "Rule 8.7 no-source-example handling",
        "No explicit Example section is provided" in ans
        and "Information not found in company knowledge base" not in ans,
        ans[-260:],
    ))

    # 9) Explanation + classification should be concise instead of dumping table/examples.
    q = "Discuss Rule 10.1 and explain prior classification."
    ctx, res = qs.retrieve_context("discuss rule 10.1 and explain prior classification", intent_question=q, source_family="misra")
    focus = svc._detect_answer_focus(q, q)
    ans = svc._deterministic_structured_answer(ctx, q, q, focus)
    checks.append(ck(
        "Rule 10.1 discussion + classification concise",
        focus.startswith("STRUCTURED EXPLANATION WITH CATEGORY:")
        and "Rule 10.1" in ans
        and "Required" in ans
        and "Amplification:" not in ans
        and "Example:" not in ans
        and len(ans) < 1800,
        ans[:320],
    ))

    # 10) Strict invalid identifiers remain blocked.
    bad_refs = extract_structured_references("Explain Rule ABC.X.")
    checks.append(ck("Malformed Rule ABC.X not parsed as numeric Rule", not bad_refs, bad_refs))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "manual_qa_round2_hotfix"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.86",
        "phase": "manual_qa_round2_hotfix_validation",
        "overall": overall,
        "checks": checks,
        "kb_rebuild_performed": False,
        "model_stack_changed": False,
        "retrieval_parameter_changes": False,
        "manual_question_retest": "PENDING" if overall == "PASS" else "BLOCKED",
    }
    path = outdir / "v6.4.86_manual_qa_round2_validation_latest.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
