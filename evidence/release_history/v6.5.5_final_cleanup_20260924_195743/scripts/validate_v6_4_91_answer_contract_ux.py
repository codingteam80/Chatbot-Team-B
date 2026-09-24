from __future__ import annotations

import ast
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from retrieval.retriever import HybridRetriever
from services.answer_service import AnswerService
from services.misra_compliance import MisraComplianceMode
from services.query_service import QueryService


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def load_records():
    corpus = Path(settings.BM25_DIR) / "corpus.pkl"
    with corpus.open("rb") as fh:
        data = pickle.load(fh)
    return data if isinstance(data, list) else []


def main():
    checks = []

    # Active source syntax only.
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
    checks.append(check("Active Python syntax", not errors, f"{count} files"))

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
        checks.append(check(f"Architecture lock: {attr}", actual == wanted, repr(actual)))

    records = load_records()
    checks.append(check("Authoritative BM25 corpus available", bool(records), f"{len(records)} records"))

    # --------------------------------------------------------------
    # YES/NO contract
    # --------------------------------------------------------------
    req_157 = None
    amp_157 = None
    req_42 = None
    for r in records:
        if not isinstance(r, dict):
            continue
        m = r.get("metadata", {}) or {}
        if m.get("rule_id") == "15.7" and m.get("section_type") == "rule":
            req_157 = r
        if m.get("parent_rule_id") == "15.7" and str(m.get("section_role", "")).casefold() == "amplification":
            amp_157 = r
        if m.get("rule_id") == "4.2" and m.get("section_type") == "rule":
            req_42 = r

    empty_else_answer = ""
    if req_157 and amp_157:
        item = {
            "text": str(req_157["text"]) + "\n\n" + str(amp_157["text"]),
            "metadata": dict(req_157.get("metadata", {}) or {}),
            "_misra_rule_body_rescue": True,
            "_misra_cue_coverage": 1.0,
            "_misra_assessment_state": "violation",
        }
        empty_else_answer = MisraComplianceMode.deterministic_yes_no_answer(
            "Can I have an empty else block?",
            [item],
        )

    checks.append(check(
        "YES/NO contract: empty else starts with No.",
        empty_else_answer.startswith("No."),
        empty_else_answer[:500],
    ))
    checks.append(check(
        "YES/NO contract: empty else remains scoped",
        "terminating `else`" in empty_else_answer and "Rule 15.7" in empty_else_answer,
        empty_else_answer[:500],
    ))

    trigraph_answer = ""
    if req_42:
        item = {
            "text": str(req_42["text"]),
            "metadata": dict(req_42.get("metadata", {}) or {}),
            "_misra_rule_body_rescue": True,
            "_misra_cue_coverage": 1.0,
            "_misra_assessment_state": "violation",
        }
        trigraph_answer = MisraComplianceMode.deterministic_yes_no_answer(
            "Is it ok to use trigraphs?",
            [item],
        )
    checks.append(check(
        "YES/NO regression: trigraphs starts with No.",
        trigraph_answer.startswith("No."),
        trigraph_answer[:400],
    ))
    checks.append(check(
        "Classification choice remains non-binary",
        MisraComplianceMode.yes_no_intent(
            "Is Rule 14.3 required, mandatory, or advisory?"
        ) == "",
    ))

    # --------------------------------------------------------------
    # Bare major Rule family clarification
    # --------------------------------------------------------------
    fake_retriever = HybridRetriever.__new__(HybridRetriever)
    fake_retriever.bm25 = SimpleNamespace(records=records)
    family = fake_retriever._retrieve_structured_rule_major_family("17")
    family_answer = AnswerService._deterministic_structured_major_family_answer(family)
    family_members = (
        family[0].get("_structured_major_family_members", [])
        if family else []
    )
    checks.append(check(
        "Bare Rule 17 family resolved from source inventory",
        family_members == [f"17.{i}" for i in range(1, 9)],
        family_members,
    ))
    checks.append(check(
        "Bare Rule 17 gives clarification instead of generic fallback",
        "no single **Rule 17** entry" in family_answer
        and "Rules 17.1–17.8" in family_answer
        and "Please specify" in family_answer,
        family_answer,
    ))

    # End-to-end structured retrieval should return the compact family anchor.
    qs = QueryService()
    family_context, family_results = qs.retrieve_context(
        "Tell me about Rule 17 and what is its rationale.",
        intent_question="Tell me about Rule 17 and what is its rationale.",
        source_family="misra",
    )
    checks.append(check(
        "Bare Rule 17 retrieval returns family clarification anchor",
        bool(family_results)
        and bool(family_results[0].get("_structured_major_family_anchor")),
        str([
            (x.get("metadata", {}) or {}).get("exact_reference")
            for x in family_results
        ]),
    ))

    # --------------------------------------------------------------
    # Structured one-item list presentation
    # --------------------------------------------------------------
    svc = AnswerService.__new__(AnswerService)
    compiler_q = "What are the rules for compiler switch?"
    _compiler_context, compiler_results = qs.retrieve_context(
        compiler_q,
        intent_question=compiler_q,
        source_family="misra",
    )
    compiler_answer = svc._deterministic_structured_topic_list_answer(compiler_results)
    compiler_final = svc._format_answer_presentation(
        compiler_answer,
        compiler_q,
        "LIST: Return every relevant explicitly stated Rule.",
    )
    checks.append(check(
        "Compiler-switch semantic disambiguation",
        "Rule 20.9" in compiler_final and "Rule 16.1" not in compiler_final,
        compiler_final[:900],
    ))
    checks.append(check(
        "Compiler-switch one-item Markdown stays clean",
        compiler_final.startswith("### Compiler option-related MISRA rules")
        and "\n\nScope:" in compiler_final
        and "\n\n- **Rule 20.9**" in compiler_final
        and not compiler_final.startswith("- ###"),
        compiler_final[:900],
    ))

    switch_q = "What are the rules for switch statement?"
    _switch_context, switch_results = qs.retrieve_context(
        switch_q,
        intent_question=switch_q,
        source_family="misra",
    )
    switch_ids = [
        str((item.get("metadata", {}) or {}).get("rule_id", "") or "")
        for item in switch_results
        if isinstance(item, dict)
    ]
    checks.append(check(
        "C switch family regression",
        switch_ids == [f"16.{i}" for i in range(1, 8)],
        switch_ids,
    ))

    # --------------------------------------------------------------
    # Rule rationale cross-reference source visibility
    # --------------------------------------------------------------
    svc.query_service = qs
    reason_q = "Why does Rule 10.6 exist?"
    reason_context, reason_results = svc._retrieve_context_for_request(
        search_question=reason_q,
        semantic_target_question=reason_q,
        misra_compliance_mode=False,
    )
    reason_sections = [
        str((item.get("metadata", {}) or {}).get("section_id", "") or "")
        for item in reason_results
        if isinstance(item, dict)
    ]
    reason_sources = svc._extract_sources(reason_results, max_sources=None)
    reason_refs = [str(src.get("reference", "") or "") for src in reason_sources]
    checks.append(check(
        "Rule 10.6 exact Section cross-reference still followed",
        "8.10.3" in reason_sections
        and "confusion" in reason_context.casefold()
        and "misconception" in reason_context.casefold(),
        reason_sections,
    ))
    checks.append(check(
        "Rule 10.6 supporting source set includes Rule and Section",
        "Rule 10.6" in reason_refs and "Section 8.10.3" in reason_refs,
        reason_refs,
    ))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "answer_contract_ux_hardening"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.91",
        "overall": overall,
        "checks": checks,
        "runtime_files_changed": [
            "services/answer_service.py",
            "services/misra_compliance.py",
            "retrieval/retriever.py",
        ],
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "chunking_changed": False,
        "kb_rebuild_performed": False,
        "manual_retest": (
            "READY_FOR_FOCUSED_8_QUESTION_RETEST"
            if overall == "PASS"
            else "BLOCKED"
        ),
    }
    (
        outdir / "v6.4.91_answer_contract_ux_validation_latest.json"
    ).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
