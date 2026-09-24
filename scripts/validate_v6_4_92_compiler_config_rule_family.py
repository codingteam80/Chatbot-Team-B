from __future__ import annotations

import ast
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService
from services.misra_compliance import MisraComplianceMode


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def load_records():
    corpus = Path(settings.BM25_DIR) / "corpus.pkl"
    with corpus.open("rb") as fh:
        data = pickle.load(fh)
    return data if isinstance(data, list) else []


def rule_ids(results):
    return [
        str((item.get("metadata", {}) or {}).get("rule_id", "") or "")
        for item in (results or [])
        if isinstance(item, dict)
        and str((item.get("metadata", {}) or {}).get("rule_id", "") or "")
    ]


def main():
    checks = []

    excluded = {
        "logs", "evidence", "venv", ".venv", "env", ".git", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
    }
    syntax_errors = []
    count = 0
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0].casefold() in excluded:
            continue
        if "__pycache__" in rel.parts:
            continue
        count += 1
        try:
            ast.parse(
                path.read_text(encoding="utf-8", errors="replace"),
                filename=str(path),
            )
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}: {exc}")
    checks.append(check("Active Python syntax", not syntax_errors, f"{count} files"))

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
        checks.append(check(
            f"Architecture lock: {attr}",
            actual == wanted,
            repr(actual),
        ))

    records = load_records()
    checks.append(check(
        "Authoritative BM25 corpus available",
        bool(records),
        f"{len(records)} records",
    ))

    retriever = CompanyRetriever.__new__(CompanyRetriever)
    retriever.bm25 = SimpleNamespace(records=records)
    answer_service = AnswerService.__new__(AnswerService)

    # ------------------------------------------------------------------
    # Generalized compiler/toolchain configuration routing
    # ------------------------------------------------------------------
    compiler_variants = [
        "What are the rules for compiler switch?",
        "What compiler options should be reviewed?",
        "List the compiler flags/settings guidance.",
        "What toolchain switches should be configured?",
    ]
    for question in compiler_variants:
        results = retriever.retrieve(
            question,
            intent_query=question,
            source_family="misra",
        )
        section_ids = [
            str((item.get("metadata", {}) or {}).get("section_id", "") or "")
            for item in results
            if isinstance(item, dict)
        ]
        compiler_anchor = bool(results) and bool(
            results[0].get("_structured_toolchain_configuration_anchor")
        )
        answer = answer_service._deterministic_named_section_answer(results)
        checks.append(check(
            f"Compiler configuration route: {question}",
            compiler_anchor
            and "5.3.1" in section_ids
            and "Rule 20.9" not in answer
            and "Compiler configuration" in answer
            and "optimization" in answer.casefold()
            and "messages" in answer.casefold(),
            answer[:1200],
        ))

    # Must remain distinct from C switch statements.
    switch_q = "What are the rules for switch statement?"
    switch_results = retriever.retrieve(
        switch_q,
        intent_query=switch_q,
        source_family="misra",
    )
    checks.append(check(
        "C switch statement family remains Rules 16.1-16.7",
        rule_ids(switch_results) == [f"16.{i}" for i in range(1, 8)],
        rule_ids(switch_results),
    ))

    # Explicit preprocessor wording must not be hijacked by compiler-config.
    pre_q = "What are the rules for preprocessing directives?"
    pre_results = retriever.retrieve(
        pre_q,
        intent_query=pre_q,
        source_family="misra",
    )
    checks.append(check(
        "Preprocessor family remains separate from compiler configuration",
        bool(pre_results)
        and not any(
            item.get("_structured_toolchain_configuration_anchor")
            for item in pre_results
            if isinstance(item, dict)
        )
        and "20.9" in rule_ids(pre_results),
        rule_ids(pre_results)[:20],
    ))

    # ------------------------------------------------------------------
    # Bare major Rule family UX
    # ------------------------------------------------------------------
    family = retriever._retrieve_structured_rule_major_family("17")
    family_answer = AnswerService._deterministic_structured_major_family_answer(
        family
    )
    family_members = (
        family[0].get("_structured_major_family_members", [])
        if family else []
    )
    checks.append(check(
        "Bare Rule 17 family still resolves exactly",
        family_members == [f"17.{i}" for i in range(1, 9)],
        family_members,
    ))
    checks.append(check(
        "Bare Rule 17 now gives useful family overview",
        "Rules 17.1–17.8" in family_answer
        and "**Rule 17.1 — Required**" in family_answer
        and "**Rule 17.3 — Mandatory**" in family_answer
        and "**Rule 17.8 — Advisory**" in family_answer
        and "ambiguous" in family_answer.casefold()
        and "individual rationale" in family_answer.casefold(),
        family_answer[:2200],
    ))

    # ------------------------------------------------------------------
    # Previously fixed YES/NO contract regression
    # ------------------------------------------------------------------
    req_42 = None
    for record in records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) or {}
        if (
            metadata.get("rule_id") == "4.2"
            and metadata.get("section_type") == "rule"
        ):
            req_42 = record
            break

    trigraph_answer = ""
    if req_42:
        item = {
            "text": str(req_42.get("text", "") or ""),
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
        "YES/NO regression: trigraphs still starts with No.",
        trigraph_answer.startswith("No."),
        trigraph_answer,
    ))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "compiler_config_rule_family"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.92",
        "overall": overall,
        "checks": checks,
        "runtime_files_changed": [
            "retrieval/retriever.py",
            "services/answer_service.py",
        ],
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "chunking_changed": False,
        "kb_rebuild_performed": False,
        "manual_retest": (
            "READY_FOR_FOCUSED_5_QUESTION_RETEST"
            if overall == "PASS"
            else "BLOCKED"
        ),
    }
    (
        outdir / "v6.4.92_compiler_config_rule_family_validation_latest.json"
    ).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
