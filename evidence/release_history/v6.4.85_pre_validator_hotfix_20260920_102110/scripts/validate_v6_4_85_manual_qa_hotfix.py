from __future__ import annotations

import ast
import importlib.util
import json
import pickle
import re
import sys
import types
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "manual_qa_hotfix"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_ast_dummy(source: Path, class_name: str, methods: set[str], globals_ns: dict):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    original = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    chosen = [
        node for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods
    ]
    dummy = ast.ClassDef(
        name="ValidationDummy",
        bases=[],
        keywords=[],
        body=chosen,
        decorator_list=[],
    )
    module = ast.Module(body=[dummy], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_ns)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["ValidationDummy"]


def result(name: str, ok: bool, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def main() -> int:
    checks = []
    sr = load_module("docubot_v6485_structured_reference", ROOT / "utils" / "structured_reference.py")
    mc = load_module("docubot_v6485_misra_compliance", ROOT / "services" / "misra_compliance.py")

    refs = sr.extract_structured_references("Explain Rules 8.7, 10.1 and 14.3.")
    ids = [(item.kind, item.identifier) for item in refs]
    checks.append(result(
        "Grouped multi-Rule shorthand parses all explicit references",
        ids == [("rule", "8.7"), ("rule", "10.1"), ("rule", "14.3")],
        ids,
    ))
    checks.append(result(
        "Malformed exact Rule token is detected before semantic substitution",
        sr.has_malformed_structured_reference("Explain Rule ABC.X."),
    ))

    Mode = mc.MisraComplianceMode
    checks.append(result(
        "Category-choice question is document information, not Yes/No compliance",
        Mode.informational_reference_query("Is Rule 14.3 required, mandatory, or advisory?")
        and not Mode.is_request("Is Rule 14.3 required, mandatory, or advisory?", "MISRA")
        and Mode.yes_no_intent("Is Rule 14.3 required, mandatory, or advisory?") == "",
    ))
    checks.append(result(
        "Rule Example request stays on structured-document path",
        Mode.informational_reference_query("Show compliant and non-compliant examples for Rule 14.4.")
        and not Mode.is_request("Show compliant and non-compliant examples for Rule 14.4.", "MISRA"),
    ))
    checks.append(result(
        "Broad rule inventories stay out of compliance-assessment mode",
        all(
            Mode.informational_catalog_query(q) and not Mode.is_request(q, "MISRA")
            for q in (
                "List rules related to goto statements.",
                "What are the rules for unused code?",
                "Which MISRA rules apply to pointers?",
                "List mandatory MISRA rules.",
            )
        ),
    ))

    corpus_path = ROOT / "storage" / "option_c_qwen3_qdrant_v4" / "bm25" / "corpus.pkl"
    with corpus_path.open("rb") as handle:
        records = pickle.load(handle)

    RetrieverDummy = build_ast_dummy(
        ROOT / "retrieval" / "retriever.py",
        "CompanyRetriever",
        {
            "_normalize_text",
            "_is_list_or_relationship_query",
            "_structured_catalog_token",
            "_retrieve_structured_rule_catalog",
        },
        {"re": re, "unicodedata": unicodedata},
    )
    retriever = RetrieverDummy()
    retriever.bm25 = types.SimpleNamespace(records=records)

    expected_catalogs = {
        "List rules related to goto statements.": ["15.1", "15.2", "15.3", "15.4"],
        "What are the rules for unused code?": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"],
        "Which MISRA rules apply to pointers?": ["11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9"],
        "List mandatory MISRA rules.": [
            "9.1", "12.5", "13.6", "17.3", "17.4", "17.6", "19.1",
            "21.13", "21.17", "21.18", "21.19", "21.20", "22.4", "22.5", "22.6",
        ],
    }
    for question, expected in expected_catalogs.items():
        items = retriever._retrieve_structured_rule_catalog(question, question)
        actual = [str((item.get("metadata", {}) or {}).get("rule_id", "")) for item in items]
        checks.append(result(
            f"Structured inventory: {question}",
            actual == expected,
            actual,
        ))

    # Preserve the separately certified switch-family route.
    checks.append(result(
        "Existing switch-family query remains delegated to certified path",
        retriever._retrieve_structured_rule_catalog(
            "List rules for switch statements.", "List rules for switch statements."
        ) == [],
    ))

    # Test answer-focus routing without importing heavy runtime dependencies.
    AnswerDummy = build_ast_dummy(
        ROOT / "services" / "answer_service.py",
        "AnswerService",
        {"_detect_answer_focus", "_structured_detail_label"},
        {
            "re": re,
            "extract_structured_reference": sr.extract_structured_reference,
            "extract_structured_references": sr.extract_structured_references,
        },
    )
    # Stubs are only for branches below the structured-reference early returns.
    AnswerDummy._is_explicit_temporal_comparison_question = lambda self, q: False
    AnswerDummy._is_explicit_temporal_order_question = lambda self, q: False
    AnswerDummy._is_compound_question = lambda self, q: False
    AnswerDummy._requested_compound_facets = lambda self, q: []
    answer = AnswerDummy()

    focus_cases = {
        "Is Rule 14.3 required, mandatory, or advisory?": "STRUCTURED DETAIL:",
        "Show compliant and non-compliant examples for Rule 14.4.": "STRUCTURED DETAIL:",
        "Explain Rules 8.7, 10.1 and 14.3.": "STRUCTURED MULTI EXPLANATION:",
        "Compare Rule 10.1 and Rule 10.3.": "STRUCTURED COMPARISON:",
        "Explain Rule 8.7 then give an example.": "STRUCTURED EXPLANATION WITH EXAMPLE:",
        "Summarize Rule 13.5.": "STRUCTURED SUMMARY:",
    }
    for question, prefix in focus_cases.items():
        focus = answer._detect_answer_focus(question, question)
        checks.append(result(
            f"Answer routing: {question}",
            focus.startswith(prefix),
            focus.split(":", 1)[0],
        ))

    # Prove the Rule 14.4 renderer preserves source code as a fenced multiline block.
    FormatDummy = build_ast_dummy(
        ROOT / "services" / "answer_service.py",
        "AnswerService",
        {"_structured_example_code_line", "_format_structured_example_answer"},
        {"re": re},
    )
    formatter = FormatDummy()
    rule_144_example = ""
    for record in records:
        metadata = record.get("metadata", {}) or {}
        if str(metadata.get("parent_rule_id", "")) == "14.4" and str(metadata.get("section_role", "")).casefold() == "example":
            text = str(record.get("text", "") or "")
            match = re.search(r"(?is)^\s*Rule\s+14\.4\s*\n\s*Example\s*\n(.+)$", text)
            rule_144_example = match.group(1) if match else text
            break
    formatted = formatter._format_structured_example_answer(rule_144_example)
    checks.append(result(
        "Rule 14.4 examples render as readable multiline code",
        "```c" in formatted and "\nwhile ( p )" in formatted and "\nif ( i != 0 )" in formatted,
    ))

    # Empty-else natural wording must retrieve Rule 15.7 source evidence and
    # produce a scoped answer instead of an empty-result fallback.
    rescue, _diag = Mode.rule_body_cue_rescue_from_authoritative_corpus(
        "Can I have an empty else block?", top_k=6
    )
    empty_else_answer = Mode.deterministic_yes_no_answer(
        "Can I have an empty else block?", rescue
    )
    checks.append(result(
        "Empty else question is grounded to Rule 15.7 with scope qualification",
        "Rule 15.7" in empty_else_answer
        and "side effect or a comment" in empty_else_answer
        and "enough context" in empty_else_answer,
        empty_else_answer,
    ))

    # The category value is verified directly from authoritative corpus text.
    rule_143 = next(
        record for record in records
        if str((record.get("metadata", {}) or {}).get("rule_id", "")) == "14.3"
        and str((record.get("metadata", {}) or {}).get("section_type", "")).casefold() == "rule"
    )
    checks.append(result(
        "Authoritative Rule 14.3 Category remains Required",
        bool(re.search(r"(?im)^\s*Category\s*$\s*^\s*Required\s*$", str(rule_143.get("text", "")))),
    ))

    # Full active-source syntax check.
    syntax_errors = []
    py_count = 0
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in {"logs", "evidence"}:
            continue
        py_count += 1
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}: {exc}")
    checks.append(result(
        "Active Python syntax",
        not syntax_errors,
        f"{py_count} files" if not syntax_errors else syntax_errors,
    ))

    # Guard architecture-lock values: this hotfix is routing/rendering only.
    settings_text = (ROOT / "config" / "settings.py").read_text(encoding="utf-8", errors="replace")
    architecture_tokens = [
        'qwen2.5:7b', 'qwen3-embedding:8b', 'BAAI/bge-reranker-v2-m3',
        'CHUNK_SIZE = 900', 'CHUNK_OVERLAP = 150', 'VECTOR_TOP_K = 10',
        'BM25_TOP_K = 10', 'FINAL_TOP_K = 3', 'MIN_RETRIEVAL_SCORE = 0.55',
    ]
    missing_tokens = [token for token in architecture_tokens if token not in settings_text]
    checks.append(result(
        "Architecture-lock settings unchanged",
        not missing_tokens,
        missing_tokens,
    ))

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {
        "version": "v6.4.85",
        "phase": "manual_qa_correctness_and_formatting_hotfix_validation",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "checks": checks,
        "model_or_kb_rebuild_performed": False,
        "manual_question_qa": "RETEST REQUIRED AFTER PATCH",
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "v6.4.85_manual_qa_hotfix_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nOverall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
