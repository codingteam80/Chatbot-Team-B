from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from config import settings
from config.prompts import MULTI_QUERY_RETRIEVAL_PROMPT
from ingestion.parser import DocumentParser
from llm.ollama_client import _is_transient_connection_error
from retrieval.multi_query import parse_multi_query_variants
from scripts.smart_build import get_update_plan
from scripts.update_server_kb import _json_safe
from services.answer_service import AnswerService
from ui.streamlit_ui import StreamlitUI
from utils.manifest import ManifestManager


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


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
            ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
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
        checks.append(check(f"Architecture lock: {attr}", actual == wanted, repr(actual)))

    checks.append(check(
        "Single canonical source root",
        Path(settings.DOCUMENT_DIR).resolve() == Path(settings.TECHNICAL_DOCUMENT_DIR).resolve()
        and not hasattr(settings, "ALL_DOCUMENT_DIR"),
        str(settings.DOCUMENT_DIR),
    ))

    run_text = (ROOT / "run.py").read_text(encoding="utf-8", errors="replace")
    checks.append(check(
        "Obsolete all_documents PDF server is not active",
        "pdf_server" not in run_text and "all_documents" not in run_text,
    ))

    parser_text = (ROOT / "ingestion" / "parser.py").read_text(encoding="utf-8", errors="replace")
    checks.append(check(
        "Unstructured parser import is lazy",
        "from unstructured.partition.auto import partition" in parser_text
        and not parser_text.lstrip().startswith("from unstructured"),
    ))
    checks.append(check(
        "Native PDF/Office loader fallback is available",
        all(token in parser_text for token in ("PDFLoader", "DOCXLoader", "XLSXLoader", "PPTXLoader")),
    ))

    setup_text = (ROOT / "scripts" / "setup_production_environment.py").read_text(
        encoding="utf-8", errors="replace"
    )
    checks.append(check(
        "Environment setup verifies ingestion dependencies",
        all(token in setup_text for token in ('"unstructured": "unstructured"', '"pymupdf": "fitz"', '"python_docx": "docx"')),
    ))

    # Portable manifest identity: the same source document must have one key
    # regardless of which Windows installation path originally built the index.
    current_pdf = Path(settings.DOCUMENT_DIR) / "MISRA_FromInternet.pdf"
    old_path = r"C:\\user_dev\\company-chatbot\\data\\technical_documents\\MISRA_FromInternet.pdf"
    other_path = r"C:\\Users\\Other User\\Documents\\DocuBot\\data\\technical_documents\\MISRA_FromInternet.pdf"
    old_key = ManifestManager.document_key(old_path)
    other_key = ManifestManager.document_key(other_path)
    current_key = ManifestManager.document_key(current_pdf)
    checks.append(check(
        "Portable manifest key across installation paths",
        old_key == other_key == current_key,
        f"old={old_key}; other={other_key}; current={current_key}",
    ))

    resolved_source = StreamlitUI._resolve_source_path(old_path)
    checks.append(check(
        "Stored source path remaps to current technical_documents",
        bool(resolved_source)
        and Path(resolved_source).name == "MISRA_FromInternet.pdf"
        and Path(resolved_source).is_file(),
        str(resolved_source),
    ))

    # The project copy itself must not be interpreted as a source-content change.
    plan = get_update_plan()
    checks.append(check(
        "Current KB does not require rebuild solely because of path migration",
        plan.get("mode") == "noop",
        plan.get("reason") or "noop",
    ))

    checks.append(check(
        "Server KB report serialization handles Paths",
        _json_safe({"documents": [current_pdf]}) == {"documents": [str(current_pdf)]},
    ))

    # v6.4.93 semantic hardening is included in this consolidated release.
    ambiguous = "What are the rules for compiler switch?"
    clarification = AnswerService._compiler_switch_ambiguity_clarification(ambiguous)
    checks.append(check(
        "Generic compiler switch remains ambiguity-aware",
        bool(clarification)
        and "ambiguous" in clarification.casefold()
        and "compiler configuration" in clarification.casefold(),
        clarification,
    ))

    prior_rule_state = {
        "context": "Rule 17.1 ... Rule 17.8",
        "misra": True,
        "anchor_question": "Tell me about Rule 17 and what is its rationale.",
        "references": ["Rules 17.1-17.8"],
    }
    checks.append(check(
        "Standalone permission question does not reuse stale follow-up state",
        not AnswerService._is_grounded_followup_candidate(
            "Is it ok to use trigraphs?", prior_rule_state
        ),
    ))

    # Connection retry policy: quick connection interruptions may retry, long
    # timeouts must not be doubled.
    checks.append(check(
        "Transient connection reset is retryable",
        _is_transient_connection_error(ConnectionError("connection reset by peer")),
    ))
    checks.append(check(
        "Timeout is not automatically retried",
        not _is_transient_connection_error(TimeoutError("timed out")),
    ))

    # Multi-query is experiment-ready but deliberately OFF until RAGAS/Golden A/B.
    variants = parse_multi_query_variants(
        "1. Rule 10 essential type operands\n2. inappropriate essential type operands\n3. MISRA Rule 10 operand type",
        "Explain Rule 10.1",
    )
    checks.append(check(
        "Multi-query experiment prompt/settings are centralized",
        "do not answer" in MULTI_QUERY_RETRIEVAL_PROMPT.casefold()
        and len(variants) == 3
        and settings.MULTI_QUERY_RETRIEVAL_ENABLED is False,
        f"enabled={settings.MULTI_QUERY_RETRIEVAL_ENABLED}; variants={variants}",
    ))

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    outdir = ROOT / "logs" / "operational_hardening"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.94",
        "overall": overall,
        "checks": checks,
        "base": "v6.4.92 current project + v6.4.93 semantic fixes",
        "kb_rebuild_performed": False,
        "model_stack_changed": False,
        "retrieval_threshold_or_topk_changed": False,
        "chunking_changed": False,
        "multi_query_production_enabled": False,
        "next": "focused operational smoke, then unseen/RAGAS baseline and MultiQuery A/B",
    }
    (outdir / "v6.4.94_operational_hardening_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
