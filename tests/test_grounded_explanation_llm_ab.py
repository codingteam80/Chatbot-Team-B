import os
import subprocess
import sys
import types
from pathlib import Path


def _cache_resource(*args, **kwargs):
    def decorator(function):
        return function
    return decorator


streamlit_stub = sys.modules.get("streamlit") or types.ModuleType("streamlit")
streamlit_stub.cache_resource = _cache_resource
sys.modules.setdefault("streamlit", streamlit_stub)

chromadb_stub = sys.modules.get("chromadb") or types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

rank_bm25_stub = sys.modules.get("rank_bm25") or types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

llama_index_stub = sys.modules.get("llama_index") or types.ModuleType("llama_index")
llama_embeddings_stub = sys.modules.get("llama_index.embeddings") or types.ModuleType("llama_index.embeddings")
llama_hf_stub = sys.modules.get("llama_index.embeddings.huggingface") or types.ModuleType("llama_index.embeddings.huggingface")
llama_hf_stub.HuggingFaceEmbedding = object
llama_llms_stub = sys.modules.get("llama_index.llms") or types.ModuleType("llama_index.llms")
llama_ollama_stub = sys.modules.get("llama_index.llms.ollama") or types.ModuleType("llama_index.llms.ollama")
llama_ollama_stub.Ollama = object
sys.modules.setdefault("llama_index", llama_index_stub)
sys.modules.setdefault("llama_index.embeddings", llama_embeddings_stub)
sys.modules.setdefault("llama_index.embeddings.huggingface", llama_hf_stub)
sys.modules.setdefault("llama_index.llms", llama_llms_stub)
sys.modules.setdefault("llama_index.llms.ollama", llama_ollama_stub)

from ingestion.pdf_structure import PDFStructureExtractor
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService
from utils.structured_reference import extract_structured_reference


ROOT = Path(__file__).resolve().parents[1]


def _service():
    return AnswerService.__new__(AnswerService)


def _actual_exact_context(question):
    pdf_path = ROOT / "data" / "all_documents" / "MISRA_FromInternet.pdf"
    sections = PDFStructureExtractor.extract(str(pdf_path))
    records = [
        {
            "text": section.text,
            "metadata": {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "chunk_id": chunk_id,
                "section_type": section.section_type,
                "rule_id": section.rule_id,
                "section_id": section.section_id,
                "page_start": section.page_start,
                "page_end": section.page_end,
            },
        }
        for chunk_id, section in enumerate(sections)
    ]
    retriever = CompanyRetriever.__new__(CompanyRetriever)
    retriever.bm25 = types.SimpleNamespace(records=records)
    reference = extract_structured_reference(question)
    results = retriever._retrieve_exact_structured_reference(reference)
    assert len(results) == 1
    return "===== DOCUMENT 1 =====\n" + results[0]["text"]


def test_generic_explain_gets_grounded_explanation_focus():
    focus = _service()._detect_answer_focus(
        "Explain the employee leave policy.",
        "explain the employee leave policy"
    )

    assert focus.startswith("GROUNDED EXPLANATION:")
    assert "2 to 5 concise sentences" in focus
    assert "outside knowledge" in focus


def test_explain_rule_1_2_covers_more_than_one_rationale_point():
    service = _service()
    context = _actual_exact_context("Explain Rule 1.2.")
    focus = service._detect_answer_focus("Explain Rule 1.2.", "rule 1.2")

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Explain Rule 1.2.",
        resolved_question="rule 1.2",
        answer_focus=focus,
        current_answer="too short"
    )

    assert answer.startswith("Rule 1.2: Language extensions should not be used.")
    assert "less portable" in answer
    assert "full description of the behaviour" in answer
    assert "project’s design documentation" in answer or "project's design documentation" in answer


def test_explain_directive_4_12_covers_scope_and_rationale():
    service = _service()
    context = _actual_exact_context("Explain Directive 4.12.")
    focus = service._detect_answer_focus("Explain Directive 4.12.", "directive 4.12")

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Explain Directive 4.12.",
        resolved_question="directive 4.12",
        answer_focus=focus,
        current_answer="too short"
    )

    assert answer.startswith("Directive 4.12: Dynamic memory allocation shall not be used.")
    assert "all dynamic memory allocation packages" in answer
    assert "Third-party packages" in answer
    assert "undefined behaviour" in answer
    assert "checked to ensure" in answer


def test_explain_section_uses_structured_explanation_focus_without_rule_only_wording():
    focus = _service()._detect_answer_focus(
        "Explain Section 6.",
        "section 6"
    )

    assert focus.startswith("STRUCTURED EXPLANATION:")
    assert "Rule, Directive, or Section" in focus


def test_grounded_explanation_wrapper_is_removed():
    service = _service()
    assert service._strip_output_wrappers(
        "Grounded Explanation:\nThis is the supported explanation."
    ) == "This is the supported explanation."


def _read_model_from_clean_process(model_value):
    env = os.environ.copy()
    env["DOCUBOT_OLLAMA_MODEL"] = model_value
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from config.settings import OLLAMA_MODEL; print(OLLAMA_MODEL)",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip().splitlines()[-1]


def test_llm_model_can_be_overridden_for_ab_test_without_code_change():
    assert _read_model_from_clean_process("qwen2.5:7b") == "qwen2.5:7b"
    assert _read_model_from_clean_process("llama3.2:3b") == "llama3.2:3b"


def test_invalid_timeout_override_falls_back_instead_of_crashing():
    env = os.environ.copy()
    env["DOCUBOT_OLLAMA_TIMEOUT"] = "not-a-number"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from config.settings import OLLAMA_TIMEOUT; print(OLLAMA_TIMEOUT)",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] == "120"


def test_default_llm_model_remains_llama3_2_3b_when_no_override_is_set():
    env = os.environ.copy()
    env.pop("DOCUBOT_OLLAMA_MODEL", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from config.settings import OLLAMA_MODEL; print(OLLAMA_MODEL)",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] == "llama3.2:3b"


def test_windows_runner_accepts_process_local_model_parameter():
    runner = (ROOT / "run_docubot_with_log.ps1").read_text(encoding="utf-8")
    assert '[string]$Model = ""' in runner
    assert '$env:DOCUBOT_OLLAMA_MODEL = $Model.Trim()' in runner
