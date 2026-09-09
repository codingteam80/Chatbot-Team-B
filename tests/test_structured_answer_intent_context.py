import sys
import types
from pathlib import Path
from unittest.mock import patch


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

from chat.chat_manager import ChatManager
from chat.conversation_resolver import ConversationResolver
from ingestion.pdf_structure import PDFStructureExtractor
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService
from utils.structured_reference import extract_structured_reference


def _service():
    return AnswerService.__new__(AnswerService)


def _retriever_with_records(records):
    retriever = CompanyRetriever.__new__(CompanyRetriever)
    retriever.bm25 = types.SimpleNamespace(records=records)
    return retriever


def _actual_misra_records():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "all_documents"
        / "MISRA_FromInternet.pdf"
    )
    sections = PDFStructureExtractor.extract(str(pdf_path))
    return [
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


def _actual_exact_context(question):
    records = _actual_misra_records()
    retriever = _retriever_with_records(records)
    reference = extract_structured_reference(question)
    results = retriever._retrieve_exact_structured_reference(reference)
    assert len(results) == 1
    return "===== DOCUMENT 1 =====\n" + results[0]["text"]


def test_followup_that_rule_resolves_to_latest_exact_rule_not_document_topic():
    history = [
        {"role": "user", "content": "What is MISRA C?"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "What does Rule 1.2 say?"},
    ]

    with patch.object(ChatManager, "get_current_topic", return_value="MISRA"):
        resolved = ConversationResolver().resolve(
            history,
            "why is that rule important"
        )

    assert resolved == "why is Rule 1.2 important"


def test_followup_that_directive_resolves_to_latest_exact_directive():
    history = [
        {"role": "user", "content": "What is Dir 4.12?"},
    ]

    with patch.object(ChatManager, "get_current_topic", return_value="MISRA"):
        resolved = ConversationResolver().resolve(
            history,
            "why is that directive important"
        )

    assert resolved == "why is Directive 4.12 important"


def test_exact_rule_say_uses_structured_statement_focus():
    focus = _service()._detect_answer_focus(
        "What does Rule 1.2 say?",
        "what does rule 1.2 say"
    )

    assert focus.startswith("STRUCTURED STATEMENT:")


def test_explain_directive_uses_structured_explanation_focus():
    focus = _service()._detect_answer_focus(
        "Explain Directive 4.12.",
        "directive 4.12"
    )

    assert focus.startswith("STRUCTURED EXPLANATION:")


def test_rule_1_2_direct_answer_returns_rule_statement_not_rationale():
    service = _service()
    context = _actual_exact_context("What does Rule 1.2 say?")
    focus = service._detect_answer_focus(
        "What does Rule 1.2 say?",
        "what does rule 1.2 say"
    )

    answer = service._apply_structured_answer_focus(
        context=context,
        question="What does Rule 1.2 say?",
        resolved_question="what does rule 1.2 say",
        answer_focus=focus,
        current_answer="A program that relies on language extensions may be less portable."
    )

    assert answer == "Rule 1.2: Language extensions should not be used"
    assert "less portable" not in answer


def test_rule_10_6_multiline_statement_is_joined_completely():
    service = _service()
    context = _actual_exact_context("What does Rule 10.6 say?")
    focus = service._detect_answer_focus(
        "What does Rule 10.6 say?",
        "what does rule 10.6 say"
    )

    answer = service._apply_structured_answer_focus(
        context=context,
        question="What does Rule 10.6 say?",
        resolved_question="what does rule 10.6 say",
        answer_focus=focus,
        current_answer="wrong draft"
    )

    assert answer == (
        "Rule 10.6: The value of a composite expression shall not be assigned "
        "to an object with wider essential type"
    )


def test_explain_directive_includes_exact_statement_and_rationale():
    service = _service()
    context = _actual_exact_context("Explain Directive 4.12.")
    focus = service._detect_answer_focus(
        "Explain Directive 4.12.",
        "directive 4.12"
    )

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Explain Directive 4.12.",
        resolved_question="directive 4.12",
        answer_focus=focus,
        current_answer="Directive 4.12: Dynamic memory allocation shall not be used."
    )

    assert answer.startswith(
        "Directive 4.12: Dynamic memory allocation shall not be used."
    )
    assert "undefined behaviour" in answer


def test_structured_reason_uses_exact_rule_rationale():
    service = _service()
    context = _actual_exact_context("What does Rule 1.2 say?")
    focus = service._detect_answer_focus(
        "Why is that rule important?",
        "why is Rule 1.2 important"
    )

    assert focus.startswith("REASON:")

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Why is that rule important?",
        resolved_question="why is Rule 1.2 important",
        answer_focus=focus,
        current_answer="generic MISRA answer"
    )

    assert answer == (
        "A program that relies on language extensions may be less portable "
        "than one that does not."
    )


def test_internal_answer_focus_label_is_removed():
    service = _service()

    assert service._strip_output_wrappers(
        'Definition or Detail:\nSection 6 refers to the "Introduction to the guidelines" section.'
    ) == 'Section 6 refers to the "Introduction to the guidelines" section.'
