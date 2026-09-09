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
from qa.test_case_registry import evaluate_answer, get_test_case
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


def test_explain_section_6_fallback_uses_actual_section_contents_not_title_only():
    service = _service()
    context = _actual_exact_context("Explain Section 6.")
    focus = service._detect_answer_focus("Explain Section 6.", "section 6")

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Explain Section 6.",
        resolved_question="section 6",
        answer_focus=focus,
        current_answer="Section 6: Introduction to the guidelines",
    )

    assert answer.startswith("Section 6: Introduction to the guidelines.")
    assert "explains the presentation of the guidelines" in answer
    assert "Guideline classification" in answer
    assert "Guideline categories" in answer
    assert len(answer) > 180


def test_substantive_section_model_answer_is_preserved():
    service = _service()
    context = _actual_exact_context("Explain Section 6.")
    focus = service._detect_answer_focus("Explain Section 6.", "section 6")
    current = (
        "Section 6 explains how the guidelines are organized and classified. "
        "It distinguishes rules from directives, describes guideline categories, "
        "and discusses how compliance analysis is scoped and presented."
    )

    answer = service._apply_structured_answer_focus(
        context=context,
        question="Explain Section 6.",
        resolved_question="section 6",
        answer_focus=focus,
        current_answer=current,
    )

    assert answer == current


def test_compact_policy_explanation_adds_omitted_eligibility_fact():
    service = _service()
    context = """===== DOCUMENT 1 =====
Employee Leave Policy
Vacation Leave
Employees are entitled to 47 days' vacation leave annually.
Sick Leave
Employees are entitled to 17 days sick leave annually.
Approval
All leave requests require manager approval.
Eligibility
All regular employee is entitled to the leave listed above.
"""
    focus = service._detect_answer_focus(
        "Explain the employee leave policy.",
        "the employee leave policy",
    )
    current = (
        "The employee leave policy provides for 47 days of vacation leave and "
        "17 days of sick leave annually, with all leave requests requiring manager approval."
    )

    answer = service._apply_grounded_explanation_coverage(
        context=context,
        answer_focus=focus,
        current_answer=current,
    )

    assert current in answer
    assert "Eligibility:" in answer
    assert "regular employee" in answer


def test_compact_policy_coverage_does_not_change_non_explain_answer():
    service = _service()
    context = """===== DOCUMENT 1 =====
Employee Leave Policy
Vacation Leave
Employees are entitled to 47 days' vacation leave annually.
Sick Leave
Employees are entitled to 17 days sick leave annually.
Approval
All leave requests require manager approval.
Eligibility
All regular employee is entitled to the leave listed above.
"""
    current = "47 days of vacation leave annually."

    assert service._apply_grounded_explanation_coverage(
        context=context,
        answer_focus="QUANTITY: Return the requested number.",
        current_answer=current,
    ) == current


def test_qa_matcher_accepts_middle_name_between_expected_two_part_name():
    case = get_test_case("Who is Jose Rizal?")
    status, notes = evaluate_answer(
        "José Protasio Rizal was a Filipino polymath, writer, and revolutionary.",
        case,
    )

    assert status == "PASS", notes


def test_qa_matcher_still_rejects_wrong_person_name():
    case = get_test_case("Who is Jose Rizal?")
    status, _ = evaluate_answer(
        "Andrés Bonifacio was a Filipino writer and polymath.",
        case,
    )

    assert status == "FAIL"
