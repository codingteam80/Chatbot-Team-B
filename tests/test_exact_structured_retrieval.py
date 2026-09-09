import sys
import types
from pathlib import Path


# Keep focused retrieval tests independent of optional runtime packages that
# are not installed in the packaging container. The production modules still
# use the real dependencies on Windows.
def _cache_resource(*args, **kwargs):
    def decorator(function):
        return function
    return decorator


streamlit_stub = types.ModuleType("streamlit")
streamlit_stub.cache_resource = _cache_resource
sys.modules.setdefault("streamlit", streamlit_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

rank_bm25_stub = types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

llama_index_stub = types.ModuleType("llama_index")
llama_embeddings_stub = types.ModuleType("llama_index.embeddings")
llama_hf_stub = types.ModuleType("llama_index.embeddings.huggingface")
llama_hf_stub.HuggingFaceEmbedding = object
sys.modules.setdefault("llama_index", llama_index_stub)
sys.modules.setdefault("llama_index.embeddings", llama_embeddings_stub)
sys.modules.setdefault("llama_index.embeddings.huggingface", llama_hf_stub)

from chat.query_enricher import QueryEnricher
from ingestion.pdf_structure import PDFStructureExtractor
from qa.evidence_logger import evidence_logger
from retrieval.retriever import CompanyRetriever
from utils.structured_reference import extract_structured_reference


class _FailingSearch:
    def search(self, *args, **kwargs):
        raise AssertionError("normal search must not run after an exact metadata match")


class _BM25WithRecords(_FailingSearch):
    def __init__(self, records):
        self.records = records


class _FailingHybrid:
    def merge(self, *args, **kwargs):
        raise AssertionError("hybrid merge must not run after an exact metadata match")


def _retriever_with_records(records):
    retriever = CompanyRetriever.__new__(CompanyRetriever)
    retriever.bm25 = _BM25WithRecords(records)
    retriever.chroma = _FailingSearch()
    retriever.hybrid = _FailingHybrid()
    retriever.reranker = None
    return retriever


def _record(chunk_id, text, *, section_type="section", rule_id="", section_id=""):
    return {
        "text": text,
        "metadata": {
            "file_name": "MISRA_FromInternet.pdf",
            "file_path": "/knowledge/MISRA_FromInternet.pdf",
            "chunk_id": chunk_id,
            "section_type": section_type,
            "rule_id": rule_id,
            "section_id": section_id,
        },
    }


def test_exact_reference_detection_supports_rule_dir_and_section():
    rule = extract_structured_reference("What does Rule 1.2 say about MISRA C?")
    directive = extract_structured_reference("Explain Directive 4.12")
    section = extract_structured_reference("What is Section 6?")

    assert (rule.kind, rule.identifier) == ("rule", "1.2")
    assert (directive.kind, directive.identifier) == ("directive", "4.12")
    assert (section.kind, section.identifier) == ("section", "6")


def test_query_enricher_does_not_dilute_exact_rule_identifier():
    query = "what does rule 1.2 say about MISRA C"

    assert QueryEnricher().enrich(
        query,
        intent_question="What does Rule 1.2 say?",
    ) == query


def test_exact_rule_retrieval_includes_subordinate_chunks_and_stops_at_next_rule():
    records = [
        _record(10, "Rule 1.1\nPrevious rule", section_type="rule", rule_id="1.1"),
        _record(
            11,
            "Rule 1.2\nLanguage extensions should not be used\nCategory\nAdvisory",
            section_type="rule",
            rule_id="1.2",
        ),
        _record(
            12,
            "Rationale\nA program that relies on language extensions may be less portable.",
        ),
        _record(
            13,
            "Rule 1.3\nThere shall be no occurrence of undefined behaviour",
            section_type="rule",
            rule_id="1.3",
        ),
    ]

    retriever = _retriever_with_records(records)
    reference = extract_structured_reference("What does Rule 1.2 say?")
    results = retriever._retrieve_exact_structured_reference(reference)

    assert len(results) == 1
    assert "Language extensions should not be used" in results[0]["text"]
    assert "Rationale" in results[0]["text"]
    assert "Rule 1.3" not in results[0]["text"]
    assert results[0]["metadata"]["exact_chunk_start"] == 11
    assert results[0]["metadata"]["exact_chunk_end"] == 12


def test_retrieve_uses_metadata_first_and_bypasses_semantic_pipeline():
    records = [
        _record(
            20,
            "Rule 10.6\nThe value of a composite expression shall not be assigned to an object with wider essential type",
            section_type="rule",
            rule_id="10.6",
        ),
        _record(21, "Rationale\nThis rule avoids potential developer confusion."),
    ]

    retriever = _retriever_with_records(records)
    previous_enabled = evidence_logger.enabled
    evidence_logger.enabled = False

    try:
        results = retriever.retrieve("What does Rule 10.6 say?")
    finally:
        evidence_logger.enabled = previous_enabled

    assert len(results) == 1
    assert results[0]["metadata"]["exact_reference"] == "Rule 10.6"
    assert results[0]["rerank_score"] == 1.0
    assert "wider essential type" in results[0]["text"]


def test_rule_id_consistency_rejects_unrelated_cross_reference_chunk():
    retriever = CompanyRetriever.__new__(CompanyRetriever)
    reference = extract_structured_reference("What does Rule 1.2 say?")

    wrong = {
        "text": "H.1 Undefined behaviour\nSee also Rule 1.2 and Rule 1.3.",
        "metadata": {
            "section_type": "section",
            "section_id": "H.1",
        },
    }
    legacy_exact = {
        "text": "Rule 1.2\nLanguage extensions should not be used",
        "metadata": {},
    }

    assert retriever._structured_reference_consistent(wrong, reference) is False
    assert retriever._structured_reference_consistent(legacy_exact, reference) is True


def test_actual_misra_pdf_rule_1_2_exact_lookup_contains_rule_and_rationale():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "all_documents"
        / "MISRA_FromInternet.pdf"
    )

    sections = PDFStructureExtractor.extract(str(pdf_path))
    records = []

    for chunk_id, section in enumerate(sections):
        records.append(
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
        )

    retriever = _retriever_with_records(records)
    reference = extract_structured_reference("What does Rule 1.2 say?")
    results = retriever._retrieve_exact_structured_reference(reference)

    assert len(results) == 1
    text = results[0]["text"]
    assert "Rule 1.2" in text
    assert "Language extensions should not be used" in text
    assert "Rationale" in text
    assert "A program that relies on language extensions may be less portable" in text
    assert "Rule 1.3\nThere shall be no occurrence" not in text


def test_actual_misra_pdf_directive_4_12_exact_lookup():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "all_documents"
        / "MISRA_FromInternet.pdf"
    )

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
            },
        }
        for chunk_id, section in enumerate(sections)
    ]

    retriever = _retriever_with_records(records)
    reference = extract_structured_reference("What is Dir 4.12?")
    results = retriever._retrieve_exact_structured_reference(reference)

    assert len(results) == 1
    assert "Dynamic memory allocation shall not be used" in results[0]["text"]
