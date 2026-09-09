import sys
import types


def _cache_resource(*args, **kwargs):
    def decorator(function):
        function.clear = lambda: None
        return function
    return decorator


streamlit_stub = sys.modules.get("streamlit") or types.ModuleType("streamlit")
streamlit_stub.cache_resource = _cache_resource
sys.modules.setdefault("streamlit", streamlit_stub)

chromadb_stub = sys.modules.get("chromadb") or types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

rank_bm25_stub = sys.modules.get("rank_bm25") or types.ModuleType("rank_bm25")


class _FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus


rank_bm25_stub.BM25Okapi = _FakeBM25
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

from config.settings import MIN_RETRIEVAL_SCORE
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService


def _retriever():
    return CompanyRetriever.__new__(CompanyRetriever)


def test_identity_ranking_deprioritizes_reference_section():
    retriever = _retriever()
    query = "who is alex rivera biography alex rivera life"

    biography = {
        "text": (
            "Alex Rivera was a company engineer and project lead. "
            "Rivera led the documented migration program and trained the team."
        ),
        "score": 0.70,
        "metadata": {
            "file_name": "Alex Rivera.pdf",
            "section_title": "Alex Rivera",
        },
    }
    citations = {
        "text": (
            "Citations\nAlex Rivera (2024). Alex Rivera biography. "
            "https://example.invalid/a https://example.invalid/b ISBN 1234."
        ),
        "score": 0.95,
        "metadata": {
            "file_name": "Alex Rivera.pdf",
            "section_title": "Citations",
        },
    }

    ranked = retriever._prioritize_informative_chunks(
        query,
        [citations, biography],
    )

    assert ranked[0] is biography
    assert ranked[-1] is citations
    assert citations["_is_reference_section"] is True


def test_identity_reranker_pool_does_not_collapse_to_one_reference_chunk():
    retriever = _retriever()
    query = "who is alex rivera biography"

    candidates = []
    for index in range(8):
        candidates.append(
            {
                "text": (
                    f"Alex Rivera was associated with documented company detail {index}. "
                    "This paragraph contains direct descriptive information."
                ),
                "score": 0.70 - (index * 0.01),
                "metadata": {
                    "file_name": "Alex Rivera.pdf",
                    "section_title": f"Profile {index}",
                },
            }
        )

    reference = {
        "text": "Citations\nAlex Rivera reference list https://example.invalid",
        "score": 0.99,
        "metadata": {
            "file_name": "Alex Rivera.pdf",
            "section_title": "Citations",
        },
    }

    ranked = retriever._prioritize_informative_chunks(
        query,
        [reference, *candidates],
    )
    filtered = retriever._remove_low_information_chunks(
        query,
        ranked,
    )
    filtered = retriever._remove_identity_noise_chunks(
        query,
        filtered,
    )
    pool = retriever._prepare_identity_reranker_candidates(
        query=query,
        ranked_candidates=ranked,
        filtered_candidates=filtered,
        limit=10,
        minimum_pool=6,
    )

    assert len(pool) >= 6
    assert reference not in pool


def test_definition_chunk_is_not_rejected_only_for_url_noise():
    retriever = _retriever()
    text = (
        "Alex Rivera was a company engineer responsible for the migration program. "
        + " ".join(
            f"https://example.invalid/{index}"
            for index in range(10)
        )
    )

    assert retriever._looks_like_low_value_chunk(text) is False


def test_verifier_transcript_returns_only_explicit_answer():
    service = AnswerService.__new__(AnswerService)
    transcript = (
        "DRAFT ANSWER: Alex Rivera was an engineer.\n\n"
        "MATCHED ANSWER FOCUS: IDENTITY OR OVERVIEW\n\n"
        "ANSWER: Alex Rivera was a company engineer and project lead."
    )

    assert service._extract_verifier_answer(
        transcript,
        "Alex Rivera was an engineer.",
    ) == "Alex Rivera was a company engineer and project lead."

    assert service._apply_output_safety_gate(
        transcript,
        "Who is Alex Rivera?",
    ) == "Alex Rivera was a company engineer and project lead."


def test_real_prompt_leak_still_falls_back():
    service = AnswerService.__new__(AnswerService)

    assert service._apply_output_safety_gate(
        "COMPANY KNOWLEDGE:\nDo not use outside knowledge.",
        "test",
    ) == "Information not found in company knowledge base."


def test_retrieval_confidence_threshold_is_no_longer_permissive():
    assert 0.50 <= MIN_RETRIEVAL_SCORE <= 0.75


def test_short_high_confidence_company_fact_is_not_filtered_only_for_word_count():
    retriever = _retriever()
    query = "the internal project code in the v6.2 incremental test document"
    item = {
        "text": (
            "DocuBot v6.2 incremental indexing validation document. "
            "The internal project code is V62-TEST-ALPHA. "
            "This file exists only to validate new-file incremental indexing."
        ),
        "score": 0.9038,
        "metadata": {
            "file_name": "v62_incremental_test.txt",
            "section_title": "",
        },
    }

    assert len(item["text"].split()) < 25
    assert retriever._short_chunk_has_strong_query_evidence(query, item) is True
    assert retriever._looks_like_low_value_chunk(
        item["text"],
        query=query,
        item=item,
    ) is False
    assert retriever._remove_low_information_chunks(query, [item]) == [item]


def test_short_weak_or_unrelated_chunk_still_uses_low_information_guardrail():
    retriever = _retriever()
    query = "the internal project code in the v6.2 incremental test document"
    item = {
        "text": "Project notes archived for historical reference only.",
        "score": 0.51,
        "metadata": {
            "file_name": "notes.txt",
            "section_title": "",
        },
    }

    assert retriever._short_chunk_has_strong_query_evidence(query, item) is False
    assert retriever._looks_like_low_value_chunk(
        item["text"],
        query=query,
        item=item,
    ) is True
