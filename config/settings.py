import os
from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================
ROOT_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# DATA
# =====================================================
DATA_DIR = ROOT_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "all_documents"

# =====================================================
# STORAGE
# =====================================================
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
BM25_DIR = STORAGE_DIR / "bm25"
METADATA_DIR = STORAGE_DIR / "metadata"
INGESTION_CACHE_DIR = STORAGE_DIR / "cache" / "ingestion"

# =====================================================
# CHROMA
# =====================================================
CHROMA_COLLECTION_NAME = "company_knowledge"

# =====================================================
# EMBEDDING MODEL
# =====================================================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

# =====================================================
# EMBEDDING SETTINGS
# =====================================================
# Normalize vectors before storing/searching (Improves semantic similarity accuracy)
DEFAULT_NORMALIZE_EMBEDDINGS = True
# Number of texts embedded per batch (Mostly affects indexing speed)
DEFAULT_BATCH_SIZE = 32

# Number of prepared vectors written to Chroma per upsert.
# Larger writes reduce database overhead without changing retrieval quality.
try:
    CHROMA_WRITE_BATCH_SIZE = max(
        16,
        int(os.getenv("DOCUBOT_CHROMA_WRITE_BATCH_SIZE", "128")),
    )
except (TypeError, ValueError):
    CHROMA_WRITE_BATCH_SIZE = 128

# Independent source files can be parsed/chunked concurrently. Keep the
# default conservative for Windows deployments and CPU/RAM stability.
try:
    INGESTION_MAX_WORKERS = max(
        1,
        min(
            8,
            int(
                os.getenv(
                    "DOCUBOT_INGEST_WORKERS",
                    str(min(4, max(1, os.cpu_count() or 1))),
                )
            ),
        ),
    )
except (TypeError, ValueError):
    INGESTION_MAX_WORKERS = min(4, max(1, os.cpu_count() or 1))

# Cache format version for parsed/cleaned/prepared chunks. Changing only the
# cache format does not invalidate the vector index; changing parsing/chunking
# semantics must still increment INDEX_SCHEMA_VERSION below.
INGESTION_CACHE_SCHEMA_VERSION = "prepared-chunks-v1"

# =====================================================
# RERANKER
# =====================================================
#RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_MODEL = "BAAI/bge-reranker-base"
#RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

#Try possible improvement on reranking
#RERANK_MAX_CHARS = 900 # Maximum characters per chunk before reranking.
#RERANK_BATCH_SIZE = 8  # Number of query-document pairs processed at once.


# =====================================================
# MULTILINGUAL RETRIEVAL
# =====================================================
# English remains the primary retrieval language.
# Non-English questions can be converted into a standalone
# English retrieval query while the final answer still uses
# the language of the user's original question.
ENABLE_MULTILINGUAL_RETRIEVAL = True

# When the bilingual query returns no accepted context,
# retry retrieval using only the canonical English query.
MULTILINGUAL_RETRY_ON_EMPTY = True

# Prevent excessively long translated retrieval queries.
MULTILINGUAL_QUERY_MAX_CHARS = 500

# =====================================================
# OLLAMA
# =====================================================
# Default remains llama3.2:3b so upgrading to this package does not
# silently change runtime behavior. For controlled A/B tests, set
# DOCUBOT_OLLAMA_MODEL before starting Streamlit (for example qwen2.5:7b).
# Model switching affects answer generation only; it does not require a KB rebuild.
OLLAMA_MODEL = os.getenv(
    "DOCUBOT_OLLAMA_MODEL",
    "llama3.2:3b"
).strip() or "llama3.2:3b"

try:
    OLLAMA_TIMEOUT = max(
        1,
        int(
            os.getenv(
                "DOCUBOT_OLLAMA_TIMEOUT",
                "120"
            )
        )
    )
except (TypeError, ValueError):
    # Invalid optional overrides must not prevent DocuBot from starting.
    OLLAMA_TIMEOUT = 120

# =====================================================
# CHUNKING
# =====================================================
# Increment this when parsing/chunking semantics change. It is stored in
# the manifest so Smart Build automatically re-indexes existing files.
INDEX_SCHEMA_VERSION = "structure-aware-pdf-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# =====================================================
# DOCUMENT VALIDATION
# =====================================================
MIN_DOCUMENT_LENGTH = 50

# =====================================================
# RETRIEVAL
# =====================================================
VECTOR_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 3

# =====================================================
# CONTEXT COMPLETENESS EXPANSION
# =====================================================
# Used for list / procedure / requirement / rule questions.
# When a relevant chunk is found, DocuBot also includes
# nearby chunks from the same file so lists/steps are not cut.
COMPLETENESS_TOP_K = 6
CONTEXT_EXPANSION_PREVIOUS_CHUNKS = 1
CONTEXT_EXPANSION_NEXT_CHUNKS = 1
CONTEXT_EXPANSION_MAX_SEEDS = 2

# Fast mode:
# False = BM25 + Vector Hybrid only
# True  = Hybrid + Reranker
ENABLE_RERANKER = True
# Score debugger
DEBUG_RETRIEVAL = True

# =====================================================
# DEBUG
# =====================================================
DEBUG_MODE = True

# =====================================================
# HYBRID WEIGHTS
# =====================================================
VECTOR_WEIGHT = 0.40
BM25_WEIGHT = 0.60

# =====================================================
# UI
# =====================================================
PAGE_TITLE = "Company Knowledge Assistant"
PAGE_ICON = r"C:\user_dev\company-chatbot\assets_logos\docubot_logo.png"
LAYOUT = "wide"

# =====================================================
# SUPPORTED FILES
# =====================================================
SUPPORTED_EXTENSIONS = [
    # Documents
    ".pdf",
    ".docx", ".doc",
    ".pptx",
    ".xlsx", ".xls",
    ".csv",
    ".txt",
    ".md",
    ".rtf",

    # Web / Structured Data
    ".html", ".htm",
    ".xml",
    ".json",

    # Email
    ".msg",
    ".eml",

    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
]

# =====================================================
# CREATE DIRECTORIES
# =====================================================
for path in [
    DATA_DIR,
    DOCUMENT_DIR,
    STORAGE_DIR,
    CHROMA_DIR,
    BM25_DIR,
    METADATA_DIR,
    INGESTION_CACHE_DIR
]:
    path.mkdir(parents=True, exist_ok=True)

# ======================================
# Retrieval Confidence Threshold
# ======================================
# Minimum reranker score required
# for a retrieved chunk to be considered relevant.
#
# Increase value  -> stricter retrieval
# Decrease value  -> more permissive retrieval
#
# Recommended range:
# 0.55 - 0.75
# BAAI/bge-reranker-base returns scores close to 1.0 for strong matches in
# this project. 0.10 was too permissive and allowed weak evidence to reach
# answer generation. Keep this conservative enough to avoid false negatives
# while rejecting clearly unrelated chunks.
MIN_RETRIEVAL_SCORE = 0.55

# ======================================
# TEMPORARY QA EVIDENCE LOGGING
# ======================================

# Enable detailed QA test logs.
# Set to False for normal or official builds.
TEST_EVIDENCE_MODE = True

# False = chunk previews only.
# True = complete chunk text in the QA log.
TEST_EVIDENCE_FULL_CHUNKS = False

# Maximum chunk preview length when full chunks are disabled.
TEST_EVIDENCE_PREVIEW_LENGTH = 700