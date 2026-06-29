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

# =====================================================
# RERANKER
# =====================================================
#RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_MODEL = "BAAI/bge-reranker-base"

#Try possible improvement on reranking
#RERANK_MAX_CHARS = 900 # Maximum characters per chunk before reranking.
#RERANK_BATCH_SIZE = 8  # Number of query-document pairs processed at once.

# =====================================================
# OLLAMA
# =====================================================
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 120

# =====================================================
# CHUNKING
# =====================================================
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80

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
#Score debugger
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
PAGE_ICON = "📚"
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
    METADATA_DIR
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
MIN_RETRIEVAL_SCORE = 0.65
