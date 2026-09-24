import os
from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================
ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "enabled"}


# =====================================================
# CENTRALIZED LAN SERVER MODE
# =====================================================
# LAN mode is enabled only by the dedicated server launcher. Normal local
# launches keep the previously certified single-PC behavior unchanged.
LAN_SERVER_MODE = _env_bool("DOCUBOT_LAN_SERVER_MODE", False)
ALLOW_WEB_KB_UPDATE = _env_bool(
    "DOCUBOT_ALLOW_WEB_KB_UPDATE",
    not LAN_SERVER_MODE,
)
try:
    LAN_SERVER_PORT = max(1, min(65535, int(os.getenv("DOCUBOT_LAN_PORT", "8501"))))
except (TypeError, ValueError):
    LAN_SERVER_PORT = 8501


# =====================================================
# DATA — FINAL PRODUCTION PROFILE
# =====================================================
DATA_DIR = ROOT_DIR / "data"
TECHNICAL_DOCUMENT_DIR = DATA_DIR / "technical_documents"

# v6.4.82 finalizes the certified technical production profile. Historical
# mixed/general-corpus material is evidence/archive-only and is no longer an
# active runtime profile. The constant is retained only as a compatibility
# path for historical tooling that may inspect it; production never selects it.
ALL_DOCUMENT_DIR = DATA_DIR / "all_documents"
KNOWLEDGE_PROFILE = "technical"
DOCUMENT_DIR = TECHNICAL_DOCUMENT_DIR

# =====================================================
# STORAGE / VECTOR BACKEND — FINAL PRODUCTION PROFILE
# =====================================================
STORAGE_DIR = ROOT_DIR / "storage"
VECTOR_BACKEND = "qdrant"
CHUNKING_PROFILE = "v4"

# Phase 6 certified and promoted this exact v4 store. Legacy Chroma, v3 Qdrant
# rollback, and experimental promotion stores are no longer active components.
OPTION_C_V4_PRODUCTION_STORAGE_DIR = STORAGE_DIR / "option_c_qwen3_qdrant_v4"
V4_PRODUCTION_READY_MARKER = (
    OPTION_C_V4_PRODUCTION_STORAGE_DIR / "metadata" / "v4_production_ready.json"
)

_OPTION_C_STORAGE_OVERRIDE = os.getenv("DOCUBOT_OPTION_C_STORAGE_DIR", "").strip()
if _OPTION_C_STORAGE_OVERRIDE:
    _option_c_candidate = Path(_OPTION_C_STORAGE_OVERRIDE).expanduser()
    OPTION_C_STORAGE_DIR = (
        _option_c_candidate
        if _option_c_candidate.is_absolute()
        else ROOT_DIR / _option_c_candidate
    ).resolve()
else:
    OPTION_C_STORAGE_DIR = OPTION_C_V4_PRODUCTION_STORAGE_DIR.resolve()

QDRANT_DIR = OPTION_C_STORAGE_DIR / "qdrant"
QDRANT_COLLECTION_NAME = "technical_knowledge"
BM25_DIR = OPTION_C_STORAGE_DIR / "bm25"
METADATA_DIR = OPTION_C_STORAGE_DIR / "metadata"
INGESTION_CACHE_DIR = OPTION_C_STORAGE_DIR / "cache" / "ingestion"

# =====================================================
# EMBEDDING MODEL — FINAL PRODUCTION PROFILE
# =====================================================
EMBEDDING_BACKEND = "ollama"
EMBED_MODEL_NAME = os.getenv(
    "DOCUBOT_EMBED_MODEL",
    "qwen3-embedding:8b",
).strip() or "qwen3-embedding:8b"
EMBED_OLLAMA_URL = os.getenv(
    "DOCUBOT_OLLAMA_URL",
    "http://127.0.0.1:11434",
).strip().rstrip("/") or "http://127.0.0.1:11434"
try:
    EMBED_OLLAMA_TIMEOUT = max(30, int(os.getenv("DOCUBOT_EMBED_TIMEOUT", "180")))
except (TypeError, ValueError):
    EMBED_OLLAMA_TIMEOUT = 180
EMBED_OLLAMA_KEEP_ALIVE = os.getenv(
    "DOCUBOT_EMBED_KEEP_ALIVE",
    "5m",
).strip() or "5m"

# =====================================================
# EMBEDDING SETTINGS
# =====================================================
# Normalize vectors before storing/searching (Improves semantic similarity accuracy)
DEFAULT_NORMALIZE_EMBEDDINGS = True
# Number of texts embedded per batch (Mostly affects indexing speed)
DEFAULT_BATCH_SIZE = 8

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
INGESTION_CACHE_SCHEMA_VERSION = "prepared-chunks-v3-option-c-office-recovery"

# =====================================================
# RERANKER
# =====================================================
#RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_MODEL = os.getenv(
    "DOCUBOT_RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
).strip() or "BAAI/bge-reranker-v2-m3"
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
# OLLAMA / GROUNDED GENERATION ROUTING
# =====================================================
# v6.4.50 production trial has only two answer execution classes:
# - deterministic finalization when authoritative evidence is sufficient;
# - qwen2.5:7b for every generative answer.
# Legacy fast/complex route labels remain only for compatibility with the
# certified pipeline and QA evidence; both labels resolve to the same model.
#
# DOCUBOT_OLLAMA_MODEL remains a backward-compatible process-local override
# for controlled diagnostics/A-B runs. When set, all *generative* routes use
# that model. Safe deterministic structured answers still bypass generation.
OLLAMA_FORCED_MODEL = os.getenv(
    "DOCUBOT_OLLAMA_MODEL",
    ""
).strip()

OLLAMA_FAST_MODEL = os.getenv(
    "DOCUBOT_FAST_LLM_MODEL",
    "qwen2.5:7b",
).strip() or "qwen2.5:7b"

OLLAMA_COMPLEX_MODEL = os.getenv(
    "DOCUBOT_COMPLEX_LLM_MODEL",
    "qwen2.5:7b"
).strip() or "qwen2.5:7b"

# Legacy name retained for existing diagnostics/tests and auxiliary code.
# With no force override it resolves to the fast/default model.
OLLAMA_MODEL = OLLAMA_FORCED_MODEL or OLLAMA_FAST_MODEL

HYBRID_LLM_ROUTING_ENABLED = True

# Route labels (fast/complex) are retained for compatibility and QA semantics,
# but both routes intentionally resolve to the same qwen2.5:7b model.  There is
# no llama fallback/model switch in the v6.4.50 production trial.
# Qwen2.5:7B receives the complete accepted evidence.  The compact-context
# optimization was designed for the former small fast model and is disabled in
# the quality-first technical profile.
GENERATION_USE_FULL_ACCEPTED_CONTEXT = True

# Certification-only switch.  Normal production keeps deterministic finalizers
# enabled.  The one-click LLM certification harness can set this process-local
# environment variable so retrieved/grounded MISRA evidence is still used, but
# the final answer must be produced by an actual routed Ollama model.
# This is deliberately OFF unless explicitly enabled by the certification run.
LLM_CERTIFICATION_FORCE_GENERATION = os.getenv(
    "DOCUBOT_LLM_CERT_FORCE_GENERATION",
    "",
).strip().casefold() in {"1", "true", "yes", "on"}

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

# Qwen2.5 generation residency/context defaults. These settings affect only
# Ollama generation memory/residency; they do not alter retrieval, embeddings,
# reranking, or grounding. Both compatibility route labels use the same 8192
# token context in the quality-first technical profile.
_DEFAULT_FAST_CONTEXT_WINDOW = 8192
try:
    OLLAMA_FAST_CONTEXT_WINDOW = max(
        2048,
        int(os.getenv("DOCUBOT_FAST_LLM_CONTEXT_WINDOW", str(_DEFAULT_FAST_CONTEXT_WINDOW))),
    )
except (TypeError, ValueError):
    OLLAMA_FAST_CONTEXT_WINDOW = _DEFAULT_FAST_CONTEXT_WINDOW

try:
    OLLAMA_COMPLEX_CONTEXT_WINDOW = max(
        4096,
        int(os.getenv("DOCUBOT_COMPLEX_LLM_CONTEXT_WINDOW", "8192")),
    )
except (TypeError, ValueError):
    OLLAMA_COMPLEX_CONTEXT_WINDOW = 8192

# Compatibility keep-alive values remain separately configurable, although
# both route labels resolve to the same Qwen2.5:7B model in this trial.
OLLAMA_KEEP_ALIVE_OVERRIDE = os.getenv(
    "DOCUBOT_OLLAMA_KEEP_ALIVE",
    "",
).strip()

OLLAMA_FAST_KEEP_ALIVE = (
    os.getenv("DOCUBOT_FAST_LLM_KEEP_ALIVE", "15m").strip() or "15m"
)
OLLAMA_COMPLEX_KEEP_ALIVE = (
    os.getenv("DOCUBOT_COMPLEX_LLM_KEEP_ALIVE", "2m").strip() or "2m"
)

# =====================================================
# STARTUP LATENCY PREWARM
# =====================================================
# `auto` keeps deterministic-only sessions lightweight on genuinely low-RAM
# machines, while moving semantic model/Ollama cold-load work out of the first
# submitted question on PCs with enough memory. Set DOCUBOT_STARTUP_PREWARM=off
# to disable or =on to force it regardless of detected RAM.
STARTUP_PREWARM_MODE = os.getenv(
    "DOCUBOT_STARTUP_PREWARM",
    "auto",
).strip() or "auto"

try:
    STARTUP_PREWARM_MIN_RAM_MB = max(
        4096,
        int(os.getenv("DOCUBOT_STARTUP_PREWARM_MIN_RAM_MB", "12288")),
    )
except (TypeError, ValueError):
    STARTUP_PREWARM_MIN_RAM_MB = 12288

try:
    STARTUP_PREWARM_MIN_AVAILABLE_RAM_MB = max(
        1024,
        int(os.getenv("DOCUBOT_STARTUP_PREWARM_MIN_AVAILABLE_RAM_MB", "4096")),
    )
except (TypeError, ValueError):
    STARTUP_PREWARM_MIN_AVAILABLE_RAM_MB = 4096

try:
    STARTUP_PREWARM_DELAY_SECONDS = max(
        0.0,
        float(os.getenv("DOCUBOT_STARTUP_PREWARM_DELAY_SECONDS", "0.75")),
    )
except (TypeError, ValueError):
    STARTUP_PREWARM_DELAY_SECONDS = 0.75

STARTUP_PREWARM_OLLAMA_URL = os.getenv(
    "DOCUBOT_OLLAMA_URL",
    "http://127.0.0.1:11434",
).strip() or "http://127.0.0.1:11434"

# =====================================================
# POST-COMPLEX FAST-MODEL RESIDENCY RECOVERY
# =====================================================
# Loading the larger complex model can evict the frequently used fast model
# on a single-GPU workstation. After a complex answer is fully generated and
# verified, DocuBot may restore the fast model in a daemon thread while the
# user reads the answer. This changes residency only: routing, prompts,
# retrieval, thresholds, and model identities remain unchanged.
_DEFAULT_POST_COMPLEX_RECOVERY = "off"
POST_COMPLEX_FAST_RECOVERY_MODE = os.getenv(
    "DOCUBOT_POST_COMPLEX_FAST_RECOVERY",
    _DEFAULT_POST_COMPLEX_RECOVERY,
).strip() or _DEFAULT_POST_COMPLEX_RECOVERY

try:
    POST_COMPLEX_FAST_RECOVERY_MIN_RAM_MB = max(
        4096,
        int(os.getenv("DOCUBOT_POST_COMPLEX_FAST_RECOVERY_MIN_RAM_MB", "12288")),
    )
except (TypeError, ValueError):
    POST_COMPLEX_FAST_RECOVERY_MIN_RAM_MB = 12288

try:
    POST_COMPLEX_FAST_RECOVERY_MIN_AVAILABLE_RAM_MB = max(
        1024,
        int(os.getenv("DOCUBOT_POST_COMPLEX_FAST_RECOVERY_MIN_AVAILABLE_RAM_MB", "3072")),
    )
except (TypeError, ValueError):
    POST_COMPLEX_FAST_RECOVERY_MIN_AVAILABLE_RAM_MB = 3072

try:
    POST_COMPLEX_FAST_RECOVERY_DELAY_SECONDS = max(
        0.0,
        float(os.getenv("DOCUBOT_POST_COMPLEX_FAST_RECOVERY_DELAY_SECONDS", "0.15")),
    )
except (TypeError, ValueError):
    POST_COMPLEX_FAST_RECOVERY_DELAY_SECONDS = 0.15

try:
    POST_COMPLEX_FAST_RECOVERY_WAIT_SECONDS = max(
        0.0,
        float(os.getenv("DOCUBOT_POST_COMPLEX_FAST_RECOVERY_WAIT_SECONDS", "12")),
    )
except (TypeError, ValueError):
    POST_COMPLEX_FAST_RECOVERY_WAIT_SECONDS = 12.0

# =====================================================
# CHUNKING
# =====================================================
# The certified rule-aware v4 representation is the only active production
# chunking schema in the finalized tree.
INDEX_SCHEMA_VERSION = "structure-aware-pdf-v4-rule-aware-option-c-qdrant-qwen3-tech"

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
DEBUG_RETRIEVAL = False

# =====================================================
# DEBUG
# =====================================================
DEBUG_MODE = False

# =====================================================
# HYBRID WEIGHTS
# =====================================================
VECTOR_WEIGHT = 0.40
BM25_WEIGHT = 0.60

# =====================================================
# UI
# =====================================================
PAGE_TITLE = "Company Knowledge Assistant"
PAGE_ICON = str(ROOT_DIR / "assets_logos" / "docubot_logo.png")
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
    TECHNICAL_DOCUMENT_DIR,
    DOCUMENT_DIR,
    STORAGE_DIR,
    OPTION_C_STORAGE_DIR,
    QDRANT_DIR,
    BM25_DIR,
    METADATA_DIR,
    INGESTION_CACHE_DIR,
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