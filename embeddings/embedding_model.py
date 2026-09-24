import streamlit as st

from config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EMBEDDING_BACKEND,
    EMBED_MODEL_NAME,
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Return the single finalized production embedding adapter.

    v6.4.82 removes the legacy HuggingFace/E5 production branch.  The active
    embedding component is Ollama qwen3-embedding:8b only.
    """

    if EMBEDDING_BACKEND != "ollama":
        raise RuntimeError(
            "Final production supports only the Ollama Qwen3 embedding backend."
        )

    print(
        f"[EMBEDDING] Loading model: {EMBED_MODEL_NAME} "
        f"({EMBEDDING_BACKEND})"
    )

    from embeddings.ollama_embedding import OllamaEmbeddingModel

    model = OllamaEmbeddingModel()

    print("[EMBEDDING] Model loaded successfully")
    print(f"[EMBEDDING] Normalize: {DEFAULT_NORMALIZE_EMBEDDINGS}")
    print(f"[EMBEDDING] Batch Size: {DEFAULT_BATCH_SIZE}")
    return model
