# Import HuggingFace embedding wrapper from LlamaIndex
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

# Embedding settings
from config.settings import (
    EMBED_MODEL_NAME,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DEFAULT_BATCH_SIZE
)

# Cached embedding model
_embedding_model = None


def get_embedding_model():

    # Access cached model
    global _embedding_model

    # Load only once
    if _embedding_model is None:

        print(
            f"[EMBEDDING] Loading model: "
            f"{EMBED_MODEL_NAME}"
        )

        # =====================================
        # EMBEDDING MODEL
        # Converts text into vectors
        # =====================================

        _embedding_model = HuggingFaceEmbedding(

            # E5 multilingual model
            model_name=EMBED_MODEL_NAME,

            # Normalize vectors
            # Improves cosine similarity search
            normalize=DEFAULT_NORMALIZE_EMBEDDINGS,

            # Number of texts embedded together
            embed_batch_size=DEFAULT_BATCH_SIZE
        )

        print(
            "[EMBEDDING] Model loaded successfully"
        )

        print(
            f"[EMBEDDING] Normalize: "
            f"{DEFAULT_NORMALIZE_EMBEDDINGS}"
        )

        print(
            f"[EMBEDDING] Batch Size: "
            f"{DEFAULT_BATCH_SIZE}"
        )

    return _embedding_model