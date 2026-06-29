from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import EMBED_MODEL_NAME


_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        _embedding_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME
        )

    return _embedding_model