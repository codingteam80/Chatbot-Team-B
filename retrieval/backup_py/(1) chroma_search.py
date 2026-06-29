import chromadb

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR
)

from embeddings.embedding_model import (
    get_embedding_model
)

from config.settings import VECTOR_TOP_K


class ChromaSearcher:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = (
            self.client.get_collection(
                CHROMA_COLLECTION_NAME
            )
        )

        self.embed_model = (
            get_embedding_model()
        )

    def search(self, query, top_k=None):
        if top_k is None:
            top_k = VECTOR_TOP_K

        query_embedding = (
            self.embed_model.get_text_embedding(
                f"query: {query}"
            )
        )

        result = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k
            )
        )

        documents = result["documents"][0]

        metadatas = result["metadatas"][0]

        distances = result["distances"][0]

        output = []

        for doc, meta, dist in zip(
            documents,
            metadatas,
            distances
        ):

            output.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - float(dist)
                }
            )

        return output