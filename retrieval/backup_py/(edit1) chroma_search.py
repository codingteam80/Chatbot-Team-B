import chromadb

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    VECTOR_TOP_K
)

from embeddings.embedding_model import get_embedding_model


class ChromaSearcher:
    """
    Semantic search using ChromaDB.
    """

    def __init__(self):

        self.embed_model = get_embedding_model()

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

    def search(self, query: str, top_k: int = None):

        top_k = top_k or VECTOR_TOP_K

        query_embedding = self.embed_model.get_text_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format(results)

    def _format(self, results):

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {
                "content": d,
                "metadata": m,
                "score": dist
            }
            for d, m, dist in zip(docs, metas, dists)
        ]