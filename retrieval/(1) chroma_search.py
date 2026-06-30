import chromadb

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    VECTOR_TOP_K
)

from embeddings.embedding_model import (
    get_embedding_model
)


class ChromaSearcher:

    def __init__(self):

        # Connect to local ChromaDB storage.
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Load document embedding collection.
        self.collection = (
            self.client.get_collection(
                CHROMA_COLLECTION_NAME
            )
        )

        # Load embedding model for semantic search.
        self.embed_model = (
            get_embedding_model()
        )

    def search(
        self,
        query,
        top_k=None
    ):

        # Use default retrieval limit from settings.
        if top_k is None:
            top_k = VECTOR_TOP_K

        # Convert user query into an embedding vector.
        query_embedding = (
            self.embed_model.get_text_embedding(
                f"query: {query}"
            )
        )

        # Retrieve similar chunks from ChromaDB.
        result = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k
            )
        )

        # Extract retrieved chunk texts.
        documents = result["documents"][0]

        # Extract chunk metadata.
        metadatas = result["metadatas"][0]

        # Extract vector distances.
        distances = result["distances"][0]

        # Store formatted search results.
        output = []

        # Convert Chroma results into standard retrieval format.
        for doc, meta, dist in zip(
            documents,
            metadatas,
            distances
        ):

            output.append(
                {
                    # Retrieved document chunk.
                    "text": doc,

                    # File and chunk information.
                    "metadata": meta,

                    # Convert distance into similarity score.
                    "score": 1 - float(dist)
                }
            )

        # Return semantic search results.
        return output