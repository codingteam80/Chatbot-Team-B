import streamlit as st
import chromadb

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    VECTOR_TOP_K
)

from embeddings.embedding_model import (
    get_embedding_model
)


@st.cache_resource(
    show_spinner=False
)
def get_chroma_collection():

    print(
        "[CHROMA] Opening database..."
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # get_or_create_collection also supports first-run and
    # empty-knowledge-base states.
    collection = (
        client.get_or_create_collection(
            CHROMA_COLLECTION_NAME
        )
    )

    print(
        "[CHROMA] Ready."
    )

    return collection


class ChromaSearcher:

    def __init__(self):

        self.collection = (
            get_chroma_collection()
        )

        self.embed_model = (
            get_embedding_model()
        )

    def search(
        self,
        query,
        top_k=None
    ):

        if top_k is None:

            top_k = VECTOR_TOP_K

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:

            return []

        query_embedding = (
            self.embed_model.get_text_embedding(
                f"query: {query}"
            )
        )

        result = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                top_k,
                collection_count
            )
        )

        documents = (
            result.get("documents")
            or [[]]
        )[0]

        metadatas = (
            result.get("metadatas")
            or [[]]
        )[0]

        distances = (
            result.get("distances")
            or [[]]
        )[0]

        output = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            output.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "score": (
                        1 - float(distance)
                    )
                }
            )

        return output
