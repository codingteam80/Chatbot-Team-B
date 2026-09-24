from __future__ import annotations

import threading

from config.settings import QDRANT_COLLECTION_NAME, QDRANT_DIR, VECTOR_TOP_K


# Qdrant local mode protects its storage directory against concurrent clients.
# Streamlit can serve multiple sessions/threads, so serialize each short-lived
# local client lifecycle to avoid intermittent file-lock errors.
_QDRANT_LOCAL_LOCK = threading.RLock()


def _qdrant_imports():
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError(
            "qdrant-client is not installed. Run Setup_DocuBot_Option_C.bat "
            "or install the updated requirements.txt before starting Option C."
        ) from error
    return QdrantClient


def open_qdrant_client(path=None):
    QdrantClient = _qdrant_imports()
    return QdrantClient(path=str(path or QDRANT_DIR))


def _embedding_model():
    from embeddings.embedding_model import get_embedding_model
    return get_embedding_model()


def qdrant_collection_count() -> int:
    with _QDRANT_LOCAL_LOCK:
        client = open_qdrant_client()
        try:
            try:
                if hasattr(client, "collection_exists") and not client.collection_exists(
                    QDRANT_COLLECTION_NAME
                ):
                    return 0
            except Exception:
                pass

            try:
                return int(client.count(QDRANT_COLLECTION_NAME, exact=True).count)
            except Exception:
                # Older qdrant-client versions may not expose exact= on local count.
                return int(client.count(QDRANT_COLLECTION_NAME).count)
        except Exception:
            return 0
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


class QdrantSearcher:
    def __init__(self):
        self.embed_model = _embedding_model()

    @staticmethod
    def _points_from_response(response):
        points = getattr(response, "points", None)
        if points is not None:
            return list(points)
        if isinstance(response, list):
            return response
        return []

    def search(self, query, top_k=None):
        top_k = max(1, int(top_k or VECTOR_TOP_K))
        embed_query = getattr(self.embed_model, "get_query_embedding", None)
        query_embedding = (
            embed_query(query)
            if callable(embed_query)
            else self.embed_model.get_text_embedding(query)
        )
        with _QDRANT_LOCAL_LOCK:
            client = open_qdrant_client()

            try:
                try:
                    if hasattr(client, "collection_exists") and not client.collection_exists(
                        QDRANT_COLLECTION_NAME
                    ):
                        return []
                except Exception:
                    pass

                response = None
                query_points = getattr(client, "query_points", None)
                if callable(query_points):
                    try:
                        response = query_points(
                            collection_name=QDRANT_COLLECTION_NAME,
                            query=query_embedding,
                            limit=top_k,
                            with_payload=True,
                        )
                    except (TypeError, AttributeError):
                        response = None

                if response is None:
                    search = getattr(client, "search", None)
                    if not callable(search):
                        raise RuntimeError(
                            "Installed qdrant-client does not expose a compatible search API."
                        )
                    response = search(
                        collection_name=QDRANT_COLLECTION_NAME,
                        query_vector=query_embedding,
                        limit=top_k,
                        with_payload=True,
                    )

                output = []
                for point in self._points_from_response(response):
                    payload = dict(getattr(point, "payload", None) or {})
                    text = str(payload.get("text") or "")
                    metadata = payload.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    output.append(
                        {
                            "text": text,
                            "metadata": metadata,
                            "score": float(getattr(point, "score", 0.0) or 0.0),
                        }
                    )
                return output
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
