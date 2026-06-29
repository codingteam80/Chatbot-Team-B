from rank_bm25 import BM25Okapi

from config.settings import BM25_TOP_K


class BM25Searcher:
    """
    Keyword-based retrieval using BM25.
    """

    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [doc.split() for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = None):
        top_k = top_k or BM25_TOP_K

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            results.append({
                "content": self.documents[idx],
                "score": float(score),
                "index": idx
            })

        return results