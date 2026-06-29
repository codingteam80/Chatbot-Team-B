from retrieval.bm25_index import BM25Searcher
from retrieval.prf_expander import PRFExpander
from retrieval.chroma_search import ChromaSearcher

from config.settings import (
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    FINAL_TOP_K,
    DEBUG_RETRIEVAL
)


class HybridRetriever:
    """
    Combines BM25 + Chroma + PRF with weighted scoring.
    """

    def __init__(self, documents):

        self.documents = documents

        self.bm25 = BM25Searcher(documents)
        self.prf = PRFExpander()
        self.chroma = ChromaSearcher()

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K):

        # 1. BM25 retrieval
        bm25_results = self.bm25.search(query)

        # 2. PRF expansion
        expanded_query = self.prf.expand_query(query, bm25_results)

        # 3. Vector retrieval
        chroma_results = self.chroma.search(expanded_query)

        # 4. Fusion
        final = self._merge(bm25_results, chroma_results)

        return final[:top_k]

    def _merge(self, bm25_results, chroma_results):

        merged = {}

        def add(results, weight):
            for r in results:

                key = r["content"]

                if key not in merged:
                    merged[key] = {
                        "content": r["content"],
                        "metadata": r.get("metadata", {}),
                        "score": 0.0
                    }

                merged[key]["score"] += r["score"] * weight

        add(bm25_results, BM25_WEIGHT)
        add(chroma_results, VECTOR_WEIGHT)

        final = sorted(
            merged.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        if DEBUG_RETRIEVAL:
            print("\n=== RETRIEVAL DEBUG ===")
            for r in final[:10]:
                print(r["score"], "->", r["content"][:80])

        return final