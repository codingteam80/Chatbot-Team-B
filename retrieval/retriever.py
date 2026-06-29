print("USING retriever.py")
from retrieval.bm25_index import (
    BM25Searcher
)

from retrieval.chroma_search import (
    ChromaSearcher
)

from retrieval.hybrid_search import (
    HybridRetriever
)

from retrieval.reranker import (
    CrossEncoderReranker
)

from config.settings import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    DEBUG_RETRIEVAL,
    MIN_RETRIEVAL_SCORE
)


class CompanyRetriever:

    def __init__(self):

        # Initialize BM25 keyword search.
        self.bm25 = BM25Searcher()

        # Initialize vector similarity search.
        self.chroma = ChromaSearcher()

        # Initialize hybrid score merger.
        self.hybrid = HybridRetriever()

        # Initialize final reranking model.
        self.reranker = (
            CrossEncoderReranker()
        )

    # =====================================
    # DIVERSITY FILTER
    # =====================================

    def _apply_diversity_filter(
        self,
        results,
        max_chunks_per_file=2
    ):

        # Store filtered retrieval results.
        filtered = []

        # Track chunk count per file.
        file_count = {}

        # Process retrieved chunks.
        for item in results:

            file_name = (
                item["metadata"]
                .get(
                    "file_name",
                    "Unknown"
                )
            )

            current_count = (
                file_count.get(
                    file_name,
                    0
                )
            )

            # Skip files exceeding chunk limit.
            if current_count >= max_chunks_per_file:
                continue

            file_count[file_name] = (
                current_count + 1
            )

            # Keep chunk in final results.
            filtered.append(item)

        # Return diversified chunks.
        return filtered

    # =====================================
    # CONFIDENCE FILTER
    # =====================================

    def _apply_confidence_filter(
        self,
        results
    ):

        # No retrieval results
        if not results:
            return []

        # Highest CrossEncoder score
        best_score = results[0].get(
            "rerank_score",
            0.0
            )

        if DEBUG_RETRIEVAL:

            print("\n===== CONFIDENCE FILTER =====")
            print(f"Best Score : {best_score:.4f}")
            print(
                f"Threshold  : {MIN_RETRIEVAL_SCORE:.4f}"
            )

        # Reject low-confidence retrievals
        if best_score < MIN_RETRIEVAL_SCORE:

            if DEBUG_RETRIEVAL:
                print("Result     : REJECTED")
                print("=============================\n")

            return []

        if DEBUG_RETRIEVAL:
            print("Result     : ACCEPTED")
            print("=============================\n")

        return results

    # =====================================
    # RETRIEVE
    # =====================================

    def retrieve(
        self,
        query
    ):

        # Retrieve keyword matching chunks.
        bm25_results = (
            self.bm25.search(
                query,
                BM25_TOP_K
            )
        )

        # Retrieve semantic matching chunks.
        vector_results = (
            self.chroma.search(
                query,
                VECTOR_TOP_K
            )
        )

        # Combine BM25 and vector results.
        merged = (
            self.hybrid.merge(
                vector_results,
                bm25_results
            )
        )

        # =====================================
        # DEBUG HYBRID
        # =====================================

        if DEBUG_RETRIEVAL:

            print("\n===== HYBRID RESULTS =====")

            for item in merged[:10]:

                print(
                    f"{item['metadata']['file_name']} "
                    f"=> {item['score']:.4f}"
                )

            print("==========================\n")

            print("\n===== BEFORE RERANK =====")

            for item in merged:

                print(
                    item["metadata"]["file_name"],
                    item["score"]
                )

        # Apply cross-encoder reranking.
        reranked = (
            self.reranker.rerank(
                query,
                merged
            )
        )

        # =====================================
        # DEBUG RERANK
        # =====================================

        if DEBUG_RETRIEVAL:

            print("\n===== RERANKED RESULTS =====")

            for item in reranked[:10]:

                score = item.get(
                    "rerank_score",
                    0.0
                )

                print(
                    f"{item['metadata']['file_name']} "
                    f"=> {score:.4f}"
                )

            print(
                "============================\n"
            )

        # Reject low-confidence retrievals.
        reranked = self._apply_confidence_filter(
            reranked
        )

        # Increase source diversity.
        diversified = (
            self._apply_diversity_filter(
                reranked,
                max_chunks_per_file=2
            )
        )

        # =====================================
        # DEBUG DIVERSIFIED
        # =====================================

        if DEBUG_RETRIEVAL:

            print(
                "\n===== DIVERSIFIED RESULTS ====="
            )

            for item in diversified[:10]:

                score = item.get(
                    "rerank_score",
                    0.0
                )

                print(
                    f"{item['metadata']['file_name']} "
                    f"=> {score:.4f}"
                )

            print(
                "===============================\n"
            )

        # Return final retrieval results.
        return diversified

    # =====================================
    # BUILD CONTEXT
    # =====================================

    def build_context(
        self,
        query
    ):

        # Retrieve best matching chunks.
        results = (
            self.retrieve(
                query
            )
        )

        # Keep only final context chunks.
        results = results[:FINAL_TOP_K]

        # Combine chunks into LLM context.
        context = "\n\n".join(
            item["text"]
            for item in results
        )

        # Return context and source chunks.
        return context, results