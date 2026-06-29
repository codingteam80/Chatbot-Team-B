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
    DEBUG_RETRIEVAL
)


class CompanyRetriever:

    def __init__(self):

        self.bm25 = BM25Searcher()

        self.chroma = ChromaSearcher()

        self.hybrid = HybridRetriever()

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

        filtered = []

        file_count = {}

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

            if current_count >= max_chunks_per_file:
                continue

            file_count[file_name] = (
                current_count + 1
            )

            filtered.append(item)

        return filtered

    # =====================================
    # RETRIEVE
    # =====================================

    def retrieve(
        self,
        query
    ):

        bm25_results = (
            self.bm25.search(
                query,
                BM25_TOP_K
            )
        )

        vector_results = (
            self.chroma.search(
                query,
                VECTOR_TOP_K
            )
        )

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

        # =====================================
        # RERANK
        # =====================================

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

            print("\n===== FINAL RANKING =====")

            for item in reranked[:10]:

                score = item.get(
                    "score",
                    0.0
                )

                print(
                    f"{item['metadata']['file_name']} "
                    f"=> {score:.4f}"
                )

            print("=========================\n")

        # =====================================
        # DIVERSITY FILTER
        # =====================================

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
                    "score",
                    0.0
                )

                print(
                    f"{item['metadata']['file_name']} "
                    f"=> {score:.4f}"
                )

            print(
                "===============================\n"
            )

        return diversified

    # =====================================
    # BUILD CONTEXT
    # =====================================

    def build_context(
        self,
        query
    ):

        results = (
            self.retrieve(
                query
            )
        )

        results = results[
            :FINAL_TOP_K
        ]

        context = "\n\n".join(
            item["text"]
            for item in results
        )

        return context, results