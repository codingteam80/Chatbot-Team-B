from sentence_transformers import (
    CrossEncoder
)

from config.settings import (
    RERANKER_MODEL,
    RERANK_MAX_CHARS,
    RERANK_BATCH_SIZE,
    FINAL_TOP_K
)


class CrossEncoderReranker:

    def __init__(self):

        # PURPOSE: Load cross-encoder reranking model.
        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query,
        candidates
    ):

        # PURPOSE: Return empty results when no candidates exist.
        if not candidates:

            return []

        # PURPOSE: Create query-document pairs for reranking.
        pairs = [
            [
                query,
                item["text"][:RERANK_MAX_CHARS]
            ]
            for item in candidates
        ]

        # PURPOSE: Calculate reranker relevance scores.
        scores = (
            self.model.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE
            )
        )

        # PURPOSE: Store reranker scores in each candidate.
        for item, score in zip(
            candidates,
            scores
        ):

            item[
                "rerank_score"
            ] = float(score)

        # PURPOSE: Sort candidates by reranker score.
        candidates.sort(
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )

        # PURPOSE: Display reranking results for debugging.
        print(
            "\n===== FINAL RANKING ====="
        )

        for item in candidates[:10]:

            print(
                f"{item['metadata']['file_name']} "
                f"=> "
                f"{item['rerank_score']:.4f}"
            )

        print(
            "=========================\n"
        )

        # PURPOSE: Get best reranker score.
        best_score = candidates[
            0
        ]["rerank_score"]

        # PURPOSE: Reject weak retrieval results.
        if best_score < 0.01:

            return []

        # PURPOSE: Return top reranked chunks.
        return candidates[
            :FINAL_TOP_K
        ]