print("USING reranker.py")
from sentence_transformers import (
    CrossEncoder
)

from config.settings import (
    RERANKER_MODEL,
    FINAL_TOP_K
)


class CrossEncoderReranker:

    def __init__(self):

        # Load cross-encoder reranking model.
        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query,
        candidates
    ):

        # Return empty results when no candidates exist.
        if not candidates:

            return []

        # Create query-document pairs for reranking.
        pairs = [
            [query, item["text"]]
            for item in candidates
        ]

        print("\n===== RERANK INPUT =====")
        for i, item in enumerate(candidates):

            print(
                i,
                item["metadata"]["file_name"]
            )

        # Calculate relevance scores for each pair.
        scores = (
            self.model.predict(
                pairs
            )
        )

        # Attach reranker scores to candidates.
        for item, score in zip(
            candidates,
            scores
        ):

            item[
                "rerank_score"
            ] = float(score)

        # Sort candidates by reranker score.
        candidates.sort(
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )

        # Display reranking results for debugging.
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

        # Get highest reranker score.
        #best_score = candidates[0]["rerank_score"]

        # Reject results with very low relevance.
        #if best_score < 0.01:
        #    return []

        # Return the top reranked chunks.
        return candidates[:FINAL_TOP_K]