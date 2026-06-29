from config.settings import (
    VECTOR_WEIGHT,
    BM25_WEIGHT
)


class HybridRetriever:

    def merge(
        self,
        vector_results,
        bm25_results
    ):

        # Store merged retrieval results.
        merged = {}

        # Get highest BM25 score for normalization.
        max_bm25 = max(
            [r["score"] for r in bm25_results],
            default=1
        )

        # Add vector search results.
        for item in vector_results:

            # Create unique chunk identifier.
            key = (
                item["metadata"]["file_name"]
                + "_"
                + str(
                    item["metadata"]["chunk_id"]
                )
            )

            merged[key] = {

                # Store chunk text.
                "text": item["text"],

                # Store chunk metadata.
                "metadata": item["metadata"],

                # Apply vector search weight.
                "score":
                    item["score"]
                    * VECTOR_WEIGHT
            }

        # Merge BM25 search results.
        for item in bm25_results:

            # Create unique chunk identifier.
            key = (
                item["metadata"]["file_name"]
                + "_"
                + str(
                    item["metadata"]["chunk_id"]
                )
            )

            # Normalize BM25 score to 0-1 range.
            normalized_score = (
                item["score"]
                / max_bm25
            )

            # Combine scores when chunk exists in both searches.
            if key in merged:

                merged[key]["score"] += (
                    normalized_score
                    * BM25_WEIGHT
                )

            else:

                merged[key] = {

                    # Store chunk text.
                    "text": item["text"],

                    # Store chunk metadata.
                    "metadata": item["metadata"],

                    # Apply BM25 search weight.
                    "score":
                        normalized_score
                        * BM25_WEIGHT
                }

        # Convert dictionary into a list.
        results = list(
            merged.values()
        )

        # Sort results by highest score first.
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # Return hybrid ranked results.
        return results