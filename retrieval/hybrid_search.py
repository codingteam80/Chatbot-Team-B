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
            default=0
        )

        # Prevent division by zero when all BM25 scores are zero.
        if max_bm25 <= 0:
            max_bm25 = 1

        # Add vector search results.
        #
        # Preserve the original channel/rank evidence. The previous merge
        # collapsed BM25 + vector into one score, so downstream code could not
        # tell whether both independent retrievers agreed on the same chunk.
        # These fields are metadata-only; the configured hybrid weights remain
        # unchanged.
        for vector_rank, item in enumerate(vector_results, start=1):

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
                    * VECTOR_WEIGHT,

                "_vector_score": float(item["score"]),
                "_vector_rank": vector_rank,
                "_bm25_score": None,
                "_bm25_normalized_score": 0.0,
                "_bm25_rank": None,
                "_retrieval_channels": ["vector"],
            }

        # Merge BM25 search results.
        for bm25_rank, item in enumerate(bm25_results, start=1):

            # Create unique chunk identifier.
            key = (
                item["metadata"]["file_name"]
                + "_"
                + str(
                    item["metadata"]["chunk_id"]
                )
            )

            # Normalize BM25 score to 0-1 range.
            if max_bm25 > 0:
                normalized_score = (
                    item["score"] / max_bm25
                )
            else:
                normalized_score = 0.0

            # Combine scores when chunk exists in both searches.
            if key in merged:

                merged[key]["score"] += (
                    normalized_score
                    * BM25_WEIGHT
                )

                merged[key]["_bm25_score"] = float(item["score"])
                merged[key]["_bm25_normalized_score"] = float(
                    normalized_score
                )
                merged[key]["_bm25_rank"] = bm25_rank
                channels = list(
                    merged[key].get("_retrieval_channels", [])
                )
                if "bm25" not in channels:
                    channels.append("bm25")
                merged[key]["_retrieval_channels"] = channels

            else:

                merged[key] = {

                    # Store chunk text.
                    "text": item["text"],

                    # Store chunk metadata.
                    "metadata": item["metadata"],

                    # Apply BM25 search weight.
                    "score":
                        normalized_score
                        * BM25_WEIGHT,

                    "_vector_score": None,
                    "_vector_rank": None,
                    "_bm25_score": float(item["score"]),
                    "_bm25_normalized_score": float(normalized_score),
                    "_bm25_rank": bm25_rank,
                    "_retrieval_channels": ["bm25"],
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