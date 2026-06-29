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

        merged = {}

        max_bm25 = max(
            [r["score"] for r in bm25_results],
            default=1
        )

        for item in vector_results:

            key = (
                item["metadata"]["file_name"]
                + "_"
                + str(
                    item["metadata"]["chunk_id"]
                )
            )

            merged[key] = {

                "text": item["text"],

                "metadata": item["metadata"],

                "score":
                    item["score"]
                    * VECTOR_WEIGHT
            }

        for item in bm25_results:

            key = (
                item["metadata"]["file_name"]
                + "_"
                + str(
                    item["metadata"]["chunk_id"]
                )
            )

            normalized_score = (
                item["score"]
                / max_bm25
            )

            if key in merged:

                merged[key]["score"] += (
                    normalized_score
                    * BM25_WEIGHT
                )

            else:

                merged[key] = {

                    "text": item["text"],

                    "metadata": item["metadata"],

                    "score":
                        normalized_score
                        * BM25_WEIGHT
                }

        results = list(
            merged.values()
        )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results