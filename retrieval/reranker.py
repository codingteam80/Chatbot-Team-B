import math

import streamlit as st
import torch

from sentence_transformers import CrossEncoder

from config.settings import (
    RERANKER_MODEL
)


print("USING reranker.py")


# ==========================================================
# GET RERANKER MODEL
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_reranker_model():

    """
    Load and cache the reranker model.

    This uses sentence-transformers CrossEncoder,
    which is compatible with:
        BAAI/bge-reranker-base

    The model is cached so Streamlit does not reload it
    on every rerun.
    """

    print(
        f"[RERANKER] Loading model: "
        f"{RERANKER_MODEL}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = CrossEncoder(
        RERANKER_MODEL,
        device=device,
        max_length=512,
        # Keep a stable 0..1 confidence scale across BGE reranker variants so
        # MIN_RETRIEVAL_SCORE remains meaningful after the v2-m3 migration.
        activation_fn=torch.nn.Sigmoid(),
    )

    print(
        f"[RERANKER] Model loaded successfully on {device}"
    )

    return model


# ==========================================================
# CROSS ENCODER RERANKER
# ==========================================================

class CrossEncoderReranker:

    def __init__(self):

        # Reuse cached model.
        self.model = get_reranker_model()

    def rerank(
        self,
        query,
        candidates
    ):

        """
        Rerank candidate chunks using query-document relevance.

        Input:
            query      - final retrieval query
            candidates - chunks from hybrid retrieval

        Output:
            candidates sorted by rerank_score descending
        """

        if not candidates:

            return []

        pairs = []

        for item in candidates:

            text = item.get(
                "text",
                ""
            )

            # Keep reranking input compact for speed.
            text = text[:3000]

            pairs.append(
                [
                    query,
                    text
                ]
            )

        print("\n===== RERANK INPUT =====")

        for index, item in enumerate(candidates):

            print(
                index,
                item["metadata"].get(
                    "file_name",
                    "Unknown"
                )
            )

        raw_scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False
        )

        # CrossEncoder inference can rarely produce NaN/Inf on some local
        # CPU/model combinations. A non-finite value must never enter Python
        # sorting or the confidence gate because NaN comparisons are not
        # ordered and can move unrelated candidates ahead of valid evidence.
        # Retry only the affected pair at batch size 1. If it is still
        # non-finite, convert it to a safe finite rejection score (0.0).
        scores = []

        for index, raw_score in enumerate(raw_scores):

            score = float(raw_score)

            if math.isfinite(score):
                scores.append(score)
                continue

            item = candidates[index]
            file_name = item.get(
                "metadata",
                {}
            ).get(
                "file_name",
                "Unknown"
            )

            print(
                "[RERANKER] Non-finite score detected for "
                f"{file_name}; retrying this candidate with batch_size=1."
            )

            retry_score = float("nan")

            try:
                retry_scores = self.model.predict(
                    [pairs[index]],
                    batch_size=1,
                    show_progress_bar=False
                )

                retry_score = float(retry_scores[0])

            except Exception as error:
                retry_score = float("nan")

                print(
                    "[RERANKER] Single-candidate retry failed safely: "
                    f"{type(error).__name__}."
                )

            if math.isfinite(retry_score):
                score = retry_score
                item["_rerank_nonfinite_recovered"] = True

                print(
                    "[RERANKER] Non-finite score recovered with a finite "
                    f"retry score: {score:.4f}."
                )

            else:
                score = 0.0
                item["_rerank_nonfinite_safe_rejected"] = True

                print(
                    "[RERANKER] Retry remained non-finite; assigning the "
                    "safe rejection score 0.0000."
                )

            scores.append(score)

        print("\n===== RAW RERANK SCORES =====")

        for index, (item, score) in enumerate(
            zip(candidates, scores)
        ):

            print(
                f"{index} | "
                f"{item['metadata'].get('file_name', 'Unknown')} | "
                f"{score:.4f}"
            )

        print("=============================\n")

        for item, score in zip(
            candidates,
            scores
        ):

            item["rerank_score"] = score

        candidates.sort(
            key=lambda item:
                item.get(
                    "rerank_score",
                    0.0
                ),
            reverse=True
        )

        print("\n===== FINAL RERANKING =====")

        for item in candidates[:10]:

            print(
                f"{item['metadata'].get('file_name', 'Unknown')} "
                f"=> rerank={item.get('rerank_score', 0.0):.4f} "
                f"hybrid={item.get('score', 0.0):.4f} "
                f"info={item.get('_info_score', 0.0):.4f}"
            )

        print("===========================\n")

        # Important:
        # Return all sorted candidates.
        # retriever.py will decide final_top_k.
        return candidates
