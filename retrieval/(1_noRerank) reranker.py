import streamlit as st
import torch

from transformers import (
    AutoModelForSequenceClassification
)

from config.settings import (
    RERANKER_MODEL,
    FINAL_TOP_K
)

print("USING reranker.py")


# ==========================================================
# GET RERANKER MODEL
#
# Jina reranker uses a custom model implementation.
# trust_remote_code=True is required.
#
# use_flash_attn=False is safer for local CPU / non-CUDA setup.
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_reranker_model():

    print(
        f"[RERANKER] Loading model: "
        f"{RERANKER_MODEL}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        trust_remote_code=True,
        torch_dtype="auto",
        use_flash_attn=False
    )

    model.to(device)

    model.eval()

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
        self.model = (
            get_reranker_model()
        )

    def rerank(
        self,
        query,
        candidates
    ):

        # Return empty results when no candidates exist.
        if not candidates:

            return []

        # Build query-document pairs.
        pairs = [

            [
                query,
                item["text"]
            ]

            for item in candidates
        ]

        print("\n===== RERANK INPUT =====")

        for index, item in enumerate(candidates):

            print(
                index,
                item["metadata"].get(
                    "file_name",
                    "Unknown"
                )
            )

        # Jina official scoring method.
        scores = self.model.compute_score(
            pairs,
            max_length=1024
        )

        print("\n===== RAW RERANK SCORES =====")

        for index, (item, score) in enumerate(
            zip(candidates, scores)
        ):

            print(
                f"{index} | "
                f"{item['metadata'].get('file_name', 'Unknown')} | "
                f"{score}"
            )

        print("=============================\n")

        # Attach scores.
        for item, score in zip(
            candidates,
            scores
        ):

            item["rerank_score"] = float(score)

        # Sort by highest reranker score.
        candidates.sort(
            key=lambda item:
            item.get(
                "rerank_score",
                0.0
            ),
            reverse=True
        )

        print("\n===== FINAL RANKING =====")

        for item in candidates[:10]:

            print(
                f"{item['metadata'].get('file_name', 'Unknown')} "
                f"=> "
                f"{item.get('rerank_score', 0.0):.4f}"
            )

        print("=========================\n")

        # Return top reranked chunks.
        return candidates[:FINAL_TOP_K]