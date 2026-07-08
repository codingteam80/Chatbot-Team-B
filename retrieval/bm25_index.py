import streamlit as st
import pickle
import os

from rank_bm25 import BM25Okapi
from config.settings import BM25_DIR
import re
import unicodedata

# Store original chunk records.
CORPUS_FILE = BM25_DIR / "corpus.pkl"

# Store trained BM25 index.
INDEX_FILE = BM25_DIR / "bm25_index.pkl"


class BM25Indexer:

    @staticmethod
    def tokenize(text):

        text = text.lower()

        # Remove accents (José -> Jose)
        text = unicodedata.normalize(
            "NFKD",
            text
        ).encode(
            "ascii",
            "ignore"
        ).decode(
            "ascii"
        )

        # Keep only letters and numbers
        return re.findall(
            r"[a-z0-9]+",
            text
        )

    def build(self, records):

        # Create BM25 storage directory.
        os.makedirs(
            BM25_DIR,
            exist_ok=True
        )

        # Tokenize all document chunks.
        corpus = [
            self.tokenize(
                record["text"]
            )
            for record in records
        ]

        # Build BM25 keyword index.
        bm25 = BM25Okapi(corpus)

        # Save original chunk records.
        with open(
            CORPUS_FILE,
            "wb"
        ) as f:
            pickle.dump(
                records,
                f
            )

        # Save BM25 index to disk.
        with open(
            INDEX_FILE,
            "wb"
        ) as f:
            pickle.dump(
                bm25,
                f
            )

        # Display indexing summary.
        print(
            f"BM25 index created: {len(records)} chunks"
        )


# ==========================================================
# GET BM25 INDEX
#
# Cache the BM25 index so it is loaded only once
# during the Streamlit session.
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_bm25_resources():

    print(
        "[BM25] Loading index..."
    )

    # Load BM25 index.
    with open(
        INDEX_FILE,
        "rb"
    ) as f:

        bm25 = pickle.load(f)

    # Load original records.
    with open(
        CORPUS_FILE,
        "rb"
    ) as f:

        records = pickle.load(f)

    print(
        "[BM25] Ready."
    )

    return bm25, records


class BM25Searcher:

    def __init__(self):

        # Reuse cached BM25 resources.
        self.bm25, self.records = (
            get_bm25_resources()
        )

    def search(
        self,
        query,
        top_k=20
    ):

        # Tokenize the search query.
        tokens = BM25Indexer.tokenize(
            query
        )
        # Calculate BM25 relevance scores.
        scores = self.bm25.get_scores(
            tokens
        )

        # Rank chunks by score.
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Store formatted search results.
        results = []

        # Collect top ranked chunks.
        for idx, score in ranked[:top_k]:

            record = self.records[idx]

            results.append(
                {
                    # Retrieved chunk text.
                    "text": record["text"],

                    # File and chunk metadata.
                    "metadata": record["metadata"],

                    # BM25 relevance score.
                    "score": float(score)
                }
            )

        # Return keyword search results.
        return results