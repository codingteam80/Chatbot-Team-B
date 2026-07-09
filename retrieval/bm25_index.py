import os
import pickle
import re
import unicodedata

import streamlit as st

from rank_bm25 import BM25Okapi
from config.settings import BM25_DIR


CORPUS_FILE = (
    BM25_DIR / "corpus.pkl"
)

INDEX_FILE = (
    BM25_DIR / "bm25_index.pkl"
)


class BM25Indexer:

    @staticmethod
    def tokenize(text):

        text = text.lower()

        text = unicodedata.normalize(
            "NFKD",
            text
        ).encode(
            "ascii",
            "ignore"
        ).decode(
            "ascii"
        )

        return re.findall(
            r"[a-z0-9]+",
            text
        )

    def build(
        self,
        records
    ):

        """
        BM25 is rebuilt from the current Chroma records.

        This operation is fast compared with document parsing and
        embedding, so unchanged documents are reused without being
        embedded again.
        """

        os.makedirs(
            BM25_DIR,
            exist_ok=True
        )

        if not records:

            # Keep valid empty resources so retrieval can safely
            # return no BM25 results after all files are deleted.
            bm25 = None

        else:

            corpus = [
                self.tokenize(
                    record.get(
                        "text",
                        ""
                    )
                )
                for record in records
            ]

            bm25 = BM25Okapi(
                corpus
            )

        with open(
            CORPUS_FILE,
            "wb"
        ) as file:

            pickle.dump(
                records,
                file
            )

        with open(
            INDEX_FILE,
            "wb"
        ) as file:

            pickle.dump(
                bm25,
                file
            )

        print(
            f"BM25 index created: "
            f"{len(records)} chunks"
        )


@st.cache_resource(
    show_spinner=False
)
def get_bm25_resources():

    print(
        "[BM25] Loading index..."
    )

    if (
        not INDEX_FILE.exists()
        or not CORPUS_FILE.exists()
    ):

        print(
            "[BM25] Index files not found. "
            "Using empty resources."
        )

        return None, []

    with open(
        INDEX_FILE,
        "rb"
    ) as file:

        bm25 = pickle.load(
            file
        )

    with open(
        CORPUS_FILE,
        "rb"
    ) as file:

        records = pickle.load(
            file
        )

    print(
        "[BM25] Ready."
    )

    return bm25, records


class BM25Searcher:

    def __init__(self):

        self.bm25, self.records = (
            get_bm25_resources()
        )

    def search(
        self,
        query,
        top_k=20
    ):

        if (
            self.bm25 is None
            or not self.records
        ):

            return []

        tokens = BM25Indexer.tokenize(
            query
        )

        if not tokens:

            return []

        scores = self.bm25.get_scores(
            tokens
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        result_limit = min(
            top_k,
            len(self.records)
        )

        for index, score in ranked[
            :result_limit
        ]:

            record = self.records[
                index
            ]

            results.append(
                {
                    "text": record["text"],
                    "metadata": (
                        record["metadata"]
                    ),
                    "score": float(score)
                }
            )

        return results
