import pickle
from pathlib import Path
import os

from rank_bm25 import BM25Okapi

from config.settings import BM25_DIR


CORPUS_FILE = BM25_DIR / "corpus.pkl"

INDEX_FILE = BM25_DIR / "bm25_index.pkl"


class BM25Indexer:

    @staticmethod
    def tokenize(text):

        return text.lower().split()

    def build(self, records):
 
        os.makedirs(
            BM25_DIR,
            exist_ok=True
        )

        corpus = [
            self.tokenize(
                record["text"]
            )
            for record in records
        ]

        bm25 = BM25Okapi(corpus)

        with open(
            CORPUS_FILE,
            "wb"
        ) as f:
            pickle.dump(
                records,
                f
            )

        with open(
            INDEX_FILE,
            "wb"
        ) as f:
            pickle.dump(
                bm25,
                f
            )

        print(
            f"BM25 index created: {len(records)} chunks"
        )


class BM25Searcher:

    def __init__(self):

        with open(
            INDEX_FILE,
            "rb"
        ) as f:
            self.bm25 = pickle.load(f)

        with open(
            CORPUS_FILE,
            "rb"
        ) as f:
            self.records = pickle.load(f)

    def search(
        self,
        query,
        top_k=20
    ):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(
            tokens
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked[:top_k]:

            record = self.records[idx]

            results.append(
                {
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "score": float(score)
                }
            )

        return results