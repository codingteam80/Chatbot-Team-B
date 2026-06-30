import pickle
import os

from rank_bm25 import BM25Okapi

from config.settings import BM25_DIR


# Store original chunk records.
CORPUS_FILE = BM25_DIR / "corpus.pkl"

# Store trained BM25 index.
INDEX_FILE = BM25_DIR / "bm25_index.pkl"


class BM25Indexer:

    @staticmethod
    def tokenize(text):

        # Convert text into searchable tokens.
        return text.lower().split()

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


class BM25Searcher:

    def __init__(self):

        # Load BM25 index from storage.
        with open(
            INDEX_FILE,
            "rb"
        ) as f:
            self.bm25 = pickle.load(f)

        # Load original chunk records.
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

        # Tokenize the search query.
        tokens = query.lower().split()

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