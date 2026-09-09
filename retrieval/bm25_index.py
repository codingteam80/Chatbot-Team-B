import os
import pickle
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

import streamlit as st

from rank_bm25 import BM25Okapi
from config.settings import BM25_DIR


CORPUS_FILE = BM25_DIR / "corpus.pkl"
INDEX_FILE = BM25_DIR / "bm25_index.pkl"


class BM25Indexer:

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return re.findall(r"[a-z0-9]+", text)

    @staticmethod
    def _write_pickle(path, value):
        with open(path, "wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

    def build(self, records):
        """Atomically rebuild BM25 from current Chroma records."""
        os.makedirs(BM25_DIR, exist_ok=True)

        if not records:
            bm25 = None
        else:
            corpus = [self.tokenize(record.get("text", "")) for record in records]
            bm25 = BM25Okapi(corpus)

        corpus_tmp = CORPUS_FILE.with_suffix(".pkl.tmp")
        index_tmp = INDEX_FILE.with_suffix(".pkl.tmp")
        corpus_backup = CORPUS_FILE.with_suffix(".pkl.bak")
        index_backup = INDEX_FILE.with_suffix(".pkl.bak")

        for path in (corpus_tmp, index_tmp, corpus_backup, index_backup):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        try:
            self._write_pickle(corpus_tmp, records)
            self._write_pickle(index_tmp, bm25)

            if CORPUS_FILE.exists():
                shutil.copy2(CORPUS_FILE, corpus_backup)
            if INDEX_FILE.exists():
                shutil.copy2(INDEX_FILE, index_backup)

            corpus_tmp.replace(CORPUS_FILE)
            index_tmp.replace(INDEX_FILE)
        except Exception:
            if corpus_backup.exists():
                corpus_backup.replace(CORPUS_FILE)
            if index_backup.exists():
                index_backup.replace(INDEX_FILE)
            raise
        finally:
            for path in (corpus_tmp, index_tmp, corpus_backup, index_backup):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        print(f"BM25 index created: {len(records)} chunks")


def snapshot_bm25_resources():
    """Create a temporary on-disk snapshot for a wider KB transaction."""
    BM25_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(
        tempfile.mkdtemp(prefix="docubot-bm25-backup-", dir=str(BM25_DIR.parent))
    )
    if CORPUS_FILE.exists():
        shutil.copy2(CORPUS_FILE, snapshot_dir / CORPUS_FILE.name)
    if INDEX_FILE.exists():
        shutil.copy2(INDEX_FILE, snapshot_dir / INDEX_FILE.name)
    return snapshot_dir


def restore_bm25_resources(snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    for active in (CORPUS_FILE, INDEX_FILE):
        backup = snapshot_dir / active.name
        if backup.exists():
            shutil.copy2(backup, active)
        else:
            active.unlink(missing_ok=True)


def discard_bm25_snapshot(snapshot_dir):
    try:
        shutil.rmtree(snapshot_dir)
    except FileNotFoundError:
        pass


@st.cache_resource(show_spinner=False)
def get_bm25_resources():
    print("[BM25] Loading index...")

    if not INDEX_FILE.exists() or not CORPUS_FILE.exists():
        print("[BM25] Index files not found. Using empty resources.")
        return None, []

    with open(INDEX_FILE, "rb") as file:
        bm25 = pickle.load(file)

    with open(CORPUS_FILE, "rb") as file:
        records = pickle.load(file)

    print("[BM25] Ready.")
    return bm25, records


class BM25Searcher:

    def __init__(self):
        self.bm25, self.records = get_bm25_resources()

    def search(self, query, top_k=20):
        if self.bm25 is None or not self.records:
            return []

        tokens = BM25Indexer.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results = []
        result_limit = min(top_k, len(self.records))

        for index, score in ranked[:result_limit]:
            record = self.records[index]
            results.append(
                {
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "score": float(score),
                }
            )

        return results
