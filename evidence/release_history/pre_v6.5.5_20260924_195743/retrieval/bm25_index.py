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
        self._coverage_cache = None

    @staticmethod
    def _light_stem(token):
        """Return a conservative morphology key for lexical coverage.

        This is intentionally tiny and dependency-free.  It exists only to
        make obvious local variants such as ``password``/``passwords``,
        ``patch``/``patched`` and ``change``/``changed`` comparable during
        lexical candidate discovery.  It is not used for answer generation.
        """
        token = str(token or "").strip().lower()
        if len(token) <= 3:
            return token

        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 6:
            base = token[:-3]
            if len(base) >= 3:
                return base
        if token.endswith("ed") and len(token) > 5:
            base = token[:-2]
            if base.endswith("i"):
                base = base[:-1] + "y"
            return base
        if token.endswith("es") and len(token) > 5:
            return token[:-2]
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        # Make base forms such as change/require comparable with the -ed
        # variants above (changed -> chang, required -> requir).
        if token.endswith("e") and len(token) > 4:
            return token[:-1]
        return token

    @classmethod
    def _coverage_stop_words(cls):
        # Keep this list intentionally language-light and domain-neutral.
        # We remove grammatical/filler words only; business/technical terms
        # such as policy, standard, employee, server, approval, etc. remain
        # eligible evidence.
        return {
            "a", "an", "the", "of", "to", "for", "in", "on", "at",
            "and", "or", "with", "from", "by", "as", "is", "are", "was",
            "were", "be", "been", "being", "do", "does", "did", "has",
            "have", "had", "can", "could", "should", "would", "will",
            "what", "which", "who", "when", "where", "why", "how",
            "give", "show", "return", "state", "tell", "please", "only",
            "all", "any", "this", "that", "these", "those", "it", "its",
            "often", "under", "must", "every", "according",
            "ang", "ng", "mga", "sa", "ay", "at", "na", "nang", "para",
            "ano", "anong", "sino", "alin", "kailan", "saan", "bakit",
            "paano", "gaano", "dapat", "ayon", "lahat", "ito", "iyon",
            "nito", "niya", "nila", "ba", "din", "rin",
        }

    @classmethod
    def _meaningful_coverage_tokens(cls, text):
        tokens = BM25Indexer.tokenize(text)
        stop_words = cls._coverage_stop_words()
        output = []
        seen = set()
        for token in tokens:
            if len(token) <= 2 or token in stop_words:
                continue
            key = cls._light_stem(token)
            if len(key) <= 2 or key in stop_words or key in seen:
                continue
            seen.add(key)
            output.append(key)
        return output

    def _build_coverage_cache(self):
        if self._coverage_cache is not None:
            return self._coverage_cache

        cached = []
        for record in self.records or []:
            text = str(record.get("text", "") or "")
            metadata = record.get("metadata", {}) or {}
            file_name = str(metadata.get("file_name", "") or "")
            section_title = str(metadata.get("section_title", "") or "")
            document_tokens = set(self._meaningful_coverage_tokens(text))
            title_tokens = set(
                self._meaningful_coverage_tokens(
                    f"{file_name} {section_title}"
                )
            )
            cached.append((document_tokens, title_tokens))

        self._coverage_cache = cached
        return cached

    def coverage_search(self, query, top_k=12, minimum_score=0.34):
        """Lightweight corpus-wide lexical coverage search.

        BM25 can under-rank a short exact company record when a very large
        document contains many generic query words.  This second lexical view
        scans the already-loaded BM25 records and ranks by *coverage of the
        user's meaningful terms*, including filename/section-title evidence.
        It uses no model and no new index, so it is suitable for low-spec PCs.

        The result is only a candidate-discovery signal; normal downstream
        grounding/reranker safeguards still decide whether it reaches context.
        """
        if not self.records:
            return []

        query_tokens = self._meaningful_coverage_tokens(query)
        if not query_tokens:
            return []

        # A one-token query is too broad for a corpus-wide rescue unless the
        # token also appears in the source title.  The scoring below naturally
        # enforces that by requiring a higher minimum for single-token input.
        query_set = set(query_tokens)
        cache = self._build_coverage_cache()
        scored = []

        for index, record in enumerate(self.records):
            document_tokens, title_tokens = cache[index]
            if not document_tokens and not title_tokens:
                continue

            matched_doc = query_set.intersection(document_tokens)
            matched_title = query_set.intersection(title_tokens)
            matched = matched_doc.union(matched_title)
            matched_count = len(matched)

            if matched_count == 0:
                continue
            if len(query_tokens) >= 3 and matched_count < 2:
                continue

            body_coverage = len(matched_doc) / len(query_tokens)
            title_coverage = len(matched_title) / len(query_tokens)
            total_coverage = matched_count / len(query_tokens)

            # Reward a source-title agreement without letting filename text
            # alone dominate the actual document content.
            score = (
                total_coverage * 0.68
                + body_coverage * 0.22
                + title_coverage * 0.10
            )

            if len(query_tokens) == 1:
                if not matched_title:
                    continue
                score = max(score, 0.55)

            if score < minimum_score:
                continue

            scored.append((score, index, matched, matched_title))

        scored.sort(
            key=lambda item: (
                item[0],
                len(item[2]),
                len(item[3]),
            ),
            reverse=True,
        )

        results = []
        for rank, (score, index, matched, matched_title) in enumerate(
            scored[: max(1, int(top_k))],
            start=1,
        ):
            record = self.records[index]
            results.append(
                {
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "score": float(score),
                    "_lexical_coverage_score": float(score),
                    "_lexical_coverage_rank": rank,
                    "_lexical_matched_tokens": sorted(matched),
                    "_lexical_title_tokens": sorted(matched_title),
                }
            )

        return results

    def meaningful_token_presence(self, query):
        """Return corpus presence for the query's meaningful lexical tokens.

        This is intentionally a *presence* signal, not a relevance score.  It
        reuses the already-loaded BM25 corpus and the same conservative
        language-light token normalization used by ``coverage_search``.  The
        Phase-7 OOD identity fast-fail uses it only to prove that a requested
        identity has no lexical footprint anywhere in the active company
        corpus before deciding to skip the expensive embedding/vector/reranker
        path.

        A token appearing in either document text or source title counts as
        present.  Callers must treat any presence as a reason to continue with
        the normal certified retrieval pipeline.
        """
        query_tokens = self._meaningful_coverage_tokens(query)
        if not query_tokens or not self.records:
            return {
                "query_tokens": list(query_tokens),
                "present_tokens": [],
                "missing_tokens": list(query_tokens),
            }

        wanted = set(query_tokens)
        present = set()
        for document_tokens, title_tokens in self._build_coverage_cache():
            present.update(wanted.intersection(document_tokens))
            present.update(wanted.intersection(title_tokens))
            if present == wanted:
                break

        return {
            "query_tokens": list(query_tokens),
            "present_tokens": sorted(present),
            "missing_tokens": sorted(wanted.difference(present)),
        }

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
