from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config.settings import (
    BM25_DIR,
    EMBED_MODEL_NAME,
    METADATA_DIR,
    MIN_RETRIEVAL_SCORE,
    MULTI_QUERY_RRF_K,
)


SEMANTIC_RULE_INDEX_SCHEMA = "misra-semantic-rule-index-v1"
SEMANTIC_RULE_METADATA = METADATA_DIR / "misra_semantic_rule_index_v1.json"
SEMANTIC_RULE_VECTORS = METADATA_DIR / "misra_semantic_rule_index_v1.npz"
_BUILD_LOCK = threading.RLock()


@dataclass(frozen=True)
class RuleProfile:
    kind: str
    identifier: str
    display_name: str
    requirement_text: str
    profile_text: str
    file_name: str
    file_path: str
    page_start: Any
    page_end: Any
    supporting_roles: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_misra_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {}) or {}
    source = " ".join(
        str(metadata.get(key, "") or "")
        for key in ("file_name", "file_path", "parent_title", "section_title")
    ).casefold()
    return "misra" in source or str(metadata.get("chunking_profile", "")).casefold() == "v4_rule_aware"


def _record_reference(record: dict[str, Any]) -> tuple[str, str]:
    metadata = record.get("metadata", {}) or {}
    section_type = str(metadata.get("section_type", "") or "").strip().casefold()
    parent_type = str(metadata.get("parent_type", "") or "").strip().casefold()
    role = str(metadata.get("section_role", "") or "").strip().casefold()

    if section_type in {"rule", "directive"} and role in {"requirement", "statement", ""}:
        identifier = str(
            metadata.get("directive_id", "")
            or metadata.get("rule_id", "")
            or metadata.get("parent_identifier", "")
            or ""
        ).strip()
        return section_type, identifier

    if parent_type in {"rule", "directive"}:
        identifier = str(
            metadata.get("parent_identifier", "")
            or metadata.get("parent_directive_id", "")
            or metadata.get("parent_rule_id", "")
            or metadata.get("rule_id", "")
            or ""
        ).strip()
        return parent_type, identifier

    if metadata.get("parent_directive_id"):
        return "directive", str(metadata.get("parent_directive_id") or "").strip()
    if metadata.get("parent_rule_id"):
        return "rule", str(metadata.get("parent_rule_id") or "").strip()
    return "", ""


def _requirement_statement(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if re.match(r"^(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)*\b", lines[0], re.I):
        lines = lines[1:]
    parts: list[str] = []
    stop = re.compile(
        r"^(?:Category|Analysis|Applies to|Rationale|Amplification|Examples?|Exceptions?|Notes?|See also)\b",
        re.I,
    )
    for line in lines:
        if stop.match(line):
            break
        parts.append(line)
    return _normalize_space(" ".join(parts))


def extract_rule_profiles(records: Iterable[dict[str, Any]]) -> list[RuleProfile]:
    """Create document-derived Rule/Directive semantic profiles.

    No query phrase or Rule mapping is encoded here.  The profile is assembled
    only from the indexed structured requirement and its own child sections.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_support = {"rationale", "amplification", "example", "examples", "exception", "exceptions", "note", "notes"}

    for raw in records or []:
        if not isinstance(raw, dict) or not _is_misra_record(raw):
            continue
        kind, identifier = _record_reference(raw)
        if kind not in {"rule", "directive"} or not identifier:
            continue
        metadata = raw.get("metadata", {}) or {}
        role = str(metadata.get("section_role", "") or "").strip().casefold()
        section_type = str(metadata.get("section_type", "") or "").strip().casefold()
        bucket = grouped.setdefault(
            (kind, identifier),
            {"requirement": None, "support": [], "roles": set()},
        )
        if section_type in {"rule", "directive"} and role in {"requirement", "statement", ""}:
            bucket["requirement"] = raw
        elif role in allowed_support:
            bucket["support"].append(raw)
            bucket["roles"].add(role)

    profiles: list[RuleProfile] = []
    role_order = {"rationale": 0, "amplification": 1, "exception": 2, "exceptions": 2, "example": 3, "examples": 3, "note": 4, "notes": 4}
    role_caps = {"rationale": 1400, "amplification": 1400, "exception": 900, "exceptions": 900, "example": 1400, "examples": 1400, "note": 700, "notes": 700}

    for (kind, identifier), bucket in sorted(grouped.items(), key=lambda pair: (pair[0][0], tuple(int(x) if x.isdigit() else x for x in pair[0][1].split(".")))):
        requirement = bucket.get("requirement")
        if not isinstance(requirement, dict):
            continue
        metadata = requirement.get("metadata", {}) or {}
        requirement_text = _requirement_statement(requirement.get("text", ""))
        if not requirement_text:
            continue
        display = f"Directive {identifier}" if kind == "directive" else f"Rule {identifier}"
        pieces = [f"{display}. Requirement: {requirement_text}"]

        support_records = sorted(
            list(bucket.get("support") or []),
            key=lambda item: (
                role_order.get(str((item.get("metadata", {}) or {}).get("section_role", "")).casefold(), 99),
                int((item.get("metadata", {}) or {}).get("chunk_id", 0) or 0),
            ),
        )
        used_role_counts: dict[str, int] = {}
        for item in support_records:
            item_meta = item.get("metadata", {}) or {}
            role = str(item_meta.get("section_role", "") or "").strip().casefold()
            # One representative block per semantic role is enough for the
            # profile and keeps the embedding compact/predictable.
            if used_role_counts.get(role, 0) >= 1:
                continue
            text = _normalize_space(item.get("text", ""))
            if not text:
                continue
            cap = role_caps.get(role, 900)
            text = text[:cap].rsplit(" ", 1)[0] if len(text) > cap else text
            label = role.title()
            pieces.append(f"{label}: {text}")
            used_role_counts[role] = used_role_counts.get(role, 0) + 1

        profiles.append(
            RuleProfile(
                kind=kind,
                identifier=identifier,
                display_name=display,
                requirement_text=requirement_text,
                profile_text="\n".join(pieces),
                file_name=str(metadata.get("file_name", "") or ""),
                file_path=str(metadata.get("file_path", "") or ""),
                page_start=metadata.get("page_start"),
                page_end=metadata.get("page_end") or metadata.get("page_start"),
                supporting_roles=tuple(sorted(set(bucket.get("roles") or []))),
            )
        )
    return profiles


def _load_bm25_records() -> list[dict[str, Any]]:
    corpus = BM25_DIR / "corpus.pkl"
    if not corpus.is_file():
        return []
    with corpus.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _index_signature() -> dict[str, str]:
    corpus = BM25_DIR / "corpus.pkl"
    return {
        "schema": SEMANTIC_RULE_INDEX_SCHEMA,
        "embedding_model": EMBED_MODEL_NAME,
        "corpus_sha256": _sha256_file(corpus) if corpus.is_file() else "",
    }


def _metadata_matches(metadata: dict[str, Any], signature: dict[str, str]) -> bool:
    return all(str(metadata.get(key, "")) == str(value) for key, value in signature.items())


def semantic_rule_index_ready() -> bool:
    if not SEMANTIC_RULE_METADATA.is_file() or not SEMANTIC_RULE_VECTORS.is_file():
        return False
    try:
        metadata = json.loads(SEMANTIC_RULE_METADATA.read_text(encoding="utf-8"))
        if not _metadata_matches(metadata, _index_signature()):
            return False
        profiles = metadata.get("profiles") or []
        vectors = np.load(SEMANTIC_RULE_VECTORS, allow_pickle=False)["vectors"]
        return bool(profiles) and vectors.ndim == 2 and vectors.shape[0] == len(profiles)
    except Exception:
        return False


def build_semantic_rule_index(*, embedding_model=None, force: bool = False, batch_size: int = 16) -> dict[str, Any]:
    """Build/update the small derived Rule-level semantic index.

    The production chunk/Qdrant/BM25 KB is never modified.  Only a derived
    metadata cache is written under the existing production metadata folder.
    """

    with _BUILD_LOCK:
        signature = _index_signature()
        if not signature.get("corpus_sha256"):
            raise RuntimeError("BM25 corpus is unavailable; semantic Rule index cannot be built.")
        if not force and semantic_rule_index_ready():
            metadata = json.loads(SEMANTIC_RULE_METADATA.read_text(encoding="utf-8"))
            return {"built": False, "ready": True, "profile_count": len(metadata.get("profiles") or []), **signature}

        records = _load_bm25_records()
        profiles = extract_rule_profiles(records)
        if len(profiles) < 10:
            raise RuntimeError(f"Only {len(profiles)} structured Rule/Directive profiles were extracted; refusing to build an incomplete semantic index.")

        if embedding_model is None:
            from embeddings.embedding_model import get_embedding_model
            embedding_model = get_embedding_model()

        texts = [profile.profile_text for profile in profiles]
        rows: list[list[float]] = []
        batch_size = max(1, int(batch_size or 1))
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors = embedding_model.get_text_embedding_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding backend returned an incomplete semantic Rule batch.")
            rows.extend(vectors)

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(profiles) or matrix.shape[1] <= 0:
            raise RuntimeError("Semantic Rule embedding matrix has an invalid shape.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(~np.isfinite(matrix)) or np.any(norms <= 0):
            raise RuntimeError("Semantic Rule embeddings contain non-finite or zero vectors.")
        matrix = matrix / norms

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        metadata_payload = {
            **signature,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile_count": len(profiles),
            "vector_dimensions": int(matrix.shape[1]),
            "profiles": [
                {
                    "kind": profile.kind,
                    "identifier": profile.identifier,
                    "display_name": profile.display_name,
                    "requirement_text": profile.requirement_text,
                    "profile_text": profile.profile_text,
                    "file_name": profile.file_name,
                    "file_path": profile.file_path,
                    "page_start": profile.page_start,
                    "page_end": profile.page_end,
                    "supporting_roles": list(profile.supporting_roles),
                }
                for profile in profiles
            ],
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(METADATA_DIR), suffix=".json.tmp") as handle:
            json.dump(metadata_payload, handle, indent=2, ensure_ascii=False)
            metadata_tmp = Path(handle.name)
        fd, vector_tmp_name = tempfile.mkstemp(dir=str(METADATA_DIR), suffix=".npz.tmp")
        os.close(fd)
        vector_tmp = Path(vector_tmp_name)
        try:
            # np.savez_compressed appends .npz only when given a string path;
            # writing through a file handle preserves our atomic temp filename.
            with vector_tmp.open("wb") as handle:
                np.savez_compressed(handle, vectors=matrix)
            metadata_tmp.replace(SEMANTIC_RULE_METADATA)
            vector_tmp.replace(SEMANTIC_RULE_VECTORS)
        finally:
            metadata_tmp.unlink(missing_ok=True)
            vector_tmp.unlink(missing_ok=True)

        return {"built": True, "ready": True, "profile_count": len(profiles), "vector_dimensions": int(matrix.shape[1]), **signature}


class SemanticRuleResolver:
    """Rule-level semantic resolver derived entirely from indexed source text."""

    def __init__(self, *, embedding_model=None, reranker=None):
        self.embedding_model = embedding_model
        self.reranker = reranker
        self._metadata: dict[str, Any] | None = None
        self._vectors: np.ndarray | None = None
        # Small process-local caches remove repeated model work when an
        # ambiguous original query is widened with MultiQuery.  They are keyed
        # only by query/profile text and never persist user content to disk.
        self._query_vector_cache: dict[str, np.ndarray] = {}
        self._rerank_score_cache: dict[tuple[str, str, str], float] = {}

    def _ensure_loaded(self) -> None:
        if not semantic_rule_index_ready():
            build_semantic_rule_index(embedding_model=self.embedding_model)
        metadata = json.loads(SEMANTIC_RULE_METADATA.read_text(encoding="utf-8"))
        vectors = np.load(SEMANTIC_RULE_VECTORS, allow_pickle=False)["vectors"].astype(np.float32, copy=False)
        profiles = metadata.get("profiles") or []
        if vectors.ndim != 2 or vectors.shape[0] != len(profiles):
            raise RuntimeError("Semantic Rule index metadata/vector row count mismatch.")
        self._metadata = metadata
        self._vectors = vectors

    def _get_embedding_model(self):
        if self.embedding_model is None:
            from embeddings.embedding_model import get_embedding_model
            self.embedding_model = get_embedding_model()
        return self.embedding_model

    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed query variants once and reuse them within the process.

        The semantic resolver intentionally performs a single-query pass before
        deciding whether MultiQuery is needed.  Without this cache, the widened
        pass embedded the original query a second time.  On CPU-only machines
        that duplicated one of the most expensive stages.  Cache entries are
        normalized vectors and the cache is deliberately small/bounded.
        """

        model = self._get_embedding_model()
        dimension = int(self._vectors.shape[1])
        output: list[np.ndarray | None] = [None] * len(queries)
        missing_indexes: list[int] = []
        missing_queries: list[str] = []

        for index, query in enumerate(queries):
            key = _normalize_space(query).casefold()
            cached = self._query_vector_cache.get(key)
            if isinstance(cached, np.ndarray) and cached.ndim == 1 and cached.shape[0] == dimension:
                output[index] = cached
            else:
                missing_indexes.append(index)
                missing_queries.append(query)

        if missing_queries:
            batch = getattr(model, "get_query_embedding_batch", None)
            if callable(batch):
                rows = batch(missing_queries)
            else:
                rows = [model.get_query_embedding(query) for query in missing_queries]
            matrix = np.asarray(rows, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(missing_queries) or matrix.shape[1] != dimension:
                raise RuntimeError("Query embeddings are incompatible with the semantic Rule index.")
            if np.any(~np.isfinite(matrix)):
                raise RuntimeError("Query embeddings contain non-finite values.")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise RuntimeError("Query embedding has zero norm.")
            matrix = matrix / norms

            for target_index, query, row in zip(missing_indexes, missing_queries, matrix):
                row = np.asarray(row, dtype=np.float32)
                output[target_index] = row
                key = _normalize_space(query).casefold()
                if len(self._query_vector_cache) >= 128:
                    oldest = next(iter(self._query_vector_cache), None)
                    if oldest is not None:
                        self._query_vector_cache.pop(oldest, None)
                self._query_vector_cache[key] = row

        if any(row is None for row in output):
            raise RuntimeError("Query embedding cache returned an incomplete batch.")
        return np.vstack(output).astype(np.float32, copy=False)

    def _rerank_rule_candidates(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rerank compact Rule requirements with process-local score reuse.

        Embeddings intentionally use the richer requirement+rationale/example
        profile for recall.  The BGE precision stage, however, compares the
        question against the concise authoritative requirement body.  This
        avoids long profile text diluting the CrossEncoder signal and also
        makes repeated single-query -> MultiQuery widening cheaper.
        """

        if not candidates:
            return []
        if self.reranker is None:
            ranked = [dict(item) for item in candidates]
            for item in ranked:
                item["rerank_score"] = float(item.get("semantic_score", 0.0) or 0.0)
            return sorted(ranked, key=lambda item: item.get("rerank_score", 0.0), reverse=True)

        query_key = _normalize_space(query).casefold()
        prepared: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        key_by_reference: dict[str, tuple[str, str, str]] = {}

        for item in candidates:
            clone = dict(item)
            profile = dict(clone.get("profile") or {})
            reference = str(profile.get("display_name", "") or "")
            requirement = _normalize_space(profile.get("requirement_text", ""))
            requirement_hash = hashlib.sha256(requirement.encode("utf-8")).hexdigest()[:16]
            cache_key = (query_key, reference.casefold(), requirement_hash)
            key_by_reference[reference] = cache_key
            clone["text"] = f"{reference}. {requirement}".strip()
            cached = self._rerank_score_cache.get(cache_key)
            if cached is None:
                missing.append(clone)
            else:
                clone["rerank_score"] = float(cached)
            prepared.append(clone)

        if missing:
            ranked_missing = self.reranker.rerank(query, missing)
            for item in ranked_missing:
                profile = dict(item.get("profile") or {})
                reference = str(profile.get("display_name", "") or "")
                cache_key = key_by_reference.get(reference)
                if cache_key is None:
                    continue
                score = float(item.get("rerank_score", 0.0) or 0.0)
                if len(self._rerank_score_cache) >= 256:
                    oldest = next(iter(self._rerank_score_cache), None)
                    if oldest is not None:
                        self._rerank_score_cache.pop(oldest, None)
                self._rerank_score_cache[cache_key] = score

        for item in prepared:
            if "rerank_score" in item:
                continue
            profile = dict(item.get("profile") or {})
            reference = str(profile.get("display_name", "") or "")
            cache_key = key_by_reference.get(reference)
            item["rerank_score"] = float(self._rerank_score_cache.get(cache_key, 0.0) or 0.0)

        return sorted(prepared, key=lambda item: item.get("rerank_score", 0.0), reverse=True)

    def resolve(
        self,
        query: str,
        *,
        query_variants: Iterable[str] | None = None,
        semantic_top_k: int = 8,
        rerank_top_k: int = 5,
    ) -> dict[str, Any]:
        original = _normalize_space(query)
        if not original:
            return {"status": "no_query", "accepted": False, "candidates": []}

        queries: list[str] = []
        seen = set()
        for value in [original, *(query_variants or [])]:
            clean = _normalize_space(value)
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                queries.append(clean)
        if not queries:
            return {"status": "no_query", "accepted": False, "candidates": []}

        try:
            self._ensure_loaded()
            q_matrix = self._embed_queries(queries)
            similarities = q_matrix @ self._vectors.T
            profile_count = similarities.shape[1]

            if len(queries) == 1:
                semantic_values = similarities[0].astype(np.float64, copy=False)
                hit_counts = np.ones(profile_count, dtype=np.int32)
            else:
                # Same rank-only RRF principle as the production MultiQuery path.
                # This fuses Rule profiles across rewrites without pretending raw
                # cosine scores from different query formulations share a perfect
                # calibrated scale.
                rrf = np.zeros(profile_count, dtype=np.float64)
                hit_counts = np.zeros(profile_count, dtype=np.int32)
                depth = min(max(20, int(semantic_top_k) * 4), profile_count)
                rrf_k = float(MULTI_QUERY_RRF_K)
                for row in similarities:
                    order = np.argsort(-row)[:depth]
                    for rank, idx in enumerate(order.tolist(), start=1):
                        rrf[idx] += 1.0 / (rrf_k + float(rank))
                        hit_counts[idx] += 1
                maximum = float(np.max(rrf)) if rrf.size else 0.0
                semantic_values = rrf / maximum if maximum > 0 else rrf

            semantic_top_k = max(2, min(int(semantic_top_k or 8), profile_count))
            indexes = np.argpartition(-semantic_values, semantic_top_k - 1)[:semantic_top_k]
            indexes = indexes[np.argsort(-semantic_values[indexes])]
            profiles = self._metadata.get("profiles") or []
            candidates = []
            for semantic_rank, idx in enumerate(indexes.tolist(), start=1):
                profile = dict(profiles[idx])
                candidates.append({
                    "text": str(profile.get("profile_text", "") or ""),
                    "metadata": {
                        "file_name": str(profile.get("file_name", "") or "MISRA semantic Rule index"),
                        "file_path": str(profile.get("file_path", "") or ""),
                        "section_type": str(profile.get("kind", "rule") or "rule"),
                        "section_title": str(profile.get("display_name", "") or ""),
                        "rule_id": str(profile.get("identifier", "") or ""),
                        "page_start": profile.get("page_start"),
                        "page_end": profile.get("page_end"),
                    },
                    "semantic_score": float(semantic_values[idx]),
                    "best_similarity": float(np.max(similarities[:, idx])),
                    "multi_query_hits": int(hit_counts[idx]),
                    "semantic_rank": semantic_rank,
                    "profile": profile,
                })

            to_rerank = candidates[: max(2, min(int(rerank_top_k or 5), len(candidates)))]
            ranked = self._rerank_rule_candidates(original, to_rerank)

            bge_top = ranked[0] if ranked else None
            bge_second = ranked[1] if len(ranked) > 1 else None
            if not bge_top:
                return {"status": "no_candidates", "accepted": False, "candidates": []}

            # The semantic leader is the Rule profile selected by corpus-derived
            # embedding/RRF recall before the CrossEncoder.  BGE remains an
            # independent precision signal, but its absolute 0.55 chunk score is
            # not treated as a universal calibration for short Rule statements.
            # v6.5.1 field evidence showed that correct Rule leaders could receive
            # very small absolute BGE scores in Tagalog/Taglish while remaining
            # stable across all three semantic queries.  Use multi-signal
            # agreement instead of lowering the production chunk threshold.
            semantic_leader = min(
                ranked,
                key=lambda item: int(item.get("semantic_rank", 999) or 999),
            )
            semantic_runner = next(
                (
                    item for item in sorted(
                        ranked,
                        key=lambda item: int(item.get("semantic_rank", 999) or 999),
                    )
                    if int(item.get("semantic_rank", 999) or 999) == 2
                ),
                None,
            )

            bge_top_score = float(bge_top.get("rerank_score", 0.0) or 0.0)
            bge_second_score = float((bge_second or {}).get("rerank_score", 0.0) or 0.0)
            bge_margin = bge_top_score - bge_second_score

            leader_score = float(semantic_leader.get("rerank_score", 0.0) or 0.0)
            leader_similarity = float(semantic_leader.get("best_similarity", 0.0) or 0.0)
            runner_similarity = float((semantic_runner or {}).get("best_similarity", -1.0) or -1.0)
            similarity_margin = leader_similarity - runner_similarity
            semantic_score = float(semantic_leader.get("semantic_score", 0.0) or 0.0)
            semantic_second = float(candidates[1].get("semantic_score", 0.0) or 0.0) if len(candidates) > 1 else -1.0
            semantic_margin = semantic_score - semantic_second
            leader_rank_in_bge = next(
                (index for index, item in enumerate(ranked, start=1) if item is semantic_leader),
                999,
            )
            bge_disagreement = max(0.0, bge_top_score - leader_score)
            full_multi_query_consensus = (
                len(queries) <= 1
                or int(semantic_leader.get("multi_query_hits", 0) or 0) >= len(queries)
            )

            selected = bge_top
            accepted = False
            acceptance_basis = ""

            # Preserve the original strict BGE acceptance path when it is
            # decisive.  This path still uses the unchanged global 0.55 floor.
            bge_semantic_rank = int(bge_top.get("semantic_rank", 999) or 999)
            if bge_top_score >= float(MIN_RETRIEVAL_SCORE) and bge_semantic_rank <= 3:
                if bge_top_score >= 0.82 and bge_margin >= 0.025:
                    accepted = True
                elif bge_top_score >= 0.68 and bge_margin >= 0.060:
                    accepted = True
                elif bge_top_score >= float(MIN_RETRIEVAL_SCORE) and bge_margin >= 0.100 and bge_semantic_rank == 1:
                    accepted = True
                elif bge_semantic_rank == 1 and semantic_margin >= 0.035 and bge_top_score >= 0.62 and bge_margin >= 0.040:
                    accepted = True
                if accepted:
                    acceptance_basis = "strict_bge"

            # v6.5.2 semantic-consensus gate.  This does NOT change
            # MIN_RETRIEVAL_SCORE for ordinary chunks.  It is specific to the
            # 173 source-derived Rule profiles and is followed by an independent
            # authoritative Rule-body lookup before any answer is finalized.
            # The Rule must lead semantically with a real cosine margin, remain
            # present in every MultiQuery arm, and either agree with BGE or lose
            # to the BGE leader by only a very small amount.
            if not accepted:
                semantic_consensus = bool(
                    len(queries) > 1
                    and int(semantic_leader.get("semantic_rank", 999) or 999) == 1
                    and leader_similarity >= 0.68
                    and similarity_margin >= 0.025
                    and full_multi_query_consensus
                    and leader_rank_in_bge <= 2
                    and leader_score >= 0.020
                    and (
                        bge_top is semantic_leader
                        or (
                            bge_disagreement <= 0.035
                            and leader_similarity
                            - float(bge_top.get("best_similarity", 0.0) or 0.0)
                            >= 0.020
                        )
                    )
                )
                if semantic_consensus:
                    selected = semantic_leader
                    accepted = True
                    acceptance_basis = "semantic_consensus"

            top = selected
            top_rerank = float(top.get("rerank_score", 0.0) or 0.0)
            competing_scores = [
                float(item.get("rerank_score", 0.0) or 0.0)
                for item in ranked
                if item is not top
            ]
            second_rerank = max(competing_scores) if competing_scores else 0.0
            rerank_margin = top_rerank - second_rerank
            semantic_rank = int(top.get("semantic_rank", 999) or 999)

            compact = [
                {
                    "reference": str((item.get("profile") or {}).get("display_name", "")),
                    "semantic_rank": int(item.get("semantic_rank", 0) or 0),
                    "semantic_score": round(float(item.get("semantic_score", 0.0) or 0.0), 6),
                    "best_similarity": round(float(item.get("best_similarity", 0.0) or 0.0), 6),
                    "multi_query_hits": int(item.get("multi_query_hits", 0) or 0),
                    "rerank_score": round(float(item.get("rerank_score", 0.0) or 0.0), 6),
                }
                for item in ranked[:5]
            ]
            return {
                "status": "accepted" if accepted else "ambiguous",
                "accepted": accepted,
                "reference": str((top.get("profile") or {}).get("display_name", "")) if accepted else "",
                "kind": str((top.get("profile") or {}).get("kind", "")) if accepted else "",
                "identifier": str((top.get("profile") or {}).get("identifier", "")) if accepted else "",
                "top_rerank_score": top_rerank,
                "rerank_margin": rerank_margin,
                "semantic_score": semantic_score,
                "semantic_margin": semantic_margin,
                "similarity_margin": similarity_margin,
                "acceptance_basis": acceptance_basis,
                "bge_top_reference": str((bge_top.get("profile") or {}).get("display_name", "")),
                "bge_top_score": bge_top_score,
                "bge_disagreement": bge_disagreement,
                "full_multi_query_consensus": full_multi_query_consensus,
                "query_count": len(queries),
                "multi_query_fusion": "RRF" if len(queries) > 1 else "single-query cosine",
                "candidates": compact,
            }
        except Exception as error:
            return {
                "status": "unavailable",
                "accepted": False,
                "error": f"{type(error).__name__}: {error}",
                "query_count": len(queries),
                "candidates": [],
            }

