import gzip
import json
import re
from pathlib import Path

from config.settings import (
    INDEX_SCHEMA_VERSION,
    INGESTION_CACHE_DIR,
    INGESTION_CACHE_SCHEMA_VERSION,
)


class IngestionCache:
    """Persistent cache for parsed/cleaned/prepared document chunks."""

    @staticmethod
    def _safe_token(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    @staticmethod
    def cache_path(file_hash):
        schema = IngestionCache._safe_token(INDEX_SCHEMA_VERSION)
        cache_schema = IngestionCache._safe_token(INGESTION_CACHE_SCHEMA_VERSION)
        return INGESTION_CACHE_DIR / f"{file_hash}.{schema}.{cache_schema}.json.gz"

    @staticmethod
    def load(file_hash):
        if not file_hash:
            return None

        cache_file = IngestionCache.cache_path(file_hash)
        if not cache_file.exists():
            return None

        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError, TypeError):
            # A damaged cache must never block indexing. Remove it and parse
            # the source document normally on this build.
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        if payload.get("file_hash") != file_hash:
            return None
        if payload.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            return None
        if payload.get("cache_schema_version") != INGESTION_CACHE_SCHEMA_VERSION:
            return None

        status = payload.get("status")
        if status not in {"ready", "skipped"}:
            return None

        return payload

    @staticmethod
    def save_ready(file_hash, prepared_chunks):
        IngestionCache._save(
            file_hash=file_hash,
            status="ready",
            prepared_chunks=prepared_chunks,
            skip_reason="",
        )

    @staticmethod
    def save_skipped(file_hash, skip_reason):
        IngestionCache._save(
            file_hash=file_hash,
            status="skipped",
            prepared_chunks=[],
            skip_reason=skip_reason or "no valid chunks",
        )

    @staticmethod
    def _save(file_hash, status, prepared_chunks, skip_reason):
        if not file_hash:
            return

        INGESTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = IngestionCache.cache_path(file_hash)
        temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")

        payload = {
            "file_hash": file_hash,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "cache_schema_version": INGESTION_CACHE_SCHEMA_VERSION,
            "status": status,
            "skip_reason": skip_reason,
            "prepared_chunks": prepared_chunks,
        }

        try:
            with gzip.open(temp_file, "wt", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
            temp_file.replace(cache_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    @staticmethod
    def prune(active_hashes):
        """Remove stale content hashes and obsolete cache-schema variants."""
        active_hashes = {value for value in active_hashes if value}
        if not INGESTION_CACHE_DIR.exists():
            return 0

        allowed_names = {
            IngestionCache.cache_path(file_hash).name
            for file_hash in active_hashes
        }

        removed = 0
        for cache_file in INGESTION_CACHE_DIR.glob("*.json.gz"):
            if cache_file.name in allowed_names:
                continue
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                # Cache cleanup is non-critical and should never fail a build.
                continue

        return removed
