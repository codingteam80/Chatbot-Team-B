import json
import os
import re
from pathlib import Path
from datetime import datetime

from config.settings import (
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DOCUMENT_DIR,
    EMBED_MODEL_NAME,
    INDEX_SCHEMA_VERSION,
    METADATA_DIR,
)
from utils.hash_utils import FileHasher


MANIFEST_FILE = METADATA_DIR / "manifest.json"


class ManifestManager:
    """Portable source tracking for the active technical-document corpus.

    Older DocuBot manifests used absolute Windows paths as dictionary keys.
    Copying an otherwise healthy project to another PC/path therefore looked
    like every source file had been deleted and re-added.  The production
    manifest now keys documents by their path *relative to DOCUMENT_DIR* while
    retaining ``indexed_file_path`` for compatibility with already-built Qdrant
    payloads.
    """

    @staticmethod
    def _relative_source_path(file_path) -> str:
        text = str(file_path or "").strip()
        if not text:
            return ""

        # First prefer the actual current DOCUMENT_DIR relationship.
        try:
            path = Path(text).expanduser().resolve()
            rel = path.relative_to(DOCUMENT_DIR.resolve())
            return rel.as_posix()
        except Exception:
            pass

        # Migrate old absolute Windows/Linux paths from another installation by
        # taking the suffix after data/technical_documents.  Normalize slashes
        # first so this works regardless of which OS created the manifest.
        normalized = text.replace("\\", "/")
        lowered = normalized.casefold()
        markers = (
            "/data/technical_documents/",
            "data/technical_documents/",
            "/technical_documents/",
            "technical_documents/",
        )
        for marker in markers:
            index = lowered.find(marker)
            if index >= 0:
                rel = normalized[index + len(marker):].lstrip("/")
                if rel:
                    return rel

        # Last-resort compatibility for very old one-folder manifests.
        return Path(normalized).name

    @staticmethod
    def _current_source_path(relative_path: str, stored_path: str = "") -> Path:
        rel = str(relative_path or "").replace("\\", "/").lstrip("/")
        if rel:
            candidate = (DOCUMENT_DIR / Path(rel)).resolve()
            if candidate.exists():
                return candidate

        # Same-machine legacy manifests may still point to a valid absolute path.
        try:
            candidate = Path(stored_path).expanduser().resolve()
            if candidate.exists():
                return candidate
        except Exception:
            pass

        # Return the canonical expected current path even when a file is missing;
        # change detection will then correctly classify the source as deleted.
        return (DOCUMENT_DIR / Path(rel or Path(str(stored_path or "")).name)).resolve()

    @staticmethod
    def document_key(file_path):
        """Create one stable, installation-independent manifest key."""
        raw = str(file_path or "").strip()
        raw_normalized = raw.replace("\\", "/").strip("/")
        # Callers may already supply a DOCUMENT_DIR-relative path. Preserve its
        # subdirectory structure instead of resolving it against the process CWD.
        looks_absolute = (
            raw_normalized.startswith("/")
            or bool(re.match(r"^[A-Za-z]:/", raw_normalized))
        )
        contains_source_root = "technical_documents/" in raw_normalized.casefold()
        relative = (
            ManifestManager._relative_source_path(raw)
            if looks_absolute or contains_source_root
            else raw_normalized
        )
        normalized = str(relative or "").replace("\\", "/").strip("/")
        return os.path.normcase(normalized).casefold()

    @staticmethod
    def current_index_signature():
        return {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": EMBED_MODEL_NAME,
            "embedding_normalize": bool(DEFAULT_NORMALIZE_EMBEDDINGS),
        }

    @staticmethod
    def index_signature(info):
        """Normalize old manifests without forcing an unnecessary migration rebuild."""
        current = ManifestManager.current_index_signature()
        return {
            "index_schema_version": info.get(
                "index_schema_version",
                current["index_schema_version"],
            ),
            "embedding_model": info.get(
                "embedding_model",
                current["embedding_model"],
            ),
            "embedding_normalize": bool(
                info.get(
                    "embedding_normalize",
                    current["embedding_normalize"],
                )
            ),
        }

    @staticmethod
    def load():
        if not MANIFEST_FILE.exists():
            return {}

        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
                raw_manifest = json.load(file)
        except (OSError, ValueError, TypeError):
            # A broken manifest must not crash DocuBot startup. Smart Build
            # will detect the existing index and run its safe migration path.
            return {}

        if not isinstance(raw_manifest, dict):
            return {}

        normalized_manifest = {}

        for old_key, info in raw_manifest.items():
            if not isinstance(info, dict):
                continue

            stored_path = str(info.get("file_path") or old_key or "")
            relative_path = (
                str(info.get("source_relative_path") or "").strip()
                or ManifestManager._relative_source_path(stored_path)
                or str(info.get("file_name") or "").strip()
            )
            if not relative_path:
                continue

            current_path = ManifestManager._current_source_path(
                relative_path,
                stored_path=stored_path,
            )
            document_key = ManifestManager.document_key(relative_path)
            normalized_info = dict(info)

            normalized_info["file_name"] = (
                normalized_info.get("file_name") or current_path.name
            )
            normalized_info["source_relative_path"] = relative_path.replace("\\", "/")
            normalized_info["file_path"] = str(current_path)
            # Keep the path that was embedded into the current vector payload so
            # health/source compatibility remains intact until the next rebuild.
            normalized_info["indexed_file_path"] = info.get(
                "indexed_file_path",
                stored_path,
            )

            signature = ManifestManager.index_signature(normalized_info)
            normalized_info.update(signature)
            normalized_manifest[document_key] = normalized_info

        return normalized_manifest

    @staticmethod
    def save(manifest):
        """Save atomically so interruption cannot corrupt the active manifest."""
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = MANIFEST_FILE.with_suffix(".json.tmp")

        try:
            with open(temporary_file, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=4, ensure_ascii=False)
            temporary_file.replace(MANIFEST_FILE)
        finally:
            if temporary_file.exists():
                try:
                    temporary_file.unlink()
                except OSError:
                    pass

    @staticmethod
    def build(documents, previous_manifest=None):
        """Build current fingerprints while avoiding SHA256 on unchanged files."""
        previous_manifest = previous_manifest or {}
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_signature = ManifestManager.current_index_signature()
        manifest = {}

        for document in documents:
            resolved_path = Path(document).resolve()
            relative_path = ManifestManager._relative_source_path(resolved_path)
            document_key = ManifestManager.document_key(relative_path)
            previous_info = previous_manifest.get(document_key, {})
            stat_info = FileHasher.stat_fingerprint(resolved_path)

            stat_unchanged = (
                previous_info.get("file_size") == stat_info["file_size"]
                and previous_info.get("modified_time_ns") == stat_info["modified_time_ns"]
                and bool(previous_info.get("hash"))
            )

            if stat_unchanged:
                file_hash = previous_info["hash"]
            else:
                file_hash = FileHasher.sha256(resolved_path)

            previous_signature = ManifestManager.index_signature(previous_info)
            unchanged = (
                previous_info.get("hash") == file_hash
                and previous_signature == current_signature
            )

            info = {
                "file_name": resolved_path.name,
                "source_relative_path": relative_path.replace("\\", "/"),
                "file_path": str(resolved_path),
                "indexed_file_path": (
                    previous_info.get("indexed_file_path", str(resolved_path))
                    if unchanged
                    else str(resolved_path)
                ),
                "hash": file_hash,
                "file_size": stat_info["file_size"],
                "modified_time_ns": stat_info["modified_time_ns"],
                **current_signature,
                "last_indexed": (
                    previous_info.get("last_indexed")
                    if unchanged
                    else now
                ) or now,
            }

            if unchanged and previous_info.get("index_status"):
                info["index_status"] = previous_info["index_status"]
            if unchanged and previous_info.get("skip_reason"):
                info["skip_reason"] = previous_info["skip_reason"]

            manifest[document_key] = info

        return manifest
