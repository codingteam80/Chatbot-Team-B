import json
import os

from pathlib import Path
from datetime import datetime

from config.settings import (
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EMBED_MODEL_NAME,
    INDEX_SCHEMA_VERSION,
    METADATA_DIR,
)
from utils.hash_utils import FileHasher


MANIFEST_FILE = METADATA_DIR / "manifest.json"


class ManifestManager:

    @staticmethod
    def document_key(file_path):
        """Create one stable manifest key per normalized absolute path."""
        path = Path(file_path).resolve()
        return os.path.normcase(str(path))

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

            stored_path = info.get("file_path") or old_key
            resolved_path = str(Path(stored_path).resolve())
            document_key = ManifestManager.document_key(resolved_path)
            normalized_info = dict(info)

            normalized_info["file_path"] = resolved_path
            normalized_info["indexed_file_path"] = info.get(
                "indexed_file_path",
                stored_path,
            )

            # Old v6.1 manifests did not record embedding identity. Treat
            # missing fields as the current v6.1 defaults so the upgrade does
            # not re-embed every unchanged document just to enrich metadata.
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
            document_key = ManifestManager.document_key(resolved_path)
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
