import json
import os

from pathlib import Path
from datetime import datetime

from config.settings import METADATA_DIR
from utils.hash_utils import FileHasher


MANIFEST_FILE = (
    METADATA_DIR / "manifest.json"
)


class ManifestManager:

    @staticmethod
    def document_key(file_path):

        """
        Create one stable manifest key per document.

        Full normalized paths are used instead of filenames so
        files with the same name in different folders do not collide.
        """

        path = Path(file_path).resolve()

        return os.path.normcase(
            str(path)
        )

    @staticmethod
    def load():

        if not MANIFEST_FILE.exists():
            return {}

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            raw_manifest = json.load(
                file
            )

        normalized_manifest = {}

        for old_key, info in raw_manifest.items():

            stored_path = (
                info.get("file_path")
                or old_key
            )

            resolved_path = str(
                Path(stored_path).resolve()
            )

            document_key = (
                ManifestManager.document_key(
                    resolved_path
                )
            )

            normalized_info = dict(
                info
            )

            # Canonical path used by the new incremental index.
            normalized_info["file_path"] = (
                resolved_path
            )

            # Preserve the exact path value previously stored in
            # Chroma metadata. This supports migration from older
            # manifests that used relative file paths.
            normalized_info[
                "indexed_file_path"
            ] = info.get(
                "indexed_file_path",
                stored_path
            )

            normalized_manifest[
                document_key
            ] = normalized_info

        return normalized_manifest

    @staticmethod
    def save(manifest):

        """
        Save atomically so an interrupted write does not corrupt
        the existing manifest.
        """

        MANIFEST_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_file = (
            MANIFEST_FILE.with_suffix(
                ".json.tmp"
            )
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary_file.replace(
            MANIFEST_FILE
        )

    @staticmethod
    def build(
        documents,
        previous_manifest=None
    ):

        """
        Build the current document manifest.

        Unchanged files keep their original last_indexed value.
        Changed files receive a new candidate timestamp, which is
        saved only after their new chunks are stored successfully.
        """

        previous_manifest = (
            previous_manifest
            or {}
        )

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        manifest = {}

        for document in documents:

            resolved_path = Path(
                document
            ).resolve()

            document_key = (
                ManifestManager.document_key(
                    resolved_path
                )
            )

            file_hash = FileHasher.sha256(
                resolved_path
            )

            previous_info = (
                previous_manifest.get(
                    document_key,
                    {}
                )
            )

            unchanged = (
                previous_info.get("hash")
                == file_hash
            )

            last_indexed = (
                previous_info.get(
                    "last_indexed"
                )
                if unchanged
                else now
            )

            manifest[
                document_key
            ] = {
                "file_name": resolved_path.name,
                "file_path": str(resolved_path),
                "indexed_file_path": str(
                    resolved_path
                ),
                "hash": file_hash,
                "last_indexed": (
                    last_indexed
                    or now
                )
            }

        return manifest
