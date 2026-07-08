import json

from pathlib import Path
from datetime import datetime

from config.settings import METADATA_DIR
from utils.hash_utils import FileHasher


MANIFEST_FILE = (
    METADATA_DIR / "manifest.json"
)


class ManifestManager:

    @staticmethod
    def load():

        if not MANIFEST_FILE.exists():
            return {}

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    @staticmethod
    def save(manifest):

        MANIFEST_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            MANIFEST_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def build(documents):

        """
        Build a manifest from the current document set.

        Key:
            filename

        Value:
            file path
            SHA256 hash
            last indexed time
        """

        manifest = {}

        for file in documents:

            manifest[file.name] = {

                "file_path": str(file),

                "hash": FileHasher.sha256(file),

                "last_indexed":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }

        return manifest