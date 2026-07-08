from pathlib import Path
import hashlib


class FileHasher:

    @staticmethod
    def sha256(file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Returns:
            Hexadecimal hash string.
        """

        sha = hashlib.sha256()

        with open(file_path, "rb") as f:

            while True:

                chunk = f.read(8192)

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()