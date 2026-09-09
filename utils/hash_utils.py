from pathlib import Path
import hashlib


class FileHasher:

    READ_SIZE = 1024 * 1024

    @staticmethod
    def sha256(file_path: Path) -> str:
        """Compute SHA256 using large buffered reads for lower I/O overhead."""
        sha = hashlib.sha256()

        with open(file_path, "rb") as file:
            while True:
                chunk = file.read(FileHasher.READ_SIZE)
                if not chunk:
                    break
                sha.update(chunk)

        return sha.hexdigest()

    @staticmethod
    def stat_fingerprint(file_path: Path):
        """Return cheap filesystem signals used to avoid unnecessary re-hashing."""
        stat = Path(file_path).stat()
        return {
            "file_size": int(stat.st_size),
            "modified_time_ns": int(stat.st_mtime_ns),
        }
