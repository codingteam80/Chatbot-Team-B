from pathlib import Path
from datetime import datetime


class MetadataBuilder:
    """Add searchable file and structure information to every chunk."""

    @staticmethod
    def _sanitize_value(value):
        # Chroma metadata values must be scalar and cannot be None.
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def build(
        file_path: str,
        chunk_id: int,
        total_chunks: int,
        extra_metadata=None,
    ):
        path = Path(file_path).resolve()

        metadata = {
            "file_name": path.name,
            "file_path": str(path),
            "folder_name": path.parent.name,
            "extension": path.suffix.lower(),
            "chunk_id": chunk_id,
            "total_chunks": total_chunks,
            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        for key, value in (extra_metadata or {}).items():
            if value is None or value == "":
                continue
            metadata[str(key)] = MetadataBuilder._sanitize_value(value)

        return metadata
