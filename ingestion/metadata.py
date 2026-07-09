from pathlib import Path
from datetime import datetime


class MetadataBuilder:
    """
    Add searchable file information to every chunk.
    """

    @staticmethod
    def build(
        file_path: str,
        chunk_id: int,
        total_chunks: int
    ):

        path = Path(
            file_path
        ).resolve()

        return {
            # Original filename
            "file_name": path.name,

            # Canonical absolute path used for source opening
            # and targeted Chroma deletion.
            "file_path": str(path),

            # Parent folder
            "folder_name": path.parent.name,

            # File extension
            "extension": path.suffix.lower(),

            # Current chunk number
            "chunk_id": chunk_id,

            # Total chunks generated
            "total_chunks": total_chunks,

            # When this chunk was indexed
            "indexed_at": (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        }
