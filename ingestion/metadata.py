from pathlib import Path
from datetime import datetime

# Adding extra information to each chunk.
class MetadataBuilder:

    @staticmethod
    def build(
        file_path: str,
        chunk_id: int,
        total_chunks: int
    ):

        path = Path(file_path)

        metadata = {

            # Original filename
            "file_name": path.name,

            # Full file path
            "file_path": str(path),

            # Parent folder
            "folder_name": (
                path.parent.name
            ),

            # File extension
            "extension": (
                path.suffix.lower()
            ),

            # Current chunk number
            "chunk_id": chunk_id,

            # Total chunks generated
            "total_chunks": total_chunks,

            # When chunk was indexed
            "indexed_at": (
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        }

        return metadata