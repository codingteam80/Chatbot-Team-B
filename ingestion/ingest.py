from pathlib import Path

from ingestion.parser import DocumentParser
from ingestion.cleaner import TextCleaner
from ingestion.splitter import DocumentSplitter
from ingestion.metadata import MetadataBuilder

from utils.file_utils import get_all_documents
from utils.logger import log

from config.settings import (
    MIN_DOCUMENT_LENGTH
)


class IngestionPipeline:

    def __init__(self):

        self.parser = DocumentParser()
        self.cleaner = TextCleaner()
        self.splitter = DocumentSplitter()

    def process_document(
        self,
        file_path
    ):

        """
        Parse, clean, split, and describe one document.

        This method is used by incremental indexing so unchanged
        documents never enter the ingestion pipeline.
        """

        file_path = Path(
            file_path
        )

        log(
            f"Processing: {file_path.name}"
        )

        raw_text = self.parser.parse(
            str(file_path)
        )

        if not raw_text:

            log(
                f"Skipped (empty): "
                f"{file_path.name}"
            )

            return []

        clean_text = self.cleaner.clean(
            raw_text
        )

        if (
            len(clean_text)
            < MIN_DOCUMENT_LENGTH
        ):

            log(
                f"Skipped (too short): "
                f"{file_path.name} "
                f"[{len(clean_text)} chars]"
            )

            return []

        chunks = self.splitter.split(
            clean_text
        )

        if not chunks:

            log(
                f"Skipped (no chunks): "
                f"{file_path.name}"
            )

            return []

        total_chunks = len(
            chunks
        )

        records = []

        for chunk_id, chunk in enumerate(
            chunks
        ):

            records.append(
                {
                    "text": chunk,
                    "metadata": (
                        MetadataBuilder.build(
                            file_path=str(
                                file_path
                            ),
                            chunk_id=chunk_id,
                            total_chunks=total_chunks
                        )
                    )
                }
            )

        log(
            f"Created {total_chunks} chunks"
        )

        return records

    def run(
        self,
        documents=None
    ):

        """
        Process all documents by default, or only the supplied
        document paths during an incremental operation.
        """

        if documents is None:

            documents = (
                get_all_documents()
            )

        documents = [
            Path(document)
            for document in documents
        ]

        log(
            f"Found {len(documents)} documents"
        )

        all_records = []

        for document in documents:

            try:

                records = (
                    self.process_document(
                        document
                    )
                )

                all_records.extend(
                    records
                )

            except Exception as error:

                # One damaged file must not stop the rest of
                # a full build.
                log(
                    f"Failed: {document.name} "
                    f"[{error}]"
                )

        log(
            f"Total Chunks: "
            f"{len(all_records)}"
        )

        return all_records


if __name__ == "__main__":

    pipeline = IngestionPipeline()

    chunks = pipeline.run()

    print()
    print("=" * 50)
    print(
        f"TOTAL CHUNKS CREATED: "
        f"{len(chunks)}"
    )
    print("=" * 50)
