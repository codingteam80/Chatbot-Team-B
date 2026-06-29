from pathlib import Path

from ingestion.parser import DocumentParser        # Extract text from files
from ingestion.cleaner import TextCleaner          # Clean extracted text
from ingestion.splitter import DocumentSplitter    # Split document into chunks
from ingestion.metadata import MetadataBuilder     # Create metadata per chunk

from utils.file_utils import get_all_documents     # Discover all files
from utils.logger import log                       # Console logger

from config.settings import (
    MIN_DOCUMENT_LENGTH
)


class IngestionPipeline:

    def __init__(self):

        # Document parser
        self.parser = DocumentParser()

        # Text cleaner
        self.cleaner = TextCleaner()

        # Chunk splitter
        self.splitter = DocumentSplitter()

    def process_document(
        self,
        file_path
    ):

        log(
            f"Processing: {file_path.name}"
        )

        # EXTRACT TEXT
        # Convert file into raw text
        raw_text = self.parser.parse(
            str(file_path)
        )

        # Skip if parser returns nothing
        if not raw_text:

            log(
                f"Skipped (empty): "
                f"{file_path.name}"
            )

            return []

        # CLEAN TEXT
        # Remove unwanted characters,
        # normalize spaces and line breaks
        clean_text = self.cleaner.clean(
            raw_text
        )

        # DOCUMENT LENGTH VALIDATION
        # Prevent indexing useless files:
        # empty pages, corrupted files,
        # "OK", "-", "N/A", etc.
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

        # CHUNKING
        # Convert large document into
        # smaller searchable chunks
        chunks = self.splitter.split(
            clean_text
        )

        # Safety check
        if not chunks:

            log(
                f"Skipped (no chunks): "
                f"{file_path.name}"
            )

            return []

        records = []

        total_chunks = len(chunks)

        # CREATE CHUNK RECORDS
        # Each chunk gets its own metadata
        for idx, chunk in enumerate(
            chunks
        ):

            metadata = MetadataBuilder.build(
                file_path=str(file_path),
                chunk_id=idx,
                total_chunks=total_chunks
            )

            records.append(
                {
                    "text": chunk,
                    "metadata": metadata
                }
            )

        log(
            f"Created "
            f"{total_chunks} chunks"
        )

        return records

    def run(self):

        all_records = []

        # DISCOVER DOCUMENTS
        # Scan document folder and collect
        # all supported files
        documents = get_all_documents()

        log(
            f"Found {len(documents)} documents"
        )

        # PROCESS DOCUMENTS
        # Parse -> Clean -> Chunk -> Metadata
        for document in documents:

            records = self.process_document(
                document
            )

            all_records.extend(
                records
            )

        # FINAL STATISTICS
        # Total chunks generated from all files
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