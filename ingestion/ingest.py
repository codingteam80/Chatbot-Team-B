from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ingestion.parser import DocumentParser
from ingestion.cleaner import TextCleaner
from ingestion.splitter import DocumentSplitter
from ingestion.metadata import MetadataBuilder
from ingestion.pdf_structure import PDFStructureExtractor
from ingestion.chunking_v4 import RuleAwareChunkingV4

from utils.file_utils import get_all_documents
from utils.hash_utils import FileHasher
from utils.ingestion_cache import IngestionCache
from utils.logger import log

from config.settings import CHUNKING_PROFILE, INGESTION_MAX_WORKERS, MIN_DOCUMENT_LENGTH


class IngestionPipeline:

    def __init__(self, use_cache=True):
        self.parser = DocumentParser()
        self.cleaner = TextCleaner()
        self.splitter = DocumentSplitter()
        self.use_cache = bool(use_cache)
        self.last_skip_reason = None
        self.last_cache_hit = False
        self.skipped_documents = {}
        self.failed_documents = {}

    def _split_pdf_structure(self, file_path: Path):
        """Create PDF chunks by logical section, then enforce size limits."""
        try:
            sections = PDFStructureExtractor.extract(str(file_path))
        except Exception as error:
            log(
                f"PDF structure parsing fallback: {file_path.name} "
                f"[{error}]"
            )
            return None

        if not sections:
            return None

        if CHUNKING_PROFILE == "v4":
            return self._split_pdf_structure_v4(file_path, sections)

        prepared = []
        cleaned_sections = []

        for section in sections:
            clean_text = self.cleaner.clean(section.text)
            if clean_text:
                cleaned_sections.append((section, clean_text))

        total_clean_length = sum(len(text) for _, text in cleaned_sections)
        if total_clean_length < MIN_DOCUMENT_LENGTH:
            self.last_skip_reason = (
                f"too short [{total_clean_length} chars; "
                f"minimum {MIN_DOCUMENT_LENGTH}]"
            )
            log(
                f"Skipped (too short): {file_path.name} "
                f"[structure-aware PDF]"
            )
            return []

        for section_index, (section, clean_text) in enumerate(cleaned_sections):
            parts = self.splitter.split(clean_text)
            if not parts:
                continue

            for part_index, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                if (
                    section.section_title
                    and not part.casefold().startswith(
                        section.section_title.casefold()
                    )
                ):
                    part = f"{section.section_title}\n{part}"

                prepared.append(
                    {
                        "text": part,
                        "metadata": {
                            "parser_strategy": "pdf_structure_aware",
                            "section_index": section_index,
                            "section_part": part_index + 1,
                            "section_parts": len(parts),
                            "section_type": section.section_type,
                            "section_title": section.section_title,
                            "rule_id": section.rule_id,
                            "section_id": section.section_id,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                        },
                    }
                )

        if not prepared:
            self.last_skip_reason = "no valid chunks"
            return []

        return prepared

    def _split_pdf_structure_v4(self, file_path: Path, sections):
        """Experimental parent-anchored Rule/Directive subsection chunks.

        This path is reachable only when DOCUBOT_CHUNKING_PROFILE=v4. The
        certified v3 branch above is intentionally left unchanged.
        """
        units = RuleAwareChunkingV4.expand(sections)
        cleaned_units = []
        total_clean_length = 0

        for unit in units:
            clean_text = self.cleaner.clean(unit.text)
            if not clean_text:
                continue
            cleaned_units.append((unit, clean_text))
            total_clean_length += len(clean_text)

        if total_clean_length < MIN_DOCUMENT_LENGTH:
            self.last_skip_reason = (
                f"too short [{total_clean_length} chars; "
                f"minimum {MIN_DOCUMENT_LENGTH}]"
            )
            log(
                f"Skipped (too short): {file_path.name} "
                f"[rule-aware PDF v4]"
            )
            return []

        prepared = []
        for unit_index, (unit, clean_text) in enumerate(cleaned_units):
            parts = self.splitter.split(clean_text)
            if not parts:
                continue

            unit_metadata = unit.metadata()
            for part_index, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # SentenceSplitter can split a long unit. Re-anchor every later
                # part to its parent/section so semantic and lexical retrieval
                # do not lose the Rule/Directive identity.
                anchor_lines = []
                parent_title = str(unit_metadata.get("parent_title") or "").strip()
                section_title = str(unit_metadata.get("section_title") or "").strip()
                if parent_title:
                    anchor_lines.append(parent_title)
                if section_title and section_title.casefold() != parent_title.casefold():
                    anchor_lines.append(section_title)

                if part_index > 0 and anchor_lines:
                    prefix = "\n".join(anchor_lines)
                    if not part.casefold().startswith(prefix.casefold()):
                        part = f"{prefix}\n{part}"

                metadata = dict(unit_metadata)
                metadata.update(
                    {
                        "section_index": unit_index,
                        "section_part": part_index + 1,
                        "section_parts": len(parts),
                        "parser_strategy": "pdf_rule_aware_v4",
                    }
                )
                prepared.append({"text": part, "metadata": metadata})

        if not prepared:
            self.last_skip_reason = "no valid chunks"
            return []
        return prepared

    def _split_generic_document(self, file_path: Path):
        raw_text = self.parser.parse(str(file_path))
        if not raw_text:
            self.last_skip_reason = "empty document"
            return []

        clean_text = self.cleaner.clean(raw_text)
        if len(clean_text) < MIN_DOCUMENT_LENGTH:
            self.last_skip_reason = (
                f"too short [{len(clean_text)} chars; "
                f"minimum {MIN_DOCUMENT_LENGTH}]"
            )
            log(
                f"Skipped (too short): {file_path.name} "
                f"[{len(clean_text)} chars]"
            )
            return []

        chunks = self.splitter.split(clean_text)
        if not chunks:
            self.last_skip_reason = "no valid chunks"
            return []

        return [
            {
                "text": chunk,
                "metadata": {
                    "parser_strategy": "generic_sentence",
                },
            }
            for chunk in chunks
            if chunk and chunk.strip()
        ]

    def _prepared_chunks_from_cache(self, file_hash):
        if not self.use_cache or not file_hash:
            return None

        payload = IngestionCache.load(file_hash)
        if payload is None:
            return None

        self.last_cache_hit = True
        if payload.get("status") == "skipped":
            self.last_skip_reason = payload.get("skip_reason") or "no valid chunks"
            return []

        prepared_chunks = payload.get("prepared_chunks")
        if not isinstance(prepared_chunks, list):
            return None

        return prepared_chunks

    def _cache_prepared_chunks(self, file_hash, prepared_chunks):
        if not self.use_cache or not file_hash:
            return

        if prepared_chunks:
            IngestionCache.save_ready(file_hash, prepared_chunks)
            return

        # Do not cache an "empty document" result because the universal
        # parser also returns empty text after transient extraction failures.
        if self.last_skip_reason and self.last_skip_reason != "empty document":
            IngestionCache.save_skipped(file_hash, self.last_skip_reason)

    def process_document(self, file_path, file_hash=None):
        """Parse and chunk exactly one source file at a time."""
        file_path = Path(file_path)
        self.last_skip_reason = None
        self.last_cache_hit = False
        log(f"Processing: {file_path.name}")

        if file_hash is None and self.use_cache:
            file_hash = FileHasher.sha256(file_path)

        prepared_chunks = self._prepared_chunks_from_cache(file_hash)

        if prepared_chunks is not None:
            log(f"Chunk cache hit: {file_path.name}")
        else:
            if file_path.suffix.lower() == ".pdf":
                prepared_chunks = self._split_pdf_structure(file_path)
            else:
                prepared_chunks = None

            if prepared_chunks is None:
                prepared_chunks = self._split_generic_document(file_path)

            self._cache_prepared_chunks(file_hash, prepared_chunks)

        if not prepared_chunks:
            log(f"Skipped (no chunks): {file_path.name}")
            return []

        total_chunks = len(prepared_chunks)
        records = []

        for chunk_id, item in enumerate(prepared_chunks):
            records.append(
                {
                    "text": item["text"],
                    "metadata": MetadataBuilder.build(
                        file_path=str(file_path),
                        chunk_id=chunk_id,
                        total_chunks=total_chunks,
                        extra_metadata=item.get("metadata"),
                    ),
                }
            )

        log(f"Created {total_chunks} chunks")
        return records

    def run(self, documents=None, document_hashes=None, max_workers=None):
        """Process independent files concurrently while preserving output order."""
        if documents is None:
            documents = get_all_documents()

        documents = [Path(document) for document in documents]
        document_hashes = document_hashes or {}
        worker_count = max(1, int(max_workers or INGESTION_MAX_WORKERS))
        worker_count = min(worker_count, max(1, len(documents)))
        log(f"Found {len(documents)} documents")

        if not documents:
            return []

        results_by_index = {}

        def _run_one(index, document):
            worker = IngestionPipeline(use_cache=self.use_cache)
            file_hash = document_hashes.get(str(document.resolve()))
            try:
                records = worker.process_document(document, file_hash=file_hash)
                return {
                    "index": index,
                    "document": document,
                    "records": records,
                    "skip_reason": worker.last_skip_reason,
                    "cache_hit": worker.last_cache_hit,
                    "error": None,
                }
            except Exception as error:
                return {
                    "index": index,
                    "document": document,
                    "records": [],
                    "skip_reason": None,
                    "cache_hit": False,
                    "error": error,
                }

        if worker_count == 1:
            completed = [_run_one(index, document) for index, document in enumerate(documents)]
        else:
            completed = []
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="docubot-ingest") as executor:
                futures = {
                    executor.submit(_run_one, index, document): index
                    for index, document in enumerate(documents)
                }
                for future in as_completed(futures):
                    completed.append(future.result())

        for result in completed:
            results_by_index[result["index"]] = result

        all_records = []
        for index in range(len(documents)):
            result = results_by_index[index]
            document = result["document"]
            error = result["error"]
            if error is not None:
                self.failed_documents[str(document.resolve())] = str(error)
                log(f"Failed: {document.name} [{error}]")
                continue

            if not result["records"] and result["skip_reason"]:
                self.skipped_documents[str(document.resolve())] = result["skip_reason"]

            all_records.extend(result["records"])

        log(f"Total Chunks: {len(all_records)}")
        return all_records


def process_document_task(file_path, file_hash=None):
    """Thread-safe one-file entry point used by Smart Build."""
    pipeline = IngestionPipeline(use_cache=True)
    try:
        records = pipeline.process_document(file_path, file_hash=file_hash)
        return {
            "records": records,
            "skip_reason": pipeline.last_skip_reason,
            "cache_hit": pipeline.last_cache_hit,
            "error": None,
        }
    except Exception as error:
        return {
            "records": [],
            "skip_reason": None,
            "cache_hit": False,
            "error": error,
        }


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    chunks = pipeline.run()

    print()
    print("=" * 50)
    print(f"TOTAL CHUNKS CREATED: {len(chunks)}")
    print("=" * 50)
