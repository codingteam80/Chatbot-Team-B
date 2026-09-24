print("USING retriever.py")

import re
import unicodedata
from pathlib import Path

from retrieval.bm25_index import BM25Searcher
from retrieval.hybrid_search import HybridRetriever
from qa.evidence_logger import evidence_logger

from utils.structured_reference import (
    StructuredReference,
    extract_structured_reference,
    extract_structured_references,
)
from utils.unicode_markers import decode_unicode_markers

from config.settings import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    DEBUG_RETRIEVAL,
    ENABLE_RERANKER,
    COMPLETENESS_TOP_K,
    CONTEXT_EXPANSION_PREVIOUS_CHUNKS,
    CONTEXT_EXPANSION_NEXT_CHUNKS,
    CONTEXT_EXPANSION_MAX_SEEDS,
    MIN_RETRIEVAL_SCORE,
    TECHNICAL_DOCUMENT_DIR,
)


class CompanyRetriever:

    def __init__(self):

        # BM25 is lightweight and also backs exact structured metadata lookup.
        # Keep it ready immediately, but defer the embedding/vector and
        # CrossEncoder stacks until a query actually needs semantic retrieval.
        # This improves new-process startup and preserves the ultra-fast exact
        # Rule/Directive/Section path without loading unused ML models.
        self.bm25 = BM25Searcher()
        self.vector_searcher = None
        self.hybrid = HybridRetriever()
        self.reranker = None

    def _get_vector_searcher(self):

        # Lazily initialize the finalized Qdrant + Qwen3 embedding path.
        if self.vector_searcher is None:
            try:
                from runtime.prewarm import wait_for_retrieval_prewarm
                wait_for_retrieval_prewarm(timeout=90.0)
            except Exception:
                pass

            import time
            started = time.perf_counter()
            component = "QdrantSearcher + Qwen3 embedding"
            evidence_logger.record_event(
                event_name="LAZY RETRIEVAL COMPONENT",
                status="VECTOR STARTED",
                details={"component": component},
            )

            from retrieval.qdrant_search import QdrantSearcher
            self.vector_searcher = QdrantSearcher()

            evidence_logger.record_event(
                event_name="LAZY RETRIEVAL COMPONENT",
                status="VECTOR READY",
                details={
                    "component": component,
                    "seconds": round(time.perf_counter() - started, 4),
                },
            )

        return self.vector_searcher

    def _get_reranker(self):

        if not ENABLE_RERANKER:
            return None

        if self.reranker is None:
            try:
                from runtime.prewarm import wait_for_retrieval_prewarm
                wait_for_retrieval_prewarm(timeout=90.0)
            except Exception:
                pass

            import time
            started = time.perf_counter()
            evidence_logger.record_event(
                event_name="LAZY RETRIEVAL COMPONENT",
                status="RERANKER STARTED",
                details={"component": "CrossEncoder reranker"},
            )
            from retrieval.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
            evidence_logger.record_event(
                event_name="LAZY RETRIEVAL COMPONENT",
                status="RERANKER READY",
                details={
                    "component": "CrossEncoder reranker",
                    "seconds": round(time.perf_counter() - started, 4),
                },
            )

        return self.reranker

    def _record_qa_stage(
        self,
        stage,
        items,
        limit=10,
        accepted=None,
        rejection_reason=""
    ):

        """
        Record retrieval evidence without modifying candidates.
        """

        if not evidence_logger.enabled:

            return

        qa_items = []

        for item in list(
            items
            or []
        )[:limit]:

            qa_item = dict(
                item
            )

            qa_item["metadata"] = dict(
                item.get(
                    "metadata",
                    {}
                )
            )

            score = item.get(
                "score",
                ""
            )

            if stage == "BM25":

                qa_item["bm25_score"] = score

            elif stage == "VECTOR":

                qa_item["vector_score"] = score

            else:

                qa_item["hybrid_score"] = score

            qa_item["informative_score"] = (
                item.get(
                    "_info_score",
                    ""
                )
            )

            qa_items.append(
                qa_item
            )

        for position, item in enumerate(
            qa_items,
            start=1
        ):

            evidence_logger.record_chunk(
                stage=stage,
                position=position,
                item=item,
                accepted=accepted,
                rejection_reason=rejection_reason
            )

    def _normalize_text(self, text):

        if not text:
            return ""

        text = text.lower()

        text = unicodedata.normalize(
            "NFKD",
            text
        )

        text = "".join(
            char for char in text
            if not unicodedata.combining(char)
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    @staticmethod
    def _normalize_structured_identifier(value):

        return str(
            value or ""
        ).strip().casefold()

    def _metadata_matches_structured_reference(
        self,
        metadata,
        reference: StructuredReference
    ):

        """Return True only for an exact structured metadata match."""

        if not metadata:
            return False

        section_type = str(
            metadata.get(
                "section_type",
                ""
            )
        ).strip().casefold()

        identifier = self._normalize_structured_identifier(
            metadata.get(
                reference.metadata_id_key,
                ""
            )
        )

        return (
            section_type == reference.section_type
            and identifier == reference.identifier
        )

    def _text_starts_with_structured_reference(
        self,
        text,
        reference: StructuredReference
    ):

        """Strict textual fallback for legacy or incomplete metadata."""

        if not text:
            return False

        escaped_id = re.escape(
            reference.identifier
        )

        if reference.kind == "rule":
            prefix = r"rule"
        elif reference.kind == "directive":
            prefix = r"(?:dir|directive)"
        elif reference.kind in {"section", "article", "chapter", "part"}:
            prefix = rf"{re.escape(reference.kind)}"
        else:
            return False

        pattern = (
            rf"^\s*{prefix}\s*"
            rf"(?:no\.?\s*|number\s*)?"
            rf"{escaped_id}(?=\s|$|[:\-–—])"
        )

        return bool(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    def _structured_reference_consistent(
        self,
        item,
        reference: StructuredReference
    ):

        """Reject a high-scoring chunk when it is not the requested ID."""

        metadata = item.get(
            "metadata",
            {}
        )

        if self._metadata_matches_structured_reference(
            metadata,
            reference
        ):
            return True

        return self._text_starts_with_structured_reference(
            item.get(
                "text",
                ""
            ),
            reference
        )

    def _same_file_records_sorted(
        self,
        file_key
    ):

        records = getattr(
            self.bm25,
            "records",
            []
        ) or []

        same_file = []

        for record in records:

            metadata = record.get(
                "metadata",
                {}
            )

            record_file_key = (
                metadata.get("file_path")
                or metadata.get("file_name")
            )

            if record_file_key != file_key:
                continue

            try:
                chunk_id = int(
                    metadata.get(
                        "chunk_id",
                        -1
                    )
                )
            except (TypeError, ValueError):
                continue

            same_file.append(
                (
                    chunk_id,
                    record
                )
            )

        same_file.sort(
            key=lambda pair: pair[0]
        )

        return same_file

    def _collect_rule_or_directive_group(
        self,
        seed_record,
        reference: StructuredReference,
        max_chunks=8
    ):

        """
        Collect an exact Rule/Directive plus its subordinate continuation chunks.

        MISRA commonly stores Rationale, Amplification, Example, Exception,
        and See also as visually separate sections after the rule heading. Those
        chunks intentionally have section_type=section, so the safest boundary
        is the next Rule/Directive heading in the same source file.
        """

        metadata = seed_record.get(
            "metadata",
            {}
        )

        file_key = (
            metadata.get("file_path")
            or metadata.get("file_name")
        )

        if not file_key:
            return [seed_record]

        try:
            seed_chunk_id = int(
                metadata.get(
                    "chunk_id"
                )
            )
        except (TypeError, ValueError):
            return [seed_record]

        ordered = self._same_file_records_sorted(
            file_key
        )

        collected = []
        started = False

        for chunk_id, record in ordered:

            if chunk_id < seed_chunk_id:
                continue

            if chunk_id == seed_chunk_id:
                started = True

            if not started:
                continue

            current_metadata = record.get(
                "metadata",
                {}
            )

            if (
                chunk_id != seed_chunk_id
                and str(
                    current_metadata.get(
                        "section_type",
                        ""
                    )
                ).strip().casefold()
                in {"rule", "directive"}
                and self._normalize_structured_identifier(
                    current_metadata.get(
                        "rule_id",
                        ""
                    )
                )
                != reference.identifier
            ):
                break

            collected.append(
                record
            )

            if len(collected) >= max_chunks:
                break

        return collected or [seed_record]

    def _collect_section_group(
        self,
        seed_record,
        reference: StructuredReference,
        max_chunks=8
    ):

        """Collect all split parts belonging to one exact Section identifier."""

        metadata = seed_record.get(
            "metadata",
            {}
        )

        file_key = (
            metadata.get("file_path")
            or metadata.get("file_name")
        )

        if not file_key:
            return [seed_record]

        matches = []

        for _, record in self._same_file_records_sorted(
            file_key
        ):

            if self._metadata_matches_structured_reference(
                record.get(
                    "metadata",
                    {}
                ),
                reference
            ):
                matches.append(
                    record
                )

            if len(matches) >= max_chunks:
                break

        return matches or [seed_record]

    def _merge_exact_structured_records(
        self,
        records,
        reference: StructuredReference
    ):

        texts = []
        seen_text = set()

        first_metadata = dict(
            records[0].get(
                "metadata",
                {}
            )
        )

        chunk_ids = []
        page_starts = []
        page_ends = []

        for record in records:

            text = str(
                record.get(
                    "text",
                    ""
                )
            ).strip()

            if text and text not in seen_text:
                texts.append(text)
                seen_text.add(text)

            metadata = record.get(
                "metadata",
                {}
            )

            try:
                chunk_ids.append(
                    int(
                        metadata.get(
                            "chunk_id"
                        )
                    )
                )
            except (TypeError, ValueError):
                pass

            for key, target in (
                ("page_start", page_starts),
                ("page_end", page_ends),
            ):
                try:
                    target.append(
                        int(
                            metadata.get(key)
                        )
                    )
                except (TypeError, ValueError):
                    pass

        if chunk_ids:
            first_metadata["exact_chunk_start"] = min(chunk_ids)
            first_metadata["exact_chunk_end"] = max(chunk_ids)

        if page_starts:
            first_metadata["page_start"] = min(page_starts)

        if page_ends:
            first_metadata["page_end"] = max(page_ends)

        first_metadata["exact_structured_match"] = True
        first_metadata["exact_reference"] = reference.display_name
        first_metadata["exact_group_chunks"] = len(records)

        return {
            "text": "\n\n".join(texts),
            "metadata": first_metadata,
            "score": 1.0,
            "rerank_score": 1.0,
            "_exact_structured_match": True,
        }

    def _retrieve_exact_structured_reference(
        self,
        reference: StructuredReference
    ):

        """Metadata-first lookup for Rule/Dir/Section identifiers."""

        records = getattr(
            self.bm25,
            "records",
            []
        ) or []

        seed_records = [
            record
            for record in records
            if self._metadata_matches_structured_reference(
                record.get(
                    "metadata",
                    {}
                ),
                reference
            )
        ]

        if not seed_records:
            return []

        # One logical result per source file/reference. If a section was split
        # into several parts, only its first chunk becomes a seed.
        def seed_sort_key(record):

            metadata = record.get(
                "metadata",
                {}
            )

            try:
                chunk_id = int(
                    metadata.get(
                        "chunk_id",
                        0
                    )
                )
            except (TypeError, ValueError):
                chunk_id = 0

            return (
                str(
                    metadata.get(
                        "file_path",
                        ""
                    )
                ),
                chunk_id,
            )

        seed_records.sort(
            key=seed_sort_key
        )

        results = []
        seen_files = set()

        for seed in seed_records:

            metadata = seed.get(
                "metadata",
                {}
            )

            file_key = (
                metadata.get("file_path")
                or metadata.get("file_name")
            )

            if file_key in seen_files:
                continue

            seen_files.add(file_key)

            if reference.kind in {"rule", "directive"}:
                group = self._collect_rule_or_directive_group(
                    seed,
                    reference
                )
            elif reference.is_section_like:
                group = self._collect_section_group(
                    seed,
                    reference
                )
            else:
                continue

            results.append(
                self._merge_exact_structured_records(
                    group,
                    reference
                )
            )

        return results

    def _filter_for_structured_reference(
        self,
        items,
        reference: StructuredReference
    ):

        return [
            item
            for item in (items or [])
            if self._structured_reference_consistent(
                item,
                reference
            )
        ]

    def _is_short_lookup_query(self, query):

        if not query:
            return False

        words = query.strip().split()

        return len(words) <= 5

    def _is_list_or_relationship_query(
        self,
        query
    ):

        """
        Detect genuine list, multi-answer, procedure,
        and relationship questions.

        Important:
        Only inspect the beginning of the enriched query
        for list/completeness terms.

        This prevents later enrichment terms such as:
            requirements
            rules
            steps
            configuration

        from incorrectly activating list mode.
        """

        if not query:

            return False

        clean = self._normalize_text(
            query
        )

        if not clean:

            return False

        tokens = clean.split()

        # The original normalized question is always
        # placed at the beginning of the enriched query.
        #
        # Inspect only the first few words when deciding
        # whether this is a list/completeness request.
        prefix_tokens = tokens[:8]

        intent_prefix = " ".join(
            prefix_tokens
        )

        # Explicit list-style question starters.
        explicit_patterns = [
            r"^(who are|what are)\b",
            r"^(list|enumerate)\b",
            r"^name the\b",
            r"^give me the list\b",
            r"^(?:give|show|return|provide)\s+(?:the\s+)?(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            r"^(?:give|show|return|provide)\s+(?:the\s+)?(?:names?|records?|entries|items|employees?|people|requirements?|steps?|rules?|options?|values?)\b",
            r"^(sino sino|ano ano|ilista)\b",
            r"^(all|examples|types|categories)\b",

            # Explicit procedure requests
            r"^how to\b",
            r"^step by step\b",
            r"^steps\b",
            r"^procedure for\b",
            r"^procedures for\b",
            r"^process for\b",
            r"^workflow for\b",
        ]

        for pattern in explicit_patterns:

            if re.search(
                pattern,
                intent_prefix
            ):

                return True

        # Completeness terms must appear near the start
        # of the original question, not only in enrichment.
        completeness_terms = {
            "steps",
            "requirements",
            "rules",
            "conditions",
            "exceptions",
            "approvers",
            "parameters",
            "options",
            "commands",
            "errors",
        }

        if set(prefix_tokens).intersection(
            completeness_terms
        ):

            return True

        # Relationship questions may contain their
        # identifying terms later in the sentence.
        relationship_patterns = [
            r"\bladies\b",
            r"\bwomen\b",
            r"\brelationship\b",
            r"\brelationships\b",
            r"\bromantic relationship\b",
            r"\bromantic relationships\b",
            r"\blove interest\b",
            r"\blove interests\b",
            r"\bgirlfriend\b",
            r"\bgirlfriends\b",
            r"\bpersonal life\b",
            r"\bassociated women\b",
            r"\bconnected to\b",
            r"\bassociated with\b",
        ]

        for pattern in relationship_patterns:

            if re.search(
                pattern,
                clean
            ):

                return True

        return False

    def _is_relationship_query(self, query):
        """Return True only for relationship/personal-life list intent."""
        if not query:
            return False
        clean = self._normalize_text(query)
        relationship_patterns = (
            r"\bladies\b", r"\bwomen\b", r"\brelationship(?:s)?\b",
            r"\bromantic\b", r"\blove interests?\b", r"\bgirlfriends?\b",
            r"\bpersonal life\b", r"\bspouses?\b", r"\bwives?\b", r"\bhusbands?\b",
        )
        return any(re.search(pattern, clean) for pattern in relationship_patterns)

    def _final_top_k_for_query(
        self,
        query
    ):

        """
        Use more final chunks for completeness-style questions.
        Keep normal questions concise.
        """

        if self._is_list_or_relationship_query(
            query
        ):

            return max(
                FINAL_TOP_K,
                COMPLETENESS_TOP_K
            )

        return FINAL_TOP_K


    def _source_pdf_rule_family_supplement(
        self,
        source_path,
        major_rule_id,
        metadata_template=None,
    ):
        """Recover missing structured Rule-family members from the source PDF.

        This is a source-backed completeness guard, not a canned answer. It is
        used only after the active BM25 corpus has already proven the relevant
        Rule family but appears incomplete. The authoritative PDF is parsed
        with the same structure extractor used by ingestion, and only explicit
        ``Rule <major>.<n>`` sections are accepted. Results are cached by file
        identity so repeated list questions do not repeatedly parse the PDF.
        """

        path = Path(str(source_path or "")).expanduser()
        if not path.exists() or not path.is_file() or path.suffix.casefold() != ".pdf":
            return []

        try:
            stat = path.stat()
        except OSError:
            return []

        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)

        cache = getattr(self, "_structured_pdf_family_cache", None)
        if cache is None:
            cache = {}
            self._structured_pdf_family_cache = cache

        cache_key = (
            resolved,
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
            str(major_rule_id or ""),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        try:
            from ingestion.pdf_structure import PDFStructureExtractor
            sections = PDFStructureExtractor.extract(str(path))
        except Exception:
            return []

        template = dict(metadata_template or {})
        recovered = []
        family_pattern = re.compile(
            rf"{re.escape(str(major_rule_id or ''))}\.\d+"
        )

        for section_index, section in enumerate(sections or []):
            rule_id = str(getattr(section, "rule_id", "") or "").strip()
            if not family_pattern.fullmatch(rule_id):
                continue

            text = str(getattr(section, "text", "") or "").strip()
            if not text or not re.match(
                rf"^\s*Rule\s+{re.escape(rule_id)}\b",
                text,
                re.IGNORECASE,
            ):
                continue

            metadata = dict(template)
            metadata.update({
                "file_name": path.name,
                "file_path": str(path),
                "folder_name": path.parent.name,
                "extension": path.suffix,
                "parser_strategy": "pdf_structure_direct_recovery",
                "section_index": section_index,
                "section_type": "rule",
                "section_title": f"Rule {rule_id}",
                "rule_id": rule_id,
                "page_start": int(getattr(section, "page_start", 1) or 1),
                "page_end": int(getattr(section, "page_end", 1) or 1),
            })

            recovered.append({
                "text": text,
                "metadata": metadata,
                "score": 1.0,
                "rerank_score": 1.0,
                "_structured_topic_family": "switch statements",
                "_structured_topic_anchor": True,
                "_structured_metadata_recovered": True,
                "_structured_source_recovered": True,
                "_structured_metadata_quality": 3,
            })

        # Keep this cache intentionally small. A normal Option-C run has one
        # technical PDF, but bounded caching avoids unbounded growth if more
        # technical sources are added later.
        if len(cache) >= 8:
            cache.clear()
        cache[cache_key] = [dict(item) for item in recovered]
        return recovered


    def _authoritative_rule_source_candidates(self, metadata_template=None):
        """Return existing authoritative PDF paths for structured recovery.

        Prefer the path stored in retrieval metadata, but never depend on it
        exclusively.  Office indexes can outlive a project move or preserve a
        path spelling that is no longer directly resolvable by the current
        process.  The active technical corpus is therefore also searched by
        source file name.  Only real PDF files are returned.
        """

        metadata = dict(metadata_template or {})
        output = []
        seen = set()

        def add(candidate):
            if not candidate:
                return
            try:
                path = Path(str(candidate)).expanduser()
            except (TypeError, ValueError):
                return
            try:
                if not path.exists() or not path.is_file():
                    return
            except OSError:
                return
            if path.suffix.casefold() != ".pdf":
                return
            try:
                key = str(path.resolve()).casefold()
            except OSError:
                key = str(path).casefold()
            if key in seen:
                return
            seen.add(key)
            output.append(path)

        add(metadata.get("file_path"))

        file_name = str(metadata.get("file_name", "") or "").strip()
        if file_name:
            add(TECHNICAL_DOCUMENT_DIR / file_name)
            try:
                for candidate in TECHNICAL_DOCUMENT_DIR.glob("*.pdf"):
                    if candidate.name.casefold() == file_name.casefold():
                        add(candidate)
            except OSError:
                pass

        # If the active technical corpus contains only one PDF, it is a safe
        # final candidate even when stale index metadata lost the source name.
        if not output:
            try:
                pdfs = [p for p in TECHNICAL_DOCUMENT_DIR.glob("*.pdf") if p.is_file()]
            except OSError:
                pdfs = []
            if len(pdfs) == 1:
                add(pdfs[0])

        return output


    def _prepared_cache_rule_family_supplement(
        self,
        source_path,
        major_rule_id,
        metadata_template=None,
    ):
        """Recover structured Rule-family members from the active ingestion cache.

        The prepared ingestion cache is the exact structure-aware source artifact
        used during index construction.  Reading it gives the runtime a bounded,
        deterministic way to recover an explicit Rule section if a downstream
        BM25/vector record was omitted or lost metadata.  Only literal structured
        Rule entries for the requested major family are accepted; no requirement
        text or member IDs are invented here.
        """

        path = Path(str(source_path or "")).expanduser()
        if not path.exists() or not path.is_file():
            return []

        major = str(major_rule_id or "").strip()
        if not major:
            return []

        try:
            from utils.hash_utils import FileHasher
            from utils.ingestion_cache import IngestionCache

            file_hash = FileHasher.sha256(path)
            payload = IngestionCache.load(file_hash)
        except Exception:
            return []

        if not isinstance(payload, dict) or payload.get("status") != "ready":
            return []

        prepared_chunks = payload.get("prepared_chunks")
        if not isinstance(prepared_chunks, list):
            return []

        family_pattern = re.compile(rf"{re.escape(major)}\.\d+")
        template = dict(metadata_template or {})
        recovered = []

        for cache_index, item in enumerate(prepared_chunks):
            if not isinstance(item, dict):
                continue

            text = str(item.get("text", "") or "").strip()
            metadata = dict(item.get("metadata", {}) or {})
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            section_type = str(metadata.get("section_type", "") or "").strip().casefold()

            if section_type != "rule" or not family_pattern.fullmatch(rule_id):
                continue
            if not text or not re.match(
                rf"^\s*Rule\s+{re.escape(rule_id)}\b",
                text,
                re.IGNORECASE,
            ):
                continue

            item_metadata = dict(template)
            item_metadata.update(metadata)
            item_metadata.update({
                "file_name": path.name,
                "file_path": str(path),
                "folder_name": path.parent.name,
                "extension": path.suffix,
                "section_type": "rule",
                "section_title": f"Rule {rule_id}",
                "rule_id": rule_id,
                "parser_strategy": "prepared_cache_rule_recovery",
                "cache_chunk_index": cache_index,
            })

            recovered.append({
                "text": text,
                "metadata": item_metadata,
                "score": 1.0,
                "rerank_score": 1.0,
                "_structured_topic_family": "switch statements",
                "_structured_topic_anchor": True,
                "_structured_metadata_recovered": True,
                "_structured_cache_recovered": True,
                "_structured_metadata_quality": 4,
            })

        by_id = {}
        for item in recovered:
            rule_id = str((item.get("metadata", {}) or {}).get("rule_id", "")).strip()
            if rule_id and rule_id not in by_id:
                by_id[rule_id] = item
        return list(by_id.values())


    def _source_pdf_text_rule_family_supplement(
        self,
        source_path,
        major_rule_id,
        metadata_template=None,
    ):
        """Recover literal Rule statements from source PDF text as a last source guard.

        This fallback deliberately does not use font/layout heading classification.
        It scans source page text for explicit ``Rule <major>.<n>`` rows and only
        accepts entries whose extracted statement contains normative ``shall`` or
        ``should`` wording.  That rejects attribute-only tables/cross-references
        while remaining dynamic for any Rule family present in the source.
        """

        path = Path(str(source_path or "")).expanduser()
        if not path.exists() or not path.is_file() or path.suffix.casefold() != ".pdf":
            return []

        major = str(major_rule_id or "").strip()
        if not major:
            return []

        try:
            import fitz
        except Exception:
            return []

        rule_line = re.compile(
            rf"^\s*Rule\s+({re.escape(major)}\.\d+)\b(?P<tail>.*)$",
            re.IGNORECASE,
        )
        any_structured = re.compile(
            r"^\s*(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)*\b",
            re.IGNORECASE,
        )
        stop_label = re.compile(
            r"^\s*(?:Category|Analysis|Applies\s+to|Rationale|Amplification|"
            r"Example|Exception|See\s+also)\b",
            re.IGNORECASE,
        )
        category_line = re.compile(r"^(Required|Advisory|Mandatory)\b(.*)$", re.IGNORECASE)
        template = dict(metadata_template or {})
        candidates = {}

        try:
            document = fitz.open(str(path))
        except Exception:
            return []

        try:
            for page_number, page in enumerate(document, start=1):
                try:
                    raw_text = page.get_text("text", sort=True)
                except Exception:
                    continue

                lines = [
                    re.sub(r"\s+", " ", line or "").strip()
                    for line in str(raw_text or "").splitlines()
                ]
                lines = [line for line in lines if line]

                for index, line in enumerate(lines):
                    match = rule_line.match(line)
                    if not match:
                        continue

                    rule_id = match.group(1)
                    tail = re.sub(r"\s+", " ", match.group("tail") or "").strip(" :-\u2013\u2014")
                    category = ""
                    statement_parts = []

                    category_match = category_line.match(tail)
                    if category_match:
                        category = category_match.group(1).title()
                        tail = re.sub(r"\s+", " ", category_match.group(2) or "").strip(" :-\u2013\u2014")

                    if tail:
                        statement_parts.append(tail)

                    for following in lines[index + 1:]:
                        if any_structured.match(following):
                            break
                        if stop_label.match(following):
                            break

                        following_category = category_line.match(following)
                        if following_category and not statement_parts:
                            category = category or following_category.group(1).title()
                            remainder = re.sub(
                                r"\s+", " ", following_category.group(2) or ""
                            ).strip(" :-\u2013\u2014")
                            if remainder:
                                statement_parts.append(remainder)
                            continue

                        # Attribute-only rows such as C90/C99 or decidability
                        # tables are not requirement text.
                        if not statement_parts and re.fullmatch(
                            r"(?:C\d+(?:\s*,\s*C\d+)*)|(?:Decidable|Undecidable).+",
                            following,
                            re.IGNORECASE,
                        ):
                            continue

                        statement_parts.append(following)
                        if len(" ".join(statement_parts)) >= 600:
                            break

                    statement = re.sub(r"\s+", " ", " ".join(statement_parts)).strip()
                    attribute_tail = re.search(
                        r"\s+(?:(?:Section\s+)?Category|RulesCategory)\s+"
                        r"(Required|Advisory|Mandatory)\b",
                        statement,
                        re.IGNORECASE,
                    )
                    if attribute_tail:
                        category = category or attribute_tail.group(1).title()
                        statement = statement[:attribute_tail.start()].strip()
                    statement = re.sub(
                        r"\s+SectionRationale\b.*$",
                        "",
                        statement,
                        flags=re.IGNORECASE,
                    ).strip()
                    if not statement or not re.search(r"\b(?:shall|should)\b", statement, re.IGNORECASE):
                        continue

                    # Keep the earliest normative source occurrence for a Rule.
                    # In MISRA-style documents this favors the primary Rule body
                    # over later summary/attributes/automatic-code appendices.
                    if rule_id in candidates:
                        continue

                    item_metadata = dict(template)
                    item_metadata.update({
                        "file_name": path.name,
                        "file_path": str(path),
                        "folder_name": path.parent.name,
                        "extension": path.suffix,
                        "section_type": "rule",
                        "section_title": f"Rule {rule_id}",
                        "rule_id": rule_id,
                        "page_start": page_number,
                        "page_end": page_number,
                        "parser_strategy": "pdf_text_rule_recovery",
                    })

                    rendered = f"Rule {rule_id}\n{statement}"
                    if category:
                        rendered += f"\nCategory\n{category}"

                    candidates[rule_id] = {
                        "text": rendered,
                        "metadata": item_metadata,
                        "score": 1.0,
                        "rerank_score": 1.0,
                        "_structured_topic_family": "switch statements",
                        "_structured_topic_anchor": True,
                        "_structured_metadata_recovered": True,
                        "_structured_pdf_text_recovered": True,
                        "_structured_metadata_quality": 3,
                    }
        finally:
            try:
                document.close()
            except Exception:
                pass

        return list(candidates.values())


    def _indexed_summary_rule_family_supplement(
        self,
        records,
        major_rule_id,
        metadata_template=None,
    ):
        """Recover missing Rule-family members from an indexed guideline summary.

        This is a second source-backed completeness guard for environments where
        a live PDF re-parse is unavailable or produces a layout-dependent miss.
        Only chunks explicitly identified as a guideline summary are considered,
        and only literal ``Rule <major>.<n>`` headings inside that summary are
        converted to structured family members.  No requirement text is canned.
        """

        major = str(major_rule_id or "").strip()
        if not major:
            return []

        family_heading = re.compile(
            rf"^\s*Rule\s+({re.escape(major)}\.\d+)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        any_rule_heading = re.compile(
            r"^\s*Rule\s+(\d+(?:\.\d+)*)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        template = dict(metadata_template or {})
        recovered = []

        for record in records or []:
            metadata = dict(record.get("metadata", {}) or {})
            text = str(record.get("text", "") or "").strip()
            if not text:
                continue

            section_title = str(metadata.get("section_title", "") or "").casefold()
            is_summary = (
                "summary of guidelines" in section_title
                or bool(re.match(
                    r"^\s*Appendix\s+A\s*:\s*Summary\s+of\s+guidelines\b",
                    text,
                    re.IGNORECASE,
                ))
            )
            if not is_summary:
                continue

            matches = list(family_heading.finditer(text))
            if not matches:
                continue

            all_headings = list(any_rule_heading.finditer(text))
            heading_starts = [match.start() for match in all_headings]

            for match in matches:
                rule_id = match.group(1)
                start = match.start()
                later_starts = [pos for pos in heading_starts if pos > start]
                end = min(later_starts) if later_starts else len(text)
                segment = text[start:end].strip()
                if not segment:
                    continue

                # A summary entry must contain more than the heading itself.
                # This avoids turning bare cross-references into requirements.
                lines = [line.strip() for line in segment.splitlines() if line.strip()]
                if len(lines) < 3:
                    continue

                item_metadata = dict(template)
                item_metadata.update(metadata)
                item_metadata.update({
                    "section_type": "rule",
                    "section_title": f"Rule {rule_id}",
                    "rule_id": rule_id,
                    "parser_strategy": "indexed_summary_rule_recovery",
                })

                recovered.append({
                    "text": segment,
                    "metadata": item_metadata,
                    "score": 1.0,
                    "rerank_score": 1.0,
                    "_structured_topic_family": "switch statements",
                    "_structured_topic_anchor": True,
                    "_structured_metadata_recovered": True,
                    "_structured_summary_recovered": True,
                    "_structured_metadata_quality": 2,
                })

        by_id = {}
        for item in recovered:
            rule_id = str((item.get("metadata", {}) or {}).get("rule_id", "")).strip()
            if rule_id and rule_id not in by_id:
                by_id[rule_id] = item
        return list(by_id.values())


    @staticmethod
    def _structured_catalog_token(value: str) -> str:
        token = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            return token[:-1]
        return token

    def _retrieve_structured_section_title_match(
        self,
        query,
        intent_query=None,
    ):
        """Resolve a near-exact document Section title without semantic models.

        This fast path is intentionally conservative.  It is for short
        noun-phrase/overview questions whose content words substantially match
        one indexed document-section title.  List/rule-family requests remain on
        their dedicated structured inventory paths.
        """

        semantic_query = self._normalize_text(intent_query or query)
        if not semantic_query:
            return []

        if extract_structured_reference(semantic_query) is not None:
            return []

        if self._is_list_or_relationship_query(semantic_query):
            return []

        raw_words = re.findall(r"[a-z0-9]+", semantic_query)
        if len(raw_words) > 14:
            return []

        stop = {
            "what", "which", "who", "where", "when", "why", "how",
            "is", "are", "was", "were", "does", "do", "did",
            "the", "a", "an", "of", "for", "about", "please", "tell",
            "me", "explain", "describe", "show", "give", "overview",
        }
        query_tokens = {
            self._structured_catalog_token(token)
            for token in raw_words
        }
        query_tokens = {
            token for token in query_tokens
            if token and token not in stop and len(token) >= 2
        }
        if len(query_tokens) < 2:
            return []

        records = getattr(self.bm25, "records", []) or []
        candidates = []

        for record in records:
            if not isinstance(record, dict):
                continue
            metadata = dict(record.get("metadata", {}) or {})
            if str(metadata.get("section_type", "") or "").casefold() != "section":
                continue
            if str(metadata.get("section_role", "") or "").casefold() not in {
                "", "document_section"
            }:
                continue

            section_id = str(metadata.get("section_id", "") or "").strip()
            title = str(metadata.get("section_title", "") or "").strip()
            if not section_id or not title:
                continue

            title_without_id = re.sub(
                rf"(?i)^\s*(?:section\s+)?{re.escape(section_id)}\s*[:\-–—]?\s*",
                "",
                title,
            ).strip()
            title_tokens = {
                self._structured_catalog_token(token)
                for token in re.findall(r"[a-z0-9]+", title_without_id.casefold())
            }
            title_tokens = {
                token for token in title_tokens
                if token and token not in stop and len(token) >= 2
            }
            if len(title_tokens) < 2:
                continue

            overlap = query_tokens.intersection(title_tokens)
            if len(overlap) < 2:
                continue

            query_coverage = len(overlap) / max(1, len(query_tokens))
            title_coverage = len(overlap) / max(1, len(title_tokens))
            if query_coverage < 0.70 or title_coverage < 0.55:
                continue

            score = (
                2.0 * query_coverage
                + 1.5 * title_coverage
                + 0.1 * len(overlap)
            )
            candidates.append((score, section_id, title))

        if not candidates:
            return []

        candidates.sort(key=lambda row: row[0], reverse=True)
        best_score, best_id, best_title = candidates[0]

        # Fail closed on materially ambiguous title matches.
        if len(candidates) >= 2 and abs(best_score - candidates[1][0]) < 0.15:
            return []

        reference = StructuredReference(kind="section", identifier=best_id)
        results = self._retrieve_exact_structured_reference(reference)
        for item in results:
            if not isinstance(item, dict):
                continue
            item["_structured_section_title_anchor"] = True
            item["_structured_section_title_score"] = best_score
            item["_structured_section_title"] = best_title
            metadata = dict(item.get("metadata", {}) or {})
            metadata["exact_reference"] = reference.display_name
            item["metadata"] = metadata
        return results


    def _retrieve_structured_rule_catalog(
        self,
        query,
        intent_query=None,
    ):
        """Return source-grounded MISRA rule inventories for natural list queries.

        This path is intentionally metadata/text driven.  It does not hardcode
        answer Rule numbers.  It can enumerate a Category (Mandatory/Required/
        Advisory), derive a rule-family major from an indexed Section 8.x title
        such as ``Unused code`` or ``Pointer type conversions``, or select Rule
        statements that explicitly contain a distinctive requested construct
        such as ``goto``.  Existing switch-statement handling remains on its
        separately certified family path.
        """

        semantic_query = self._normalize_text(intent_query or query)
        if not semantic_query or not self._is_list_or_relationship_query(semantic_query):
            return []
        if not re.search(r"\b(?:rule|rules|guideline|guidelines|misra)\b", semantic_query):
            return []
        if re.search(r"\bswitch\s+(?:statement|statements|case|cases)\b", semantic_query):
            return []

        records = getattr(self.bm25, "records", []) or []
        rule_records = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            metadata = dict(record.get("metadata", {}) or {})
            text = str(record.get("text", "") or "").strip()
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            if (
                str(metadata.get("section_type", "") or "").casefold() != "rule"
                or not re.fullmatch(r"\d+\.\d+", rule_id)
                or not text
            ):
                continue
            # Keep the main requirement chunk, not its rationale/example child.
            role = str(metadata.get("section_role", "") or "").casefold()
            if role and role != "requirement":
                continue
            if not re.match(rf"^\s*Rule\s+{re.escape(rule_id)}\b", text, re.IGNORECASE):
                continue
            rule_records.setdefault(rule_id, {"text": text, "metadata": metadata})

        if not rule_records:
            return []

        family_label = ""
        family_heading = ""
        family_scope_note = ""
        compact_ids = False
        chosen_ids = []

        category_match = re.search(r"\b(mandatory|required|advisory)\b", semantic_query)
        if category_match:
            category = category_match.group(1).capitalize()
            for rule_id, record in rule_records.items():
                match = re.search(
                    r"(?im)^\s*Category\s*$\s*^\s*(Mandatory|Required|Advisory)\s*$",
                    record["text"],
                )
                if match and match.group(1).casefold() == category.casefold():
                    chosen_ids.append(rule_id)
            family_label = f"{category} MISRA"
            family_heading = f"{category} MISRA rules"
        else:
            stop = {
                "what", "which", "are", "the", "misra", "rule", "rules", "guideline", "guidelines",
                "for", "related", "relation", "to", "about", "apply", "applies", "applicable", "list",
                "enumerate", "name", "all", "show", "give", "me", "statements", "statement",
            }
            query_tokens = {
                self._structured_catalog_token(token)
                for token in re.findall(r"[a-z0-9]+", semantic_query)
            }
            query_tokens = {token for token in query_tokens if token and token not in stop and len(token) >= 3}

            # Keep compiler/toolchain switches/options distinct from the C
            # language ``switch`` statement.  Search source-grounded child
            # sections for explicit compiler command-line option/flag wording,
            # then return the parent Rule requirement(s).  This is terminology
            # resolution, not a canned answer list.
            toolchain_option_query = bool(
                re.search(
                    r"\b(?:compiler|toolchain|build|command[- ]?line)\b",
                    semantic_query,
                )
                and re.search(
                    r"\b(?:switch|switches|option|options|flag|flags)\b",
                    semantic_query,
                )
            )
            if toolchain_option_query:
                parent_rule_ids = []
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    metadata = dict(record.get("metadata", {}) or {})
                    parent_rule_id = str(
                        metadata.get("parent_rule_id")
                        or (
                            metadata.get("parent_identifier")
                            if str(metadata.get("parent_type", "") or "").casefold() == "rule"
                            else ""
                        )
                        or ""
                    ).strip()
                    if not re.fullmatch(r"\d+\.\d+", parent_rule_id):
                        continue
                    source_text = self._normalize_text(
                        str(record.get("text", "") or "")
                    )
                    if not re.search(r"\bcompiler\b", source_text):
                        continue
                    if not re.search(
                        r"\b(?:command[- ]?line|option|options|flag|flags)\b",
                        source_text,
                    ):
                        continue
                    if parent_rule_id in rule_records:
                        parent_rule_ids.append(parent_rule_id)

                chosen_ids.extend(parent_rule_ids)
                if chosen_ids:
                    family_label = "compiler option-related MISRA"
                    family_heading = "Compiler option-related MISRA rules"
                    family_scope_note = (
                        "Scope: Rules whose source text explicitly discusses "
                        "compiler/toolchain command-line options or flags. "
                        "This is separate from C switch-statement rules."
                    )
                else:
                    # Do not reinterpret ``compiler switch`` as the C language
                    # construct merely because both contain the word switch.
                    return []

            # A broad query such as "Which MISRA rules apply to pointers?"
            # must not be silently narrowed to the Section 8.11 "Pointer type
            # conversions" family. Build a conservative source-grounded
            # inventory from Rule requirements whose own statement explicitly
            # mentions pointer/pointers. This is broader than Rule 11 while
            # still avoiding inference from rationale/examples/cross-references.
            broad_pointer_query = bool(
                "pointer" in query_tokens
                and re.search(
                    r"\b(?:apply|applies|applicable|related|about|for|which|what)\b",
                    semantic_query,
                )
                and not re.search(
                    r"\b(?:conversion|conversions|convert|casting|cast)\b",
                    semantic_query,
                )
            )
            if broad_pointer_query:
                for rule_id, record in rule_records.items():
                    lines = [
                        line.strip()
                        for line in record["text"].splitlines()
                        if line.strip()
                    ]
                    statement_parts = []
                    for line in lines[1:]:
                        if re.fullmatch(
                            r"(?i)(?:Category|Analysis|Applies to|Rationale|Amplification|"
                            r"Example|Examples|Exception|Exceptions|See also)",
                            line,
                        ):
                            break
                        statement_parts.append(line)
                    statement = self._normalize_text(" ".join(statement_parts))
                    statement_tokens = {
                        self._structured_catalog_token(token)
                        for token in re.findall(r"[a-z0-9]+", statement)
                    }
                    if "pointer" in statement_tokens:
                        chosen_ids.append(rule_id)

                if chosen_ids:
                    family_label = "pointer-related MISRA"
                    family_heading = "Pointer-related MISRA rules"
                    family_scope_note = (
                        "Scope: Rules whose requirement statement explicitly "
                        "mentions a pointer or pointer type. Other MISRA Rules "
                        "may still apply to pointer-using code depending on context."
                    )
                    compact_ids = True

            # Derive Section 8.x topic headings from indexed text, then use the
            # section number as the Rule-family major only when its title shares
            # meaningful words with the user's requested topic.
            heading_scores = {}
            heading_titles = {}
            heading_re = re.compile(r"(?im)^\s*8\.(\d+)\s*(?:\n|\s+)\s*([^\n]{2,90})$")
            for record in records:
                text = str(record.get("text", "") or "")
                for match in heading_re.finditer(text):
                    major = match.group(1)
                    title = re.sub(r"\s+", " ", match.group(2)).strip(" .:-")
                    title_tokens = {
                        self._structured_catalog_token(token)
                        for token in re.findall(r"[a-z0-9]+", title.casefold())
                    }
                    title_tokens = {t for t in title_tokens if t and len(t) >= 3}
                    score = len(query_tokens.intersection(title_tokens))
                    if score > heading_scores.get(major, 0):
                        heading_scores[major] = score
                        heading_titles[major] = title

            if not chosen_ids and heading_scores:
                best_major, best_score = max(heading_scores.items(), key=lambda pair: pair[1])
                if best_score >= 1:
                    chosen_ids = [
                        rule_id for rule_id in rule_records
                        if rule_id.startswith(best_major + ".")
                    ]
                    family_label = heading_titles.get(best_major, "MISRA topic")
                    family_heading = f"{family_label} rules"

            # If no section title maps the natural topic, use only explicit
            # wording in the authoritative Rule statement itself.  Requiring a
            # distinctive token avoids generic list words selecting unrelated
            # rules.
            if not chosen_ids and query_tokens:
                distinctive = {
                    token for token in query_tokens
                    if token not in {"code", "type", "value", "object", "function", "expression"}
                }
                for rule_id, record in rule_records.items():
                    lines = [line.strip() for line in record["text"].splitlines() if line.strip()]
                    statement_parts = []
                    for line in lines[1:]:
                        if re.fullmatch(
                            r"(?i)(?:Category|Analysis|Applies to|Rationale|Amplification|Example|Examples|Exception|Exceptions|See also)",
                            line,
                        ):
                            break
                        statement_parts.append(line)
                    statement = self._normalize_text(" ".join(statement_parts))
                    statement_tokens = {
                        self._structured_catalog_token(token)
                        for token in re.findall(r"[a-z0-9]+", statement)
                    }
                    if distinctive and distinctive.intersection(statement_tokens):
                        # For a multi-token topic, require at least half the
                        # distinctive words; a one-token construct (e.g. goto)
                        # must appear literally in the Rule statement.
                        required = max(1, (len(distinctive) + 1) // 2)
                        if len(distinctive.intersection(statement_tokens)) >= required:
                            chosen_ids.append(rule_id)
                if chosen_ids:
                    topic_words = sorted(distinctive)
                    family_label = " ".join(topic_words) if topic_words else "MISRA topic"
                    family_heading = f"{family_label.capitalize()}-related rules"

        if not chosen_ids:
            return []

        def rule_key(rule_id):
            return tuple(int(part) for part in rule_id.split("."))

        output = []
        for rule_id in sorted(set(chosen_ids), key=rule_key):
            record = rule_records.get(rule_id)
            if not record:
                continue
            metadata = dict(record["metadata"])
            metadata["exact_reference"] = f"Rule {rule_id}"
            output.append({
                "text": record["text"],
                "metadata": metadata,
                "score": 1.0,
                "rerank_score": 1.0,
                "_structured_topic_family": family_label or "MISRA topic",
                "_structured_topic_heading": family_heading or "MISRA rules",
                "_structured_topic_scope_note": family_scope_note,
                "_structured_topic_compact_ids": compact_ids,
                "_structured_topic_anchor": True,
                "_structured_metadata_quality": 2,
            })
        return output

    def _retrieve_structured_rule_topic_family(
        self,
        query,
        intent_query=None,
    ):
        """Return a complete MISRA rule family for explicit topic-list requests.

        v6.4.52: a list request such as "rules for switch statement" can
        already retrieve several correct Rule 16.x chunks, but generic list
        expansion used to reorder those anchors by document position and evict
        them before context construction.  This conservative structured path
        recognizes the well-defined MISRA switch-statement family directly
        from authoritative BM25 metadata and returns every Rule 16.x header.

        The trigger intentionally requires an explicit *switch statement* topic
        phrase so compiler-option wording such as "compiler switch" cannot be
        silently reinterpreted as a C switch statement.
        """

        semantic_query = self._normalize_text(intent_query or query)
        if not semantic_query:
            return []

        # Accept both classic list wording ("what are the rules...") and
        # prospective reviewer guidance ("what rules should I watch before I
        # convert/refactor...").  This is intent-level matching, not a canned
        # question string.
        list_request = self._is_list_or_relationship_query(semantic_query)
        guidance_request = bool(
            re.search(
                r"\b(?:rule|rules|guideline|guidelines|requirements?|misra|checks?)\b",
                semantic_query,
            )
            and re.search(
                r"\b(?:watch|consider|keep\s+in\s+mind|need|kailangan|bantayan|"
                r"applicable|relevant|apply|applies|convert|converting|refactor|refactoring|"
                r"before|check|checks|gawin|gawing|gusto)\b",
                semantic_query,
            )
        )
        if not (list_request or guidance_request):
            return []

        # Keep compiler-option terminology distinct from the C language switch
        # statement. Natural Taglish/English refactoring language can still name
        # the construct without literally saying "switch statement".
        if re.search(
            r"\b(?:compiler|command[- ]?line|build)\s+(?:switch|option|flag)s?\b",
            semantic_query,
        ):
            return []

        c_switch_topic = bool(
            re.search(r"\bswitch\s+(?:statement|statements|case|cases)\b", semantic_query)
            or (
                re.search(r"\bswitch\b", semantic_query)
                and re.search(
                    r"\b(?:if[- ]?else|case|default|convert|refactor|gawin|gawing|gusto)\b",
                    semantic_query,
                )
            )
        )
        if not c_switch_topic:
            return []

        if not re.search(
            r"\b(?:rule|rules|guideline|guidelines|requirements?|misra|checks?)\b",
            semantic_query,
        ):
            return []

        records = getattr(self.bm25, "records", []) or []
        family_by_id = {}

        for record in records:
            metadata = record.get("metadata", {}) or {}
            text = str(record.get("text", "") or "").strip()
            if not text:
                continue

            section_type = str(
                metadata.get("section_type", "") or ""
            ).strip().casefold()
            metadata_rule_id = str(
                metadata.get("rule_id", "") or ""
            ).strip()

            # Prefer exact structured metadata. Older/prepared chunk caches can
            # occasionally preserve the Rule heading text while losing the
            # section_type/rule_id metadata on one boundary chunk. Recover only
            # when the chunk itself starts with an explicit Rule 16.x heading;
            # do not infer a Rule number from nearby prose or cross-references.
            textual_match = re.match(
                r"^\s*Rule\s+(16\.\d+)\b",
                text,
                re.IGNORECASE,
            )
            metadata_valid = (
                section_type == "rule"
                and bool(re.fullmatch(r"16\.\d+", metadata_rule_id))
            )

            if metadata_valid:
                rule_id = metadata_rule_id
                recovered_metadata = dict(metadata)
                quality = 2
            elif textual_match:
                rule_id = textual_match.group(1)
                recovered_metadata = dict(metadata)
                recovered_metadata["section_type"] = "rule"
                recovered_metadata["rule_id"] = rule_id
                recovered_metadata["section_title"] = f"Rule {rule_id}"
                quality = 1
            else:
                continue

            item = {
                "text": text,
                "metadata": recovered_metadata,
                "score": 1.0,
                "rerank_score": 1.0,
                "_structured_topic_family": "switch statements",
                "_structured_topic_anchor": True,
                "_structured_metadata_recovered": quality == 1,
                "_structured_metadata_quality": quality,
            }

            current = family_by_id.get(rule_id)
            if (
                current is None
                or quality > int(current.get("_structured_metadata_quality", 0))
            ):
                family_by_id[rule_id] = item

        # v6.4.53: Windows evidence showed that one otherwise valid family
        # member (Rule 16.7) could be absent from the active BM25 records even
        # after a fresh rebuild, while the authoritative PDF still contained
        # the explicit structured Rule section. Reconcile the already-proven
        # family against the source PDF so completeness does not depend on a
        # single prepared/index record. This remains dynamic/source-grounded:
        # no answer text or Rule 16.7 requirement is hardcoded here.
        if family_by_id:
            template_item = next(iter(family_by_id.values()))
            template_metadata = dict(template_item.get("metadata", {}) or {})
            # v6.4.56: do not depend on the historical absolute path stored in
            # the index.  Resolve the authoritative source from the active
            # technical corpus as well, then reconcile the family from every
            # valid source candidate.  The family major is derived from the
            # structured records rather than from a canned missing member.
            observed_rule_ids = [
                str((item.get("metadata", {}) or {}).get("rule_id", "")).strip()
                for item in family_by_id.values()
            ]
            observed_majors = {
                value.split(".", 1)[0]
                for value in observed_rule_ids
                if re.fullmatch(r"\d+\.\d+", value)
            }
            major_rule_id = next(iter(observed_majors)) if len(observed_majors) == 1 else "16"
            source_candidates = self._authoritative_rule_source_candidates(
                template_metadata
            )
            recovered_ids = set()

            for source_candidate in source_candidates:
                for cache_item in self._prepared_cache_rule_family_supplement(
                    source_path=source_candidate,
                    major_rule_id=major_rule_id,
                    metadata_template=template_metadata,
                ):
                    cache_rule_id = str(
                        (cache_item.get("metadata", {}) or {}).get("rule_id", "")
                    ).strip()
                    if cache_rule_id and cache_rule_id not in family_by_id:
                        family_by_id[cache_rule_id] = cache_item
                        recovered_ids.add(cache_rule_id)

                for source_item in self._source_pdf_rule_family_supplement(
                    source_path=source_candidate,
                    major_rule_id=major_rule_id,
                    metadata_template=template_metadata,
                ):
                    source_rule_id = str(
                        (source_item.get("metadata", {}) or {}).get("rule_id", "")
                    ).strip()
                    if source_rule_id and source_rule_id not in family_by_id:
                        family_by_id[source_rule_id] = source_item
                        recovered_ids.add(source_rule_id)

                for text_item in self._source_pdf_text_rule_family_supplement(
                    source_path=source_candidate,
                    major_rule_id=major_rule_id,
                    metadata_template=template_metadata,
                ):
                    text_rule_id = str(
                        (text_item.get("metadata", {}) or {}).get("rule_id", "")
                    ).strip()
                    if text_rule_id and text_rule_id not in family_by_id:
                        family_by_id[text_rule_id] = text_item
                        recovered_ids.add(text_rule_id)

            evidence_logger.record_event(
                event_name="STRUCTURED FAMILY SOURCE RECONCILIATION",
                status="COMPLETED",
                details={
                    "major_rule_id": major_rule_id,
                    "source_candidates": [str(path) for path in source_candidates],
                    "recovered_references": sorted(recovered_ids),
                },
            )

            # Indexed Appendix A remains a final source-backed fallback.
            for summary_item in self._indexed_summary_rule_family_supplement(
                records=records,
                major_rule_id="16",
                metadata_template=template_metadata,
            ):
                summary_rule_id = str(
                    (summary_item.get("metadata", {}) or {}).get("rule_id", "")
                ).strip()
                if summary_rule_id and summary_rule_id not in family_by_id:
                    family_by_id[summary_rule_id] = summary_item

        family = list(family_by_id.values())
        family.sort(
            key=lambda item: tuple(
                int(part)
                for part in str(item.get("metadata", {}).get("rule_id", "0")).split(".")
            )
        )
        return family

    def _anchor_preserving_list_results(self, ranked, final_top_k):
        """Expand list context without ever evicting accepted anchors."""

        anchors = list(ranked or [])[: max(1, int(final_top_k or 1))]
        if not anchors:
            return []

        expanded = self._expand_with_neighbor_chunks(anchors)
        output = []
        seen = set()

        # Accepted anchors are authoritative.  Neighbor expansion may enrich
        # them but must never replace them merely because a neighboring chunk
        # appears earlier in the source document.
        for item in anchors:
            key = self._chunk_key(item.get("metadata", {}) or {}) or ("id", id(item))
            if key in seen:
                continue
            seen.add(key)
            output.append(item)

        for item in expanded:
            if len(output) >= max(1, int(final_top_k or 1)):
                break
            key = self._chunk_key(item.get("metadata", {}) or {}) or ("id", id(item))
            if key in seen:
                continue
            seen.add(key)
            output.append(item)

        return output

    def _chunk_key(
        self,
        metadata
    ):

        """
        Build a stable key for a chunk using file path/name + chunk_id.
        """

        if not metadata:

            return None

        file_key = (
            metadata.get("file_path")
            or metadata.get("file_name")
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if file_key is None or chunk_id is None:

            return None

        try:

            chunk_id = int(
                chunk_id
            )

        except:

            return None

        return f"{file_key}::{chunk_id}"

    def _get_neighbor_lookup(
        self
    ):

        """
        Build lookup of all indexed chunks from BM25 records.

        BM25 stores the original records, so we can use it
        to fetch chunk_id - 1 / chunk_id + 1 / chunk_id + 2.
        """

        if hasattr(
            self,
            "_neighbor_lookup"
        ):

            return self._neighbor_lookup

        lookup = {}

        records = getattr(
            self.bm25,
            "records",
            []
        )

        for record in records:

            metadata = record.get(
                "metadata",
                {}
            )

            key = self._chunk_key(
                metadata
            )

            if key:

                lookup[key] = record

        self._neighbor_lookup = lookup

        return self._neighbor_lookup

    def _record_to_result(
        self,
        record,
        base_score=0.0,
        expanded_from=None
    ):

        """
        Convert a stored BM25 record back to retrieval result format.
        """

        metadata = dict(
            record.get(
                "metadata",
                {}
            )
        )

        return {
            "text": record.get(
                "text",
                ""
            ),
            "metadata": metadata,
            "score": float(
                base_score
            ),
            "_expanded_neighbor": True,
            "_expanded_from": expanded_from,
        }

    def _expand_with_neighbor_chunks(
        self,
        results
    ):

        """
        Generic context completeness expansion.

        For list/procedure/requirement/rule questions,
        the relevant answer may continue in nearby chunks.

        Example:
        - chunk 10 has the section heading
        - chunk 11 has items 1-5
        - chunk 12 has items 6-10

        This method adds nearby chunks from the same file.
        """

        if not results:

            return results

        lookup = self._get_neighbor_lookup()

        if not lookup:

            return results

        expanded = []
        seen = set()

        seed_results = results[
            :CONTEXT_EXPANSION_MAX_SEEDS
        ]

        for seed in seed_results:

            metadata = seed.get(
                "metadata",
                {}
            )

            file_key = (
                metadata.get("file_path")
                or metadata.get("file_name")
            )

            try:

                chunk_id = int(
                    metadata.get(
                        "chunk_id"
                    )
                )

                total_chunks = int(
                    metadata.get(
                        "total_chunks",
                        chunk_id + 1
                    )
                )

            except:

                key = self._chunk_key(
                    metadata
                )

                if key and key not in seen:

                    seen.add(
                        key
                    )

                    expanded.append(
                        seed
                    )

                continue

            start_chunk = max(
                0,
                chunk_id - CONTEXT_EXPANSION_PREVIOUS_CHUNKS
            )

            end_chunk = min(
                total_chunks - 1,
                chunk_id + CONTEXT_EXPANSION_NEXT_CHUNKS
            )

            for neighbor_id in range(
                start_chunk,
                end_chunk + 1
            ):

                neighbor_metadata = dict(
                    metadata
                )

                neighbor_metadata["chunk_id"] = neighbor_id

                neighbor_key = self._chunk_key(
                    neighbor_metadata
                )

                if not neighbor_key:

                    continue

                if neighbor_key in seen:

                    continue

                record = lookup.get(
                    neighbor_key
                )

                if record:

                    seen.add(
                        neighbor_key
                    )

                    if neighbor_id == chunk_id:

                        seed["_expanded_neighbor"] = False

                        expanded.append(
                            seed
                        )

                    else:

                        expanded.append(
                            self._record_to_result(
                                record=record,
                                base_score=seed.get(
                                    "score",
                                    0.0
                                ) * 0.95,
                                expanded_from=chunk_id
                            )
                        )

        if DEBUG_RETRIEVAL:

            print("\n===== NEIGHBOR CHUNK EXPANSION =====")
            print(f"Seeds used       : {len(seed_results)}")
            print(f"Before expansion : {len(results)}")
            print(f"After expansion  : {len(expanded)}")

            for item in expanded[:15]:

                metadata = item.get(
                    "metadata",
                    {}
                )

                print(
                    f"{metadata.get('file_name', 'Unknown')} "
                    f"chunk={metadata.get('chunk_id', '?')} "
                    f"expanded={item.get('_expanded_neighbor', False)}"
                )

            print("====================================\n")

        return expanded

    def _list_relationship_content_score(
        self,
        query,
        item
    ):

        """
        Score chunks for list / relationship questions.

        For relationship/list questions, intro biography chunks
        should not automatically win. Chunks with relationship,
        wife, romance, marriage, sweetheart, or similar terms
        should rank higher.
        """

        text = item.get(
            "text",
            ""
        )

        if not text:

            item["_list_score"] = 0.0

            return 0.0

        normalized = self._normalize_text(
            text
        )

        padded = f" {normalized} "

        score = float(
            item.get(
                "score",
                0.0
            )
        )

        score += self._query_match_score(
            query,
            text
        )

        positive_terms = {
            " relationship ": 1.20,
            " relationships ": 1.20,
            " romantic ": 1.20,
            " romance ": 1.20,
            " love ": 1.00,
            " lover ": 1.00,
            " sweetheart ": 1.40,
            " courtship ": 1.30,
            " courted ": 1.20,
            " girlfriend ": 1.20,
            " girlfriends ": 1.20,
            " wife ": 1.50,
            " common law wife ": 1.80,
            " married ": 1.20,
            " marriage ": 1.20,
            " fiancee ": 1.20,
            " fiance ": 1.20,
            " engaged ": 1.10,
            " women ": 0.90,
            " woman ": 0.90,
            " ladies ": 0.90,
            " lady ": 0.90,
            " personal life ": 0.90,

            # Useful in biography documents where relationship
            # information may appear as names, family terms,
            # or personal-life references rather than explicit
            # "relationship" wording.
            " common law ": 1.20,
            " partner ": 1.00,
            " spouse ": 1.00,
            " beloved ": 1.00,
            " affection ": 0.90,
            " suitor ": 0.90,
            " admirer ": 0.90,
            " daughter ": 0.70,
            " family ": 0.50,
        }

        positive_hit = False

        for term, weight in positive_terms.items():

            if term in padded:

                score += weight

                positive_hit = True

        negative_terms = [
            " monument ",
            " monuments ",
            " statue ",
            " park ",
            " popular culture ",
            " film ",
            " tv series ",
            " portrayed ",
            " actor ",
            " actress ",
            " frigate ",
            " ship ",
            " coin ",
            " bust ",
            " painting ",
            " portrait ",
            " opera ",
            " movie ",
            " awards ",
            " external links ",
            " references ",
            " bibliography ",
            " further reading ",
            " retrieved ",
            " archived ",
            " http ",
            " https ",
            " www ",
        ]

        for term in negative_terms:

            if term in padded:

                score -= 0.80

        section_signals = [
            " personal life ",
            " early life ",
            " education ",
            " women ",
            " relationship ",
            " relationships ",
            " marriage ",
            " wife ",
            " family ",
        ]

        if any(
            signal in padded
            for signal in section_signals
        ):

            score += 0.80

            positive_hit = True

        # If this is a relationship/list question and the chunk
        # has no relationship signal, lower its priority.
        if not positive_hit:

            score -= 1.00

        item["_list_score"] = score

        return score

    def _prioritize_list_relationship_chunks(
        self,
        query,
        results
    ):

        """
        Prioritize chunks that are useful for relationship/list answers.
        """

        if not results:

            return results

        ranked = sorted(
            results,
            key=lambda item:
                self._list_relationship_content_score(
                    query,
                    item
                ),
            reverse=True
        )

        useful_results = [
            item for item in ranked
            if item.get(
                "_list_score",
                0.0
            ) > -0.50
        ]

        if DEBUG_RETRIEVAL:

            print("\n===== LIST RELATIONSHIP RANKING =====")

            for item in ranked[:15]:

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:180]

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> list_score={item.get('_list_score', 0.0):.4f} "
                    f"hybrid={item.get('score', 0.0):.4f} "
                    f"| {preview}"
                )

            print("====================================\n")

        if useful_results:

            return useful_results

        return ranked

    def _generic_list_content_score(self, query, item):
        """Domain-neutral ranking for list/procedure/table-style questions."""
        if not item:
            return 0.0
        text = str(item.get("text", "") or "")
        if not text:
            item["_generic_list_score"] = 0.0
            return 0.0
        score = float(item.get("_info_score", item.get("score", 0.0)) or 0.0)
        score += self._query_match_score(query, text)
        normalized = self._normalize_text(text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 3:
            score += 0.35
        if re.search(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", text):
            score += 0.35
        if text.count(",") >= 2 or text.count(";") >= 2:
            score += 0.20
        if re.search(
            r"\b(?:requirements?|steps?|items?|names?|employees?|records?|positions?|departments?|options?|categories?|types?|rules?)\b",
            normalized,
        ):
            score += 0.20
        if self._is_reference_section(item) and not self._query_requests_reference_material(query):
            score -= 3.0
        item["_generic_list_score"] = score
        return score

    def _prioritize_generic_list_chunks(self, query, results):
        if not results:
            return results
        return sorted(
            results,
            key=lambda item: self._generic_list_content_score(query, item),
            reverse=True,
        )

    def _focus_top_source_for_short_query(self, query, results):

        if not results:
            return results

        if not self._is_short_lookup_query(query):
            return results

        top_file = results[0]["metadata"].get(
            "file_name",
            "Unknown"
        )

        focused = [
            item for item in results
            if item["metadata"].get(
                "file_name",
                "Unknown"
            ) == top_file
        ]

        if DEBUG_RETRIEVAL:

            print("\n===== SHORT QUERY SOURCE FOCUS =====")
            print(f"Query    : {query}")
            print(f"Top File : {top_file}")
            print(f"Chunks   : {len(focused)}")
            print("===================================\n")

        if focused:
            return focused

        return results

    def _sentence_count(self, text):

        if not text:
            return 0

        return len(
            re.findall(
                r"[.!?]",
                text
            )
        )

    def _has_definition_signal(self, text):

        if not text:
            return False

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        # Important:
        # Do NOT treat plain "is", "was", or "are" as definition.
        # Example low-value chunk:
        # "It is located at Earl Bales Park..."
        #
        # That is not a useful definition for "Who is Jose Rizal?"
        definition_signals = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " refers to ",
            " known as ",
            " known for ",
            " served as ",
            " born ",
            " died ",
            " filipino ",
            " nationalist ",
            " writer ",
            " polymath ",
            " employee ",
            " employees ",
            " policy ",
            " procedure ",
            " standard ",
            " requirement ",
            " requirements ",
            " must ",
            " shall ",
        ]

        return any(
            signal in padded
            for signal in definition_signals
        )

    def _reference_noise_score(self, text):

        if not text:
            return 0

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        noise_terms = [
            " retrieved ",
            " archived ",
            " archive ",
            " http ",
            " https ",
            " www ",
            " isbn ",
            " issn ",
            " doi ",
            " references ",
            " external links ",
            " bibliography ",
            " further reading ",
            " web archive ",
            " wayback machine ",
            " google ",
            " abs cbn ",
            " gma news ",
            " inquirer ",
            " news ",
        ]

        score = 0

        for term in noise_terms:

            if term in padded:
                score += 1

        url_count = (
            text.lower().count("http")
            + text.lower().count("www.")
        )

        score += min(
            url_count,
            5
        )

        return score

    def _query_requests_reference_material(self, query):

        if not query:
            return False

        normalized = f" {self._normalize_text(query)} "

        reference_terms = (
            " citation ",
            " citations ",
            " reference ",
            " references ",
            " bibliography ",
            " external links ",
            " source list ",
            " sources list ",
            " works cited ",
        )

        return any(
            term in normalized
            for term in reference_terms
        )

    def _is_reference_section(self, item):

        """Return True only for chunks that are clearly reference material.

        This deliberately uses section metadata when available and a
        conservative first-line fallback. URLs inside a real biography or
        policy paragraph do not automatically make that paragraph a
        reference section.
        """

        if not item:
            return False

        metadata = item.get(
            "metadata",
            {}
        ) or {}

        section_title = self._normalize_text(
            str(
                metadata.get(
                    "section_title",
                    ""
                )
            )
        )

        text = str(
            item.get(
                "text",
                ""
            )
        ).strip()

        first_line = ""

        if text:
            first_line = self._normalize_text(
                text.splitlines()[0]
            )

        reference_headings = {
            "citation",
            "citations",
            "reference",
            "references",
            "bibliography",
            "external links",
            "further reading",
            "sources",
            "source",
            "notes",
            "works cited",
        }

        def is_heading(value):

            if not value:
                return False

            if value in reference_headings:
                return True

            return any(
                value.startswith(
                    heading + " "
                )
                for heading in reference_headings
            )

        return (
            is_heading(section_title)
            or is_heading(first_line)
        )

    def _citation_marker_score(self, text):

        if not text:
            return 0

        clean_text = text.strip()

        citation_markers = re.findall(
            r"\[\d+\]",
            clean_text
        )

        score = min(
            len(citation_markers),
            8
        )

        # Stronger penalty only when chunk starts like a reference item.
        if re.match(r"^\[\d+\]", clean_text):
            score += 5

        return score

    def _query_match_score(self, query, text):

        if not query or not text:
            return 0.0

        query_tokens = self._normalize_text(query).split()
        text_normalized = self._normalize_text(text)

        if not query_tokens:
            return 0.0

        # Remove duplicate query tokens.
        # This prevents enriched queries like:
        # "jose rizal jose rizal biography jose rizal life..."
        # from over-boosting chunks just because they repeat the name.
        unique_tokens = []

        seen = set()

        for token in query_tokens:

            if token not in seen:

                seen.add(token)
                unique_tokens.append(token)

        score = 0.0
        first_part = text_normalized[:900]

        for token in unique_tokens:

            if len(token) <= 2:
                continue

            if token in first_part:
                score += 0.25

            elif token in text_normalized:
                score += 0.08

        # Cap query match score so it does not overpower
        # intro/definition quality.
        return min(
            score,
            1.25
        )

    def _meaningful_query_tokens(
        self,
        query
    ):

        """
        Extract useful tokens from the query for source/topic matching.

        Removes filler words so:
            "the treaty of paris in detail"
        becomes:
            ["treaty", "paris"]
        """

        normalized = self._normalize_text(
            query
        )

        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "with",
            "about",
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "can",
            "you",
            "please",
            "explain",
            "describe",
            "tell",
            "me",
            "detail",
            "details",
            "definition",
            "overview",
            "description",
            "summary",
            "background",
            "important",
            "information",
            "key",
            "facts",
            "biography",
            "life",
            "known",
        }

        tokens = []

        seen = set()

        for token in normalized.split():

            if len(token) <= 2:

                continue

            if token in stop_words:

                continue

            if token not in seen:

                seen.add(
                    token
                )

                tokens.append(
                    token
                )

        return tokens

    def _file_topic_tokens(
        self,
        metadata
    ):

        """
        Extract meaningful tokens from the source file name.

        Examples:
            Treaty of Paris (1898) - Wikipedia.pdf
            -> ["treaty", "paris"]

            MISRA_FromInternet.pdf
            -> ["misra"]
        """

        if not metadata:

            return []

        file_name = decode_unicode_markers(
            metadata.get(
                "file_name",
                ""
            )
        ).strip()

        if not file_name:

            return []

        # Remove extension.
        topic = re.sub(
            r"\.[^.]+$",
            "",
            file_name
        )

        # Remove common source suffixes.
        topic = re.sub(
            r"\s*-\s*Wikipedia$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        topic = re.sub(
            r"[_\s-]*FromInternet$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        # Remove year or extra text in parentheses.
        topic = re.sub(
            r"\([^)]*\)",
            " ",
            topic
        )

        normalized = self._normalize_text(
            topic
        )

        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "with",
            "about",
            "wikipedia",
            "frominternet",
            "pdf",
            "docx",
            "txt",
        }

        tokens = []

        seen = set()

        for token in normalized.split():

            if len(token) <= 2:

                continue

            if token in stop_words:

                continue

            if token not in seen:

                seen.add(
                    token
                )

                tokens.append(
                    token
                )

        return tokens

    def _source_topic_score(
        self,
        query,
        item
    ):

        """
        Boost chunks whose source file name strongly matches the query topic.

        This helps exact-topic questions prefer the exact document.

        Example:
            Query: Treaty of Paris in detail
            File : Treaty of Paris (1898) - Wikipedia.pdf
            -> boosted

            File : Spanish-American War - Wikipedia.pdf
            -> not boosted
        """

        if not query or not item:

            return 0.0

        metadata = item.get(
            "metadata",
            {}
        )

        file_tokens = self._file_topic_tokens(
            metadata
        )

        query_tokens = self._meaningful_query_tokens(
            query
        )

        if not file_tokens or not query_tokens:

            return 0.0

        query_set = set(
            query_tokens
        )

        matched_tokens = [
            token for token in file_tokens
            if token in query_set
        ]

        # Single-token technical/source topic.
        # Example:
        # MISRA_FromInternet.pdf + query "misra"
        if (
            len(file_tokens) == 1
            and file_tokens[0] in query_set
            and len(file_tokens[0]) >= 4
        ):

            return 2.00

        # Multi-token exact topic.
        # Example:
        # Treaty of Paris -> treaty + paris
        if len(file_tokens) >= 2:

            coverage = (
                len(matched_tokens)
                / len(file_tokens)
            )

            if coverage >= 0.70:

                return 2.50

            if len(matched_tokens) >= 2:

                return 1.25

        return 0.0

    def _intro_definition_score(self, query, text):

        if not query or not text:
            return 0.0

        normalized_query = self._normalize_text(query)
        normalized_text = self._normalize_text(text)

        first_part = normalized_text[:1000]
        padded_first = f" {first_part} "

        score = 0.0

        if normalized_query in first_part:
            score += 0.40

        intro_patterns = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " filipino ",
            " employee ",
            " employees ",
            " policy ",
            " procedure ",
        ]

        if any(
            pattern in padded_first
            for pattern in intro_patterns
        ):
            score += 0.80

        return score

    def _is_identity_lookup_query(self, query):

        if not query:
            return False

        normalized = self._normalize_text(query)
        padded = f" {normalized} "

        identity_terms = [
            " biography ",
            " profile ",
            " description ",
            " described ",
            " identified ",
            " life ",
            " known for ",
            " important facts ",
            " who is ",
            " who was ",
            " sino si ",
            " sino ang ",
        ]

        return any(
            term in padded
            for term in identity_terms
        )


    def _identity_noise_score(self, text):

        if not text:
            return 0

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        # These terms are usually not useful for answering:
        # "Who is X?"
        identity_noise_terms = [
            " monument ",
            " statue ",
            " park ",
            " located at ",
            " unveiled ",
            " gift from ",
            " community of ",
            " city of ",
            " film ",
            " tv series ",
            " portrayed ",
            " actor ",
            " actress ",
            " plays ",
            " book ",
            " books ",
            " authored ",
            " google books ",
            " retrieved ",
            " archived ",
            " references ",
            " bibliography ",
            " external links ",
            " further reading ",
            " isbn ",
            " issn ",
            " doi ",
            " http ",
            " https ",
            " www ",
        ]

        score = 0

        for term in identity_noise_terms:

            if term in padded:
                score += 1

        return score


    def _strong_intro_score(self, query, text):

        if not query or not text:
            return 0.0

        normalized_query = self._normalize_text(query)
        normalized_text = self._normalize_text(text)

        first_part = normalized_text[:700]
        padded_first = f" {first_part} "

        score = 0.0

        query_tokens = normalized_query.split()

        meaningful_tokens = [
            token for token in query_tokens
            if len(token) > 2
            and token not in {
                "biography",
                "life",
                "known",
                "for",
                "background",
                "important",
                "facts",
                "overview",
                "summary",
                "description",
            }
        ]

        meaningful_tokens = list(
            dict.fromkeys(
                meaningful_tokens
            )
        )

        topic_match_count = 0

        for token in meaningful_tokens:

            if token in first_part:
                topic_match_count += 1

        if topic_match_count >= 2:
            score += 0.80

        strong_intro_patterns = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " was a filipino ",
            " was an filipino ",
            " filipino nationalist ",
            " writer and polymath ",
            " born ",
            " died ",
        ]

        if any(
            pattern in padded_first
            for pattern in strong_intro_patterns
        ):
            score += 1.20

        # Strong boost if this looks like the first paragraph
        # of an article/profile.
        if (
            topic_match_count >= 2
            and self._sentence_count(text) >= 1
            and len(text.split()) >= 40
        ):
            score += 0.50

        return score

    def _short_chunk_has_strong_query_evidence(self, query, item):
        """Preserve short factual chunks only when retrieval evidence is unusually strong.

        This prevents a legitimate compact company record from being discarded only
        because it contains fewer than 25 words, while keeping the existing noise and
        confidence guardrails intact.
        """
        if not query or not item:
            return False

        text = str(item.get("text", "")).strip()
        if not text or len(text.split()) >= 60:
            return False

        if (
            self._is_reference_section(item)
            and not self._query_requests_reference_material(query)
        ):
            return False

        if self._reference_noise_score(text) >= 4:
            return False

        if self._citation_marker_score(text) >= 6:
            return False

        retrieval_score = float(item.get("score", 0.0) or 0.0)
        query_match_score = self._query_match_score(query, text)

        meaningful_tokens = self._meaningful_query_tokens(query)
        normalized_text = self._normalize_text(text)
        matched_tokens = sum(
            1 for token in meaningful_tokens
            if token in normalized_text
        )
        token_coverage = (
            matched_tokens / len(meaningful_tokens)
            if meaningful_tokens
            else 0.0
        )

        return (
            retrieval_score >= 0.75
            and query_match_score >= 0.50
            and token_coverage >= 0.50
        )

    def _looks_like_low_value_chunk(self, text, query=None, item=None):

        if not text:
            return True

        clean_text = text.strip()
        words = clean_text.split()

        noise_score = self._reference_noise_score(clean_text)
        citation_score = self._citation_marker_score(clean_text)
        has_definition = self._has_definition_signal(clean_text)
        strong_short_evidence = self._short_chunk_has_strong_query_evidence(
            query,
            item,
        )

        # Reject strong reference / URL chunks,
        # but do not reject real intro/definition paragraphs.
        if noise_score >= 8 and not has_definition:
            return True

        # Reject chunks that start like citation/reference list items.
        if citation_score >= 8 and re.match(r"^\[\d+\]", clean_text):
            return True

        if (
            len(words) < 25
            and not has_definition
            and not strong_short_evidence
        ):
            return True

        if (
            len(words) < 60
            and self._sentence_count(clean_text) == 0
            and not has_definition
            and not strong_short_evidence
        ):
            return True

        return False

    def _information_score(self, query, item):

        text = item.get(
            "text",
            ""
        )

        base_score = float(
            item.get(
                "score",
                0.0
            )
        )

        words = text.split()

        noise_score = self._reference_noise_score(text)
        citation_score = self._citation_marker_score(text)
        identity_noise_score = self._identity_noise_score(text)

        has_definition = self._has_definition_signal(text)

        is_identity_lookup = self._is_identity_lookup_query(
            query
        )

        is_reference_section = self._is_reference_section(
            item
        )

        reference_requested = (
            self._query_requests_reference_material(
                query
            )
        )

        info_score = base_score

        info_score += self._query_match_score(
            query,
            text
        )

        source_topic_score = self._source_topic_score(
            query,
            item
        )

        info_score += source_topic_score

        info_score += self._intro_definition_score(
            query,
            text
        )

        # Extra boost for "Who is X?" / biography-style queries.
        if is_identity_lookup:

            info_score += self._strong_intro_score(
                query,
                text
            )

        if len(words) >= 40:
            info_score += 0.10

        if len(words) >= 100:
            info_score += 0.15

        if has_definition:
            info_score += 0.45

        if self._sentence_count(text) >= 2:
            info_score += 0.10

        # Penalize URL/reference noise.
        info_score -= min(
            noise_score * 0.18,
            1.20
        )

        # Extra penalty for identity/person lookup when the chunk
        # is about monuments, parks, films, books, references, etc.
        if is_identity_lookup:

            info_score -= min(
                identity_noise_score * 0.35,
                2.10
            )

        # Reference sections can contain the subject name many times and
        # therefore score well lexically, but they are normally poor evidence
        # for identity/overview questions. Penalize confirmed reference
        # sections before reranking while still allowing explicit citation or
        # bibliography queries to retrieve them.
        if (
            is_reference_section
            and not reference_requested
        ):

            info_score -= (
                6.00
                if is_identity_lookup
                else 2.50
            )

        # Citation markers are normal in Wikipedia intro paragraphs.
        # Penalize lightly when the chunk has a real definition.
        if has_definition:

            info_score -= min(
                citation_score * 0.05,
                0.40
            )

        else:

            info_score -= min(
                citation_score * 0.20,
                1.50
            )

        if self._looks_like_low_value_chunk(
            text,
            query=query,
            item=item,
        ):

            info_score -= 1.00

        item["_source_topic_score"] = source_topic_score
        item["_info_score"] = info_score
        item["_noise_score"] = noise_score
        item["_citation_score"] = citation_score
        item["_identity_noise_score"] = identity_noise_score
        item["_has_definition"] = has_definition
        item["_is_reference_section"] = (
            is_reference_section
        )

        return info_score

    def _prioritize_informative_chunks(self, query, results):

        if not results:
            return results

        ranked = sorted(
            results,
            key=lambda item: self._information_score(
                query,
                item
            ),
            reverse=True
        )

        if DEBUG_RETRIEVAL:

            print("\n===== INFORMATIVE CHUNK RANKING =====")

            for item in ranked[:10]:

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:150]

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> hybrid={item.get('score', 0.0):.4f} "
                    f"info={item.get('_info_score', 0.0):.4f} "
                    f"source={item.get('_source_topic_score', 0.0):.2f} "
                    f"noise={item.get('_noise_score', 0)} "
                    f"id_noise={item.get('_identity_noise_score', 0)} "
                    f"cite={item.get('_citation_score', 0)} "
                    f"def={item.get('_has_definition', False)} "
                    f"| {preview}"
                )

            print("====================================\n")

        return ranked

    def _remove_low_information_chunks(
        self,
        query,
        results
    ):

        if not results:
            return results

        reference_requested = (
            self._query_requests_reference_material(
                query
            )
        )

        useful_results = []

        for item in results:

            if (
                not reference_requested
                and self._is_reference_section(
                    item
                )
            ):

                continue

            if self._looks_like_low_value_chunk(
                item.get(
                    "text",
                    ""
                ),
                query=query,
                item=item,
            ):

                continue

            useful_results.append(
                item
            )

        # If everything is filtered out,
        # do not return empty. Keep the best sorted candidates.
        if useful_results:
            return useful_results

        if self._is_identity_lookup_query(
            query
        ):

            return []

        return results

    def _remove_identity_noise_chunks(
        self,
        query,
        results
    ):

        if not results:
            return results

        if not self._is_identity_lookup_query(
            query
        ):
            return results

        useful_results = []

        for item in results:

            if (
                self._is_reference_section(
                    item
                )
                and not self._query_requests_reference_material(
                    query
                )
            ):

                continue

            identity_noise_score = item.get(
                "_identity_noise_score",
                0
            )

            info_score = item.get(
                "_info_score",
                0.0
            )

            text = item.get(
                "text",
                ""
            )

            normalized_text = self._normalize_text(
                text
            )

            first_part = normalized_text[:900]
            padded_first = f" {first_part} "

            # These are strong person-biography signals.
            # Do NOT include generic "is a" / "was a" here,
            # because monument chunks can also contain that.
            true_biography_patterns = [
                " filipino nationalist ",
                " writer and polymath ",
                " ophthalmologist ",
                " national hero ",
                " propaganda movement ",
                " born ",
                " died ",
                " execution ",
                " spanish colonial period ",
                " political reforms ",
            ]

            has_true_biography_signal = any(
                pattern in padded_first
                for pattern in true_biography_patterns
            )

            # Strict rule:
            # If this chunk has high identity noise and does NOT have
            # strong biography signals, remove it immediately.
            if (
                identity_noise_score >= 6
                and not has_true_biography_signal
            ):

                continue

            # Remove weak medium-noise chunks.
            if (
                identity_noise_score >= 4
                and info_score < 1.50
                and not has_true_biography_signal
            ):

                continue

            useful_results.append(
                item
            )

        if useful_results:
            return useful_results

        # For an identity lookup, returning a known noisy/reference chunk is
        # worse than returning no context because it invites a hallucinated
        # biography. Let the normal no-context fallback handle this safely.
        return []

    def _prepare_identity_reranker_candidates(
        self,
        query,
        ranked_candidates,
        filtered_candidates,
        limit=8,
        minimum_pool=6
    ):

        """Keep a useful multi-candidate pool for identity reranking.

        Earlier filtering could collapse a biography lookup to one reference
        chunk before the CrossEncoder had a chance to compare alternatives.
        This method keeps filtered candidates first, then backfills from the
        informative ranking while excluding confirmed reference sections.
        """

        selected = []
        selected_ids = set()

        def add_candidate(item):

            item_id = id(item)

            if item_id in selected_ids:
                return

            if (
                self._is_reference_section(
                    item
                )
                and not self._query_requests_reference_material(
                    query
                )
            ):

                return

            selected_ids.add(
                item_id
            )

            selected.append(
                item
            )

        for item in filtered_candidates:

            add_candidate(
                item
            )

            if len(selected) >= limit:
                return selected[:limit]

        target_pool = min(
            minimum_pool,
            limit
        )

        if len(selected) < target_pool:

            for item in ranked_candidates:

                add_candidate(
                    item
                )

                if len(selected) >= target_pool:
                    break

        return selected[:limit]

    def _extract_compound_facets(self, intent_query):

        """Return independently requested clauses for a compound question.

        This intentionally works from the non-enriched resolved question so
        topic-expansion terms cannot create artificial facets. It supports
        common English and Tagalog interrogatives and only splits when the
        conjunction is followed by another explicit question word.
        """

        if not intent_query:
            return []

        clean = re.sub(
            r"\s+",
            " ",
            str(intent_query).strip()
        )

        clean = re.sub(
            r"^(?:please\s+)?(?:explain|describe|ipaliwanag|ilarawan)\s+",
            "",
            clean,
            flags=re.IGNORECASE,
        )

        interrogative = (
            r"(?:who|what|when|where|why|how|which|"
            r"sino|ano|anong|kailan|saan|bakit|paano|alin)"
        )
        imperative = (
            r"(?:explain|describe|summari[sz]e|define|list|enumerate|"
            r"give|show|provide|tell|clarify|elaborate|"
            r"ipaliwanag|ilarawan|ibuod|ilista|ibigay)"
        )
        request_starter = rf"(?:{interrogative}|{imperative})"

        # Preserve independently requested clauses whether the second request
        # follows a conjunction or begins after a question/semicolon boundary.
        parts = re.split(
            rf"(?:\s+\b(?:and|at)\b\s+(?:also\s+|din\s+|rin\s+)?"
            rf"(?={request_starter}\b)|"
            rf"[?;]\s*(?:please\s+|paki\s*)?(?={request_starter}\b))",
            clean,
            flags=re.IGNORECASE,
        )

        facets = [
            part.strip(" .?;:")
            for part in parts
            if part.strip(" .?;:")
        ]

        if len(facets) < 2:
            # Explicit two-event temporal comparisons are compound retrieval
            # intents even though they often contain only one interrogative.
            # Keep both event phrases so one high-scoring event cannot crowd
            # the other out before reranking.
            tagalog_match = re.match(
                r"^\s*(?:mas\s+)?(?:nauna|sumunod)\s+(?:ba\s+)?(.+?)\s+kaysa\s+(.+?)[?!.]*$",
                clean,
                flags=re.IGNORECASE,
            )
            if tagalog_match:
                facets = [tagalog_match.group(1).strip(), tagalog_match.group(2).strip()]

        if len(facets) < 2:
            tagalog_choice = re.match(
                r"^\s*alin\s+ang\s+(?:nauna|sumunod)\s*:\s*(.+?)\s+(?:o|or)\s+(.+?)[?!.]*$",
                clean,
                flags=re.IGNORECASE,
            )
            if tagalog_choice:
                facets = [tagalog_choice.group(1).strip(), tagalog_choice.group(2).strip()]

        if len(facets) < 2:
            english_match = re.match(
                r"^\s*(?:was|is|did)\s+(.+?)\s+"
                r"(?:before\s+or\s+after|before|after|earlier\s+than|later\s+than)\s+"
                r"(.+?)[?!.]*$",
                clean,
                flags=re.IGNORECASE,
            )
            if english_match:
                facets = [english_match.group(1).strip(), english_match.group(2).strip()]

        if len(facets) < 2:
            choice_match = re.match(
                r"^\s*which\s+came\s+(?:later|first|earlier)\s*:\s*(.+?)\s+or\s+(.+?)[?!.]*$",
                clean,
                flags=re.IGNORECASE,
            )
            if choice_match:
                facets = [choice_match.group(1).strip(), choice_match.group(2).strip()]

        if len(facets) < 2:
            compare_match = re.match(
                r"^\s*compare\s+(.+?)\s+(?:with|to|and)\s+(.+?)(?:[.?!]|$)",
                clean,
                flags=re.IGNORECASE,
            )
            if compare_match:
                facets = [compare_match.group(1).strip(), compare_match.group(2).strip()]

        if len(facets) < 2:
            chronological_match = re.match(
                r"^\s*(?:put|place|arrange|order)\b.*?\bchronological\s+order\b.*?:\s*(.+?)\s+(?:and|or|at|o)\s+(.+?)(?:[.?!]|$)",
                clean,
                flags=re.IGNORECASE,
            )
            if chronological_match:
                facets = [chronological_match.group(1).strip(), chronological_match.group(2).strip()]

        if len(facets) < 2:
            return []

        return facets[:4]

    @staticmethod
    def _is_temporal_comparison_intent(intent_query):
        clean = re.sub(r"\s+", " ", str(intent_query or "")).strip(" .?!")
        if not clean:
            return False

        patterns = (
            r"\b(?:before|after|earlier|later)\b.+\bthan\b",
            r"\bbefore\s+or\s+after\b",
            r"^\s*(?:was|is|did)\s+.+\s+(?:before|after|earlier\s+than|later\s+than)\s+.+$",
            r"\b(?:nauna|sumunod)\b.+\bkaysa\b",
            r"\bmas\s+nauna\b.+\bkaysa\b",
            r"^\s*which\s+came\s+(?:later|first|earlier)\s*:",
            r"^\s*alin\s+ang\s+(?:nauna|sumunod)\s*:",
            r"^\s*compare\s+.+\s+(?:with|to|and)\s+.+\b(?:date|dates|first|earlier|later|chronological)\b",
            r"\bchronological\s+order\b.*?:\s*.+\s+(?:and|or|at|o)\s+.+",
        )
        return any(re.search(pattern, clean, re.IGNORECASE) for pattern in patterns)

    def _compound_anchor_terms(self, first_facet):

        """Extract a small shared subject anchor from the first facet."""

        if not first_facet:
            return ""

        stopwords = {
            "who", "what", "when", "where", "why", "how", "which",
            "sino", "ano", "anong", "kailan", "saan", "bakit", "paano", "alin",
            "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "can", "could", "should", "would", "will",
            "has", "have", "had", "the", "a", "an", "of", "to", "for",
            "in", "on", "at", "and", "or", "about", "say", "says", "said",
            "tell", "me", "please", "explain", "describe",
        }

        tokens = [
            token
            for token in self._normalize_text(first_facet).split()
            if len(token) > 2 and token not in stopwords
        ]

        # A compact anchor is enough to make a pronoun-heavy later facet
        # retrievable without overwhelming its own intent words.
        return " ".join(tokens[:5])

    def _compound_facet_queries(self, intent_query):

        facets = self._extract_compound_facets(intent_query)

        if not facets:
            return []

        anchor = self._compound_anchor_terms(facets[0])
        queries = []

        for index, facet in enumerate(facets):
            query = facet

            # Preserve relation meaning across multilingual temporal facets.
            # These additions are generic retrieval synonyms only; final
            # answers still require the normal reranker/confidence/grounding
            # gates.
            normalized_facet = self._normalize_text(facet)
            if re.search(r"\b(?:pagpirma|nilagdaan|lumagda|pinirmahan)\b", normalized_facet):
                query = f"{query} signed signing date"
            if re.search(r"\b(?:deklarasyon|idineklara|declaration|declared)\b", normalized_facet):
                query = f"{query} declaration declared date"

            if index > 0 and anchor:
                normalized = f" {self._normalize_text(facet)} "
                pronoun_heavy = any(
                    token in normalized
                    for token in (
                        " he ", " she ", " it ", " they ", " them ",
                        " his ", " her ", " its ", " their ",
                        " siya ", " nito ", " niya ", " nila ",
                    )
                )

                meaningful = self._meaningful_query_tokens(facet)

                # Append the shared subject only when the later clause is
                # genuinely reference-dependent. A complete short clause such
                # as "who must approve a leave request" already has its own
                # subject and should not inherit residue such as
                # "much sick leave provided" from the first facet.
                if pronoun_heavy or len(meaningful) <= 2:
                    query = f"{facet} {anchor}".strip()

            queries.append(query)

        return queries

    @staticmethod
    def _candidate_identity(item):

        metadata = item.get("metadata", {}) or {}
        return (
            metadata.get("file_path")
            or metadata.get("file_name")
            or "Unknown",
            str(metadata.get("chunk_id", "")),
        )

    def _augment_compound_candidates(
        self,
        intent_query,
        merged,
        per_facet_top_k=8,
    ):

        """Add candidates that cover each explicit facet of a compound query.

        The final answer is still protected by the normal informative filters,
        CrossEncoder reranker, confidence threshold, and grounding verifier.
        This helper only prevents one clause from crowding another clause out
        before reranking.
        """

        facet_queries = self._compound_facet_queries(intent_query)

        if not facet_queries:
            return merged, []

        by_key = {
            self._candidate_identity(item): item
            for item in merged
        }
        added = 0

        for facet_index, facet_query in enumerate(facet_queries, start=1):
            bm25_results = self.bm25.search(
                facet_query,
                per_facet_top_k
            )
            vector_results = self._get_vector_searcher().search(
                facet_query,
                per_facet_top_k
            )
            facet_results = self.hybrid.merge(
                vector_results,
                bm25_results
            )

            for item in facet_results:
                key = self._candidate_identity(item)
                existing = by_key.get(key)

                if existing is None:
                    cloned = dict(item)
                    cloned["metadata"] = dict(item.get("metadata", {}))
                    cloned["_compound_facet_hits"] = [facet_index]
                    by_key[key] = cloned
                    added += 1
                    continue

                hits = list(existing.get("_compound_facet_hits", []))
                if facet_index not in hits:
                    hits.append(facet_index)
                existing["_compound_facet_hits"] = hits
                existing["score"] = max(
                    float(existing.get("score", 0.0)),
                    float(item.get("score", 0.0)),
                )

        augmented = list(by_key.values())
        augmented.sort(
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )

        evidence_logger.record_event(
            event_name="COMPOUND FACET COVERAGE",
            status="APPLIED",
            details={
                "facets": facet_queries,
                "added_candidates": added,
                "candidate_count": len(augmented),
            },
        )

        return augmented, facet_queries

    def _select_compound_facet_coverage(
        self,
        ranked,
        facet_queries,
        limit,
    ):
        """Keep the strongest above-threshold candidate for every explicit facet.

        Whole-question reranking can otherwise place several chunks for the first
        event ahead of the second event.  This selector never promotes a result
        below the existing confidence threshold; it only preserves facet coverage
        among already-qualified reranker results.
        """

        if not facet_queries:
            return list(ranked[:limit]), True

        qualified = [
            item
            for item in ranked
            if float(item.get("rerank_score", 0.0)) >= MIN_RETRIEVAL_SCORE
        ]

        selected = []
        selected_ids = set()
        covered_facets = set()

        for facet_index in range(1, len(facet_queries) + 1):
            candidate = next(
                (
                    item
                    for item in qualified
                    if facet_index in item.get("_compound_facet_hits", [])
                ),
                None,
            )
            if candidate is None:
                continue

            covered_facets.add(facet_index)
            identity = self._candidate_identity(candidate)
            if identity not in selected_ids:
                selected.append(candidate)
                selected_ids.add(identity)

        for item in qualified:
            if len(selected) >= limit:
                break
            identity = self._candidate_identity(item)
            if identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)

        complete = len(covered_facets) == len(facet_queries)

        evidence_logger.record_event(
            event_name="COMPOUND FACET FINAL COVERAGE",
            status="COMPLETE" if complete else "INCOMPLETE",
            details={
                "facet_count": len(facet_queries),
                "covered_facets": sorted(covered_facets),
                "selected_count": len(selected[:limit]),
                "confidence_threshold": MIN_RETRIEVAL_SCORE,
            },
        )

        return selected[:limit], complete

    def _select_compound_reranker_pool(
        self,
        candidates,
        facet_queries,
        limit=16,
    ):
        """Ensure each explicit facet reaches reranking when a candidate exists.

        This operates before CrossEncoder scoring and therefore does not decide
        final relevance.  It only prevents the pre-reranker pool limit from
        discarding every candidate for a later facet.
        """

        if not facet_queries:
            return list(candidates[:limit])

        selected = []
        selected_ids = set()
        covered_facets = set()

        for facet_index in range(1, len(facet_queries) + 1):
            candidate = next(
                (
                    item
                    for item in candidates
                    if facet_index in item.get("_compound_facet_hits", [])
                ),
                None,
            )
            if candidate is None:
                continue
            covered_facets.add(facet_index)
            identity = self._candidate_identity(candidate)
            if identity not in selected_ids:
                selected.append(candidate)
                selected_ids.add(identity)

        for item in candidates:
            if len(selected) >= limit:
                break
            identity = self._candidate_identity(item)
            if identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)

        evidence_logger.record_event(
            event_name="COMPOUND FACET RERANKER POOL",
            status=(
                "COMPLETE"
                if len(covered_facets) == len(facet_queries)
                else "INCOMPLETE"
            ),
            details={
                "facet_count": len(facet_queries),
                "covered_facets": sorted(covered_facets),
                "selected_count": len(selected[:limit]),
            },
        )

        return selected[:limit]

    def _is_compound_intent(self, intent_query):

        return len(self._extract_compound_facets(intent_query)) >= 2

    def _max_chunks_per_file_for_query(
        self,
        query,
        has_compound_facets=False
    ):

        if self._is_short_lookup_query(query) or has_compound_facets:
            return FINAL_TOP_K

        return 2

    def _apply_diversity_filter(
        self,
        results,
        max_chunks_per_file=2
    ):

        filtered = []
        file_count = {}

        for item in results:

            file_name = item["metadata"].get(
                "file_name",
                "Unknown"
            )

            current_count = file_count.get(
                file_name,
                0
            )

            if current_count >= max_chunks_per_file:
                continue

            file_count[file_name] = current_count + 1
            filtered.append(item)

        return filtered

    def _identity_fast_path_target(self, query, intent_query=None):

        base = str(intent_query or query or "").strip()
        if not base:
            return ""

        tokens = self._meaningful_query_tokens(base)
        if not tokens:
            return ""

        # Identity fast-path safety must be tied to the actual requested
        # subject, not to enriched helper words such as biography/life/facts.
        return " ".join(tokens[:8])

    def _identity_fast_path_definition_score(self, target, text):

        """Score only a direct subject-bound introductory definition.

        A same-file narrative chunk is not enough. The requested subject must
        occur near the beginning of the chunk and the SAME subject occurrence
        must lead into a definitional/biographical predicate. This prevents
        sections such as Marriages, References, Commemoration, or Education
        from qualifying merely because they mention the person and contain a
        generic word such as ``was`` or ``founding`` elsewhere.
        """

        if not target or not text:
            return 0.0

        normalized_target = self._normalize_text(target)
        normalized_text = self._normalize_text(text)

        if not normalized_target or not normalized_text:
            return 0.0

        # Generic narrative/reference headings are not identity introductions.
        first_raw_line = next(
            (line.strip() for line in str(text).splitlines() if line.strip()),
            "",
        )
        first_heading = self._normalize_text(first_raw_line)
        blocked_headings = {
            "references",
            "bibliography",
            "notes",
            "citations",
            "external links",
            "further reading",
            "marriages",
            "marriage",
            "personal life",
            "commemoration",
            "legacy",
            "awards",
            "education",
            "early life",
            "etymology",
        }
        if first_heading in blocked_headings:
            return 0.0

        first_part = normalized_text[:1400]

        target_tokens = normalized_target.split()
        if len(target_tokens) >= 2:
            # Biographical leads often expand a familiar two-token name with
            # middle names/surnames (for example, "Jose Protasio Rizal ...").
            # Match the requested first/last anchor in order while allowing a
            # bounded number of intervening name tokens. This is still strict:
            # the expanded name must occur early and itself lead into the
            # definitional predicate.
            subject_pattern = (
                re.escape(target_tokens[0])
                + r"(?:\s+[a-z0-9]+){0,6}\s+"
                + re.escape(target_tokens[-1])
            )
        else:
            subject_pattern = re.escape(normalized_target)

        target_matches = list(re.finditer(rf"\b{subject_pattern}\b", first_part))
        if not target_matches:
            return 0.0

        best_score = 0.0
        for target_match in target_matches[:5]:
            if target_match.start() > 520:
                continue

            target_pos = target_match.start()
            segment = first_part[target_pos:target_pos + 700]

            direct_definition = re.search(
                rf"^{subject_pattern}\b.{{0,320}}?\b(?:is|was|are|were)\s+"
                rf"(?:a\s+|an\s+|the\s+)[a-z]",
                segment,
            )

            # Some clean company profiles use an active role verb instead of a
            # copular definition (for example, 'X serves as ...'). Keep this
            # narrow and still require the exact requested subject at the start
            # of the same relation window.
            active_profile = re.search(
                rf"^{subject_pattern}\b.{{0,240}}?\b"
                rf"(?:serves|served|leads|led|founded|established|heads|headed)\b",
                segment,
            )

            relation = direct_definition or active_profile
            if not relation:
                continue

            relation_end = relation.end()
            if target_pos <= 80 and relation_end <= 300:
                score = 3.0
            elif target_pos <= 220 and relation_end <= 460:
                score = 2.5
            else:
                score = 2.0
            best_score = max(best_score, score)

        return best_score

    def _bm25_identity_fast_path(self, query, bm25_results, intent_query=None):

        """Return one subject-bound identity intro without semantic model load.

        This path is deliberately strict. It activates only when BM25 finds a
        chunk from the exact source topic where the requested subject itself
        appears near the opening and directly participates in a definition or
        profile relation. Any ambiguity falls back to vector + reranker.
        """

        if (
            not query
            or not bm25_results
            or not self._is_identity_lookup_query(query)
            or self._is_compound_intent(intent_query)
            or self._is_list_or_relationship_query(query)
        ):
            return []

        target = self._identity_fast_path_target(
            query=query,
            intent_query=intent_query,
        )
        if not target:
            return []

        candidates = []

        for original in bm25_results:
            item = dict(original)
            item["metadata"] = dict(original.get("metadata", {}))
            text = str(item.get("text", "") or "").strip()

            if not text:
                continue

            if (
                self._is_reference_section(item)
                and not self._query_requests_reference_material(query)
            ):
                continue

            if self._identity_noise_score(text) >= 4:
                continue

            source_score = self._source_topic_score(target, item)
            definition_score = self._identity_fast_path_definition_score(
                target,
                text,
            )
            query_match_score = self._query_match_score(target, text)

            file_tokens = self._file_topic_tokens(item.get("metadata", {}))
            query_tokens = set(self._meaningful_query_tokens(target))
            normalized_head = self._normalize_text(text[:1000])

            if not file_tokens or not query_tokens:
                continue

            matched_file_tokens = [
                token for token in file_tokens
                if token in query_tokens and token in normalized_head
            ]
            file_topic_coverage = (
                len(matched_file_tokens) / len(file_tokens)
                if file_tokens
                else 0.0
            )

            # Exact source-topic match + subject-bound definition are mandatory.
            if source_score < 2.0:
                continue
            if file_topic_coverage < 0.80:
                continue
            if definition_score < 2.0:
                continue
            if query_match_score < 0.25:
                continue

            safety_score = (
                source_score
                + definition_score
                + query_match_score
            )
            item["_bm25_fast_path_score"] = safety_score
            item["_bm25_fast_path_definition_score"] = definition_score
            item["_bm25_identity_fast_path"] = True
            item["_bm25_identity_target"] = target
            candidates.append(item)

        if not candidates:
            return []

        candidates.sort(
            key=lambda item: (
                float(item.get("_bm25_fast_path_definition_score", 0.0)),
                float(item.get("_bm25_fast_path_score", 0.0)),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )

        best = candidates[0]

        evidence_logger.record_event(
            event_name="BM25 FIRST PASS",
            status="IDENTITY FAST PATH",
            details={
                "reason": (
                    "The exact requested subject was bound to a direct "
                    "introductory definition in the selected same-topic chunk; "
                    "vector and reranker loading were safely skipped."
                ),
                "source": best.get("metadata", {}).get("file_name", "Unknown"),
                "chunk_id": best.get("metadata", {}).get("chunk_id", ""),
                "target": target,
                "definition_score": round(
                    float(best.get("_bm25_fast_path_definition_score", 0.0)), 4
                ),
                "safety_score": round(
                    float(best.get("_bm25_fast_path_score", 0.0)), 4
                ),
            },
        )

        return [best]

    def _bm25_identity_ood_fast_fail(self, query, bm25_results, intent_query=None):
        """Fail closed before ML retrieval for a lexically absent identity.

        This Phase-7 path is deliberately much narrower than a generic OOD
        classifier. It activates only for identity-style questions (for
        example ``Who was Jose Rizal?``) after the existing BM25 identity fast
        path has already failed. The requested subject must contribute at
        least two meaningful tokens and *none* of those tokens may appear
        anywhere in the active company corpus or source titles.

        Any lexical footprint at all sends the request through the normal
        vector + reranker pipeline. This keeps paraphrased company questions
        safe while eliminating the known 30-second cold-path cost for clearly
        absent named identities on lower-spec PCs.
        """
        if (
            not query
            or not self._is_identity_lookup_query(query)
            or self._is_compound_intent(intent_query)
            or self._is_list_or_relationship_query(query)
        ):
            return False, {}

        target = self._identity_fast_path_target(
            query=query,
            intent_query=intent_query,
        )
        if not target:
            return False, {}

        presence = self.bm25.meaningful_token_presence(target)
        target_tokens = list(presence.get("query_tokens") or [])
        present_tokens = list(presence.get("present_tokens") or [])

        # Two subject tokens keeps this path out of broad one-word identity or
        # role questions where semantic retrieval may legitimately be needed.
        if len(target_tokens) < 2 or present_tokens:
            return False, presence

        # Defense in depth: if BM25 itself produced a positive-scoring hit that
        # literally contains the normalized target, do not fast-fail even if a
        # future coverage-token normalization change missed it.
        normalized_target = self._normalize_text(target)
        for item in list(bm25_results or [])[:20]:
            if float(item.get("score", 0.0) or 0.0) <= 0.0:
                continue
            haystack = self._normalize_text(
                " ".join(
                    (
                        str(item.get("text", "") or ""),
                        str((item.get("metadata", {}) or {}).get("file_name", "") or ""),
                        str((item.get("metadata", {}) or {}).get("section_title", "") or ""),
                    )
                )
            )
            if normalized_target and normalized_target in haystack:
                return False, presence

        return True, {
            **presence,
            "target": target,
            "reason": "No meaningful target token exists anywhere in the active company corpus.",
        }

    def _direct_relation_kind(self, query):

        """Classify only narrow factual relations that can be BM25-proven.

        The classifier intentionally excludes explanation, comparison, list,
        and free-form semantic questions.  Its only purpose is deciding
        whether a lightweight BM25 candidate can safely avoid loading the
        embedding/reranker stack for a direct fact lookup.
        """

        clean = self._normalize_text(query)
        if not clean:
            return ""

        patterns = (
            (
                "organization",
                r"\b(?:organization|organisation|group|association|society|movement)\b"
                r".*\b(?:found|founded|cofound|co founder|co founded|help found|helped found)\b"
                r"|\b(?:found|founded|cofound|co founder|co founded|help found|helped found)\b"
                r".*\b(?:organization|organisation|group|association|society|movement)\b",
            ),
            (
                "position",
                r"\b(?:position|role|office|title)\b.*\b(?:hold|held|have|serve|served|occupy|occupied)\b"
                r"|\b(?:hold|held|serve|served|appointed)\b.*\b(?:position|role|office|title)\b",
            ),
            (
                "approver",
                r"\b(?:approve|approves|approved|approval|approver|authorize|authorizes|authorization)\b",
            ),
            (
                "quantity",
                r"\b(?:how many|how much|number|amount|limit|allotment|allowance|entitlement|annual|annually|yearly|days?|hours?|credits?|percentage|percent)\b",
            ),
            (
                "purpose",
                r"\b(?:purpose|objective|goal|aim|mission)\b",
            ),
            (
                "eligibility",
                r"\b(?:eligible|eligibility|qualified|qualification|qualify|qualifies|entitled|entitlement|covered)\b",
            ),
            (
                "time",
                r"\b(?:when|date|year|signed|effective|deadline|issued|started|ended|approved on)\b",
            ),
        )

        for kind, pattern in patterns:
            if re.search(pattern, clean, re.IGNORECASE):
                return kind

        return ""

    def _direct_relation_anchor_tokens(self, query, kind):

        clean = self._normalize_text(query)
        if not clean:
            return []

        common = {
            "the", "a", "an", "of", "in", "on", "at", "for", "to",
            "and", "or", "with", "what", "which", "who", "when", "where",
            "why", "how", "did", "does", "do", "is", "are", "was", "were",
            "has", "have", "had", "its", "his", "her", "their", "he", "she",
            "they", "it", "this", "that", "from", "by", "as", "be", "been",
        }

        relation_words = {
            "position": {
                "position", "role", "office", "title", "hold", "held", "serve",
                "served", "occupy", "occupied", "appointed", "appointment",
            },
            "organization": {
                "organization", "organisation", "group", "association", "society",
                "movement", "found", "founded", "founder", "founding", "cofound",
                "help", "helped",
            },
            "approver": {
                "approve", "approves", "approved", "approval", "approver",
                "authorize", "authorizes", "authorized", "authorization", "required",
                "requires", "require",
            },
            "quantity": {
                "many", "much", "number", "amount", "limit", "allotment",
                "allowance", "annual", "annually", "yearly", "day", "days",
                "hour", "hours", "credit", "credits", "percentage", "percent",
                "provided", "provide", "provides", "given", "give", "entitled",
                "entitlement",
            },
            "purpose": {
                "purpose", "objective", "goal", "aim", "mission", "main", "primary",
                "organized", "organised", "formed", "created", "pursue", "pursued",
                "seek", "sought", "intended", "outcome",
            },
            "eligibility": {
                "eligible", "eligibility", "qualified", "qualification", "qualify",
                "qualifies", "entitled", "entitlement", "covered", "benefit", "benefits",
            },
            "time": {
                "date", "year", "signed", "sign", "effective", "deadline", "issued",
                "started", "start", "ended", "end", "approved",
            },
        }.get(kind, set())

        tokens = []
        seen = set()

        for token in clean.split():
            if len(token) <= 2:
                continue
            if token in common or token in relation_words:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)

        return tokens[:8]

    @staticmethod
    def _direct_relation_number_present(text):

        return bool(
            re.search(
                r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _direct_relation_local_score(self, kind, anchor_tokens, text):

        if not kind or not anchor_tokens or not text:
            return 0.0

        normalized = self._normalize_text(text)
        if not normalized:
            return 0.0

        # Direct-fact fast paths should never be sourced from reference-only
        # tails even when BM25 happens to rank them highly.
        first_line = next(
            (line.strip() for line in str(text).splitlines() if line.strip()),
            "",
        )
        if self._normalize_text(first_line) in {
            "references", "bibliography", "notes", "citations", "external links",
            "further reading",
        }:
            return 0.0

        positions = {
            token: normalized.find(token)
            for token in anchor_tokens
        }
        matched = [
            token for token, position in positions.items()
            if position >= 0
        ]

        required = (
            len(anchor_tokens)
            if len(anchor_tokens) <= 3
            else max(3, int(len(anchor_tokens) * 0.75 + 0.999))
        )
        if len(matched) < required:
            return 0.0

        matched_positions = [positions[token] for token in matched]
        first = min(matched_positions)
        last = max(matched_positions)

        # If the anchor words are scattered across a long narrative, the
        # candidate is not a compact relation statement and should fall back
        # to semantic retrieval.
        if last - first > 720:
            return 0.0

        start = max(0, first - 220)
        end = min(len(normalized), last + 620)
        window = normalized[start:end]

        cue_patterns = {
            "position": (
                r"\b(?:served|serve)\b.{0,90}\bas\b",
                r"\b(?:was|is|were|are)\s+appointed\b",
                r"\b(?:appointed|appointment|position|role|office|title)\b",
            ),
            "organization": (
                r"\b(?:co\s*founder|co\s*founded|founded|founding member|helped found|help found)\b",
            ),
            "approver": (
                r"\b(?:approve|approves|approved|approval|approver|authorize|authorized|authorization)\b",
            ),
            "quantity": (
                r"\b(?:day|days|hour|hours|credit|credits|percent|percentage|amount|limit|allotment|allowance|entitled|entitlement|provided|annual|annually|yearly)\b",
            ),
            "purpose": (
                r"\b(?:purpose|objective|goal|aim|mission|primary\s+objective|seek|sought|intended)\b",
            ),
            "eligibility": (
                r"\b(?:eligible|eligibility|qualified|qualify|entitled|entitlement|covered|regular\s+employees?)\b",
            ),
            "time": (
                r"\b(?:signed|effective|issued|started|ended|approved|dated|date)\b",
            ),
        }.get(kind, ())

        if not any(re.search(pattern, window, re.IGNORECASE) for pattern in cue_patterns):
            return 0.0

        if kind == "quantity" and not self._direct_relation_number_present(window):
            return 0.0

        if kind == "time":
            has_date_value = bool(
                re.search(
                    r"\b(?:18|19|20)\d{2}\b|"
                    r"\b(?:january|february|march|april|may|june|july|august|"
                    r"september|october|november|december)\b|"
                    r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b",
                    window,
                    re.IGNORECASE,
                )
            )
            if not has_date_value:
                return 0.0

        coverage = len(matched) / max(1, len(anchor_tokens))
        proximity_bonus = 1.0 if last - first <= 240 else 0.5
        return (coverage * 4.0) + 2.0 + proximity_bonus

    def _bm25_direct_relation_fast_path(self, query, bm25_results, intent_query=None):

        """Return one compact BM25-proven direct-fact chunk before vector load.

        This path is intentionally narrower than normal hybrid retrieval.  It
        requires a recognized factual relation, strong local anchor coverage,
        an explicit relation cue/value in the same window, and no close
        cross-source ambiguity.  Otherwise the existing vector + reranker path
        remains authoritative.
        """

        semantic_query = str(intent_query or query or "").strip()
        if (
            not semantic_query
            or not bm25_results
            or self._is_identity_lookup_query(query)
            or self._is_list_or_relationship_query(query)
            or self._is_compound_intent(intent_query)
            or extract_structured_reference(query)
        ):
            return []

        kind = self._direct_relation_kind(semantic_query)
        if not kind:
            return []

        anchors = self._direct_relation_anchor_tokens(semantic_query, kind)
        if not anchors:
            return []

        strongest_bm25 = max(
            (float(item.get("score", 0.0) or 0.0) for item in bm25_results),
            default=0.0,
        ) or 1.0
        candidates = []

        for original in bm25_results:
            item = dict(original)
            item["metadata"] = dict(original.get("metadata", {}))
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            if (
                self._is_reference_section(item)
                and not self._query_requests_reference_material(semantic_query)
            ):
                continue

            local_score = self._direct_relation_local_score(kind, anchors, text)
            if local_score <= 0.0:
                continue

            source_score = self._source_topic_score(semantic_query, item)
            query_score = self._query_match_score(semantic_query, text)
            normalized_bm25 = min(
                1.0,
                max(0.0, float(item.get("score", 0.0) or 0.0) / strongest_bm25),
            )
            safety_score = local_score + source_score + query_score + normalized_bm25

            # The local relation proof is mandatory; filename/source matching
            # improves confidence but is not required for generic policy docs.
            if safety_score < 7.0:
                continue

            item["_bm25_direct_relation_fast_path"] = True
            item["_bm25_direct_relation_kind"] = kind
            item["_bm25_direct_relation_score"] = safety_score
            item["_bm25_direct_relation_anchors"] = list(anchors)
            candidates.append(item)

        if not candidates:
            return []

        candidates.sort(
            key=lambda item: (
                float(item.get("_bm25_direct_relation_score", 0.0)),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        if second is not None:
            best_source = str(best.get("metadata", {}).get("file_name", "")).casefold()
            second_source = str(second.get("metadata", {}).get("file_name", "")).casefold()
            score_gap = float(best.get("_bm25_direct_relation_score", 0.0)) - float(
                second.get("_bm25_direct_relation_score", 0.0)
            )

            # Close, independent sources are ambiguous enough to justify the
            # normal semantic/reranker path instead of guessing cheaply.
            if best_source != second_source and score_gap < 0.75:
                return []

        evidence_logger.record_event(
            event_name="BM25 FIRST PASS",
            status="DIRECT RELATION FAST PATH",
            details={
                "reason": (
                    "A narrow direct-fact relation was explicitly supported by "
                    "one compact BM25 chunk; vector and reranker loading were "
                    "safely skipped."
                ),
                "relation_kind": kind,
                "anchors": anchors,
                "source": best.get("metadata", {}).get("file_name", "Unknown"),
                "chunk_id": best.get("metadata", {}).get("chunk_id", ""),
                "safety_score": round(
                    float(best.get("_bm25_direct_relation_score", 0.0)), 4
                ),
            },
        )

        return [best]

    def _general_corpus_empty_result_rescue(
        self,
        query,
        intent_query=None,
        final_top_k=3,
    ):
        """Second-pass rescue for arbitrary indexed documents.

        The certified v6.4.14.2.4 retrieval path remains the primary path and
        is not reordered, widened, or reweighted.  This rescue runs only after
        that path would otherwise return no accepted context.  It uses the
        lightweight corpus-wide lexical coverage index to discover compact
        exact records, then applies the existing CrossEncoder and the unchanged
        global confidence threshold before any chunk can reach generation.
        """
        coverage_search = getattr(self.bm25, "coverage_search", None)
        if not callable(coverage_search):
            return []

        rescue_query = str(intent_query or query or "").strip()
        if not rescue_query:
            return []

        coverage_results = coverage_search(
            rescue_query,
            top_k=max(BM25_TOP_K, 12),
            minimum_score=0.40,
        )
        if not coverage_results:
            return []

        candidates = []
        for item in coverage_results:
            if self._is_reference_section(item) and not self._query_requests_reference_material(query):
                continue
            if self._reference_noise_score(str(item.get("text", "") or "")) >= 5:
                continue
            candidates.append(item)
            if len(candidates) >= 10:
                break

        if not candidates:
            return []

        evidence_logger.record_event(
            event_name="GENERAL CORPUS EMPTY-RESULT RESCUE",
            status="STARTED",
            details={
                "coverage_candidates": len(coverage_results),
                "reranker_candidates": len(candidates),
                "confidence_threshold": MIN_RETRIEVAL_SCORE,
                "policy": (
                    "Runs only after the certified retrieval path would return "
                    "no context; normal successful queries are untouched."
                ),
            },
        )

        if ENABLE_RERANKER:
            # Candidate discovery above already prefers ``intent_query`` when a
            # canonical multilingual relation target is available.  Score the
            # same semantic query here as well; reranking the original mixed-
            # language surface form can otherwise reject an exact English
            # source that the canonical query correctly discovered.  This path
            # remains fallback-only and still uses the unchanged confidence
            # threshold.
            ranked = self._get_reranker().rerank(rescue_query, candidates)
            accepted = [
                item
                for item in ranked
                if float(item.get("rerank_score", 0.0) or 0.0)
                >= MIN_RETRIEVAL_SCORE
            ]
        else:
            # Without the configured reranker, keep this rescue conservative.
            accepted = [
                item
                for item in candidates
                if float(item.get("_lexical_coverage_score", item.get("score", 0.0)) or 0.0)
                >= 0.80
            ]

        if not accepted:
            evidence_logger.record_event(
                event_name="GENERAL CORPUS EMPTY-RESULT RESCUE",
                status="REJECTED",
                details=(
                    "Coverage found candidates, but none passed the existing "
                    "confidence requirements."
                ),
            )
            return []

        max_chunks_per_file = self._max_chunks_per_file_for_query(
            query,
            has_compound_facets=False,
        )
        accepted = self._apply_diversity_filter(
            accepted,
            max_chunks_per_file=max_chunks_per_file,
        )[:max(1, int(final_top_k or 1))]

        for item in accepted:
            item["_general_corpus_rescue"] = True

        self._record_qa_stage(
            "GENERAL CORPUS RESCUE",
            accepted,
            limit=len(accepted),
            accepted=True,
        )
        evidence_logger.record_event(
            event_name="GENERAL CORPUS EMPTY-RESULT RESCUE",
            status="ACCEPTED",
            details={
                "accepted_chunks": len(accepted),
                "sources": [
                    (item.get("metadata", {}) or {}).get("file_name", "Unknown")
                    for item in accepted
                ],
            },
        )
        return accepted

    @staticmethod
    def _matches_source_family(item, source_family):
        if not source_family:
            return True
        family = str(source_family).strip().casefold()
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        labels = " ".join(
            str(metadata.get(key, "") or "")
            for key in ("file_name", "file_path", "section_title")
        ).casefold()
        if family == "misra":
            return "misra" in labels
        return True

    def _filter_source_family(self, results, source_family):
        if not source_family:
            return list(results or [])
        return [
            item for item in (results or [])
            if self._matches_source_family(item, source_family)
        ]

    def retrieve(
        self,
        query,
        intent_query=None,
        source_family=None,
        final_top_k_override=None,
    ):

        if not query:

            evidence_logger.record_event(
                event_name="RETRIEVAL",
                details="Empty query",
                status="SKIPPED"
            )

            return []

        structured_references = extract_structured_references(query)
        structured_reference = structured_references[0] if structured_references else None

        # v6.4.60: natural structured comparisons such as
        # ``What is the difference between Rule 16.4 and Rule 16.5?`` need
        # both authoritative anchors. The older single-reference path stopped
        # after the first Rule and could therefore synthesize an incomplete or
        # contaminated comparison. Retrieve each explicit Rule/Directive
        # independently and return the exact grounded anchors together.
        comparable_references = [
            reference for reference in structured_references
            if reference.kind in {"rule", "directive"}
        ]
        if len(comparable_references) >= 2:
            comparison_results = []
            seen = set()
            for reference in comparable_references:
                for item in self._retrieve_exact_structured_reference(reference):
                    item = dict(item)
                    item["metadata"] = dict(item.get("metadata", {}) or {})
                    item["_structured_comparison_anchor"] = True
                    item["_structured_comparison_reference"] = reference.display_name
                    key = (
                        str(item["metadata"].get("file_path") or item["metadata"].get("file_name") or ""),
                        reference.kind,
                        reference.identifier,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    comparison_results.append(item)

            comparison_results = self._filter_source_family(
                comparison_results, source_family
            )
            if comparison_results:
                self._record_qa_stage(
                    "EXACT STRUCTURED COMPARISON",
                    comparison_results,
                    limit=len(comparison_results),
                    accepted=True,
                )
                evidence_logger.record_event(
                    event_name="EXACT STRUCTURED COMPARISON RETRIEVAL",
                    status="MATCHED",
                    details={
                        "references": [
                            reference.display_name for reference in comparable_references
                        ],
                        "matches": len(comparison_results),
                    },
                )
                evidence_logger.record_retrieval_summary(
                    bm25_candidates=0,
                    vector_candidates=0,
                    hybrid_candidates=0,
                    reranker_candidates=0,
                    accepted_chunks=len(comparison_results),
                    rejected_chunks=0,
                    final_chunks=len(comparison_results),
                    configured_top_k=len(comparison_results),
                    confidence_threshold=None,
                )
                return comparison_results

        if structured_reference:

            exact_results = (
                self._retrieve_exact_structured_reference(
                    structured_reference
                )
            )

            exact_results = self._filter_source_family(
                exact_results, source_family
            )

            if exact_results:

                if DEBUG_RETRIEVAL:

                    print(
                        "\n===== EXACT STRUCTURED LOOKUP ====="
                    )
                    print(
                        f"Requested : "
                        f"{structured_reference.display_name}"
                    )
                    print(
                        f"Matches   : {len(exact_results)}"
                    )

                    for item in exact_results:

                        metadata = item.get(
                            "metadata",
                            {}
                        )

                        print(
                            f"- "
                            f"{metadata.get('file_name', 'Unknown')} "
                            f"chunks "
                            f"{metadata.get('exact_chunk_start', metadata.get('chunk_id', '?'))}"
                            f"-"
                            f"{metadata.get('exact_chunk_end', metadata.get('chunk_id', '?'))}"
                        )

                    print(
                        "===================================\n"
                    )

                self._record_qa_stage(
                    "EXACT STRUCTURED",
                    exact_results,
                    limit=len(exact_results),
                    accepted=True
                )

                evidence_logger.record_event(
                    event_name="EXACT STRUCTURED RETRIEVAL",
                    status="MATCHED",
                    details={
                        "reference": (
                            structured_reference.display_name
                        ),
                        "matches": len(exact_results),
                    }
                )

                evidence_logger.record_retrieval_summary(
                    bm25_candidates=0,
                    vector_candidates=0,
                    hybrid_candidates=0,
                    reranker_candidates=0,
                    accepted_chunks=len(exact_results),
                    rejected_chunks=0,
                    final_chunks=len(exact_results),
                    configured_top_k=len(exact_results),
                    confidence_threshold=None
                )

                return exact_results

            evidence_logger.record_event(
                event_name="EXACT STRUCTURED RETRIEVAL",
                status="NO METADATA MATCH",
                details={
                    "reference": (
                        structured_reference.display_name
                    ),
                }
            )

            # Exact Rule/Directive identifiers are authoritative.  The
            # structured inventory already contains every supported indexed
            # Rule/Directive; if the exact metadata lookup misses, broad
            # semantic retrieval must not substitute a neighboring rule or pay
            # the vector/reranker/LLM cost only to fall back later.  Section-like
            # references retain the historical semantic fallback because some
            # document families expose looser section metadata.
            if structured_reference.kind in {"rule", "directive"}:
                evidence_logger.record_event(
                    event_name="EXACT STRUCTURED RETRIEVAL",
                    status="AUTHORITATIVE IDENTIFIER MISS; STOPPED",
                    details={
                        "reference": structured_reference.display_name,
                        "reason": (
                            "An explicit Rule/Directive identifier had no exact "
                            "structured metadata match; semantic substitution was "
                            "blocked for grounding safety and latency."
                        ),
                    },
                )
                evidence_logger.record_retrieval_summary(
                    bm25_candidates=0,
                    vector_candidates=0,
                    hybrid_candidates=0,
                    reranker_candidates=0,
                    accepted_chunks=0,
                    rejected_chunks=0,
                    final_chunks=0,
                    configured_top_k=0,
                    confidence_threshold=None,
                )
                return []

        named_section_results = self._retrieve_structured_section_title_match(
            query=query,
            intent_query=intent_query,
        )
        named_section_results = self._filter_source_family(
            named_section_results,
            source_family,
        )
        if named_section_results:
            self._record_qa_stage(
                "STRUCTURED SECTION TITLE",
                named_section_results,
                limit=len(named_section_results),
                accepted=True,
            )
            evidence_logger.record_event(
                event_name="STRUCTURED SECTION TITLE RETRIEVAL",
                status="MATCHED",
                details={
                    "reference": str(
                        (named_section_results[0].get("metadata", {}) or {}).get(
                            "exact_reference", ""
                        )
                    ),
                    "title": str(
                        named_section_results[0].get(
                            "_structured_section_title", ""
                        )
                    ),
                    "matches": len(named_section_results),
                },
            )
            evidence_logger.record_retrieval_summary(
                bm25_candidates=0,
                vector_candidates=0,
                hybrid_candidates=0,
                reranker_candidates=0,
                accepted_chunks=len(named_section_results),
                rejected_chunks=0,
                final_chunks=len(named_section_results),
                configured_top_k=len(named_section_results),
                confidence_threshold=None,
            )
            return named_section_results

        structured_topic_results = self._retrieve_structured_rule_catalog(
            query=query,
            intent_query=intent_query,
        )
        if not structured_topic_results:
            structured_topic_results = self._retrieve_structured_rule_topic_family(
                query=query,
                intent_query=intent_query,
            )
        if structured_topic_results:
            self._record_qa_stage(
                "STRUCTURED TOPIC FAMILY",
                structured_topic_results,
                limit=len(structured_topic_results),
                accepted=True,
            )
            self._record_qa_stage(
                "FINAL",
                structured_topic_results,
                limit=len(structured_topic_results),
                accepted=True,
            )
            evidence_logger.record_event(
                event_name="STRUCTURED TOPIC FAMILY RETRIEVAL",
                status="MATCHED",
                details={
                    "topic": next((
                        str(item.get("_structured_topic_family", "") or "")
                        for item in structured_topic_results
                        if isinstance(item, dict) and item.get("_structured_topic_family")
                    ), "structured MISRA rules"),
                    "references": [
                        str((item.get("metadata", {}) or {}).get("rule_id", ""))
                        for item in structured_topic_results
                    ],
                },
            )
            evidence_logger.record_retrieval_summary(
                bm25_candidates=0,
                vector_candidates=0,
                hybrid_candidates=0,
                reranker_candidates=0,
                accepted_chunks=len(structured_topic_results),
                rejected_chunks=0,
                final_chunks=len(structured_topic_results),
                configured_top_k=len(structured_topic_results),
                confidence_threshold=None,
            )
            return structured_topic_results

        reranker_candidates_count = 0
        accepted_chunks_count = 0
        rejected_chunks_count = 0

        is_list_mode = self._is_list_or_relationship_query(
            query
        )

        is_identity_mode = self._is_identity_lookup_query(
            query
        )

        final_top_k = (
            max(1, int(final_top_k_override))
            if final_top_k_override is not None
            else self._final_top_k_for_query(query)
        )

        if source_family:
            # Source-scoped modes (currently MISRA compliance) need a wider
            # initial pool because the corpus-wide top results may otherwise
            # crowd the requested source family before filtering.
            bm25_top_k = max(BM25_TOP_K, 80)
            vector_top_k = max(VECTOR_TOP_K, 80)

        elif structured_reference:

            # Legacy/incomplete metadata fallback: search a wider pool, then
            # keep only chunks that explicitly begin with the requested ID.
            bm25_top_k = max(
                BM25_TOP_K,
                40
            )

            vector_top_k = max(
                VECTOR_TOP_K,
                40
            )

        elif is_list_mode:

            bm25_top_k = max(
                BM25_TOP_K,
                60
            )

            vector_top_k = max(
                VECTOR_TOP_K,
                60
            )

        elif is_identity_mode:

            # Identity/overview questions need a slightly wider initial pool.
            # Wikipedia-like sources can otherwise fill the first ten hits
            # with citations/references before the biography introduction is
            # seen by the reranker.
            bm25_top_k = max(
                BM25_TOP_K,
                20
            )

            vector_top_k = max(
                VECTOR_TOP_K,
                20
            )

        else:

            bm25_top_k = BM25_TOP_K
            vector_top_k = VECTOR_TOP_K

        bm25_results = self.bm25.search(
            query,
            bm25_top_k
        )
        bm25_results = self._filter_source_family(
            bm25_results, source_family
        )

        bm25_fast_results = self._bm25_identity_fast_path(
            query=query,
            bm25_results=bm25_results,
            intent_query=intent_query,
        )

        if bm25_fast_results:
            self._record_qa_stage(
                "BM25",
                bm25_results
            )
            self._record_qa_stage(
                "ACCEPTED",
                bm25_fast_results,
                limit=len(bm25_fast_results),
                accepted=True
            )
            self._record_qa_stage(
                "FINAL",
                bm25_fast_results,
                limit=len(bm25_fast_results),
                accepted=True
            )
            evidence_logger.record_retrieval_summary(
                bm25_candidates=len(bm25_results),
                vector_candidates=0,
                hybrid_candidates=0,
                reranker_candidates=0,
                accepted_chunks=len(bm25_fast_results),
                rejected_chunks=0,
                final_chunks=len(bm25_fast_results),
                configured_top_k=1,
                confidence_threshold=None,
            )
            return bm25_fast_results

        direct_relation_fast_results = self._bm25_direct_relation_fast_path(
            query=query,
            bm25_results=bm25_results,
            intent_query=intent_query,
        )

        if direct_relation_fast_results:
            self._record_qa_stage(
                "BM25",
                bm25_results
            )
            self._record_qa_stage(
                "ACCEPTED",
                direct_relation_fast_results,
                limit=1,
                accepted=True
            )
            self._record_qa_stage(
                "FINAL",
                direct_relation_fast_results,
                limit=1,
                accepted=True
            )
            evidence_logger.record_retrieval_summary(
                bm25_candidates=len(bm25_results),
                vector_candidates=0,
                hybrid_candidates=0,
                reranker_candidates=0,
                accepted_chunks=1,
                rejected_chunks=0,
                final_chunks=1,
                configured_top_k=1,
                confidence_threshold=None,
            )
            return direct_relation_fast_results

        # Phase 7 low-spec optimization: an identity target with no meaningful
        # lexical footprint anywhere in the active corpus cannot be rescued by
        # the expensive embedding/vector/reranker path without inventing
        # external knowledge. Keep this deliberately identity-only; every
        # other semantic/paraphrase query retains the certified retrieval path.
        if not source_family and not structured_reference:
            identity_ood_fast_fail, identity_ood_details = (
                self._bm25_identity_ood_fast_fail(
                    query=query,
                    bm25_results=bm25_results,
                    intent_query=intent_query,
                )
            )
            if identity_ood_fast_fail:
                self._record_qa_stage(
                    "BM25",
                    bm25_results,
                )
                evidence_logger.record_event(
                    event_name="BM25 IDENTITY OOD FAST FAIL",
                    status="NO CORPUS FOOTPRINT; ML RETRIEVAL SKIPPED",
                    details=identity_ood_details,
                )
                evidence_logger.record_retrieval_summary(
                    bm25_candidates=len(bm25_results),
                    vector_candidates=0,
                    hybrid_candidates=0,
                    reranker_candidates=0,
                    accepted_chunks=0,
                    rejected_chunks=0,
                    final_chunks=0,
                    configured_top_k=0,
                    confidence_threshold=None,
                )
                return []

        vector_results = self._get_vector_searcher().search(
            query,
            vector_top_k
        )
        vector_results = self._filter_source_family(
            vector_results, source_family
        )

        merged = self.hybrid.merge(
            vector_results,
            bm25_results
        )

        compound_facet_queries = []

        if self._is_compound_intent(intent_query):
            merged, compound_facet_queries = (
                self._augment_compound_candidates(
                    intent_query=intent_query,
                    merged=merged,
                )
            )

        evidence_logger.record_event(
            event_name="RETRIEVAL CANDIDATE COUNTS",
            status="COLLECTED",
            details={
                "BM25 Candidates":
                    len(
                        bm25_results
                    ),

                "Vector Candidates":
                    len(
                        vector_results
                    ),

                "Hybrid Candidates":
                    len(
                        merged
                    ),

                "Configured Top-K":
                    final_top_k,

                "Confidence Threshold":
                    (
                        MIN_RETRIEVAL_SCORE
                        if ENABLE_RERANKER
                        else None
                    ),
            }
        )

        self._record_qa_stage(
            "BM25",
            bm25_results
        )

        self._record_qa_stage(
            "VECTOR",
            vector_results
        )

        self._record_qa_stage(
            "HYBRID",
            merged
        )

        if DEBUG_RETRIEVAL:

            print("\n===== HYBRID RESULTS =====")

            for item in merged[:10]:

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> "
                    f"{item.get('score', 0.0):.4f}"
                )

            print("==========================\n")

            if is_list_mode:

                print("===== LIST MODE ENABLED =====")
                print(f"Query       : {query}")
                print(f"Final Top K : {final_top_k}")
                print("=============================\n")

        if structured_reference:

            structured_fallback_candidate_count = len(
                merged
            )

            merged = self._filter_for_structured_reference(
                merged,
                structured_reference
            )

            if not merged:

                if DEBUG_RETRIEVAL:
                    print(
                        "\n===== STRUCTURED ID CONSISTENCY ====="
                    )
                    print(
                        f"Requested : "
                        f"{structured_reference.display_name}"
                    )
                    print(
                        "Result    : NO CONSISTENT FALLBACK CHUNK"
                    )
                    print(
                        "=====================================\n"
                    )

                evidence_logger.record_event(
                    event_name="STRUCTURED ID CONSISTENCY",
                    status="REJECTED",
                    details={
                        "reference": (
                            structured_reference.display_name
                        ),
                        "reason": (
                            "No fallback chunk explicitly starts with "
                            "the requested structured identifier."
                        ),
                    }
                )

                evidence_logger.record_retrieval_summary(
                    bm25_candidates=len(bm25_results),
                    vector_candidates=len(vector_results),
                    hybrid_candidates=0,
                    reranker_candidates=0,
                    accepted_chunks=0,
                    rejected_chunks=(
                        structured_fallback_candidate_count
                    ),
                    final_chunks=0,
                    configured_top_k=final_top_k,
                    confidence_threshold=(
                        MIN_RETRIEVAL_SCORE
                        if ENABLE_RERANKER
                        else None
                    )
                )

                return []

        candidates = self._focus_top_source_for_short_query(
            query,
            merged
        )

        candidates = self._prioritize_informative_chunks(
            query,
            candidates
        )

        informative_candidates = list(
            candidates
        )

        if is_list_mode:

            if self._is_relationship_query(intent_query or query):
                candidates = self._prioritize_list_relationship_chunks(
                    query,
                    candidates
                )
            else:
                candidates = self._prioritize_generic_list_chunks(
                    query,
                    candidates
                )

        else:

            candidates = self._remove_low_information_chunks(
                query,
                candidates
            )

            candidates = self._remove_identity_noise_chunks(
                query,
                candidates
            )

            if is_identity_mode:

                candidates = (
                    self._prepare_identity_reranker_candidates(
                        query=query,
                        ranked_candidates=informative_candidates,
                        filtered_candidates=candidates,
                        limit=8,
                        minimum_pool=6
                    )
                )

        # List questions need more candidates because answers
        # may be spread across several chunks.
        if is_list_mode:

            candidates = candidates[:20]

        elif compound_facet_queries:

            # Keep a slightly wider pool only for explicit multi-part
            # questions so each clause has a fair chance to reach reranking.
            candidates = self._select_compound_reranker_pool(
                candidates=candidates,
                facet_queries=compound_facet_queries,
                limit=16,
            )

        else:

            candidates = candidates[:10]

        reranker = (
            self._get_reranker()
            if ENABLE_RERANKER and not is_list_mode and candidates
            else None
        )

        if reranker is not None:

            reranker_candidates_count = len(
                candidates
            )

            rerank_query = str(intent_query or query or "").strip()
            ranked = reranker.rerank(
                rerank_query,
                candidates
            )

            nonfinite_recovered = [
                item
                for item in ranked
                if item.get("_rerank_nonfinite_recovered")
            ]
            nonfinite_safe_rejected = [
                item
                for item in ranked
                if item.get("_rerank_nonfinite_safe_rejected")
            ]

            if nonfinite_recovered or nonfinite_safe_rejected:
                evidence_logger.record_event(
                    event_name="RERANKER NUMERICAL SAFETY",
                    status=(
                        "RECOVERED"
                        if nonfinite_recovered and not nonfinite_safe_rejected
                        else "SAFE REJECTION APPLIED"
                    ),
                    details={
                        "recovered_candidates": len(nonfinite_recovered),
                        "safe_rejected_candidates": len(nonfinite_safe_rejected),
                        "policy": (
                            "Retry non-finite CrossEncoder scores once at "
                            "batch_size=1; if still non-finite, use finite "
                            "score 0.0 so the candidate cannot bypass the "
                            "confidence threshold."
                        ),
                    }
                )

            compound_coverage_complete = True
            if compound_facet_queries:
                ranked, compound_coverage_complete = self._select_compound_facet_coverage(
                    ranked=ranked,
                    facet_queries=compound_facet_queries,
                    limit=final_top_k,
                )

                # For an explicit two-event temporal comparison, returning only
                # one side is more harmful than falling back.  Require both
                # facets to have an above-threshold reranker candidate before
                # any context is allowed downstream.
                if (
                    self._is_temporal_comparison_intent(intent_query)
                    and not compound_coverage_complete
                ):
                    ranked = []
            else:
                ranked = ranked[:final_top_k]

            pre_confidence_ranked = list(
                ranked
            )

            self._record_qa_stage(
                "RERANKED",
                pre_confidence_ranked,
                limit=final_top_k
            )

            # ======================================
            # RERANKER CONFIDENCE GATE
            # ======================================
            # If the best reranker score is too low,
            # treat the query as not found.
            #
            # This prevents unrelated chunks from being
            # sent to the LLM for questions outside
            # the knowledge base.
            best_rerank_score = 0.0

            if ranked:

                best_rerank_score = float(
                    ranked[0].get(
                        "rerank_score",
                        0.0
                    )
                )

            if best_rerank_score < MIN_RETRIEVAL_SCORE:

                if DEBUG_RETRIEVAL:

                    print(
                        "\n===== RETRIEVAL CONFIDENCE GATE ====="
                    )

                    print(
                        f"Best rerank score : "
                        f"{best_rerank_score:.4f}"
                    )

                    print(
                        f"Minimum required  : "
                        f"{MIN_RETRIEVAL_SCORE:.4f}"
                    )

                    print(
                        "Result            : NO CONTEXT"
                    )

                    print(
                        "=====================================\n"
                    )

                rejected_chunks_count = len(
                    pre_confidence_ranked
                )

                self._record_qa_stage(
                    "REJECTED",
                    pre_confidence_ranked,
                    limit=final_top_k,
                    accepted=False,
                    rejection_reason=(
                        "Best reranker score is below "
                        "the confidence threshold."
                    )
                )

                rescue_results = []
                if not source_family:
                    rescue_results = self._general_corpus_empty_result_rescue(
                        query=query,
                        intent_query=intent_query,
                        final_top_k=final_top_k,
                    )
                if rescue_results:
                    evidence_logger.record_retrieval_summary(
                        bm25_candidates=len(bm25_results),
                        vector_candidates=len(vector_results),
                        hybrid_candidates=len(merged),
                        reranker_candidates=reranker_candidates_count,
                        accepted_chunks=len(rescue_results),
                        rejected_chunks=rejected_chunks_count,
                        final_chunks=len(rescue_results),
                        configured_top_k=final_top_k,
                        confidence_threshold=MIN_RETRIEVAL_SCORE,
                    )
                    return rescue_results

                evidence_logger.record_retrieval_summary(
                    bm25_candidates=len(
                        bm25_results
                    ),
                    vector_candidates=len(
                        vector_results
                    ),
                    hybrid_candidates=len(
                        merged
                    ),
                    reranker_candidates=(
                        reranker_candidates_count
                    ),
                    accepted_chunks=0,
                    rejected_chunks=(
                        rejected_chunks_count
                    ),
                    final_chunks=0,
                    configured_top_k=final_top_k,
                    confidence_threshold=(
                        MIN_RETRIEVAL_SCORE
                    )
                )

                return []

            # Keep only chunks that individually pass the
            # reranker confidence threshold.
            #
            # The previous logic checked only the best result.
            # That allowed low-confidence unrelated chunks to
            # enter the final context whenever the top result
            # passed the gate.
            ranked = [
                item
                for item in ranked
                if float(
                    item.get(
                        "rerank_score",
                        0.0
                    )
                ) >= MIN_RETRIEVAL_SCORE
            ]

            accepted_ids = {
                id(
                    item
                )
                for item in ranked
            }

            rejected_ranked = [
                item
                for item in pre_confidence_ranked
                if id(
                    item
                ) not in accepted_ids
            ]

            accepted_chunks_count = len(
                ranked
            )

            rejected_chunks_count = len(
                rejected_ranked
            )

            self._record_qa_stage(
                "ACCEPTED",
                ranked,
                limit=final_top_k,
                accepted=True
            )

            self._record_qa_stage(
                "REJECTED",
                rejected_ranked,
                limit=final_top_k,
                accepted=False,
                rejection_reason=(
                    "Reranker score is below "
                    "the confidence threshold."
                )
            )

            if DEBUG_RETRIEVAL:

                print(
                    "\n===== PER-RESULT CONFIDENCE FILTER ====="
                )

                print(
                    f"Minimum required : "
                    f"{MIN_RETRIEVAL_SCORE:.4f}"
                )

                print(
                    f"Accepted chunks  : "
                    f"{len(ranked)}"
                )

                for item in ranked:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"{item.get('rerank_score', 0.0):.4f}"
                    )

                print(
                    "========================================\n"
                )

                print("\n===== RERANKED RESULTS =====")

                for item in ranked[:10]:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"rerank={item.get('rerank_score', 0.0):.4f} "
                        f"hybrid={item.get('score', 0.0):.4f} "
                        f"info={item.get('_info_score', 0.0):.4f}"
                    )

                print("============================\n")

        else:

            ranked = candidates[:final_top_k]

            accepted_chunks_count = len(
                ranked
            )

            rejected_chunks_count = max(
                0,
                len(
                    candidates
                ) - len(
                    ranked
                )
            )

            self._record_qa_stage(
                "ACCEPTED",
                ranked,
                limit=final_top_k,
                accepted=True
            )

            if DEBUG_RETRIEVAL:

                print("\n===== FINAL HYBRID RESULTS =====")

                for item in ranked:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"{item.get('score', 0.0):.4f} "
                        f"| info={item.get('_info_score', 0.0):.4f} "
                        f"noise={item.get('_noise_score', 0)} "
                        f"cite={item.get('_citation_score', 0)} "
                        f"def={item.get('_has_definition', False)}"
                    )

                print("===============================\n")

        if is_list_mode:

            ranked = self._anchor_preserving_list_results(
                ranked,
                final_top_k=final_top_k,
            )

            max_chunks_per_file = final_top_k

        else:

            if source_family:
                # Multiple relevant MISRA rules normally live in the same PDF.
                # Do not let the ordinary cross-document diversity cap hide
                # additional applicable rules in a source-scoped assessment.
                max_chunks_per_file = final_top_k
            else:
                max_chunks_per_file = (
                    self._max_chunks_per_file_for_query(
                        query,
                        has_compound_facets=bool(
                            compound_facet_queries
                        )
                    )
                )

        diversified = self._apply_diversity_filter(
            ranked,
            max_chunks_per_file=max_chunks_per_file
        )

        diversified = diversified[:final_top_k]

        if DEBUG_RETRIEVAL:

            print("\n===== DIVERSIFIED RESULTS =====")

            for item in diversified[:10]:

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> "
                    f"{item.get('rerank_score', item.get('score', 0.0)):.4f}"
                )

            print("===============================\n")

        self._record_qa_stage(
            "FINAL",
            diversified,
            limit=final_top_k,
            accepted=True
        )

        evidence_logger.record_retrieval_summary(
            bm25_candidates=len(
                bm25_results
            ),
            vector_candidates=len(
                vector_results
            ),
            hybrid_candidates=len(
                merged
            ),
            reranker_candidates=(
                reranker_candidates_count
            ),
            accepted_chunks=(
                accepted_chunks_count
            ),
            rejected_chunks=(
                rejected_chunks_count
            ),
            final_chunks=len(
                diversified
            ),
            configured_top_k=final_top_k,
            confidence_threshold=(
                MIN_RETRIEVAL_SCORE
                if reranker is not None
                else None
            )
        )

        if not diversified and not structured_reference and not source_family:
            rescue_results = self._general_corpus_empty_result_rescue(
                query=query,
                intent_query=intent_query,
                final_top_k=final_top_k,
            )
            if rescue_results:
                return rescue_results

        return diversified
    
    def build_context(
        self,
        query,
        intent_query=None,
        source_family=None,
        final_top_k_override=None,
    ):

        results = self.retrieve(
            query,
            intent_query=intent_query,
            source_family=source_family,
            final_top_k_override=final_top_k_override,
        )

        context_limit = (
            max(1, int(final_top_k_override))
            if final_top_k_override is not None
            else self._final_top_k_for_query(query)
        )

        # v6.4.58: a structured topic-family inventory is already a bounded,
        # source-grounded result set assembled by the retriever.  Do not apply
        # the generic context Top-K a second time: that can silently truncate
        # a complete family (for example seven explicit Rule members) after
        # successful reconciliation.  Normal hybrid/list results keep the
        # existing context cap unchanged.
        structured_family_results = [
            item for item in results
            if isinstance(item, dict) and item.get("_structured_topic_family")
        ]
        if results and len(structured_family_results) == len(results):
            context_limit = max(context_limit, len(results))

        results = results[:context_limit]

        context = ""

        for index, item in enumerate(
            results,
            start=1
        ):

            context += (
                f"\n\n"
                f"===== DOCUMENT {index} =====\n"
                f"{item['text']}"
            )

        evidence_logger.record_context(
            context=context,
            chunk_count=len(
                results
            )
        )

        if DEBUG_RETRIEVAL:

            print("\n===== FINAL CONTEXT PREVIEW =====")

            for index, item in enumerate(
                results,
                start=1
            ):

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:300]

                print(
                    f"\nDOCUMENT {index}: "
                    f"{item['metadata'].get('file_name', 'Unknown')}"
                )

                print(preview)

            print("=================================\n")

        return context, results