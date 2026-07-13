"""
Temporary QA evidence logger for DocuBot.

Purpose:
- Record actual test execution evidence.
- Save readable technical logs.
- Save one CSV summary row per completed test.
- Record queries, retrieved chunks, scores, answers, sources, duration,
  expected result, actual result, and status.

Removal:
- Delete the entire qa folder.
- Remove EvidenceLogger imports/calls from instrumented files.
- No retrieval or answer behavior is changed by this module.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import uuid

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ==========================================================
# SAFE SETTINGS IMPORT
# ==========================================================
# Safe fallback values are used so this module will not crash
# when the temporary QA settings have not yet been added.
try:
    import config.settings as app_settings

    TEST_EVIDENCE_MODE = getattr(
        app_settings,
        "TEST_EVIDENCE_MODE",
        False
    )

    TEST_EVIDENCE_FULL_CHUNKS = getattr(
        app_settings,
        "TEST_EVIDENCE_FULL_CHUNKS",
        False
    )

    TEST_EVIDENCE_PREVIEW_LENGTH = getattr(
        app_settings,
        "TEST_EVIDENCE_PREVIEW_LENGTH",
        700
    )

    ROOT_DIR = Path(
        getattr(
            app_settings,
            "ROOT_DIR",
            Path(__file__).resolve().parents[1]
        )
    )

except Exception:

    TEST_EVIDENCE_MODE = False
    TEST_EVIDENCE_FULL_CHUNKS = False
    TEST_EVIDENCE_PREVIEW_LENGTH = 700

    ROOT_DIR = Path(
        __file__
    ).resolve().parents[1]


# ==========================================================
# LOG DIRECTORY
# ==========================================================
LOG_DIR = ROOT_DIR / "logs"

EVIDENCE_LOG_DIR = (
    LOG_DIR
    / "test_evidence"
)


class EvidenceLogger:
    """
    Temporary test evidence recorder.

    Important:
    - Does not modify retrieval results.
    - Does not modify prompts.
    - Does not modify generated answers.
    - Does not decide whether a test passed unless requested.
    - Safe no-op when TEST_EVIDENCE_MODE is False.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None
    ):

        self.enabled = (
            TEST_EVIDENCE_MODE
            if enabled is None
            else bool(enabled)
        )

        self.full_chunks = bool(
            TEST_EVIDENCE_FULL_CHUNKS
        )

        self.preview_length = max(
            100,
            int(
                TEST_EVIDENCE_PREVIEW_LENGTH
            )
        )

        self._lock = threading.Lock()

        self._run: Dict[str, Any] = {}

        self._started_at_monotonic: Optional[
            float
        ] = None

        self._session_stamp = (
            datetime.now()
            .strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        )

        self.log_path = (
            EVIDENCE_LOG_DIR
            / (
                "test_evidence_"
                f"{self._session_stamp}.log"
            )
        )

        self.csv_path = (
            EVIDENCE_LOG_DIR
            / (
                "test_evidence_"
                f"{self._session_stamp}.csv"
            )
        )

        if self.enabled:

            EVIDENCE_LOG_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            self._initialize_log()

    # ======================================================
    # INITIALIZATION
    # ======================================================
    def _initialize_log(self):

        header = [
            "=" * 70,
            "DOCUBOT QA TEST EVIDENCE LOG",
            "=" * 70,
            (
                "Created At : "
                f"{self._timestamp()}"
            ),
            (
                "Log File   : "
                f"{self.log_path}"
            ),
            (
                "CSV File   : "
                f"{self.csv_path}"
            ),
            (
                "Full Chunks: "
                f"{self.full_chunks}"
            ),
            "=" * 70,
            "",
        ]

        self._append_lines(
            header
        )

    # ======================================================
    # TEST RUN
    # ======================================================
    def start_run(
        self,
        test_case_id: str = "",
        category: str = "",
        description: str = "",
        question: str = "",
        expected_result: str = "",
        run_number: int = 1,
        total_runs: int = 1,
        environment: Optional[
            Dict[str, Any]
        ] = None
    ) -> str:

        """
        Start one test execution.

        Returns:
            Unique run ID.
        """

        if not self.enabled:

            return ""

        run_id = self._make_run_id()

        self._started_at_monotonic = (
            time.perf_counter()
        )

        self._run = {
            "run_id": run_id,
            "test_case_id": (
                test_case_id
                or "UNASSIGNED"
            ),
            "category": (
                category
                or "Uncategorized"
            ),
            "description": description,
            "question": question,
            "expected_result": (
                expected_result
            ),
            "actual_result": "",
            "status": "RUNNING",
            "run_number": run_number,
            "total_runs": total_runs,
            "started_at": self._timestamp(),
            "finished_at": "",
            "duration_seconds": "",
            "error": "",
            "sources": [],
            "accepted_chunks": 0,
            "final_chunks": 0,
            "fallback_used": False,
        }

        self._write_section(
            "TEST EXECUTION",
            {
                "Run ID": run_id,
                "Test Case ID": (
                    self._run[
                        "test_case_id"
                    ]
                ),
                "Category": category,
                "Description": description,
                "Run": (
                    f"{run_number} "
                    f"of {total_runs}"
                ),
                "Started At": (
                    self._run[
                        "started_at"
                    ]
                ),
                "Question": question,
                "Expected Result": (
                    expected_result
                ),
            }
        )

        if environment:

            self.record_environment(
                environment
            )

        return run_id

    def finish_run(
        self,
        status: str = "",
        actual_result: str = "",
        error: str = "",
        notes: str = ""
    ):

        """
        Finish the active test execution and append one CSV row.

        Supported statuses:
        - PASS
        - FAIL
        - MANUAL CHECK
        - ERROR

        When no status is provided:
        - ERROR is used when error exists.
        - MANUAL CHECK is used otherwise.
        """

        if not self.enabled:

            return

        if not self._run:

            return

        finished_at = self._timestamp()

        duration = ""

        if (
            self._started_at_monotonic
            is not None
        ):

            duration = round(
                time.perf_counter()
                - self._started_at_monotonic,
                4
            )

        final_status = (
            status.strip().upper()
            if status
            else (
                "ERROR"
                if error
                else "MANUAL CHECK"
            )
        )

        self._run.update(
            {
                "actual_result":
                    actual_result,

                "status":
                    final_status,

                "finished_at":
                    finished_at,

                "duration_seconds":
                    duration,

                "error":
                    error,
            }
        )

        self._write_section(
            "TEST RESULT",
            {
                "Expected Result":
                    self._run.get(
                        "expected_result",
                        ""
                    ),

                "Actual Result":
                    actual_result,

                "Status":
                    final_status,

                "Duration":
                    (
                        f"{duration} seconds"
                        if duration != ""
                        else ""
                    ),

                "Error":
                    error or "None",

                "Notes":
                    notes,
            }
        )

        self._append_csv_row()

        self._append_lines(
            [
                "",
                "=" * 70,
                "END OF TEST EXECUTION",
                "=" * 70,
                "",
            ]
        )

        self._run = {}

        self._started_at_monotonic = None

    # ======================================================
    # ENVIRONMENT
    # ======================================================
    def record_environment(
        self,
        environment: Dict[str, Any]
    ):

        if not self.enabled:

            return

        self._write_section(
            "TEST ENVIRONMENT",
            environment
        )

    # ======================================================
    # QUESTION PIPELINE
    # ======================================================
    def record_question_pipeline(
        self,
        original_question: str = "",
        normalized_question: str = "",
        resolved_question: str = "",
        search_question: str = "",
        answer_focus: str = "",
        current_topic: str = "",
        history: str = ""
    ):

        if not self.enabled:

            return

        self._write_section(
            "QUESTION PIPELINE",
            {
                "Original Question":
                    original_question,

                "Normalized Question":
                    normalized_question,

                "Resolved Question":
                    resolved_question,

                "Search Question":
                    search_question,

                "Answer Focus":
                    answer_focus,

                "Current Topic":
                    current_topic,

                "History":
                    history,
            }
        )

    # ======================================================
    # DOCUMENT INGESTION
    # ======================================================
    def record_document(
        self,
        file_name: str,
        file_path: str = "",
        extension: str = "",
        loader: str = "",
        status: str = "",
        character_count: Optional[
            int
        ] = None,
        chunk_count: Optional[
            int
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        error: str = ""
    ):

        if not self.enabled:

            return

        data = {
            "File Name": file_name,
            "File Path": file_path,
            "Extension": extension,
            "Loader": loader,
            "Status": status,
            "Character Count":
                character_count,
            "Chunk Count": chunk_count,
            "Error": error or "None",
        }

        if metadata:

            data["Metadata"] = metadata

        self._write_section(
            "DOCUMENT INGESTION",
            data
        )

    # ======================================================
    # INDEX BUILD
    # ======================================================
    def record_index_summary(
        self,
        discovered_files: Optional[
            int
        ] = None,
        processed_files: Optional[
            int
        ] = None,
        skipped_files: Optional[
            int
        ] = None,
        failed_files: Optional[
            int
        ] = None,
        total_chunks: Optional[
            int
        ] = None,
        chroma_count: Optional[
            int
        ] = None,
        bm25_count: Optional[
            int
        ] = None,
        added_files: Optional[
            int
        ] = None,
        updated_files: Optional[
            int
        ] = None,
        deleted_files: Optional[
            int
        ] = None,
        unchanged_files: Optional[
            int
        ] = None,
        extra: Optional[
            Dict[str, Any]
        ] = None
    ):

        if not self.enabled:

            return

        data = {
            "Discovered Files":
                discovered_files,

            "Processed Files":
                processed_files,

            "Skipped Files":
                skipped_files,

            "Failed Files":
                failed_files,

            "Total Chunks":
                total_chunks,

            "Chroma Count":
                chroma_count,

            "BM25 Count":
                bm25_count,

            "Added Files":
                added_files,

            "Updated Files":
                updated_files,

            "Deleted Files":
                deleted_files,

            "Unchanged Files":
                unchanged_files,
        }

        if extra:

            data.update(
                extra
            )

        self._write_section(
            "INDEX BUILD SUMMARY",
            data
        )

    # ======================================================
    # RETRIEVAL
    # ======================================================
    def record_retrieval_summary(
        self,
        bm25_candidates: Optional[
            int
        ] = None,
        vector_candidates: Optional[
            int
        ] = None,
        hybrid_candidates: Optional[
            int
        ] = None,
        reranker_candidates: Optional[
            int
        ] = None,
        accepted_chunks: Optional[
            int
        ] = None,
        rejected_chunks: Optional[
            int
        ] = None,
        final_chunks: Optional[
            int
        ] = None,
        configured_top_k: Optional[
            int
        ] = None,
        confidence_threshold: Optional[
            float
        ] = None,
        context_character_count: Optional[
            int
        ] = None
    ):

        if not self.enabled:

            return

        if accepted_chunks is not None:

            self._run[
                "accepted_chunks"
            ] = accepted_chunks

        if final_chunks is not None:

            self._run[
                "final_chunks"
            ] = final_chunks

        self._write_section(
            "RETRIEVAL SUMMARY",
            {
                "BM25 Candidates":
                    bm25_candidates,

                "Vector Candidates":
                    vector_candidates,

                "Hybrid Candidates":
                    hybrid_candidates,

                "Reranker Candidates":
                    reranker_candidates,

                "Accepted Chunks":
                    accepted_chunks,

                "Rejected Chunks":
                    rejected_chunks,

                "Final Chunks":
                    final_chunks,

                "Configured Top-K":
                    configured_top_k,

                "Confidence Threshold":
                    confidence_threshold,

                "Context Characters":
                    context_character_count,
            }
        )

    def record_chunk(
        self,
        stage: str,
        position: int,
        item: Dict[str, Any],
        accepted: Optional[
            bool
        ] = None,
        rejection_reason: str = ""
    ):

        """
        Record one retrieval candidate or final chunk.

        The item can contain:
        - text
        - metadata
        - bm25_score
        - vector_score
        - hybrid_score
        - informative_score
        - rerank_score
        - final_score
        """

        if not self.enabled:

            return

        metadata = item.get(
            "metadata",
            {}
        ) or {}

        text = (
            item.get("text")
            or item.get("document")
            or item.get("content")
            or ""
        )

        chunk_data = {
            "Stage": stage,
            "Position": position,

            "Source":
                metadata.get(
                    "file_name",
                    "Unknown"
                ),

            "File Path":
                metadata.get(
                    "file_path",
                    ""
                ),

            "Chunk ID":
                metadata.get(
                    "chunk_id",
                    item.get(
                        "chunk_id",
                        ""
                    )
                ),

            "Page":
                metadata.get(
                    "page_number",
                    metadata.get(
                        "page",
                        ""
                    )
                ),

            "Section":
                metadata.get(
                    "section",
                    metadata.get(
                        "heading",
                        ""
                    )
                ),

            "BM25 Score":
                item.get(
                    "bm25_score",
                    ""
                ),

            "Vector Score":
                item.get(
                    "vector_score",
                    ""
                ),

            "Hybrid Score":
                item.get(
                    "hybrid_score",
                    ""
                ),

            "Informative Score":
                item.get(
                    "informative_score",
                    item.get(
                        "info_score",
                        ""
                    )
                ),

            "Reranker Score":
                item.get(
                    "rerank_score",
                    ""
                ),

            "Final Score":
                item.get(
                    "final_score",
                    ""
                ),

            "Accepted":
                (
                    accepted
                    if accepted is not None
                    else ""
                ),

            "Rejection Reason":
                rejection_reason,

            "Text":
                (
                    self._clean_text(
                        text
                    )
                    if self.full_chunks
                    else self._preview_text(
                        text
                    )
                ),
        }

        self._write_section(
            (
                f"CHUNK {position} "
                f"[{stage}]"
            ),
            chunk_data
        )

    def record_chunks(
        self,
        stage: str,
        items: Iterable[
            Dict[str, Any]
        ],
        accepted: Optional[
            bool
        ] = None
    ):

        if not self.enabled:

            return

        for position, item in enumerate(
            items,
            start=1
        ):

            self.record_chunk(
                stage=stage,
                position=position,
                item=item,
                accepted=accepted
            )

    # ======================================================
    # FINAL CONTEXT
    # ======================================================
    def record_context(
        self,
        context: str,
        chunk_count: Optional[
            int
        ] = None
    ):

        if not self.enabled:

            return

        context_text = (
            self._clean_text(
                context
            )
            if self.full_chunks
            else self._preview_text(
                context,
                max_length=2000
            )
        )

        self._write_section(
            "FINAL CONTEXT",
            {
                "Chunk Count":
                    chunk_count,

                "Character Count":
                    len(
                        context or ""
                    ),

                "Context":
                    context_text,
            }
        )

    # ======================================================
    # ANSWER
    # ======================================================
    def record_answer(
        self,
        draft_answer: str = "",
        verified_answer: str = "",
        final_answer: str = "",
        fallback_used: bool = False,
        prompt_leak_detected: bool = False,
        sources: Optional[
            List[Any]
        ] = None,
        generation_seconds: Optional[
            float
        ] = None,
        error: str = ""
    ):

        if not self.enabled:

            return

        clean_sources = self._normalize_sources(
            sources or []
        )

        self._run[
            "fallback_used"
        ] = bool(
            fallback_used
        )

        self._run[
            "sources"
        ] = clean_sources

        self._write_section(
            "ANSWER RESULT",
            {
                "Draft Answer":
                    draft_answer,

                "Verified Answer":
                    verified_answer,

                "Final Answer":
                    final_answer,

                "Fallback Used":
                    fallback_used,

                "Prompt Leak Detected":
                    prompt_leak_detected,

                "Sources":
                    clean_sources,

                "Generation Duration":
                    (
                        f"{generation_seconds} seconds"
                        if generation_seconds
                        is not None
                        else ""
                    ),

                "Error":
                    error or "None",
            }
        )

    # ======================================================
    # GENERAL EVENT
    # ======================================================
    def record_event(
        self,
        event_name: str,
        details: Any = "",
        status: str = ""
    ):

        """
        Record a general event such as:
        - New Chat clicked
        - Model loaded
        - Source button clicked
        - Unsupported file skipped
        """

        if not self.enabled:

            return

        self._write_section(
            "EVENT",
            {
                "Event": event_name,
                "Status": status,
                "Details": details,
                "Timestamp":
                    self._timestamp(),
            }
        )

    def record_error(
        self,
        location: str,
        error: Any,
        details: Any = ""
    ):

        if not self.enabled:

            return

        self._write_section(
            "ERROR",
            {
                "Location": location,
                "Error": str(error),
                "Details": details,
                "Timestamp":
                    self._timestamp(),
            }
        )

    # ======================================================
    # INTERNAL FILE WRITING
    # ======================================================
    def _write_section(
        self,
        title: str,
        data: Dict[str, Any]
    ):

        lines = [
            "",
            "=" * 70,
            title,
            "=" * 70,
        ]

        for key, value in data.items():

            if value is None:

                continue

            formatted = self._format_value(
                value
            )

            if "\n" in formatted:

                lines.append(
                    f"{key}:"
                )

                lines.append(
                    formatted
                )

            else:

                lines.append(
                    f"{key:<24}: "
                    f"{formatted}"
                )

        self._append_lines(
            lines
        )

    def _append_lines(
        self,
        lines: Iterable[str]
    ):

        if not self.enabled:

            return

        with self._lock:

            with self.log_path.open(
                "a",
                encoding="utf-8"
            ) as log_file:

                for line in lines:

                    log_file.write(
                        f"{line}\n"
                    )

    def _append_csv_row(self):

        if not self.enabled:

            return

        columns = [
            "run_id",
            "test_case_id",
            "category",
            "description",
            "question",
            "expected_result",
            "actual_result",
            "status",
            "run_number",
            "total_runs",
            "started_at",
            "finished_at",
            "duration_seconds",
            "accepted_chunks",
            "final_chunks",
            "fallback_used",
            "sources",
            "error",
            "evidence_log",
        ]

        row = {
            "run_id":
                self._run.get(
                    "run_id",
                    ""
                ),

            "test_case_id":
                self._run.get(
                    "test_case_id",
                    ""
                ),

            "category":
                self._run.get(
                    "category",
                    ""
                ),

            "description":
                self._run.get(
                    "description",
                    ""
                ),

            "question":
                self._run.get(
                    "question",
                    ""
                ),

            "expected_result":
                self._run.get(
                    "expected_result",
                    ""
                ),

            "actual_result":
                self._run.get(
                    "actual_result",
                    ""
                ),

            "status":
                self._run.get(
                    "status",
                    ""
                ),

            "run_number":
                self._run.get(
                    "run_number",
                    ""
                ),

            "total_runs":
                self._run.get(
                    "total_runs",
                    ""
                ),

            "started_at":
                self._run.get(
                    "started_at",
                    ""
                ),

            "finished_at":
                self._run.get(
                    "finished_at",
                    ""
                ),

            "duration_seconds":
                self._run.get(
                    "duration_seconds",
                    ""
                ),

            "accepted_chunks":
                self._run.get(
                    "accepted_chunks",
                    ""
                ),

            "final_chunks":
                self._run.get(
                    "final_chunks",
                    ""
                ),

            "fallback_used":
                self._run.get(
                    "fallback_used",
                    False
                ),

            "sources":
                json.dumps(
                    self._run.get(
                        "sources",
                        []
                    ),
                    ensure_ascii=False
                ),

            "error":
                self._run.get(
                    "error",
                    ""
                ),

            "evidence_log":
                str(
                    self.log_path
                ),
        }

        write_header = (
            not self.csv_path.exists()
            or self.csv_path.stat().st_size
            == 0
        )

        with self._lock:

            with self.csv_path.open(
                "a",
                encoding="utf-8-sig",
                newline=""
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=columns
                )

                if write_header:

                    writer.writeheader()

                writer.writerow(
                    row
                )

    # ======================================================
    # HELPERS
    # ======================================================
    def _make_run_id(self) -> str:

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d-%H%M%S"
            )
        )

        suffix = (
            uuid.uuid4()
            .hex[:6]
            .upper()
        )

        return (
            f"QA-{timestamp}-{suffix}"
        )

    def _timestamp(self) -> str:

        return (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    def _format_value(
        self,
        value: Any
    ) -> str:

        if isinstance(
            value,
            (dict, list, tuple)
        ):

            try:

                return json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )

            except Exception:

                return str(value)

        return str(value)

    def _clean_text(
        self,
        text: Any
    ) -> str:

        if text is None:

            return ""

        text = str(text)

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    def _preview_text(
        self,
        text: Any,
        max_length: Optional[
            int
        ] = None
    ) -> str:

        clean = self._clean_text(
            text
        )

        limit = (
            max_length
            if max_length is not None
            else self.preview_length
        )

        if len(clean) <= limit:

            return clean

        return (
            clean[:limit].rstrip()
            + "\n...[TRUNCATED]"
        )

    def _normalize_sources(
        self,
        sources: List[Any]
    ) -> List[Any]:

        output = []

        for source in sources:

            if isinstance(
                source,
                dict
            ):

                output.append(
                    {
                        "name":
                            source.get(
                                "name",
                                source.get(
                                    "file_name",
                                    "Unknown"
                                )
                            ),

                        "path":
                            source.get(
                                "path",
                                source.get(
                                    "file_path",
                                    ""
                                )
                            ),
                    }
                )

            else:

                output.append(
                    str(source)
                )

        return output


# ==========================================================
# SHARED LOGGER INSTANCE
# ==========================================================
# Import this shared instance from other files:
#
# from qa.evidence_logger import evidence_logger
#
# It becomes a no-op automatically when TEST_EVIDENCE_MODE=False.
evidence_logger = EvidenceLogger()