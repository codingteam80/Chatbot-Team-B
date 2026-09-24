"""DocuBot knowledge-base coverage/health report.

This probe is intentionally read-only.  It never rebuilds the knowledge base,
loads an embedding model, or calls an LLM.  Its job is to answer one operational
question: are the supported files currently present under
``data/technical_documents`` represented in the active manifest and Qdrant
collection?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DOCUMENT_DIR,
    METADATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
    SUPPORTED_EXTENSIONS,
    VECTOR_BACKEND,
)

MANIFEST_FILE = METADATA_DIR / "manifest.json"
REPORT_DIR = Path("logs") / "kb_health"


@dataclass
class FileHealth:
    name: str
    extension: str
    supported: bool
    manifest_status: str
    chunk_count: int | None
    note: str = ""

    @property
    def healthy(self) -> bool:
        return (
            self.supported
            and self.manifest_status == "indexed"
            and self.chunk_count is not None
            and self.chunk_count > 0
        )


def _all_source_files(document_dir: Path = DOCUMENT_DIR):
    if not document_dir.exists():
        return []
    return sorted(
        (path for path in document_dir.rglob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )


def _load_manifest(path: Path = MANIFEST_FILE):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_by_filename(manifest):
    by_name = {}
    for key, info in manifest.items():
        if not isinstance(info, dict):
            continue
        name = info.get("file_name") or Path(str(info.get("file_path") or key)).name
        if name:
            by_name[str(name).casefold()] = info
    return by_name


class _QdrantHealthCollection:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=str(QDRANT_DIR))

    def get(self, where=None, include=None):
        from qdrant_client import models

        where = where or {}
        conditions = []
        for field, value in where.items():
            conditions.append(
                models.FieldCondition(
                    key=f"metadata.{field}",
                    match=models.MatchValue(value=value),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        ids = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(point.id) for point in points)
            if offset is None:
                break
        return {"ids": ids, "metadatas": []}

    def close(self):
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _open_collection():
    """Open the finalized Qdrant production collection without loading ML models."""
    try:
        collection = _QdrantHealthCollection()
        if hasattr(collection.client, "collection_exists"):
            if not collection.client.collection_exists(QDRANT_COLLECTION_NAME):
                collection.close()
                return None
        return collection
    except Exception:
        return None


def _chunk_count_for_file(collection, source_path: Path, manifest_info):
    if collection is None:
        return None

    candidates = []
    if isinstance(manifest_info, dict):
        for field in ("indexed_file_path", "file_path"):
            value = manifest_info.get(field)
            if value:
                candidates.append(("file_path", str(value)))
    candidates.append(("file_path", str(source_path.resolve())))
    candidates.append(("file_name", source_path.name))

    seen = set()
    for field, value in candidates:
        marker = (field, value)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            result = collection.get(where={field: value}, include=["metadatas"])
        except Exception:
            continue
        ids = result.get("ids") or []
        if ids:
            return len(ids)
    return 0


def build_health_rows(source_files, manifest, collection):
    manifest_by_name = _manifest_by_filename(manifest)
    rows = []

    for path in source_files:
        extension = path.suffix.lower()
        supported = extension in SUPPORTED_EXTENSIONS
        info = manifest_by_name.get(path.name.casefold())

        if not supported:
            rows.append(
                FileHealth(
                    name=path.name,
                    extension=extension or "(none)",
                    supported=False,
                    manifest_status="unsupported",
                    chunk_count=None,
                    note="File type is not in SUPPORTED_EXTENSIONS.",
                )
            )
            continue

        if not info:
            rows.append(
                FileHealth(
                    name=path.name,
                    extension=extension,
                    supported=True,
                    manifest_status="missing",
                    chunk_count=0 if collection is not None else None,
                    note="Supported file is not tracked by the active manifest.",
                )
            )
            continue

        index_status = str(info.get("index_status") or "indexed").strip().lower()
        if index_status == "skipped":
            rows.append(
                FileHealth(
                    name=path.name,
                    extension=extension,
                    supported=True,
                    manifest_status="skipped",
                    chunk_count=0,
                    note=str(info.get("skip_reason") or "Validation/indexing skipped this file."),
                )
            )
            continue

        chunk_count = _chunk_count_for_file(collection, path, info)
        note = ""
        if collection is None:
            note = f"Active {VECTOR_BACKEND} collection could not be opened; chunk count unavailable."
        elif chunk_count == 0:
            note = f"Manifest says indexed, but no active {VECTOR_BACKEND} chunks were found."

        rows.append(
            FileHealth(
                name=path.name,
                extension=extension,
                supported=True,
                manifest_status="indexed",
                chunk_count=chunk_count,
                note=note,
            )
        )

    return rows



def _is_expected_nonblocking_skip(row: FileHealth) -> bool:
    """Return True for intentional ingestion-policy skips, not failures.

    A supported text file below the configured minimum length is deliberately
    excluded by Smart Build because it cannot form a useful knowledge chunk.
    It should remain visible in the health report, but it must not make the
    entire corpus unhealthy.  Other skipped reasons remain blocking.
    """
    if row.manifest_status != "skipped":
        return False
    note = str(row.note or "").strip().casefold()
    return note.startswith("too short [") and "minimum" in note and "chars" in note

def _render_report(rows, manifest, collection_available):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    supported = [row for row in rows if row.supported]
    unsupported = [row for row in rows if not row.supported]
    healthy = [row for row in supported if row.healthy]
    skipped = [row for row in supported if row.manifest_status == "skipped"]
    expected_skips = [row for row in skipped if _is_expected_nonblocking_skip(row)]
    blocking_skips = [row for row in skipped if row not in expected_skips]
    missing = [row for row in supported if row.manifest_status == "missing"]
    zero_chunk = [
        row for row in supported
        if row.manifest_status == "indexed" and row.chunk_count == 0
    ]
    unknown_chunk = [
        row for row in supported
        if row.manifest_status == "indexed" and row.chunk_count is None
    ]
    total_chunks = sum(row.chunk_count or 0 for row in supported)

    lines = [
        "DocuBot Knowledge Base Health Report",
        "=" * 76,
        f"Created                 : {now}",
        f"Knowledge folder        : {DOCUMENT_DIR}",
        "Vector backend            : qdrant",
        f"Active vector path        : {QDRANT_DIR}",
        f"Vector collection         : {QDRANT_COLLECTION_NAME}",
        f"Vector readable           : {'YES' if collection_available else 'NO'}",
        "",
        "Coverage Summary",
        "-" * 76,
        f"Files discovered         : {len(rows)}",
        f"Supported files          : {len(supported)}",
        f"Unsupported files        : {len(unsupported)}",
        f"Healthy indexed files    : {len(healthy)}",
        f"Manifest-missing files   : {len(missing)}",
        f"Validation-skipped files : {len(skipped)}",
        f"Expected benign skips    : {len(expected_skips)}",
        f"Blocking skipped files   : {len(blocking_skips)}",
        f"Zero-chunk indexed files : {len(zero_chunk)}",
        f"Unknown chunk counts     : {len(unknown_chunk)}",
        f"Active chunks counted    : {total_chunks}",
        f"Manifest entries         : {len(manifest)}",
        "",
        "Per-file Coverage",
        "-" * 76,
    ]

    for row in rows:
        if not row.supported:
            state = "UNSUPPORTED"
        elif row.healthy:
            state = "OK"
        elif row.manifest_status == "skipped":
            state = "SKIPPED"
        elif row.manifest_status == "missing":
            state = "MISSING"
        elif row.chunk_count == 0:
            state = "ZERO-CHUNK"
        else:
            state = "CHECK"
        chunks = "n/a" if row.chunk_count is None else str(row.chunk_count)
        lines.append(f"[{state:11}] chunks={chunks:>6} | {row.name}")
        if row.note:
            lines.append(f"              note: {row.note}")

    ext_counts = Counter(row.extension for row in rows)
    if ext_counts:
        lines.extend(["", "Discovered File Types", "-" * 76])
        for extension, count in sorted(ext_counts.items()):
            lines.append(f"{extension:12} : {count}")

    blocking = missing + blocking_skips + zero_chunk
    if not collection_available:
        overall = "CHECK"
    elif blocking:
        overall = "ISSUES FOUND"
    else:
        overall = "HEALTHY"

    lines.extend([
        "",
        "Result",
        "-" * 76,
        f"Overall                 : {overall}",
    ])
    if blocking:
        lines.append(
            "Action                  : Run Update_DocuBot_Knowledge_Base.bat and review the listed file(s)."
        )
    elif not collection_available:
        lines.append(
            "Action                  : Run this check inside the DocuBot venv where qdrant-client is installed."
        )
    else:
        if expected_skips:
            lines.append(
                "Note                    : Intentional too-short files are reported but do not block retrieval validation."
            )
        lines.append(
            "Action                  : Knowledge-base coverage is ready for retrieval/manual QA."
        )

    return "\n".join(lines) + "\n", overall


def run_health_check(report_dir: Path = REPORT_DIR):
    source_files = _all_source_files()
    manifest = _load_manifest()

    collection = None
    collection_error = ""
    try:
        collection = _open_collection()
    except Exception as error:
        collection_error = f"{type(error).__name__}: {error}"

    try:
        rows = build_health_rows(source_files, manifest, collection)
        report, overall = _render_report(rows, manifest, collection is not None)
    finally:
        close = getattr(collection, "close", None) if collection is not None else None
        if callable(close):
            close()

    if collection_error:
        report += f"Vector open error       : {collection_error}\n"

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"kb_health_{timestamp}.txt"
    path.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Report saved            : {path.resolve()}")

    return 0 if overall == "HEALTHY" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only DocuBot KB coverage check")
    parser.parse_args(argv)
    return run_health_check()


if __name__ == "__main__":
    raise SystemExit(main())
