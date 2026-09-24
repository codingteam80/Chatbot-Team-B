"""Generate read-only chunking evidence from the active DocuBot KB.

The BM25 corpus is used as the authoritative text-record snapshot because the
production updater rebuilds BM25 from the same final chunk records that are
activated in Qdrant.  When qdrant-client is available, this tool also performs a
read-only active Qdrant point-count cross-check.

No parsing, embedding, indexing, deletion, or mutation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    BM25_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_PROFILE,
    INDEX_SCHEMA_VERSION,
    METADATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
)

CORPUS_FILE = BM25_DIR / "corpus.pkl"
MANIFEST_FILE = METADATA_DIR / "manifest.json"
OUT_ROOT = ROOT / "logs" / "chunking_evidence"
REF_RE = re.compile(r"\b(?P<kind>Rule|Directive)\s+(?P<id>\d+(?:\.\d+)+)\b", re.I)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        return {}
    value = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_records() -> list[dict[str, Any]]:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Active BM25 corpus not found: {CORPUS_FILE}")
    with CORPUS_FILE.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, list):
        raise TypeError("Active BM25 corpus is not a list of chunk records.")
    return [item for item in value if isinstance(item, dict)]


def _canonical_ref(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    match = REF_RE.search(text)
    if not match:
        return ""
    kind = "Directive" if match.group("kind").casefold().startswith("dir") else "Rule"
    return f"{kind} {match.group('id')}"


def _metadata_reference(metadata: dict[str, Any]) -> str:
    section_type = str(metadata.get("section_type", "") or "").casefold()
    if section_type == "rule":
        rid = str(metadata.get("rule_id", "") or "").strip()
        return f"Rule {rid}" if rid else ""
    if section_type == "directive":
        did = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or metadata.get("section_id", "") or "").strip()
        return f"Directive {did}" if did else ""
    parent_type = str(metadata.get("parent_type", "") or "").casefold()
    parent_id = str(metadata.get("parent_identifier", "") or "").strip()
    if parent_type == "rule" and parent_id:
        return f"Rule {parent_id}"
    if parent_type == "directive" and parent_id:
        return f"Directive {parent_id}"
    return ""


def _latest_retrieval_references() -> tuple[list[str], str]:
    diagnostics = ROOT / "logs" / "pipeline_diagnostics"
    candidates = sorted(diagnostics.glob("retrieval*_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if diagnostics.exists() else []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        refs: list[str] = []
        for case in payload.get("cases") or []:
            for raw in case.get("final_references") or []:
                ref = _canonical_ref(raw)
                if ref and ref not in refs:
                    refs.append(ref)
        if refs:
            return refs, str(path)
    return [], ""


def _qdrant_count() -> dict[str, Any]:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        return {"available": False, "count": None, "note": f"qdrant-client unavailable: {type(exc).__name__}"}

    client = None
    try:
        client = QdrantClient(path=str(QDRANT_DIR))
        count = int(client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count)
        return {"available": True, "count": count, "note": "read-only exact count"}
    except Exception as exc:
        return {"available": False, "count": None, "note": f"Qdrant count unavailable: {type(exc).__name__}: {exc}"}
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_+#.-]+", str(text or "").casefold())


def _suffix_prefix_overlap_words(left: str, right: str, maximum: int = 220) -> int:
    a = _word_tokens(left)
    b = _word_tokens(right)
    limit = min(maximum, len(a), len(b))
    for size in range(limit, 0, -1):
        if a[-size:] == b[:size]:
            return size
    return 0


def _sort_key(record: dict[str, Any]):
    meta = record.get("metadata", {}) or {}
    return (
        str(meta.get("file_name", "") or "").casefold(),
        int(meta.get("chunk_id", 0) or 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate read-only DocuBot chunking evidence")
    parser.add_argument("--focus-reference", action="append", default=[], help="Optional Rule/Directive reference to highlight; may be repeated")
    args = parser.parse_args()

    created = datetime.now()
    stamp = created.strftime("%Y%m%d_%H%M%S")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    records = _load_records()
    manifest = _load_manifest()
    records_sorted = sorted(records, key=_sort_key)

    focus_refs = []
    for raw in args.focus_reference:
        ref = _canonical_ref(raw)
        if ref and ref not in focus_refs:
            focus_refs.append(ref)
    focus_source = "cli"
    if not focus_refs:
        focus_refs, latest_path = _latest_retrieval_references()
        focus_source = latest_path or "none"

    per_file = defaultdict(list)
    role_counts = Counter()
    parser_counts = Counter()
    profile_counts = Counter()
    ref_counts = Counter()
    section_type_counts = Counter()
    empty_chunks = []

    inventory_rows = []
    for record in records_sorted:
        text = str(record.get("text", "") or "")
        meta = record.get("metadata", {}) or {}
        file_name = str(meta.get("file_name", "") or "")
        per_file[file_name].append(record)
        role_counts[str(meta.get("section_role", "") or "(none)")] += 1
        parser_counts[str(meta.get("parser_strategy", "") or "(none)")] += 1
        profile_counts[str(meta.get("chunking_profile", "") or "(none)")] += 1
        section_type_counts[str(meta.get("section_type", "") or "(none)")] += 1
        ref = _metadata_reference(meta)
        if ref:
            ref_counts[ref] += 1
        if not text.strip():
            empty_chunks.append({"file_name": file_name, "chunk_id": meta.get("chunk_id")})
        inventory_rows.append({
            "file_name": file_name,
            "chunk_id": meta.get("chunk_id", ""),
            "total_chunks": meta.get("total_chunks", ""),
            "parser_strategy": meta.get("parser_strategy", ""),
            "chunking_profile": meta.get("chunking_profile", ""),
            "section_type": meta.get("section_type", ""),
            "section_title": meta.get("section_title", ""),
            "section_role": meta.get("section_role", ""),
            "reference": ref,
            "parent_type": meta.get("parent_type", ""),
            "parent_identifier": meta.get("parent_identifier", ""),
            "page_start": meta.get("page_start", ""),
            "page_end": meta.get("page_end", ""),
            "section_index": meta.get("section_index", ""),
            "section_part": meta.get("section_part", ""),
            "section_parts": meta.get("section_parts", ""),
            "char_count": len(text),
            "word_count": len(_word_tokens(text)),
            "preview": re.sub(r"\s+", " ", text).strip()[:240],
        })

    file_summaries = []
    contiguous_checks = []
    total_chunk_field_checks = []
    for file_name, items in sorted(per_file.items()):
        ids = sorted(int((item.get("metadata", {}) or {}).get("chunk_id", 0) or 0) for item in items)
        expected_ids = list(range(len(ids)))
        contiguous = ids == expected_ids
        totals = {int((item.get("metadata", {}) or {}).get("total_chunks", 0) or 0) for item in items}
        total_ok = totals == {len(items)}
        contiguous_checks.append(contiguous)
        total_chunk_field_checks.append(total_ok)
        file_summaries.append({
            "file_name": file_name,
            "chunk_count": len(items),
            "chunk_ids_contiguous_0_based": contiguous,
            "total_chunks_metadata_consistent": total_ok,
            "min_chars": min((len(str(item.get("text", "") or "")) for item in items), default=0),
            "max_chars": max((len(str(item.get("text", "") or "")) for item in items), default=0),
            "average_chars": round(sum(len(str(item.get("text", "") or "")) for item in items) / len(items), 2) if items else 0,
        })

    # Heuristic overlap evidence only for chunks that are known parts of the same
    # structure-aware unit.  CHUNK_OVERLAP is a tokenizer setting; this report
    # intentionally labels word overlap as an estimate rather than claiming an
    # exact token-level measurement.
    multipart_groups = defaultdict(list)
    for record in records_sorted:
        meta = record.get("metadata", {}) or {}
        parts = int(meta.get("section_parts", 1) or 1)
        if parts <= 1:
            continue
        key = (str(meta.get("file_name", "")), str(meta.get("section_index", "")))
        multipart_groups[key].append(record)

    overlap_pairs = []
    for (file_name, section_index), items in sorted(multipart_groups.items()):
        items.sort(key=lambda r: int((r.get("metadata", {}) or {}).get("section_part", 0) or 0))
        for left, right in zip(items, items[1:]):
            lm, rm = left.get("metadata", {}) or {}, right.get("metadata", {}) or {}
            overlap_pairs.append({
                "file_name": file_name,
                "section_index": section_index,
                "left_chunk_id": lm.get("chunk_id"),
                "right_chunk_id": rm.get("chunk_id"),
                "left_section_part": lm.get("section_part"),
                "right_section_part": rm.get("section_part"),
                "word_overlap_estimate": _suffix_prefix_overlap_words(left.get("text", ""), right.get("text", "")),
            })

    qdrant = _qdrant_count()
    bm25_count = len(records)
    qdrant_count_match = (qdrant.get("count") == bm25_count) if qdrant.get("available") else None

    checks = {
        "bm25_corpus_nonempty": bm25_count > 0,
        "no_empty_chunk_text": not empty_chunks,
        "chunk_ids_contiguous_per_file": all(contiguous_checks) if contiguous_checks else False,
        "total_chunks_metadata_consistent_per_file": all(total_chunk_field_checks) if total_chunk_field_checks else False,
        "active_profile_is_v4": CHUNKING_PROFILE == "v4",
        "configured_chunk_size_900": int(CHUNK_SIZE) == 900,
        "configured_chunk_overlap_150": int(CHUNK_OVERLAP) == 150,
        "qdrant_count_matches_bm25_when_available": qdrant_count_match,
    }
    required_checks = [
        checks["bm25_corpus_nonempty"],
        checks["no_empty_chunk_text"],
        checks["chunk_ids_contiguous_per_file"],
        checks["total_chunks_metadata_consistent_per_file"],
        checks["active_profile_is_v4"],
        checks["configured_chunk_size_900"],
        checks["configured_chunk_overlap_150"],
    ]
    if qdrant_count_match is False:
        required_checks.append(False)
    overall = all(required_checks)

    focus_chunks = []
    focus_set = {ref.casefold() for ref in focus_refs}
    for record in records_sorted:
        ref = _metadata_reference(record.get("metadata", {}) or {})
        if ref and ref.casefold() in focus_set:
            focus_chunks.append(record)

    summary = {
        "version": "v6.5.5.7",
        "created": created.isoformat(timespec="seconds"),
        "mode": "read_only_active_chunking_evidence",
        "overall": "PASS" if overall else "FAIL",
        "configuration": {
            "chunking_profile": CHUNKING_PROFILE,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_size_note": "SentenceSplitter token target configured by production settings.",
            "overlap_evidence_note": "word_overlap_estimate is a heuristic display only; configured overlap remains the authoritative tokenizer setting.",
        },
        "sources": {
            "bm25_corpus": str(CORPUS_FILE),
            "manifest": str(MANIFEST_FILE),
            "qdrant_path": str(QDRANT_DIR),
            "qdrant_collection": QDRANT_COLLECTION_NAME,
        },
        "counts": {
            "bm25_chunks": bm25_count,
            "qdrant": qdrant,
            "manifest_entries": len(manifest),
            "files_in_corpus": len(per_file),
            "structured_references": len(ref_counts),
            "multipart_units": len(multipart_groups),
            "multipart_overlap_pairs": len(overlap_pairs),
        },
        "checks": checks,
        "file_summaries": file_summaries,
        "chunking_profiles": dict(profile_counts),
        "parser_strategies": dict(parser_counts),
        "section_types": dict(section_type_counts),
        "section_roles": dict(role_counts),
        "focus": {
            "references": focus_refs,
            "source": focus_source,
            "matching_chunk_count": len(focus_chunks),
        },
        "reference_chunk_counts": dict(sorted(ref_counts.items())),
        "multipart_overlap_evidence": overlap_pairs,
        "empty_chunks": empty_chunks,
    }

    work = Path(tempfile.mkdtemp(prefix="docubot-chunk-evidence-"))
    try:
        summary_path = work / f"chunking_evidence_summary_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest_path = work / "manifest_snapshot.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        inventory_path = work / "chunk_inventory.csv"
        fieldnames = list(inventory_rows[0].keys()) if inventory_rows else []
        with inventory_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory_rows)

        full_path = work / "chunk_text_full.txt"
        with full_path.open("w", encoding="utf-8") as f:
            for index, record in enumerate(records_sorted, 1):
                meta = record.get("metadata", {}) or {}
                f.write("=" * 96 + "\n")
                f.write(f"RECORD {index}/{len(records_sorted)}\n")
                for key in (
                    "file_name", "chunk_id", "total_chunks", "parser_strategy", "chunking_profile",
                    "section_type", "section_title", "section_role", "rule_id", "directive_id",
                    "parent_type", "parent_identifier", "page_start", "page_end", "section_index",
                    "section_part", "section_parts",
                ):
                    if key in meta and meta.get(key) not in (None, ""):
                        f.write(f"{key}: {meta.get(key)}\n")
                f.write(f"char_count: {len(str(record.get('text', '') or ''))}\n")
                f.write(f"word_count: {len(_word_tokens(record.get('text', '')))}\n")
                f.write("-" * 96 + "\n")
                f.write(str(record.get("text", "") or "").rstrip() + "\n\n")

        focus_path = work / "focus_retrieval_rule_chunks.txt"
        with focus_path.open("w", encoding="utf-8") as f:
            f.write("DocuBot Focus Chunk Evidence\n")
            f.write("=" * 80 + "\n")
            f.write("Focus references are taken from the newest retrieval diagnostic when available.\n")
            f.write("No phrase-to-Rule mapping is used by this evidence generator.\n\n")
            f.write(f"Focus source: {focus_source}\n")
            f.write(f"References: {', '.join(focus_refs) if focus_refs else '(none found)'}\n\n")
            for record in focus_chunks:
                meta = record.get("metadata", {}) or {}
                ref = _metadata_reference(meta)
                f.write("=" * 80 + "\n")
                f.write(f"{ref} | chunk_id={meta.get('chunk_id')} | role={meta.get('section_role')} | pages={meta.get('page_start')}-{meta.get('page_end')}\n")
                f.write("-" * 80 + "\n")
                f.write(str(record.get("text", "") or "").rstrip() + "\n\n")

        readme = work / "README_CHUNKING_EVIDENCE.txt"
        readme.write_text(
            "DocuBot v6.5.5.7 Chunking Evidence\n"
            "==================================\n"
            "This package is generated read-only from the active BM25 chunk corpus and manifest.\n"
            "The production updater builds BM25 from the same final chunk records activated in Qdrant.\n"
            "If qdrant-client can open the local collection, an exact point-count cross-check is also recorded.\n\n"
            "Files:\n"
            "- chunking_evidence_summary_*.json : configuration, QA checks, counts, per-file statistics\n"
            "- chunk_inventory.csv              : one row per active chunk with metadata and preview\n"
            "- chunk_text_full.txt              : complete text of every active chunk\n"
            "- focus_retrieval_rule_chunks.txt  : chunks for Rules/Directives seen in latest retrieval QA\n"
            "- manifest_snapshot.json           : active manifest snapshot\n\n"
            "This tool does not rebuild, re-embed, modify, or delete KB data.\n",
            encoding="utf-8",
        )

        zip_path = OUT_ROOT / f"DocuBot_v6.5.5.7_Chunking_Evidence_{stamp}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(work.iterdir()):
                zf.write(path, arcname=path.name)
        sha_path = Path(str(zip_path) + ".sha256.txt")
        sha_path.write_text(f"{_sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")

        print("DocuBot v6.5.5.7 CHUNKING EVIDENCE")
        print("=" * 64)
        print(f"Overall           : {'PASS' if overall else 'FAIL'}")
        print(f"BM25 chunks       : {bm25_count}")
        print(f"Qdrant count      : {qdrant.get('count') if qdrant.get('available') else 'unavailable'}")
        print(f"Files             : {len(per_file)}")
        print(f"Chunking profile  : {CHUNKING_PROFILE}")
        print(f"Chunk size/overlap: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
        print(f"Focus refs        : {', '.join(focus_refs) if focus_refs else '(none)'}")
        print(f"Evidence ZIP      : {zip_path}")
        print(f"SHA256            : {sha_path}")
        return 0 if overall else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
