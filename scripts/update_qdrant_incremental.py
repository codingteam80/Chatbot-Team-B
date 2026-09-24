from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import (
    DEFAULT_BATCH_SIZE,
    EMBED_MODEL_NAME,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
)
from embeddings.embedding_model import get_embedding_model
from ingestion.ingest import IngestionPipeline
from retrieval.bm25_index import (
    BM25Indexer,
    discard_bm25_snapshot,
    get_bm25_resources,
    restore_bm25_resources,
    snapshot_bm25_resources,
)
from retrieval.qdrant_search import _qdrant_imports
from utils.file_utils import get_all_documents
from utils.hash_utils import FileHasher
from utils.ingestion_cache import IngestionCache
from utils.manifest import ManifestManager


def _models_import():
    try:
        from qdrant_client import models
    except ImportError as error:
        raise RuntimeError(
            "qdrant-client is not installed. Run Setup_DocuBot_Production_Environment.bat first."
        ) from error
    return models


def _safe_close(client):
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _remove_tree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _activate_staging_directory(staging: Path, active: Path):
    backup = active.with_name(
        active.name + "__backup__" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    _remove_tree(backup)
    had_active = active.exists()
    if had_active:
        active.replace(backup)
    try:
        staging.replace(active)
    except Exception:
        if had_active and backup.exists() and not active.exists():
            backup.replace(active)
        raise
    return backup, had_active


def _rollback_active_directory(active: Path, backup: Path, had_active: bool):
    _remove_tree(active)
    if had_active and backup.exists():
        backup.replace(active)
    else:
        _remove_tree(backup)


def _commit_active_directory(backup: Path):
    _remove_tree(backup)


def _batches(items, size):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _record_id(record) -> str:
    metadata = record.get("metadata", {}) or {}
    identity = f"{metadata.get('file_path', '')}|{metadata.get('chunk_id', 0)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _norm_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/").casefold()


def _manifest_info_paths(info: dict[str, Any]) -> set[str]:
    paths = set()
    for field in ("indexed_file_path", "file_path"):
        value = _norm_path(info.get(field))
        if value:
            paths.add(value)
    return paths


def _scroll_all_points(client, *, with_vectors: bool = False):
    points = []
    offset = None
    while True:
        response = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        if isinstance(response, tuple):
            batch, next_offset = response
        else:
            batch = getattr(response, "points", None) or []
            next_offset = getattr(response, "next_page_offset", None)
        points.extend(list(batch or []))
        if next_offset is None:
            break
        offset = next_offset
    return points


def _point_payload(point) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    return dict(payload or {})


def _point_metadata(point) -> dict[str, Any]:
    metadata = _point_payload(point).get("metadata") or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _point_records(points) -> list[dict[str, Any]]:
    records = []
    for point in points:
        payload = _point_payload(point)
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        records.append({
            "text": str(payload.get("text") or ""),
            "metadata": metadata,
        })
    return records


def _point_ids_for_document(point_list, document_key: str, manifest_info: dict[str, Any]):
    candidate_paths = _manifest_info_paths(manifest_info)
    ids = []
    for point in point_list:
        metadata = _point_metadata(point)
        metadata_path = _norm_path(metadata.get("file_path"))
        matches = bool(metadata_path and metadata_path in candidate_paths)
        if not matches and metadata_path:
            try:
                matches = ManifestManager.document_key(metadata_path) == document_key
            except Exception:
                matches = False
        if matches:
            ids.append(getattr(point, "id"))
    return ids


def _delete_ids(client, ids):
    ids = list(ids or [])
    if not ids:
        return
    models = _models_import()
    try:
        selector = models.PointIdsList(points=ids)
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=selector,
            wait=True,
        )
    except (TypeError, AttributeError):
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=ids,
            wait=True,
        )


def _copy_qdrant_tree(active: Path, staging: Path, *, retries: int = 8):
    if not active.exists():
        raise RuntimeError("Active Qdrant directory is missing; incremental update cannot proceed safely.")
    _remove_tree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            shutil.copytree(active, staging, copy_function=shutil.copy2)
            return
        except Exception as error:
            last_error = error
            _remove_tree(staging)
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(
        "Could not create the transactional Qdrant staging copy. Close other local "
        f"DocuBot/Python processes and retry. Detail: {type(last_error).__name__}: {last_error}"
    )


def _source_state_matches(expected_manifest: dict[str, Any], previous_manifest: dict[str, Any]):
    current_documents = get_all_documents()
    current_manifest = ManifestManager.build(
        current_documents,
        previous_manifest=previous_manifest,
    )
    if set(current_manifest) != set(expected_manifest):
        return False, "Source file set changed while the KB update was running."
    for key, expected in expected_manifest.items():
        current = current_manifest.get(key, {})
        if current.get("hash") != expected.get("hash"):
            return False, f"Source content changed during update: {expected.get('file_name') or key}"
        if ManifestManager.index_signature(current) != ManifestManager.index_signature(expected):
            return False, f"Index identity changed during update: {expected.get('file_name') or key}"
    return True, ""


def _snapshot_changed_sources(changes, new_manifest):
    snapshot_root = Path(
        tempfile.mkdtemp(prefix="docubot-kb-source-snapshot-", dir=str(QDRANT_DIR.parent))
    )
    snapshot_paths = {}
    try:
        for change_type in ("added", "updated"):
            for document_key in changes.get(change_type, []):
                info = new_manifest[document_key]
                source = Path(info["file_path"])
                if not source.is_file():
                    raise RuntimeError(f"Changed source disappeared before preparation: {source}")
                target_dir = snapshot_root / uuid.uuid5(uuid.NAMESPACE_URL, document_key).hex[:16]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / source.name
                shutil.copy2(source, target)
                copied_hash = FileHasher.sha256(target)
                if copied_hash != info.get("hash"):
                    raise RuntimeError(
                        f"Source changed while its transaction snapshot was being created: {source.name}"
                    )
                snapshot_paths[document_key] = target
        return snapshot_root, snapshot_paths
    except Exception:
        _remove_tree(snapshot_root)
        raise


def _prepare_changed_documents(changes, new_manifest):
    snapshot_root, snapshot_paths = _snapshot_changed_sources(changes, new_manifest)
    prepared = {}
    cache_hits = 0
    try:
        for change_type in ("added", "updated"):
            for document_key in changes.get(change_type, []):
                info = new_manifest[document_key]
                snapshot = snapshot_paths[document_key]
                pipeline = IngestionPipeline(use_cache=True)
                records = pipeline.process_document(snapshot, file_hash=info.get("hash"))
                if pipeline.last_cache_hit:
                    cache_hits += 1

                original = Path(info["file_path"])
                for record in records:
                    metadata = dict(record.get("metadata", {}) or {})
                    metadata["file_path"] = str(original.resolve())
                    metadata["file_name"] = original.name
                    metadata["folder_name"] = original.parent.name
                    metadata["extension"] = original.suffix.lower()
                    record["metadata"] = metadata

                prepared[document_key] = {
                    "change_type": change_type,
                    "file_name": original.name,
                    "records": records,
                    "status": "ready" if records else "skipped",
                    "skip_reason": pipeline.last_skip_reason or ("no valid chunks" if not records else None),
                    "record_count": len(records),
                }
        return prepared, cache_hits, snapshot_root
    except Exception:
        _remove_tree(snapshot_root)
        raise


def _upsert_records(client, records):
    if not records:
        return 0, 0
    models = _models_import()
    embed_model = get_embedding_model()
    vector_size = None
    total = 0
    for batch in _batches(records, DEFAULT_BATCH_SIZE):
        texts = [str(record.get("text", "")) for record in batch]
        embeddings = list(embed_model.get_text_embedding_batch(texts))
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding count did not match changed chunk count.")
        if embeddings and vector_size is None:
            vector_size = len(embeddings[0])
        points = []
        for record, embedding in zip(batch, embeddings):
            if vector_size is None or len(embedding) != vector_size:
                raise RuntimeError("Embedding dimensionality changed during incremental update.")
            metadata = dict(record.get("metadata", {}) or {})
            points.append(
                models.PointStruct(
                    id=_record_id(record),
                    vector=embedding,
                    payload={"text": str(record.get("text", "")), "metadata": metadata},
                )
            )
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points, wait=True)
        total += len(points)
    return total, int(vector_size or 0)


def build_qdrant_incremental(plan: dict[str, Any]) -> dict[str, Any]:
    """Apply only new/modified/deleted files while preserving unchanged vectors.

    Expensive parsing/embedding is restricted to ``added`` and ``updated`` source
    files.  The active Qdrant directory is copied to a sibling staging directory,
    delta mutations are applied there, BM25 is regenerated from the final staging
    payload text, and Qdrant/BM25/manifest are committed as one rollback-capable
    transaction.  Unchanged documents are never re-embedded.
    """

    changes = plan.get("changes") or {}
    old_manifest = dict(plan.get("old_manifest") or {})
    new_manifest = dict(plan.get("new_manifest") or {})
    transaction = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    staging_dir = QDRANT_DIR.with_name(QDRANT_DIR.name + "__incremental__" + transaction)
    snapshot_root = None
    client = None
    bm25_snapshot = None
    qdrant_backup = None
    qdrant_had_active = False
    qdrant_activated = False
    manifest_committed = False
    result = {
        "mode": "incremental",
        "added": len(changes.get("added") or []),
        "updated": len(changes.get("updated") or []),
        "deleted": len(changes.get("deleted") or []),
        "unchanged_skipped": len(changes.get("unchanged") or []),
        "embedded_changed_chunks": 0,
        "removed_old_chunks": 0,
        "cache_hits": 0,
        "source_stability_guard": False,
    }

    try:
        has_document_changes = any(
            changes.get(name) for name in ("added", "updated", "deleted")
        )
        if not has_document_changes:
            print("[1/3] No source changes; repairing BM25 from active Qdrant text only...")
            QdrantClient = _qdrant_imports()
            read_client = QdrantClient(path=str(QDRANT_DIR))
            try:
                points = _scroll_all_points(read_client, with_vectors=False)
                final_records = _point_records(points)
            finally:
                _safe_close(read_client)
            stable, detail = _source_state_matches(new_manifest, old_manifest)
            if not stable:
                raise RuntimeError(detail)
            result["source_stability_guard"] = True
            print("[2/3] Rebuilding BM25 without touching Qdrant vectors...")
            bm25_snapshot = snapshot_bm25_resources()
            BM25Indexer().build(final_records)
            ManifestManager.save(new_manifest)
            manifest_committed = True
            try:
                get_bm25_resources.clear()
            except Exception:
                pass
            result["final_qdrant_chunks"] = len(points)
            result["ok"] = True
            print("[3/3] BM25 companion repair completed; Qdrant remained unchanged.")
            return result

        print("[1/8] Preparing only new/modified source files...")
        prepared, cache_hits, snapshot_root = _prepare_changed_documents(changes, new_manifest)
        result["cache_hits"] = cache_hits

        print("[2/8] Creating transactional copy of the active Qdrant index...")
        _copy_qdrant_tree(QDRANT_DIR, staging_dir)
        QdrantClient = _qdrant_imports()
        client = QdrantClient(path=str(staging_dir))

        try:
            if hasattr(client, "collection_exists") and not client.collection_exists(QDRANT_COLLECTION_NAME):
                raise RuntimeError("Active Qdrant collection is missing; incremental update requires a full rebuild.")
        except AttributeError:
            pass

        print("[3/8] Removing only modified/deleted document vectors from staging...")
        before_points = _scroll_all_points(client, with_vectors=False)
        remove_keys = list(changes.get("updated") or []) + list(changes.get("deleted") or [])
        remove_ids = []
        for document_key in remove_keys:
            old_info = old_manifest.get(document_key) or {}
            ids = _point_ids_for_document(before_points, document_key, old_info)
            old_status = str(old_info.get("index_status") or "indexed").casefold()
            if not ids and old_status != "skipped":
                raise RuntimeError(
                    "Could not locate the existing Qdrant chunks for changed document "
                    f"'{old_info.get('file_name') or document_key}'. Refusing to risk stale duplicate vectors."
                )
            remove_ids.extend(ids)
        _delete_ids(client, remove_ids)
        result["removed_old_chunks"] = len(remove_ids)

        print("[4/8] Embedding/upserting only new/modified chunks...")
        final_manifest = {key: dict(value) for key, value in new_manifest.items()}
        vector_size = 0
        for change_type in ("added", "updated"):
            for document_key in changes.get(change_type, []):
                item = prepared[document_key]
                info = final_manifest[document_key]
                if item["status"] == "skipped":
                    info["index_status"] = "skipped"
                    info["skip_reason"] = item["skip_reason"]
                    print(f"[SKIPPED] {item['file_name']}: {item['skip_reason']}")
                    continue
                written, current_vector_size = _upsert_records(client, item["records"])
                result["embedded_changed_chunks"] += written
                vector_size = vector_size or current_vector_size
                info["index_status"] = "indexed"
                info.pop("skip_reason", None)
                print(f"[{change_type.upper()}] {item['file_name']} ({written} chunks)")

        for document_key in changes.get("unchanged") or []:
            # ManifestManager.build already preserves unchanged status/skip reason.
            info = final_manifest.get(document_key)
            if info is not None and document_key in old_manifest:
                old_info = old_manifest[document_key]
                if old_info.get("index_status") and not info.get("index_status"):
                    info["index_status"] = old_info["index_status"]
                if old_info.get("skip_reason") and not info.get("skip_reason"):
                    info["skip_reason"] = old_info["skip_reason"]

        print("[5/8] Verifying source snapshot stability before any active commit...")
        stable, detail = _source_state_matches(new_manifest, old_manifest)
        if not stable:
            raise RuntimeError(detail)
        result["source_stability_guard"] = True

        final_points = _scroll_all_points(client, with_vectors=False)
        final_records = _point_records(final_points)
        final_count = len(final_points)
        actual_count = int(client.count(QDRANT_COLLECTION_NAME, exact=True).count)
        if actual_count != final_count:
            raise RuntimeError(
                f"Qdrant staging count mismatch: scroll={final_count}, count={actual_count}."
            )
        _safe_close(client)
        client = None

        print("[6/8] Rebuilding BM25 from final staging text (no unchanged re-embedding)...")
        bm25_snapshot = snapshot_bm25_resources()
        BM25Indexer().build(final_records)

        print("[7/8] Atomically activating Qdrant + manifest transaction...")
        qdrant_backup, qdrant_had_active = _activate_staging_directory(staging_dir, QDRANT_DIR)
        qdrant_activated = True
        ManifestManager.save(final_manifest)
        manifest_committed = True

        if qdrant_backup is not None:
            _commit_active_directory(qdrant_backup)
            qdrant_backup = None

        active_hashes = [info.get("hash") for info in final_manifest.values()]
        try:
            IngestionCache.prune(active_hashes)
        except Exception as cleanup_error:
            print(f"[WARN] Ingestion-cache cleanup skipped: {type(cleanup_error).__name__}: {cleanup_error}")
        try:
            get_bm25_resources.clear()
        except Exception:
            pass

        print("[8/8] Incremental Option-C update activated successfully.")
        print(f"Unchanged skipped       : {result['unchanged_skipped']}")
        print(f"Changed chunks embedded : {result['embedded_changed_chunks']}")
        print(f"Old chunks removed      : {result['removed_old_chunks']}")
        print(f"Final Qdrant chunks     : {final_count}")
        print(f"Embedding model         : {EMBED_MODEL_NAME}")
        result["final_qdrant_chunks"] = final_count
        result["vector_dimensions"] = vector_size
        result["ok"] = True
        return result

    except Exception as error:
        print(f"[ERROR] Incremental Option-C update failed: {type(error).__name__}: {error}")
        result["ok"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        if client is not None:
            _safe_close(client)
        _remove_tree(staging_dir)

        if qdrant_activated and qdrant_backup is not None:
            try:
                _rollback_active_directory(QDRANT_DIR, qdrant_backup, qdrant_had_active)
                qdrant_backup = None
            except Exception as rollback_error:
                result["qdrant_rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"

        if bm25_snapshot is not None:
            try:
                restore_bm25_resources(bm25_snapshot)
            except Exception as rollback_error:
                result["bm25_rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"

        if manifest_committed:
            try:
                ManifestManager.save(old_manifest)
            except Exception as rollback_error:
                result["manifest_rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"

        print("Previous active Option-C state was preserved where rollback succeeded.")
        return result
    finally:
        if bm25_snapshot is not None:
            discard_bm25_snapshot(bm25_snapshot)
        if snapshot_root is not None:
            _remove_tree(Path(snapshot_root))
