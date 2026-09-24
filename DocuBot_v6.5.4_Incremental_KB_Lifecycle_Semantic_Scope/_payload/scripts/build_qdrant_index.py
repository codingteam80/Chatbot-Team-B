from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

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
from utils.ingestion_cache import IngestionCache
from utils.manifest import ManifestManager
from utils.logger import separator


def _models_import():
    try:
        from qdrant_client import models
    except ImportError as error:
        raise RuntimeError(
            "qdrant-client is not installed. Run Setup_DocuBot_Production_Environment.bat first."
        ) from error
    return models


def _record_id(record) -> str:
    metadata = record.get("metadata", {}) or {}
    identity = f"{metadata.get('file_path', '')}|{metadata.get('chunk_id', 0)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _batches(items, size):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _safe_close(client):
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _remove_tree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _activate_staging_directory(staging: Path, active: Path):
    """Activate staging while retaining a rollback directory until commit."""
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
    """Restore the exact pre-build Qdrant directory after a commit-stage failure."""
    _remove_tree(active)
    if had_active and backup.exists():
        backup.replace(active)
    else:
        _remove_tree(backup)


def _commit_active_directory(backup: Path):
    """Discard the rollback directory after all companion metadata commits."""
    _remove_tree(backup)




def _source_state_matches(expected_manifest, previous_manifest):
    current_documents = get_all_documents()
    current_manifest = ManifestManager.build(
        current_documents,
        previous_manifest=previous_manifest,
    )
    if set(current_manifest) != set(expected_manifest):
        return False, "Source file set changed while the full rebuild was running."
    for key, expected in expected_manifest.items():
        current = current_manifest.get(key, {})
        if current.get("hash") != expected.get("hash"):
            return False, f"Source content changed during rebuild: {expected.get('file_name') or key}"
        if ManifestManager.index_signature(current) != ManifestManager.index_signature(expected):
            return False, f"Index identity changed during rebuild: {expected.get('file_name') or key}"
    return True, ""

def build_qdrant_index():
    """Full transactional Option-C rebuild for the technical corpus.

    Qdrant is built in a sibling staging directory and becomes active only
    after every source chunk has a valid Qwen3 embedding and BM25 has also
    built successfully.
    """

    separator("OPTION C QDRANT FULL REBUILD STARTED")
    documents = get_all_documents()
    print(f"[1/7] Technical documents discovered: {len(documents)}")
    if not documents:
        print("[ERROR] No technical source documents found. Active indexes were preserved.")
        return False

    old_manifest = ManifestManager.load()
    candidate_manifest = ManifestManager.build(documents, previous_manifest=old_manifest)
    document_hashes = {
        info["file_path"]: info["hash"] for info in candidate_manifest.values()
    }

    print("[2/7] Parsing/chunking source files...")
    pipeline = IngestionPipeline(use_cache=True)
    records = pipeline.run(documents, document_hashes=document_hashes)
    if pipeline.failed_documents:
        print("[ERROR] Source extraction failed; active indexes were preserved.")
        for path, reason in pipeline.failed_documents.items():
            print(f"- {Path(path).name}: {reason}")
        return False
    if not records:
        print("[ERROR] No valid chunks generated; active indexes were preserved.")
        return False

    print(f"[3/7] Loading embedding adapter: {EMBED_MODEL_NAME}")
    embed_model = get_embedding_model()

    transaction = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    staging_dir = QDRANT_DIR.with_name(QDRANT_DIR.name + "__staging__" + transaction)
    _remove_tree(staging_dir)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)

    client = None
    bm25_snapshot = None
    qdrant_backup = None
    qdrant_had_active = False
    qdrant_activated = False
    manifest_committed = False
    try:
        QdrantClient = _qdrant_imports()
        models = _models_import()
        print("[4/7] Embedding with Qwen3 and writing staging Qdrant index...")
        client = QdrantClient(path=str(staging_dir))
        collection_created = False
        vector_size = None
        indexed_records = []

        for batch in _batches(records, DEFAULT_BATCH_SIZE):
            # Qwen3 Embedding recommends an instruction for queries only;
            # retrieval documents/passages are embedded as their source text.
            texts = [str(record.get("text", "")) for record in batch]
            embeddings = list(embed_model.get_text_embedding_batch(texts))
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding count did not match chunk count.")

            if not collection_created:
                vector_size = len(embeddings[0]) if embeddings else 0
                if vector_size <= 0:
                    raise RuntimeError("Qwen3 embedding returned an empty vector.")
                client.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                collection_created = True

            points = []
            for record, embedding in zip(batch, embeddings):
                if len(embedding) != vector_size:
                    raise RuntimeError("Embedding dimensionality changed during one rebuild.")
                metadata = dict(record.get("metadata", {}) or {})
                points.append(
                    models.PointStruct(
                        id=_record_id(record),
                        vector=embedding,
                        payload={
                            "text": str(record.get("text", "")),
                            "metadata": metadata,
                        },
                    )
                )
                indexed_records.append({
                    "text": str(record.get("text", "")),
                    "metadata": metadata,
                })

            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points,
                wait=True,
            )

        count = int(client.count(QDRANT_COLLECTION_NAME, exact=True).count)
        if count != len(records):
            raise RuntimeError(
                f"Qdrant count mismatch: expected {len(records)}, found {count}."
            )
        _safe_close(client)
        client = None

        # Fail closed before mutating BM25 or activating Qdrant if an operator
        # added/removed/modified source files while the long embedding pass ran.
        stable, detail = _source_state_matches(candidate_manifest, old_manifest)
        if not stable:
            raise RuntimeError(detail)

        print("[5/7] Rebuilding BM25 from the same technical chunks...")
        bm25_snapshot = snapshot_bm25_resources()
        BM25Indexer().build(indexed_records)

        print("[6/7] Atomically activating the new Qdrant directory...")
        qdrant_backup, qdrant_had_active = _activate_staging_directory(
            staging_dir,
            QDRANT_DIR,
        )
        qdrant_activated = True

        indexed_paths = {
            str(Path(record.get("metadata", {}).get("file_path", "")).resolve())
            for record in indexed_records
            if record.get("metadata", {}).get("file_path")
        }
        # Reuse the exact pre-embedding source snapshot instead of touching the
        # live source tree again after activation. The stability guard above
        # guarantees this candidate still describes the current source set.
        final_manifest = {key: dict(info) for key, info in candidate_manifest.items()}
        for key, info in final_manifest.items():
            resolved = str(Path(info.get("file_path", "")).resolve())
            if resolved in indexed_paths:
                info["index_status"] = "indexed"
            else:
                reason = pipeline.skipped_documents.get(resolved) or "no valid chunks"
                info["index_status"] = "skipped"
                info["skip_reason"] = reason
        ManifestManager.save(final_manifest)
        manifest_committed = True

        # Only now is the three-part transaction (Qdrant + BM25 + manifest)
        # committed.  Keep cleanup non-fatal so a healthy index is never rolled
        # back merely because an old cache/backup could not be deleted.
        if qdrant_backup is not None:
            _commit_active_directory(qdrant_backup)
            qdrant_backup = None

        active_hashes = [info.get("hash") for info in final_manifest.values()]
        try:
            IngestionCache.prune(active_hashes)
        except Exception as cleanup_error:
            print(
                "[WARN] Non-critical ingestion-cache cleanup was skipped: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        for cache_name, clear_cache in (
            ("BM25 cache", get_bm25_resources.clear),
            ("embedding cache", get_embedding_model.clear),
        ):
            try:
                clear_cache()
            except Exception as cleanup_error:
                print(
                    f"[WARN] Non-critical {cache_name} clear was skipped: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

        print("[7/7] Option-C index activated successfully.")
        print(f"Technical documents : {len(documents)}")
        print(f"Qdrant chunks       : {count}")
        print(f"Vector dimensions   : {vector_size}")
        print(f"Embedding model     : {EMBED_MODEL_NAME}")
        return True

    except Exception as error:
        print(f"[ERROR] Option-C rebuild failed: {type(error).__name__}: {error}")
        if client is not None:
            _safe_close(client)
        _remove_tree(staging_dir)

        if qdrant_activated and qdrant_backup is not None:
            try:
                _rollback_active_directory(
                    QDRANT_DIR,
                    qdrant_backup,
                    qdrant_had_active,
                )
                qdrant_backup = None
            except Exception as rollback_error:
                print(
                    "[ERROR] Qdrant rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )

        if bm25_snapshot is not None:
            try:
                restore_bm25_resources(bm25_snapshot)
            except Exception as rollback_error:
                print(
                    "[ERROR] BM25 rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )

        if manifest_committed:
            try:
                ManifestManager.save(old_manifest)
            except Exception as rollback_error:
                print(
                    "[ERROR] Manifest rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )

        print("Previous active Option-C state was preserved where rollback succeeded.")
        return False
    finally:
        if bm25_snapshot is not None:
            discard_bm25_snapshot(bm25_snapshot)


if __name__ == "__main__":
    raise SystemExit(0 if build_qdrant_index() else 1)
