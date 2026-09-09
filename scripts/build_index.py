import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import chromadb

from ingestion.ingest import IngestionPipeline
from retrieval.bm25_index import (
    BM25Indexer,
    discard_bm25_snapshot,
    restore_bm25_resources,
    snapshot_bm25_resources,
)
from embeddings.embedding_model import get_embedding_model
from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHROMA_WRITE_BATCH_SIZE,
    DEFAULT_BATCH_SIZE,
)
from utils.file_utils import get_all_documents
from utils.ingestion_cache import IngestionCache
from utils.manifest import ManifestManager
from utils.logger import separator, summary


def build_record_id(record):
    """Create a deterministic chunk ID so retries cannot duplicate chunks."""
    metadata = record.get("metadata", {})
    identity = f"{metadata.get('file_path', '')}|{metadata.get('chunk_id', 0)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _embedding_batches(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield start, records[start:start + batch_size]


def prepare_records_for_storage(records, embed_model):
    """Batch-embed a complete document before any active-index mutation."""
    payload = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
        "records": [],
    }

    if not records:
        return payload, 0

    failed_embeddings = 0
    batch_size = max(1, int(DEFAULT_BATCH_SIZE))
    batch_method = getattr(embed_model, "get_text_embedding_batch", None)

    for start, batch in _embedding_batches(records, batch_size):
        texts = [f"passage: {record.get('text', '')}" for record in batch]

        try:
            if callable(batch_method):
                embeddings = list(batch_method(texts))
            else:
                embeddings = [
                    embed_model.get_text_embedding(text)
                    for text in texts
                ]
        except Exception as error:
            print(
                f"[ERROR] Embedding batch failed at chunks "
                f"{start}-{start + len(batch) - 1}: {error}"
            )
            failed_embeddings += len(batch)
            continue

        if len(embeddings) != len(batch):
            print(
                f"[ERROR] Embedding batch returned {len(embeddings)} vectors "
                f"for {len(batch)} chunks."
            )
            failed_embeddings += len(batch)
            continue

        for record, embedding in zip(batch, embeddings):
            text = record.get("text", "")
            metadata = dict(record.get("metadata", {}))
            stored_record = {"text": text, "metadata": metadata}

            payload["ids"].append(build_record_id(stored_record))
            payload["documents"].append(text)
            payload["metadatas"].append(metadata)
            payload["embeddings"].append(embedding)
            payload["records"].append(stored_record)

    return payload, failed_embeddings


def upsert_prepared_records(collection, payload):
    """Write prepared vectors to Chroma in bulk batches."""
    total_records = len(payload["ids"])
    for start in range(0, total_records, CHROMA_WRITE_BATCH_SIZE):
        end = start + CHROMA_WRITE_BATCH_SIZE
        collection.upsert(
            ids=payload["ids"][start:end],
            documents=payload["documents"][start:end],
            embeddings=payload["embeddings"][start:end],
            metadatas=payload["metadatas"][start:end],
        )


def load_records_from_collection(collection):
    """Read current Chroma text/metadata without parsing or embedding."""
    if collection.count() == 0:
        return []

    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        {"text": document, "metadata": metadata}
        for document, metadata in zip(documents, metadatas)
    ]


def _get_collection_if_exists(client, name):
    try:
        return client.get_collection(name)
    except Exception:
        return None


def recover_active_collection_if_needed(client):
    """Recover the previous active collection after an interrupted rename swap."""
    active = _get_collection_if_exists(client, CHROMA_COLLECTION_NAME)
    if active is not None:
        return active

    try:
        listed = client.list_collections()
    except Exception:
        return None

    backup_prefix = f"{CHROMA_COLLECTION_NAME}__backup__"
    backup_names = []
    for item in listed:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, str):
            name = item
        if name and str(name).startswith(backup_prefix):
            backup_names.append(str(name))

    if not backup_names:
        return None

    # Transaction names start with a sortable timestamp in v6.2.
    backup_name = sorted(backup_names)[-1]
    backup = _get_collection_if_exists(client, backup_name)
    if backup is None:
        return None

    backup.modify(name=CHROMA_COLLECTION_NAME)
    print(
        "[RECOVERY] Restored the previous Chroma collection after an "
        "interrupted full-rebuild swap."
    )
    return backup


def _delete_collection_if_exists(client, name):
    try:
        client.delete_collection(name)
    except Exception:
        pass


def _swap_staging_collection(client, staging_name, active_name, backup_name):
    """Rename collections so the prior active vector index remains recoverable."""
    active = _get_collection_if_exists(client, active_name)
    staging = client.get_collection(staging_name)
    had_active = active is not None

    if had_active:
        active.modify(name=backup_name)

    try:
        staging.modify(name=active_name)
    except Exception:
        if had_active:
            backup = _get_collection_if_exists(client, backup_name)
            if backup is not None:
                backup.modify(name=active_name)
        raise

    return had_active


def _rollback_collection_swap(client, active_name, backup_name, failed_name, had_active):
    current = _get_collection_if_exists(client, active_name)
    if current is not None:
        try:
            current.modify(name=failed_name)
        except Exception:
            _delete_collection_if_exists(client, active_name)

    if had_active:
        backup = _get_collection_if_exists(client, backup_name)
        if backup is not None:
            backup.modify(name=active_name)

    _delete_collection_if_exists(client, failed_name)


def build_index():
    """Perform a cache-aware, batched and rollback-capable full rebuild."""
    separator("FULL REBUILD STARTED")
    documents = get_all_documents()

    print(f"[STEP 1] Discovering {len(documents)} documents...")
    if not documents:
        print("[ERROR] No source documents found. Existing indexes were preserved.")
        return False

    old_manifest = ManifestManager.load()
    candidate_manifest = ManifestManager.build(
        documents,
        previous_manifest=old_manifest,
    )
    document_hashes = {
        info["file_path"]: info["hash"]
        for info in candidate_manifest.values()
    }

    print("[STEP 2] Parsing/chunking with cache + controlled parallelism...")
    pipeline = IngestionPipeline(use_cache=True)
    records = pipeline.run(documents, document_hashes=document_hashes)

    if pipeline.failed_documents:
        print("[ERROR] One or more documents failed extraction. Existing indexes were preserved.")
        for path, reason in pipeline.failed_documents.items():
            print(f"- {Path(path).name}: {reason}")
        return False

    if not records:
        print("[ERROR] No valid chunks generated. Existing indexes were preserved.")
        return False

    print("[STEP 3] Loading embedding model once...")
    embed_model = get_embedding_model()

    grouped_records = {}
    for record in records:
        file_path = record.get("metadata", {}).get("file_path", "")
        grouped_records.setdefault(file_path, []).append(record)

    prepared_payloads = []
    indexed_documents = []

    print("[STEP 4] Batch-embedding documents...")
    for file_path, file_records in grouped_records.items():
        payload, failed = prepare_records_for_storage(file_records, embed_model)
        if failed > 0 or len(payload["records"]) != len(file_records):
            print(
                f"[ERROR] {Path(file_path).name}: embedding incomplete. "
                "Existing indexes were preserved."
            )
            return False
        prepared_payloads.append(payload)
        indexed_documents.append(Path(file_path))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    recover_active_collection_if_needed(client)
    transaction_id = (
        datetime.now().strftime("%Y%m%d%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    staging_name = f"{CHROMA_COLLECTION_NAME}__staging__{transaction_id}"
    backup_name = f"{CHROMA_COLLECTION_NAME}__backup__{transaction_id}"
    failed_name = f"{CHROMA_COLLECTION_NAME}__failed__{transaction_id}"
    _delete_collection_if_exists(client, staging_name)

    print("[STEP 5] Writing a staging Chroma collection...")
    staging = client.create_collection(staging_name)
    stored_records = []

    try:
        for payload in prepared_payloads:
            upsert_prepared_records(staging, payload)
            stored_records.extend(payload["records"])
    except Exception as error:
        _delete_collection_if_exists(client, staging_name)
        print(f"[ERROR] Staging Chroma write failed: {error}")
        print("Existing indexes were preserved.")
        return False

    manifest_documents = list(indexed_documents)
    manifest_documents.extend(Path(path) for path in pipeline.skipped_documents)
    rebuild_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest = {}
    for path in manifest_documents:
        key = ManifestManager.document_key(path)
        candidate = candidate_manifest.get(key)
        if candidate is None:
            continue
        manifest[key] = dict(candidate)
        manifest[key]["indexed_file_path"] = manifest[key]["file_path"]
        manifest[key]["last_indexed"] = rebuild_time

    for path, reason in pipeline.skipped_documents.items():
        key = ManifestManager.document_key(Path(path))
        if key in manifest:
            manifest[key]["index_status"] = "skipped"
            manifest[key]["skip_reason"] = reason

    try:
        bm25_snapshot = snapshot_bm25_resources()
    except Exception as error:
        _delete_collection_if_exists(client, staging_name)
        print(f"[ERROR] Could not create BM25 safety snapshot: {error}")
        print("Existing indexes were preserved.")
        return False

    had_active = False

    try:
        print("[STEP 6] Atomically switching the active Chroma collection...")
        had_active = _swap_staging_collection(
            client,
            staging_name,
            CHROMA_COLLECTION_NAME,
            backup_name,
        )

        print("[STEP 7] Building BM25 atomically...")
        BM25Indexer().build(stored_records)

        print("[STEP 8] Saving manifest atomically...")
        ManifestManager.save(manifest)

    except Exception as error:
        print(f"[ERROR] Full rebuild commit failed: {error}")
        try:
            _rollback_collection_swap(
                client,
                CHROMA_COLLECTION_NAME,
                backup_name,
                failed_name,
                had_active,
            )
        finally:
            restore_bm25_resources(bm25_snapshot)
            try:
                ManifestManager.save(old_manifest)
            except Exception:
                pass
        print("Previous active indexes were restored.")
        return False
    finally:
        discard_bm25_snapshot(bm25_snapshot)

    _delete_collection_if_exists(client, backup_name)
    _delete_collection_if_exists(client, failed_name)

    active_hashes = [info.get("hash") for info in manifest.values()]
    pruned_cache_entries = IngestionCache.prune(active_hashes)

    from retrieval.chroma_search import get_chroma_collection
    from retrieval.bm25_index import get_bm25_resources

    get_chroma_collection.clear()
    get_bm25_resources.clear()

    print()
    print("===== FULL REBUILD COMPLETE =====")
    print(f"Documents discovered : {len(documents)}")
    print(f"Documents indexed    : {len(indexed_documents)}")
    print(f"Validation skipped   : {len(pipeline.skipped_documents)}")
    print(f"Total chunks         : {len(stored_records)}")
    print(f"Cache entries pruned : {pruned_cache_entries}")
    print("=================================")
    print()

    summary(
        Documents=len(indexed_documents),
        Chunks=len(stored_records),
        Embeddings=len(stored_records),
    )
    return True


if __name__ == "__main__":
    build_index()
