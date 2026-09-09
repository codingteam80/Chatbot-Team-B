import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb

from utils.file_utils import get_all_documents
from utils.ingestion_cache import IngestionCache
from utils.manifest import ManifestManager
from ingestion.ingest import process_document_task
from embeddings.embedding_model import get_embedding_model
from scripts.build_index import (
    build_index,
    load_records_from_collection,
    recover_active_collection_if_needed,
    prepare_records_for_storage,
    upsert_prepared_records,
)
from retrieval.bm25_index import (
    BM25Indexer,
    CORPUS_FILE,
    INDEX_FILE,
    discard_bm25_snapshot,
    get_bm25_resources,
    restore_bm25_resources,
    snapshot_bm25_resources,
)
from retrieval.chroma_search import get_chroma_collection
from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    INGESTION_MAX_WORKERS,
)


def compare_manifests(old_manifest, new_manifest):
    added = []
    updated = []
    deleted = []
    unchanged = []

    for document_key, new_info in new_manifest.items():
        old_info = old_manifest.get(document_key)
        if old_info is None:
            added.append(document_key)
            continue

        changed = (
            new_info.get("hash") != old_info.get("hash")
            or ManifestManager.index_signature(new_info)
            != ManifestManager.index_signature(old_info)
        )

        if changed:
            updated.append(document_key)
        else:
            unchanged.append(document_key)

    for document_key in old_manifest:
        if document_key not in new_manifest:
            deleted.append(document_key)

    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "unchanged": unchanged,
    }


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    recovered = recover_active_collection_if_needed(client)
    if recovered is not None:
        return recovered
    return client.get_or_create_collection(CHROMA_COLLECTION_NAME)


def bm25_is_ready():
    return CORPUS_FILE.exists() and INDEX_FILE.exists()


def _manifest_requires_full_rebuild(old_manifest):
    """Return True when the stored vectors use a different global index identity."""
    if not old_manifest:
        return False

    current_signature = ManifestManager.current_index_signature()
    return any(
        ManifestManager.index_signature(info) != current_signature
        for info in old_manifest.values()
    )


def get_update_plan():
    """Classify the next KB action without mutating active indexes.

    Modes:
      - noop: no source/index work is required
      - incremental: add/update/delete or auxiliary BM25 repair can proceed safely
      - full_rebuild: the active vector index must be regenerated as one transaction
    """
    documents = get_all_documents()
    old_manifest = ManifestManager.load()
    new_manifest = ManifestManager.build(documents, previous_manifest=old_manifest)
    changes = compare_manifests(old_manifest, new_manifest)

    collection_count = None
    collection_error = None
    try:
        collection_count = get_collection().count()
    except Exception as error:
        collection_error = str(error)

    full_rebuild_reason = None

    if _manifest_requires_full_rebuild(old_manifest):
        full_rebuild_reason = (
            "The embedding/index identity changed, so existing vectors are no longer compatible."
        )
    elif collection_error and documents:
        full_rebuild_reason = (
            "The active Chroma index could not be opened safely."
        )
    elif not old_manifest and documents and (collection_count or 0) > 0:
        full_rebuild_reason = (
            "An existing vector index has no usable tracking manifest."
        )
    elif old_manifest and documents and collection_count == 0:
        full_rebuild_reason = (
            "The active vector index is missing or empty while indexed documents are tracked."
        )

    has_document_changes = any(
        changes[name] for name in ("added", "updated", "deleted")
    )

    if full_rebuild_reason:
        mode = "full_rebuild"
    elif has_document_changes or (documents and not bm25_is_ready()):
        mode = "incremental"
    else:
        mode = "noop"

    return {
        "mode": mode,
        "reason": full_rebuild_reason,
        "documents": documents,
        "old_manifest": old_manifest,
        "new_manifest": new_manifest,
        "changes": changes,
        "collection_count": collection_count,
        "collection_error": collection_error,
    }


def check_changes():
    plan = get_update_plan()

    if plan["mode"] != "noop":
        return True

    old_manifest = plan["old_manifest"]
    new_manifest = plan["new_manifest"]

    # Enrich legacy manifests with cheap stat fingerprints and embedding identity
    # once, without parsing or embedding unchanged source documents.
    if new_manifest != old_manifest:
        try:
            ManifestManager.save(new_manifest)
        except Exception:
            # Metadata enrichment is an optimization and must not block startup.
            pass

    return False


def canonical_path(file_path):
    return os.path.normcase(str(Path(file_path).resolve()))


def append_result_to_backup(backup, result):
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")

    for index, chunk_id in enumerate(ids):
        if chunk_id in backup["seen_ids"]:
            continue

        backup["seen_ids"].add(chunk_id)
        backup["ids"].append(chunk_id)
        backup["documents"].append(documents[index])
        backup["metadatas"].append(metadatas[index])
        backup["embeddings"].append(
            embeddings[index] if embeddings is not None else None
        )


def get_existing_file_payload(collection, manifest_info):
    """Read old vectors before mutation so an update can be rolled back."""
    backup = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
        "records": [],
        "seen_ids": set(),
    }

    candidate_paths = []
    for field_name in ("indexed_file_path", "file_path"):
        candidate_path = manifest_info.get(field_name)
        if candidate_path and candidate_path not in candidate_paths:
            candidate_paths.append(candidate_path)

    for candidate_path in candidate_paths:
        result = collection.get(
            where={"file_path": candidate_path},
            include=["documents", "metadatas", "embeddings"],
        )
        append_result_to_backup(backup, result)

    if not backup["ids"]:
        file_name = manifest_info.get("file_name") or Path(
            manifest_info.get("file_path", "")
        ).name

        if file_name:
            result = collection.get(
                where={"file_name": file_name},
                include=["documents", "metadatas", "embeddings"],
            )
            target_path = canonical_path(manifest_info.get("file_path", ""))
            filtered = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }
            result_embeddings = result.get("embeddings")

            for index, metadata in enumerate(result.get("metadatas") or []):
                if canonical_path(metadata.get("file_path", "")) != target_path:
                    continue

                filtered["ids"].append(result["ids"][index])
                filtered["documents"].append(result["documents"][index])
                filtered["metadatas"].append(metadata)
                filtered["embeddings"].append(
                    result_embeddings[index] if result_embeddings is not None else None
                )

            append_result_to_backup(backup, filtered)

    backup.pop("seen_ids", None)
    backup["records"] = [
        {"text": document, "metadata": metadata}
        for document, metadata in zip(backup["documents"], backup["metadatas"])
    ]
    return backup


def delete_payload(collection, payload):
    if payload.get("ids"):
        collection.delete(ids=payload["ids"])


def restore_payload(collection, payload):
    if not payload.get("ids"):
        return

    if any(embedding is None for embedding in payload.get("embeddings", [])):
        raise RuntimeError(
            "Old Chroma payload cannot be restored because embeddings were not returned."
        )

    collection.upsert(
        ids=payload["ids"],
        documents=payload["documents"],
        metadatas=payload["metadatas"],
        embeddings=payload["embeddings"],
    )


def rebuild_bm25_from_chroma(collection):
    records = load_records_from_collection(collection)
    BM25Indexer().build(records)
    return len(records)


def _prepare_changed_documents(changes, new_manifest):
    """Parse files concurrently and batch-embed while other parses continue."""
    work = []
    for change_type in ("added", "updated"):
        for document_key in changes[change_type]:
            info = new_manifest[document_key]
            work.append((change_type, document_key, Path(info["file_path"]), info["hash"]))

    if not work:
        return {}, [], 0

    worker_count = min(max(1, INGESTION_MAX_WORKERS), len(work))
    prepared = {}
    failures = []
    cache_hits = 0
    embed_model = None

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="docubot-update",
    ) as executor:
        futures = {
            executor.submit(process_document_task, file_path, file_hash): (
                change_type,
                document_key,
                file_path,
            )
            for change_type, document_key, file_path, file_hash in work
        }

        for future in as_completed(futures):
            change_type, document_key, file_path = futures[future]
            result = future.result()
            file_name = file_path.name

            if result.get("cache_hit"):
                cache_hits += 1

            if result.get("error") is not None:
                failures.append((file_name, str(result["error"])))
                continue

            records = result.get("records") or []
            skip_reason = result.get("skip_reason")

            if not records:
                if not skip_reason:
                    failures.append((file_name, "no valid chunks generated"))
                    continue

                prepared[document_key] = {
                    "change_type": change_type,
                    "file_name": file_name,
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "payload": None,
                    "record_count": 0,
                }
                continue

            if embed_model is None:
                print("Loading embedding model once for changed documents...")
                embed_model = get_embedding_model()

            payload, failed_count = prepare_records_for_storage(records, embed_model)
            if failed_count > 0 or len(payload["records"]) != len(records):
                failures.append((file_name, "one or more chunk embeddings failed"))
                continue

            prepared[document_key] = {
                "change_type": change_type,
                "file_name": file_name,
                "status": "ready",
                "skip_reason": None,
                "payload": payload,
                "record_count": len(records),
            }

    return prepared, failures, cache_hits


def _rollback_chroma(collection, rollback_actions):
    rollback_errors = []

    for action in reversed(rollback_actions):
        kind = action[0]
        try:
            if kind == "delete_new":
                ids = action[1]
                if ids:
                    collection.delete(ids=ids)
            elif kind == "restore_old":
                restore_payload(collection, action[1])
            elif kind == "replace_new_with_old":
                new_ids, old_payload = action[1], action[2]
                if new_ids:
                    collection.delete(ids=new_ids)
                restore_payload(collection, old_payload)
        except Exception as error:
            rollback_errors.append(str(error))

    return rollback_errors


def smart_build():
    plan = get_update_plan()
    documents = plan["documents"]
    old_manifest = plan["old_manifest"]
    new_manifest = plan["new_manifest"]
    changes = plan["changes"]

    print()
    print("===== SMART KNOWLEDGE BASE UPDATE =====")
    print()
    print(f"Mode            : {plan['mode']}")
    print(f"Documents found : {len(documents)}")
    print(f"Added           : {len(changes['added'])}")
    print(f"Updated         : {len(changes['updated'])}")
    print(f"Deleted         : {len(changes['deleted'])}")
    print(f"Unchanged/skip  : {len(changes['unchanged'])}")
    print()

    if plan["mode"] == "full_rebuild":
        print(f"Full rebuild required: {plan['reason']}")
        return build_index()

    collection = get_collection()

    has_document_changes = any(
        changes[name] for name in ("added", "updated", "deleted")
    )

    if not has_document_changes and bm25_is_ready():
        if new_manifest != old_manifest:
            ManifestManager.save(new_manifest)
        pruned = IngestionCache.prune(
            [info.get("hash") for info in new_manifest.values()]
        )
        print("No document changes detected.")
        print("All unchanged files were skipped; no parsing or embedding was run.")
        if pruned:
            print(f"Removed {pruned} stale cache entries.")
        return True

    prepared, preparation_failures, cache_hits = _prepare_changed_documents(
        changes,
        new_manifest,
    )

    if preparation_failures:
        print("[ERROR] Update preparation failed. Active indexes were not changed.")
        for file_name, reason in preparation_failures:
            print(f"- {file_name}: {reason}")
        return False

    # Enrich unchanged entries with cheap stat/embedding signature metadata.
    final_manifest = dict(old_manifest)
    for document_key in changes["unchanged"]:
        final_manifest[document_key] = new_manifest[document_key]

    rollback_actions = []
    try:
        bm25_snapshot = snapshot_bm25_resources()
    except Exception as error:
        print(f"[ERROR] Could not create BM25 safety snapshot: {error}")
        print("Active indexes were not changed.")
        return False

    successful_added = []
    successful_updated = []
    successful_deleted = []
    skipped_documents = []

    try:
        # Apply deletions only after all new/updated documents are safely parsed
        # and embedded. This keeps the active KB untouched during expensive work.
        for document_key in changes["deleted"]:
            old_info = old_manifest[document_key]
            file_name = old_info.get("file_name") or Path(
                old_info.get("file_path", "")
            ).name
            old_payload = get_existing_file_payload(collection, old_info)
            delete_payload(collection, old_payload)
            rollback_actions.append(("restore_old", old_payload))
            final_manifest.pop(document_key, None)
            successful_deleted.append(file_name)
            print(f"[DELETED] {file_name} ({len(old_payload['ids'])} chunks)")

        for change_type in ("added", "updated"):
            for document_key in changes[change_type]:
                item = prepared[document_key]
                new_info = dict(new_manifest[document_key])
                file_name = item["file_name"]
                old_payload = None

                if change_type == "updated":
                    old_payload = get_existing_file_payload(
                        collection,
                        old_manifest[document_key],
                    )
                    delete_payload(collection, old_payload)

                if item["status"] == "skipped":
                    if change_type == "updated":
                        rollback_actions.append(("restore_old", old_payload))
                    new_info["index_status"] = "skipped"
                    new_info["skip_reason"] = item["skip_reason"]
                    final_manifest[document_key] = new_info
                    skipped_documents.append((file_name, item["skip_reason"]))
                    print(f"[SKIPPED] {file_name}: {item['skip_reason']}")
                    continue

                payload = item["payload"]
                try:
                    upsert_prepared_records(collection, payload)
                except Exception:
                    if change_type == "updated":
                        restore_payload(collection, old_payload)
                    raise

                if change_type == "updated":
                    rollback_actions.append(
                        ("replace_new_with_old", payload["ids"], old_payload)
                    )
                    successful_updated.append(file_name)
                    print(f"[UPDATED] {file_name} ({item['record_count']} chunks)")
                else:
                    rollback_actions.append(("delete_new", payload["ids"]))
                    successful_added.append(file_name)
                    print(f"[ADDED] {file_name} ({item['record_count']} chunks)")

                final_manifest[document_key] = new_info

        print()
        print("Rebuilding BM25 from current Chroma text only...")
        bm25_chunk_count = rebuild_bm25_from_chroma(collection)

        ManifestManager.save(final_manifest)

    except Exception as error:
        print(f"[ERROR] Incremental update commit failed: {error}")
        rollback_errors = _rollback_chroma(collection, rollback_actions)
        restore_bm25_resources(bm25_snapshot)
        try:
            ManifestManager.save(old_manifest)
        except Exception:
            pass

        if rollback_errors:
            print("[ERROR] One or more rollback operations also failed:")
            for reason in rollback_errors:
                print(f"- {reason}")
        else:
            print("Previous Chroma/BM25/manifest state was restored.")

        return False
    finally:
        discard_bm25_snapshot(bm25_snapshot)

    active_hashes = [info.get("hash") for info in final_manifest.values()]
    pruned_cache_entries = IngestionCache.prune(active_hashes)

    get_chroma_collection.clear()
    get_bm25_resources.clear()

    print()
    print("===== INCREMENTAL UPDATE SUMMARY =====")
    print(f"Added successfully   : {len(successful_added)}")
    print(f"Updated successfully : {len(successful_updated)}")
    print(f"Deleted successfully : {len(successful_deleted)}")
    print(f"Unchanged skipped    : {len(changes['unchanged'])}")
    print(f"Validation skipped   : {len(skipped_documents)}")
    print(f"Chunk cache hits     : {cache_hits}")
    print(f"BM25 chunks          : {bm25_chunk_count}")
    print(f"Cache entries pruned : {pruned_cache_entries}")

    if skipped_documents:
        print()
        print("Validation-skipped documents:")
        for file_name, reason in skipped_documents:
            print(f"- {file_name}: {reason}")

    print("======================================")
    print()
    return True


if __name__ == "__main__":
    smart_build()
