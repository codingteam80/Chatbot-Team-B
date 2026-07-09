import os

from pathlib import Path

import chromadb

from utils.file_utils import (
    get_all_documents
)

from utils.manifest import (
    ManifestManager
)

from ingestion.ingest import (
    IngestionPipeline
)

from embeddings.embedding_model import (
    get_embedding_model
)

from scripts.build_index import (
    build_index,
    prepare_records_for_storage,
    upsert_prepared_records,
    load_records_from_collection
)

from retrieval.bm25_index import (
    BM25Indexer,
    CORPUS_FILE,
    INDEX_FILE,
    get_bm25_resources
)

from retrieval.chroma_search import (
    get_chroma_collection
)

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR
)


def compare_manifests(
    old_manifest,
    new_manifest
):

    added = []
    updated = []
    deleted = []
    unchanged = []

    for document_key, new_info in (
        new_manifest.items()
    ):

        if document_key not in old_manifest:

            added.append(
                document_key
            )

        elif (
            new_info.get("hash")
            != old_manifest[
                document_key
            ].get("hash")
        ):

            updated.append(
                document_key
            )

        else:

            unchanged.append(
                document_key
            )

    for document_key in old_manifest:

        if document_key not in new_manifest:

            deleted.append(
                document_key
            )

    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "unchanged": unchanged
    }


def get_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_or_create_collection(
        CHROMA_COLLECTION_NAME
    )


def bm25_is_ready():

    return (
        CORPUS_FILE.exists()
        and INDEX_FILE.exists()
    )


def check_changes():

    documents = get_all_documents()

    old_manifest = (
        ManifestManager.load()
    )

    new_manifest = (
        ManifestManager.build(
            documents,
            previous_manifest=old_manifest
        )
    )

    changes = compare_manifests(
        old_manifest,
        new_manifest
    )

    has_document_changes = any(
        changes[name]
        for name in (
            "added",
            "updated",
            "deleted"
        )
    )

    if has_document_changes:

        return True

    if not documents:

        return False

    try:

        collection = get_collection()

        if collection.count() == 0:

            return True

    except Exception:

        return True

    return not bm25_is_ready()


def canonical_path(
    file_path
):

    return os.path.normcase(
        str(
            Path(file_path).resolve()
        )
    )


def append_result_to_backup(
    backup,
    result
):

    ids = (
        result.get("ids")
        or []
    )

    documents = (
        result.get("documents")
        or []
    )

    metadatas = (
        result.get("metadatas")
        or []
    )

    embeddings = result.get(
        "embeddings"
    )

    for index, chunk_id in enumerate(
        ids
    ):

        if chunk_id in backup["seen_ids"]:

            continue

        backup["seen_ids"].add(
            chunk_id
        )

        backup["ids"].append(
            chunk_id
        )

        backup["documents"].append(
            documents[index]
        )

        backup["metadatas"].append(
            metadatas[index]
        )

        backup["embeddings"].append(
            embeddings[index]
            if embeddings is not None
            else None
        )


def get_existing_file_payload(
    collection,
    manifest_info
):

    """
    Retrieve the existing Chroma chunks before updating or deleting
    a document. The payload can be restored if an update write fails.
    """

    backup = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
        "records": [],
        "seen_ids": set()
    }

    candidate_paths = []

    for field_name in (
        "indexed_file_path",
        "file_path"
    ):

        candidate_path = (
            manifest_info.get(
                field_name
            )
        )

        if (
            candidate_path
            and candidate_path
            not in candidate_paths
        ):

            candidate_paths.append(
                candidate_path
            )

    for candidate_path in candidate_paths:

        result = collection.get(
            where={
                "file_path":
                    candidate_path
            },
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        append_result_to_backup(
            backup,
            result
        )

    # Migration fallback for older metadata whose relative path
    # differs from the canonical path stored in the new manifest.
    if not backup["ids"]:

        file_name = (
            manifest_info.get(
                "file_name"
            )
            or Path(
                manifest_info.get(
                    "file_path",
                    ""
                )
            ).name
        )

        if file_name:

            result = collection.get(
                where={
                    "file_name":
                        file_name
                },
                include=[
                    "documents",
                    "metadatas",
                    "embeddings"
                ]
            )

            target_path = canonical_path(
                manifest_info.get(
                    "file_path",
                    ""
                )
            )

            filtered = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": []
            }

            result_embeddings = result.get(
                "embeddings"
            )

            for index, metadata in enumerate(
                result.get(
                    "metadatas"
                )
                or []
            ):

                metadata_path = metadata.get(
                    "file_path",
                    ""
                )

                if (
                    canonical_path(
                        metadata_path
                    )
                    != target_path
                ):

                    continue

                filtered["ids"].append(
                    result["ids"][
                        index
                    ]
                )

                filtered["documents"].append(
                    result["documents"][
                        index
                    ]
                )

                filtered["metadatas"].append(
                    metadata
                )

                filtered["embeddings"].append(
                    result_embeddings[index]
                    if result_embeddings
                    is not None
                    else None
                )

            append_result_to_backup(
                backup,
                filtered
            )

    backup.pop(
        "seen_ids",
        None
    )

    backup["records"] = [
        {
            "text": document,
            "metadata": metadata
        }
        for document, metadata in zip(
            backup["documents"],
            backup["metadatas"]
        )
    ]

    return backup


def delete_payload(
    collection,
    payload
):

    if payload["ids"]:

        collection.delete(
            ids=payload["ids"]
        )


def restore_payload(
    collection,
    payload
):

    if not payload["ids"]:

        return

    if any(
        embedding is None
        for embedding in payload[
            "embeddings"
        ]
    ):

        raise RuntimeError(
            "Old Chroma payload cannot be "
            "restored because embeddings "
            "were not returned."
        )

    collection.upsert(
        ids=payload["ids"],
        documents=payload[
            "documents"
        ],
        metadatas=payload[
            "metadatas"
        ],
        embeddings=payload[
            "embeddings"
        ]
    )


def rebuild_bm25_from_chroma(
    collection
):

    records = load_records_from_collection(
        collection
    )

    BM25Indexer().build(
        records
    )

    return len(
        records
    )


def smart_build():

    print()
    print(
        "===== TRUE INCREMENTAL BUILD ====="
    )
    print()

    documents = get_all_documents()

    old_manifest = (
        ManifestManager.load()
    )

    new_manifest = (
        ManifestManager.build(
            documents,
            previous_manifest=old_manifest
        )
    )

    changes = compare_manifests(
        old_manifest,
        new_manifest
    )

    print(
        f"Documents found : "
        f"{len(documents)}"
    )
    print(
        f"Added           : "
        f"{len(changes['added'])}"
    )
    print(
        f"Updated         : "
        f"{len(changes['updated'])}"
    )
    print(
        f"Deleted         : "
        f"{len(changes['deleted'])}"
    )
    print(
        f"Unchanged/skip  : "
        f"{len(changes['unchanged'])}"
    )
    print()

    collection = get_collection()

    # One-time safety migration:
    # an existing Chroma index without a usable manifest cannot
    # reliably distinguish unchanged files.
    if (
        not old_manifest
        and collection.count() > 0
    ):

        print(
            "Existing index has no manifest."
        )
        print(
            "Running one safe full rebuild "
            "to establish incremental tracking..."
        )

        return build_index()

    # Restore a missing vector index from source documents.
    if (
        old_manifest
        and documents
        and collection.count() == 0
    ):

        print(
            "Chroma index is missing or empty."
        )
        print(
            "Running a safe full rebuild..."
        )

        return build_index()

    has_document_changes = any(
        changes[name]
        for name in (
            "added",
            "updated",
            "deleted"
        )
    )

    if (
        not has_document_changes
        and bm25_is_ready()
    ):

        print(
            "No changes detected."
        )
        print(
            "Existing documents were skipped."
        )

        return True

    pipeline = IngestionPipeline()

    embed_model = None

    if (
        changes["added"]
        or changes["updated"]
    ):

        print(
            "Loading embedding model..."
        )

        embed_model = (
            get_embedding_model()
        )

    final_manifest = dict(
        old_manifest
    )

    successful_added = []
    successful_updated = []
    successful_deleted = []
    failed_documents = []

    index_changed = False

    # ------------------------------------------------------
    # DELETE
    # ------------------------------------------------------

    for document_key in changes[
        "deleted"
    ]:

        old_info = old_manifest[
            document_key
        ]

        file_name = (
            old_info.get("file_name")
            or Path(
                old_info.get(
                    "file_path",
                    ""
                )
            ).name
        )

        try:

            old_payload = (
                get_existing_file_payload(
                    collection,
                    old_info
                )
            )

            delete_payload(
                collection,
                old_payload
            )

            final_manifest.pop(
                document_key,
                None
            )

            successful_deleted.append(
                file_name
            )

            index_changed = True

            print(
                f"[DELETED] {file_name} "
                f"({len(old_payload['ids'])} chunks)"
            )

        except Exception as error:

            failed_documents.append(
                (
                    file_name,
                    f"delete failed: {error}"
                )
            )

            print(
                f"[FAILED DELETE] "
                f"{file_name}: {error}"
            )

    # ------------------------------------------------------
    # ADD / UPDATE
    # ------------------------------------------------------

    for change_type in (
        "added",
        "updated"
    ):

        for document_key in changes[
            change_type
        ]:

            new_info = new_manifest[
                document_key
            ]

            file_path = Path(
                new_info["file_path"]
            )

            file_name = file_path.name

            try:

                records = (
                    pipeline.process_document(
                        file_path
                    )
                )

                if not records:

                    raise RuntimeError(
                        "no valid chunks generated"
                    )

                payload, failed_count = (
                    prepare_records_for_storage(
                        records,
                        embed_model
                    )
                )

                if (
                    failed_count > 0
                    or len(payload["records"])
                    != len(records)
                ):

                    raise RuntimeError(
                        "one or more chunk "
                        "embeddings failed"
                    )

                old_payload = {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "embeddings": [],
                    "records": []
                }

                if change_type == "updated":

                    old_payload = (
                        get_existing_file_payload(
                            collection,
                            old_manifest[
                                document_key
                            ]
                        )
                    )

                    delete_payload(
                        collection,
                        old_payload
                    )

                try:

                    upsert_prepared_records(
                        collection,
                        payload
                    )

                except Exception:

                    if (
                        change_type
                        == "updated"
                    ):

                        restore_payload(
                            collection,
                            old_payload
                        )

                    raise

                final_manifest[
                    document_key
                ] = new_info

                index_changed = True

                if change_type == "added":

                    successful_added.append(
                        file_name
                    )

                    print(
                        f"[ADDED] {file_name} "
                        f"({len(records)} chunks)"
                    )

                else:

                    successful_updated.append(
                        file_name
                    )

                    print(
                        f"[UPDATED] {file_name} "
                        f"({len(records)} chunks)"
                    )

            except Exception as error:

                failed_documents.append(
                    (
                        file_name,
                        str(error)
                    )
                )

                print(
                    f"[FAILED {change_type.upper()}] "
                    f"{file_name}: {error}"
                )

    # Rebuild BM25 from already stored Chroma text.
    # No unchanged document is parsed or embedded here.
    bm25_chunk_count = None

    if (
        index_changed
        or not bm25_is_ready()
    ):

        print()
        print(
            "Rebuilding BM25 from current "
            "Chroma records..."
        )

        bm25_chunk_count = (
            rebuild_bm25_from_chroma(
                collection
            )
        )

    if (
        index_changed
        or final_manifest
        != old_manifest
    ):

        ManifestManager.save(
            final_manifest
        )

    get_chroma_collection.clear()
    get_bm25_resources.clear()

    print()
    print(
        "===== INCREMENTAL BUILD SUMMARY ====="
    )
    print(
        f"Added successfully   : "
        f"{len(successful_added)}"
    )
    print(
        f"Updated successfully : "
        f"{len(successful_updated)}"
    )
    print(
        f"Deleted successfully : "
        f"{len(successful_deleted)}"
    )
    print(
        f"Unchanged skipped    : "
        f"{len(changes['unchanged'])}"
    )
    print(
        f"Failed documents     : "
        f"{len(failed_documents)}"
    )

    if bm25_chunk_count is not None:

        print(
            f"BM25 chunks          : "
            f"{bm25_chunk_count}"
        )

    if failed_documents:

        print()
        print(
            "Failed document details:"
        )

        for file_name, reason in (
            failed_documents
        ):

            print(
                f"- {file_name}: {reason}"
            )

    print(
        "====================================="
    )
    print()

    # Return False when one or more requested changes failed.
    return not failed_documents


if __name__ == "__main__":

    smart_build()
