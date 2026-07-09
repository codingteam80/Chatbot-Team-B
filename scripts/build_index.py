import hashlib
from pathlib import Path

import chromadb

from ingestion.ingest import (
    IngestionPipeline
)

from retrieval.bm25_index import (
    BM25Indexer
)

from embeddings.embedding_model import (
    get_embedding_model
)

from config.settings import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME
)

from utils.file_utils import (
    get_all_documents
)

from utils.manifest import (
    ManifestManager
)

from utils.logger import (
    separator,
    summary
)


CHROMA_WRITE_BATCH_SIZE = 64


def build_record_id(
    record
):

    """
    Create a deterministic chunk ID.

    Repeating the same operation cannot create duplicate chunks.
    """

    metadata = record.get(
        "metadata",
        {}
    )

    identity = (
        f"{metadata.get('file_path', '')}|"
        f"{metadata.get('chunk_id', 0)}"
    )

    return hashlib.sha256(
        identity.encode(
            "utf-8"
        )
    ).hexdigest()


def prepare_records_for_storage(
    records,
    embed_model
):

    """
    Generate embeddings without writing to Chroma yet.

    Preparing first prevents an updated document's previous chunks
    from being deleted when its new parsing or embedding fails.
    """

    payload = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
        "records": []
    }

    failed_embeddings = 0

    for chunk_index, record in enumerate(
        records
    ):

        text = record.get(
            "text",
            ""
        )

        metadata = dict(
            record.get(
                "metadata",
                {}
            )
        )

        try:

            embedding = (
                embed_model.get_text_embedding(
                    f"passage: {text}"
                )
            )

        except Exception as error:

            print(
                f"[ERROR] Embedding failed "
                f"at chunk {chunk_index}: "
                f"{error}"
            )

            failed_embeddings += 1

            continue

        stored_record = {
            "text": text,
            "metadata": metadata
        }

        payload["ids"].append(
            build_record_id(
                stored_record
            )
        )

        payload["documents"].append(
            text
        )

        payload["metadatas"].append(
            metadata
        )

        payload["embeddings"].append(
            embedding
        )

        payload["records"].append(
            stored_record
        )

    return (
        payload,
        failed_embeddings
    )


def upsert_prepared_records(
    collection,
    payload
):

    """
    Store prepared records in small batches.
    """

    total_records = len(
        payload["ids"]
    )

    for start in range(
        0,
        total_records,
        CHROMA_WRITE_BATCH_SIZE
    ):

        end = start + (
            CHROMA_WRITE_BATCH_SIZE
        )

        collection.upsert(
            ids=payload["ids"][
                start:end
            ],
            documents=payload[
                "documents"
            ][start:end],
            embeddings=payload[
                "embeddings"
            ][start:end],
            metadatas=payload[
                "metadatas"
            ][start:end]
        )


def load_records_from_collection(
    collection
):

    """
    Read current Chroma documents and metadata for rebuilding BM25.
    No document parsing or embedding is performed.
    """

    if collection.count() == 0:

        return []

    result = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = (
        result.get("documents")
        or []
    )

    metadatas = (
        result.get("metadatas")
        or []
    )

    return [
        {
            "text": document,
            "metadata": metadata
        }
        for document, metadata in zip(
            documents,
            metadatas
        )
    ]


def build_index():

    """
    Perform a safe full rebuild.

    This remains available for the first migration, damaged indexes,
    or an intentional complete rebuild. Normal updates should use
    scripts.smart_build.smart_build().
    """

    separator(
        "BUILD STARTED"
    )

    documents = get_all_documents()

    print(
        f"[STEP 1] Loading "
        f"{len(documents)} documents..."
    )

    pipeline = IngestionPipeline()

    records = pipeline.run(
        documents
    )

    if not records:

        print(
            "[ERROR] No chunks generated. "
            "The existing index was not deleted."
        )

        return False

    print(
        "[STEP 2] Loading embedding model..."
    )

    embed_model = (
        get_embedding_model()
    )

    # Group records per file so a document is indexed only
    # when every one of its chunks embeds successfully.
    grouped_records = {}

    for record in records:

        file_path = (
            record.get(
                "metadata",
                {}
            ).get(
                "file_path",
                ""
            )
        )

        grouped_records.setdefault(
            file_path,
            []
        ).append(
            record
        )

    prepared_payloads = []
    indexed_documents = []

    for file_path, file_records in (
        grouped_records.items()
    ):

        payload, failed = (
            prepare_records_for_storage(
                file_records,
                embed_model
            )
        )

        if (
            failed > 0
            or len(payload["records"])
            != len(file_records)
        ):

            print(
                f"[FAILED DOCUMENT] "
                f"{Path(file_path).name}: "
                f"embedding incomplete"
            )

            continue

        prepared_payloads.append(
            payload
        )

        indexed_documents.append(
            Path(file_path)
        )

    if not prepared_payloads:

        print(
            "[ERROR] No complete document "
            "embeddings were generated. "
            "The existing index was not deleted."
        )

        return False

    print(
        "[STEP 3] Replacing Chroma index..."
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        client.delete_collection(
            CHROMA_COLLECTION_NAME
        )

    except Exception:

        pass

    collection = client.create_collection(
        CHROMA_COLLECTION_NAME
    )

    stored_records = []

    for payload in prepared_payloads:

        upsert_prepared_records(
            collection,
            payload
        )

        stored_records.extend(
            payload["records"]
        )

    print(
        f"[SUCCESS] Stored "
        f"{len(stored_records)} embeddings"
    )

    print(
        "[STEP 4] Building BM25 index..."
    )

    BM25Indexer().build(
        stored_records
    )

    manifest = ManifestManager.build(
        indexed_documents
    )

    ManifestManager.save(
        manifest
    )

    # Clear cached resources only after all index files are ready.
    from retrieval.chroma_search import (
        get_chroma_collection
    )

    from retrieval.bm25_index import (
        get_bm25_resources
    )

    get_chroma_collection.clear()
    get_bm25_resources.clear()

    print()
    print(
        "===== BUILD COMPLETE ====="
    )
    print(
        f"Documents discovered: "
        f"{len(documents)}"
    )
    print(
        f"Documents indexed: "
        f"{len(indexed_documents)}"
    )
    print(
        f"Total chunks: "
        f"{len(stored_records)}"
    )
    print(
        "=========================="
    )
    print()

    summary(
        Documents=len(
            indexed_documents
        ),
        Chunks=len(
            stored_records
        ),
        Embeddings=len(
            stored_records
        )
    )

    return True


if __name__ == "__main__":

    build_index()
