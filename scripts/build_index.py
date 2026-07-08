from uuid import uuid4
import chromadb

from ingestion.ingest import IngestionPipeline
# PURPOSE:
# - loads raw documents
# - cleans + chunks text into "records"

from retrieval.bm25_index import BM25Indexer
# PURPOSE:
# - builds keyword search index (BM25)
# - separate system from vector DB

from embeddings.embedding_model import get_embedding_model
# PURPOSE:
# - converts text → embedding vectors

from config.settings import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME
)

from utils.logger import (
    separator,
    summary
)

def build_index():

    separator("BUILD STARTED")
   
    # STEP 1: LOAD + CHUNK DOCUMENTS ======================
    print("[STEP 1] Loading documents...")

    pipeline = IngestionPipeline()

    from utils.file_utils import get_all_documents

    document_count = len(
        get_all_documents()
    )

    records = pipeline.run()

    # DEBUG CHECKPOINT
    print(f"[DEBUG] Total chunks created: {len(records)}")
    # SAMPLE:
    # 10 PDFs → 500 chunks

    if len(records) == 0:
        print("[ERROR] No chunks generated. Check ingestion pipeline.")
        return

    # STEP 2: LOAD EMBEDDING MODEL ========================
    print("[STEP 2] Loading embedding model...")

    embed_model = get_embedding_model()

    print("[DEBUG] Embedding model ready")

    # STEP 3: INIT CHROMA DB ==============================
    print("[STEP 3] Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # RESET COLLECTION (SAFE REBUILD)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print("[DEBUG] Old collection deleted")
    except:
        print("[DEBUG] No existing collection to delete")

    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    print("[DEBUG] New collection created")

    # STEP 4: EMBEDDING PREPARATION =======================
    print("[STEP 4] Creating embeddings...")

    documents = []
    metadatas = []
    embeddings = []
    ids = []

    # DEBUG COUNTERS
    failed_embeddings = 0

    for i, record in enumerate(records):

        text = record["text"]

        # EMBEDDING STEP (CRITICAL) -----------------------
        try:
            emb = embed_model.get_text_embedding(
                f"passage: {text}"
            )
        except Exception as e:
            print(f"[ERROR] Embedding failed at chunk {i}: {e}")
            failed_embeddings += 1
            continue

        documents.append(text)
        metadatas.append(record["metadata"])
        embeddings.append(emb)
        ids.append(str(uuid4()))

        # DEBUG SAMPLE (first 2 chunks only)
        if i < 2:
            print(f"[DEBUG SAMPLE] Chunk {i}: {text[:80]}...")

    print(f"[DEBUG] Successful embeddings: {len(embeddings)}")
    print(f"[DEBUG] Failed embeddings: {failed_embeddings}")

    # SAFETY CHECK
    if len(embeddings) == 0:
        print("[ERROR] No embeddings generated. STOP.")
        return

    # STEP 5: STORE TO CHROMA VECTOR DB ===================
    print("[STEP 5] Storing to ChromaDB...")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("[SUCCESS] ChromaDB index completed")

    # STEP 6: BUILD BM25 INDEX ============================
    print("[STEP 6] Building BM25 index...")

    BM25Indexer().build(records)

    print("[SUCCESS] BM25 index completed")

    # FINAL STATUS ========================================
    print("\n===== BUILD COMPLETE =====")
    print(f"Total chunks: {len(records)}")
    print(f"Stored embeddings: {len(embeddings)}")
    print("==========================\n")

    summary(
        Documents=document_count,
        Chunks=len(records),
        Embeddings=len(embeddings)
    )

if __name__ == "__main__":
    build_index()