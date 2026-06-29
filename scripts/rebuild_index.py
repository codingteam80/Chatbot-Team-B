import shutil

from config.settings import (
    CHROMA_DIR,
    BM25_DIR
)

from scripts.build_index import (
    main
)

# REBUILD INDEX SCRIPT ================================
# PURPOSE: Delete old indexes - Recreate ChromaDB index - Recreate BM25 index
# WHEN TO USE: Added new documents - Removed documents - Changed chunk size - Changed embedding model - Corrupted index
# =====================================================
print("\n===== REBUILD INDEX =====")

# STEP 1: DELETE CHROMA VECTOR DATABASE ===============
print("\n[STEP 1] Removing ChromaDB...")

try:

    shutil.rmtree(
        CHROMA_DIR
    )

    print(
        f"[SUCCESS] Deleted: {CHROMA_DIR}"
    )

except FileNotFoundError:

    print(
        "[DEBUG] ChromaDB folder not found"
    )

except Exception as e:

    print(
        f"[ERROR] Failed deleting ChromaDB: {e}"
    )

# STEP 2: DELETE BM25 INDEX ===========================
print("\n[STEP 2] Removing BM25 index...")

try:

    shutil.rmtree(
        BM25_DIR
    )

    print(
        f"[SUCCESS] Deleted: {BM25_DIR}"
    )

except FileNotFoundError:

    print(
        "[DEBUG] BM25 folder not found"
    )

except Exception as e:

    print(
        f"[ERROR] Failed deleting BM25 index: {e}"
    )

# STEP 3: REBUILD EVERYTHING ==========================
print("\n[STEP 3] Building fresh indexes...")

main()

# FINISHED ============================================
print("\n===== REBUILD COMPLETE =====")

print(
    """
        Done - Old ChromaDB removed
        Done - Old BM25 index removed
        Done - Documents reloaded
        Done - Chunks regenerated
        Done - Embeddings recreated
        Done - ChromaDB rebuilt
        Done - BM25 rebuilt
    """
)