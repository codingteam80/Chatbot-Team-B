from scripts.build_qdrant_index import build_qdrant_index


print("\n===== FULL KNOWLEDGE BASE REBUILD =====")
print(
    "This operation transactionally rebuilds the finalized Qdrant v4 + BM25 "
    "production knowledge base. The current working index is preserved until "
    "the replacement is ready."
)

success = build_qdrant_index()

if success:
    print("\n===== FULL REBUILD COMPLETE =====")
else:
    print("\n===== FULL REBUILD FAILED =====")
    print("The previous working index was preserved where recovery was possible.")
