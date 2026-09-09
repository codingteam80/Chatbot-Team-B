from scripts.build_index import build_index


print("\n===== FULL KNOWLEDGE BASE REBUILD =====")
print(
    "This operation rebuilds all source documents. "
    "The current working indexes are kept until the replacement is ready."
)

success = build_index()

if success:
    print("\n===== FULL REBUILD COMPLETE =====")
else:
    print("\n===== FULL REBUILD FAILED =====")
    print("The previous working index was preserved where recovery was possible.")
