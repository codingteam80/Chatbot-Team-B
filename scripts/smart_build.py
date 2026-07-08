from utils.file_utils import get_all_documents
from utils.manifest import ManifestManager

from scripts.build_index import build_index
from retrieval.chroma_search import get_chroma_collection
from retrieval.bm25_index import get_bm25_resources

def check_changes():

    documents = get_all_documents()

    old_manifest = ManifestManager.load()

    new_manifest = ManifestManager.build(
        documents
    )

    added = []
    updated = []
    deleted = []

    for file_name, info in new_manifest.items():

        if file_name not in old_manifest:

            added.append(file_name)

        elif (
            info["hash"]
            !=
            old_manifest[file_name]["hash"]
        ):

            updated.append(file_name)

    for file_name in old_manifest:

        if file_name not in new_manifest:

            deleted.append(file_name)

    return (
        len(added) > 0
        or
        len(updated) > 0
        or
        len(deleted) > 0
    )

def smart_build():

    print("\n===== SMART BUILD =====\n")

    # Scan current documents
    documents = get_all_documents()

    print(
        f"Documents found: {len(documents)}"
    )

    # Load previous manifest
    old_manifest = (
        ManifestManager.load()
    )

    # Build current manifest
    new_manifest = (
        ManifestManager.build(
            documents
        )
    )

    # -------------------------------------
    # Compare
    # -------------------------------------

    added = []
    updated = []
    deleted = []

    # New / Updated
    for file_name, info in new_manifest.items():

        if file_name not in old_manifest:

            added.append(file_name)

        elif (

            info["hash"]

            !=

            old_manifest[file_name]["hash"]

        ):

            updated.append(file_name)

    # Deleted
    for file_name in old_manifest:

        if file_name not in new_manifest:

            deleted.append(file_name)

    # -------------------------------------
    # Summary
    # -------------------------------------

    print()

    print(f"Added   : {len(added)}")
    print(f"Updated : {len(updated)}")
    print(f"Deleted : {len(deleted)}")

    # -------------------------------------
    # No changes
    # -------------------------------------

    if (

        not added

        and

        not updated

        and

        not deleted

    ):

        print()

        print(
            "No changes detected."
        )

        print(
            "Index is already up-to-date."
        )

        return

    print()

    print(
        "Changes detected."
    )

    print(
        "Rebuilding indexes..."
    )

    print()

    # Build indexes
    build_index()

    # Save latest manifest
    ManifestManager.save(
        new_manifest
    )

    # Refresh cached indexes
    get_chroma_collection.clear()
    get_bm25_resources.clear()

    print()

    print(
        "Manifest updated."
    )

    print()

    print(
        "===== SMART BUILD COMPLETE ====="
    )


if __name__ == "__main__":

    smart_build()